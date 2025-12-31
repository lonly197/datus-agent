# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Preflight Orchestrator for Text2SQL pipeline.

Coordinates execution of preflight tools, manages caching, emits events,
and injects results into workflow context for evidence-based SQL generation.
"""

import time
import uuid
from typing import Any, AsyncGenerator, Dict, List

from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionStatus
from datus.tools.func_tool.database import db_function_tool_instance
from datus.tools.func_tool.context_search import ContextSearchTools
from datus.tools.date_tools.date_parser import DateParserTool
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class PreflightToolResult:
    """Result of a preflight tool execution."""

    def __init__(self, tool_name: str, success: bool, result: Any = None,
                 error: str = None, execution_time: float = 0.0, cache_hit: bool = False):
        self.tool_name = tool_name
        self.success = success
        self.result = result
        self.error = error
        self.execution_time = execution_time
        self.cache_hit = cache_hit

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for context injection."""
        return {
            "tool_name": self.tool_name,
            "success": self.success,
            "result": self.result,
            "error": self.error,
            "execution_time": self.execution_time,
            "cache_hit": self.cache_hit
        }


class PreflightOrchestrator:
    """
    Orchestrates preflight tool execution for Text2SQL pipeline.

    Runs required tools in sequence, manages caching and monitoring,
    emits events, and injects results into workflow context.
    """

    def __init__(self, agent_config: Any, plan_hooks: Any = None):
        """
        Initialize the preflight orchestrator.

        Args:
            agent_config: Agent configuration
            plan_hooks: Plan hooks for caching and monitoring
        """
        self.agent_config = agent_config
        self.plan_hooks = plan_hooks
        self.db_func_tool = None
        self.context_search_tools = None
        self.date_parsing_tools = None

    def _get_db_func_tool(self):
        """Lazy initialization of DB function tool."""
        if self.db_func_tool is None:
            self.db_func_tool = db_function_tool_instance(
                self.agent_config,
                self.agent_config.current_database
            )
        return self.db_func_tool

    def _get_context_search_tools(self):
        """Lazy initialization of context search tools."""
        if self.context_search_tools is None:
            self.context_search_tools = ContextSearchTools()
        return self.context_search_tools

    def _get_date_parsing_tools(self):
        """Lazy initialization of date parsing tools."""
        if self.date_parsing_tools is None:
            self.date_parsing_tools = DateParserTool()
        return self.date_parsing_tools

    async def run_preflight_tools(
        self,
        workflow,
        action_history_manager: ActionHistoryManager,
        execution_id: str = None,
        required_tools: List[str] = None
    ) -> AsyncGenerator[ActionHistory, None]:
        """
        Run preflight tools and yield ActionHistory events.

        Args:
            workflow: Workflow instance
            action_history_manager: Manager for action history
            execution_id: Execution ID for event tracking
            required_tools: List of tool names to execute

        Yields:
            ActionHistory: Events for tool calls and results
        """
        if not required_tools:
            required_tools = ["search_table", "describe_table", "search_reference_sql", "parse_temporal_expressions"]

        # Extract SQL query and table info from workflow
        sql_query = self._extract_sql_from_workflow(workflow)
        table_names = self._extract_table_names_from_workflow(workflow)
        catalog = workflow.task.catalog_name or "default_catalog"
        database = workflow.task.database_name or ""
        schema = workflow.task.schema_name or ""

        logger.info(f"Starting preflight execution for {len(required_tools)} tools: {required_tools}")

        preflight_results = []

        for tool_name in required_tools:
            tool_call_id = str(uuid.uuid4())

            # Send tool call start event
            await self._send_tool_call_event(
                tool_name, tool_call_id, {
                    "sql_query": sql_query,
                    "table_names": table_names,
                    "catalog": catalog,
                    "database": database,
                    "schema": schema
                }, execution_id
            )

            # Create and yield action for tool start
            tool_action = ActionHistory.create_action(
                role=ActionRole.TOOL,
                action_type=f"preflight_{tool_name}",
                messages=f"Executing preflight tool: {tool_name}",
                input_data={
                    "tool_name": tool_name,
                    "sql_query": sql_query,
                    "table_names": table_names,
                    "catalog": catalog,
                    "database": database,
                    "schema": schema,
                },
                status=ActionStatus.PROCESSING,
            )
            action_history_manager.add_action(tool_action)
            yield tool_action

            # Execute tool with caching
            start_time = time.time()
            cache_hit = False
            result = None
            error = None

            try:
                # Check cache first
                if self.plan_hooks and self.plan_hooks.enable_query_caching:
                    cache_key_params = {
                        "catalog": catalog,
                        "database": database,
                        "schema": schema
                    }

                    # Add tool-specific parameters
                    if tool_name == "search_table":
                        cache_key_params["query"] = workflow.task.task[:200]
                    elif tool_name == "describe_table" and table_names:
                        cache_key_params["table_name"] = table_names[0]
                    elif tool_name == "search_reference_sql":
                        cache_key_params["query"] = workflow.task.task[:200]
                    elif tool_name == "parse_temporal_expressions":
                        cache_key_params["text"] = workflow.task.task

                    cached_result = self.plan_hooks.query_cache.get(tool_name, **cache_key_params)
                    if cached_result is not None:
                        logger.debug(f"Cache hit for {tool_name}")
                        cache_hit = True
                        result = cached_result
                    else:
                        logger.debug(f"Cache miss for {tool_name}")

                # Execute tool if not cached
                if not cache_hit:
                    result = await self._execute_preflight_tool(
                        tool_name, sql_query, table_names, catalog, database, schema
                    )

                    # Cache successful results
                    if result and result.get("success") and self.plan_hooks and self.plan_hooks.enable_query_caching:
                        cache_key_params = {
                            "catalog": catalog,
                            "database": database,
                            "schema": schema
                        }
                        if tool_name == "search_table":
                            cache_key_params["query"] = workflow.task.task[:200]
                        elif tool_name == "describe_table" and table_names:
                            cache_key_params["table_name"] = table_names[0]
                        elif tool_name == "search_reference_sql":
                            cache_key_params["query"] = workflow.task.task[:200]
                        elif tool_name == "parse_temporal_expressions":
                            cache_key_params["text"] = workflow.task.task

                        self.plan_hooks.query_cache.set(
                            tool_name, result, **cache_key_params)

            except Exception as e:
                logger.warning(f"Preflight tool {tool_name} failed: {e}")
                error = str(e)
                result = {"success": False, "error": error}

            execution_time = time.time() - start_time

            # Create result
            tool_result = PreflightToolResult(
                tool_name=tool_name,
                success=result.get("success", False) if result else False,
                result=result,
                error=error,
                execution_time=execution_time,
                cache_hit=cache_hit
            )
            preflight_results.append(tool_result)

            # Send tool call result event
            await self._send_tool_call_result_event(
                tool_call_id=tool_call_id,
                result=tool_result.to_dict(),
                execution_time=execution_time,
                cache_hit=cache_hit,
                execution_id=execution_id
            )

            # Create and yield result action
            result_action = ActionHistory.create_action(
                role=ActionRole.TOOL,
                action_type=f"preflight_{tool_name}_result",
                messages=f"Preflight tool {tool_name} completed ({'cached' if cache_hit else 'executed'})",
                input_data=tool_result.to_dict(),
                output_data=tool_result.to_dict(),
                status=ActionStatus.COMPLETED if tool_result.success else ActionStatus.FAILED,
            )
            action_history_manager.add_action(result_action)
            yield result_action

            # Update monitoring
            if self.plan_hooks and hasattr(self.plan_hooks, 'monitor'):
                self.plan_hooks.monitor.record_preflight_tool_call(
                    execution_id, tool_name, tool_result.success, cache_hit, execution_time, error
                )

        # Inject results into workflow context
        self._inject_preflight_results_into_context(workflow, preflight_results)

        logger.info(f"Preflight execution completed for {len(preflight_results)} tools")

    async def _execute_preflight_tool(
        self, tool_name: str, sql_query: str, table_names: List[str],
        catalog: str, database: str, schema: str
    ) -> Dict[str, Any]:
        """Execute a specific preflight tool."""
        try:
            if tool_name == "search_table":
                return await self._execute_search_table(sql_query, catalog, database, schema)
            elif tool_name == "describe_table":
                return await self._execute_describe_table(table_names, catalog, database, schema)
            elif tool_name == "search_reference_sql":
                return await self._execute_search_reference_sql(sql_query)
            elif tool_name == "parse_temporal_expressions":
                return await self._execute_parse_temporal_expressions(sql_query)
            else:
                return {"success": False, "error": f"Unknown tool: {tool_name}"}
        except Exception as e:
            logger.error(f"Error executing preflight tool {tool_name}: {e}")
            return {"success": False, "error": str(e)}

    async def _execute_search_table(self, query: str, catalog: str, database: str, schema: str) -> Dict[str, Any]:
        """Execute table search based on query intent."""
        db_tool = self._get_db_func_tool()
        try:
            # Use semantic search to find relevant tables
            result = db_tool.search_table(
                query=query,
                catalog=catalog,
                database=database,
                schema_name=schema,
                limit=10
            )
            return {
                "success": True,
                "tables": result.result if hasattr(result, 'result') else [],
                "count": len(result.result) if hasattr(result, 'result') else 0
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _execute_describe_table(
        self, table_names: List[str], catalog: str, database: str, schema: str
    ) -> Dict[str, Any]:
        """Execute table description for given tables."""
        if not table_names:
            return {"success": False, "error": "No table names provided"}

        db_tool = self._get_db_func_tool()
        results = []

        for table_name in table_names[:3]:  # Limit to first 3 tables
            try:
                result = db_tool.describe_table(
                    table_name=table_name,
                    catalog=catalog,
                    database=database,
                    schema_name=schema
                )
                if hasattr(result, 'result') and result.result:
                    results.append({
                        "table_name": table_name,
                        "schema": result.result
                    })
            except Exception as e:
                logger.warning(f"Failed to describe table {table_name}: {e}")

        return {
            "success": len(results) > 0,
            "tables_described": results,
            "count": len(results)
        }

    async def _execute_search_reference_sql(self, query: str) -> Dict[str, Any]:
        """Execute reference SQL search."""
        search_tools = self._get_context_search_tools()
        try:
            result = search_tools.search_reference_sql(
                query_text=query,
                domain="general",
                layer1="queries",
                layer2="examples",
                top_n=5
            )
            return {
                "success": True,
                "reference_sqls": result.result if hasattr(result, 'result') else [],
                "count": len(result.result) if hasattr(result, 'result') else 0
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _execute_parse_temporal_expressions(self, text: str) -> Dict[str, Any]:
        """Execute temporal expression parsing."""
        date_tools = self._get_date_parsing_tools()
        try:
            result = date_tools.extract_and_parse_dates(text)
            return {
                "success": True,
                "temporal_expressions": result.result if hasattr(result, 'result') else [],
                "count": len(result.result) if hasattr(result, 'result') else 0
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _extract_sql_from_workflow(self, workflow) -> str:
        """Extract SQL query from workflow task."""
        if hasattr(workflow, 'task') and workflow.task:
            return getattr(workflow.task, 'task', '')
        return ''

    def _extract_table_names_from_workflow(self, workflow) -> List[str]:
        """Extract table names from workflow (placeholder implementation)."""
        # This would need to be enhanced to actually parse table names from SQL
        # For now, return empty list
        return []

    def _inject_preflight_results_into_context(self, workflow, results: List[PreflightToolResult]):
        """Inject preflight results into workflow context for prompt access."""
        if not hasattr(workflow, 'context'):
            workflow.context = type('Context', (), {})()

        # Initialize preflight_results if not exists
        if not hasattr(workflow.context, 'preflight_results'):
            workflow.context.preflight_results = []

        # Convert results to dict format for template access
        workflow.context.preflight_results = [result.to_dict() for result in results]

        # Extract specific data for template convenience
        self._extract_structured_data_for_template(workflow, results)

    def _extract_structured_data_for_template(self, workflow, results: List[PreflightToolResult]):
        """Extract structured data from results for template access."""
        # Schema info from describe_table
        schema_info = []
        for result in results:
            if result.tool_name == "describe_table" and result.success and result.result:
                tables = result.result.get("tables_described", [])
                for table in tables:
                    schema_info.append(f"Table: {table['table_name']}\n{table['schema']}")

        if schema_info:
            workflow.context.schema_info = "\n\n".join(schema_info)

        # Reference SQL from search_reference_sql
        reference_sqls = []
        for result in results:
            if result.tool_name == "search_reference_sql" and result.success and result.result:
                sqls = result.result.get("reference_sqls", [])
                reference_sqls.extend(sqls)

        if reference_sqls:
            workflow.context.reference_sqls = reference_sqls

        # Temporal expressions from parse_temporal_expressions
        temporal_expressions = []
        for result in results:
            if result.tool_name == "parse_temporal_expressions" and result.success and result.result:
                expressions = result.result.get("temporal_expressions", [])
                temporal_expressions.extend(expressions)

        if temporal_expressions:
            workflow.context.temporal_expressions = temporal_expressions

    async def _send_tool_call_event(
        self, tool_name: str, tool_call_id: str, input_data: Dict[str, Any], execution_id: str = None
    ):
        """Send tool call start event."""
        # This would integrate with execution_event_manager if available
        logger.debug(f"Tool call started: {tool_name} ({tool_call_id})")

    async def _send_tool_call_result_event(self, tool_call_id: str, result: Dict[str, Any],
                                           execution_time: float, cache_hit: bool,
                                           execution_id: str = None):
        """Send tool call result event."""
        # This would integrate with execution_event_manager if available
        logger.debug(f"Tool call completed: {tool_call_id} (cache_hit: {cache_hit}, time: {execution_time:.2f}s)")
