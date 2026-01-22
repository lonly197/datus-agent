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
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, List, Optional

from datus.schemas.action_history import (ActionHistory, ActionHistoryManager,
                                          ActionRole, ActionStatus)
from datus.tools.date_tools.date_parser import DateParserTool
from datus.tools.func_tool.context_search import ContextSearchTools
from datus.tools.func_tool.database import db_function_tool_instance
from datus.utils.loggings import get_logger

if TYPE_CHECKING:
    from datus.agent.workflow import Workflow

logger = get_logger(__name__)


class PreflightToolResult:
    """Result of a preflight tool execution."""

    def __init__(
        self,
        tool_name: str,
        success: bool,
        result: Any = None,
        error: Optional[str] = None,
        execution_time: float = 0.0,
        cache_hit: bool = False,
    ):
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
            "cache_hit": self.cache_hit,
        }


class PreflightOrchestrator:
    """
    Orchestrates preflight tool execution for Text2SQL pipeline.

    Runs required tools in sequence, manages caching and monitoring,
    emits events, and injects results into workflow context.
    """

    def __init__(self, agent_config: Any, plan_hooks: Any = None, execution_event_manager: Any = None):
        """
        Initialize the preflight orchestrator.

        Args:
            agent_config: Agent configuration
            plan_hooks: Plan hooks for caching and monitoring
            execution_event_manager: Event manager for sending events
        """
        self.agent_config = agent_config
        self.plan_hooks = plan_hooks
        self.execution_event_manager = execution_event_manager
        self.db_func_tool = None
        self.context_search_tools = None
        self.date_parsing_tools = None

    def _get_db_func_tool(self) -> Any:
        """Lazy initialization of DB function tool."""
        if self.db_func_tool is None:
            self.db_func_tool = db_function_tool_instance(self.agent_config, self.agent_config.current_database)
        return self.db_func_tool

    def _get_context_search_tools(self) -> ContextSearchTools:
        """Lazy initialization of context search tools."""
        if self.context_search_tools is None:
            self.context_search_tools = ContextSearchTools(self.agent_config)
        return self.context_search_tools

    def _get_date_parsing_tools(self) -> DateParserTool:
        """Lazy initialization of date parsing tools."""
        if self.date_parsing_tools is None:
            self.date_parsing_tools = DateParserTool()
        return self.date_parsing_tools

    async def run_preflight_tools(
        self,
        workflow: "Workflow",
        action_history_manager: ActionHistoryManager,
        execution_id: Optional[str] = None,
        required_tools: Optional[List[str]] = None,
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

        catalog = "default_catalog"
        database = ""
        schema = ""

        if workflow.task:
            catalog = workflow.task.catalog_name or "default_catalog"
            database = workflow.task.database_name or ""
            schema = workflow.task.schema_name or ""

        # Get continue_on_failure setting from agent config (default: true for text2sql)
        continue_on_failure = True  # Default to continue on failure
        if self.agent_config and hasattr(self.agent_config, "scenarios"):
            text2sql_config = self.agent_config.scenarios.get("text2sql", {})
            continue_on_failure = text2sql_config.get("continue_on_failure", True)

        # Get tool timeout from agent config (default: 30 seconds)
        tool_timeout = 30.0
        if self.agent_config and hasattr(self.agent_config, "scenarios"):
            text2sql_config = self.agent_config.scenarios.get("text2sql", {})
            tool_timeout = text2sql_config.get("tool_timeout_seconds", 30.0)

        # Ensure plan_hooks are available for caching support
        if not self.plan_hooks:
            logger.warning("Plan hooks not provided, creating ephemeral plan_hooks for basic functionality")
            # Create minimal plan_hooks for caching support
            try:
                from datus.cli.plan_hooks import PlanModeHooks

                # Create a basic plan_hooks instance (may not have full functionality)
                self.plan_hooks = PlanModeHooks.__new__(PlanModeHooks)
                self.plan_hooks.enable_query_caching = False  # Disable caching without full setup
                self.plan_hooks.enable_batch_processing = False
                logger.info("Created ephemeral plan_hooks with limited functionality")
            except Exception as e:
                logger.warning(f"Failed to create ephemeral plan_hooks: {e}")

        logger.info(f"Starting preflight execution for {len(required_tools)} tools: {required_tools}")
        logger.info(f"Continue on failure: {continue_on_failure}, Tool timeout: {tool_timeout}s")

        preflight_results: List[PreflightToolResult] = []

        for tool_name in required_tools:
            tool_call_id = str(uuid.uuid4())

            # Send tool call start event
            await self._send_tool_call_event(
                tool_name,
                tool_call_id,
                {
                    "sql_query": sql_query,
                    "table_names": table_names,
                    "catalog": catalog,
                    "database": database,
                    "schema": schema,
                },
                execution_id,
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

            # Execute tool with caching and timeout
            start_time = time.time()
            cache_hit = False
            result = None
            error = None
            timeout_occurred = False

            try:
                # Check cache first (with error handling)
                cache_key_params = self._generate_cache_key(tool_name, workflow, catalog, database, schema, table_names)
                cache_check_start = time.time()

                if self.plan_hooks and getattr(self.plan_hooks, "enable_query_caching", False):
                    try:
                        if hasattr(self.plan_hooks, "query_cache"):
                            cached_result = self.plan_hooks.query_cache.get(tool_name, **cache_key_params)
                            if cached_result is not None:
                                logger.debug(
                                    f"Cache hit for {tool_name} (cache check took {time.time() - cache_check_start:.3f}s)"
                                )
                                cache_hit = True
                                result = cached_result
                            else:
                                logger.debug(
                                    f"Cache miss for {tool_name} (cache check took {time.time() - cache_check_start:.3f}s)"
                                )
                    except Exception as cache_error:
                        logger.warning(
                            f"Cache operation failed for {tool_name}: {cache_error}, proceeding without cache"
                        )
                        # Continue without cache - don't fail the tool execution

                # Execute tool if not cached (with timeout)
                if not cache_hit:
                    import asyncio

                    try:
                        tool_exec_start = time.time()
                        result = await asyncio.wait_for(
                            self._execute_preflight_tool(tool_name, sql_query, table_names, catalog, database, schema),
                            timeout=tool_timeout,
                        )
                        logger.debug(f"Tool {tool_name} executed successfully in {time.time() - tool_exec_start:.3f}s")

                        # Cache successful results (with error handling)
                        if (
                            result
                            and result.get("success")
                            and self.plan_hooks
                            and getattr(self.plan_hooks, "enable_query_caching", False)
                        ):
                            try:
                                cache_set_start = time.time()
                                if hasattr(self.plan_hooks, "query_cache"):
                                    self.plan_hooks.query_cache.set(tool_name, result, **cache_key_params)
                                    logger.debug(
                                        f"Successfully cached result for {tool_name} (cache set took {time.time() - cache_set_start:.3f}s)"
                                    )
                            except Exception as cache_error:
                                logger.warning(
                                    f"Failed to cache result for {tool_name}: {cache_error}, continuing without caching"
                                )

                    except asyncio.TimeoutError:
                        timeout_occurred = True
                        error = f"Tool execution timed out after {tool_timeout} seconds"
                        logger.warning(f"Preflight tool {tool_name} timed out")
                        result = {"success": False, "error": error}

            except Exception as e:
                logger.warning(f"Preflight tool {tool_name} failed: {e}")
                error = str(e)
                if not timeout_occurred:  # Don't overwrite timeout error
                    result = {"success": False, "error": error}

            execution_time = time.time() - start_time

            # Create result
            tool_result = PreflightToolResult(
                tool_name=tool_name,
                success=result.get("success", False) if result else False,
                result=result,
                error=error,
                execution_time=execution_time,
                cache_hit=cache_hit,
            )
            preflight_results.append(tool_result)

            # Send tool call result event
            await self._send_tool_call_result_event(
                tool_call_id=tool_call_id,
                result=tool_result.to_dict(),
                execution_time=execution_time,
                cache_hit=cache_hit,
                execution_id=execution_id,
            )

            # Create and yield result action
            result_action = ActionHistory.create_action(
                role=ActionRole.TOOL,
                action_type=f"preflight_{tool_name}_result",
                messages=f"Preflight tool {tool_name} completed ({'cached' if cache_hit else 'executed'})",
                input_data=tool_result.to_dict(),
                output_data=tool_result.to_dict(),
                status=ActionStatus.SUCCESS if tool_result.success else ActionStatus.FAILED,
            )
            action_history_manager.add_action(result_action)
            yield result_action

            # Update monitoring
            if self.plan_hooks and hasattr(self.plan_hooks, "monitor"):
                self.plan_hooks.monitor.record_preflight_tool_call(
                    execution_id, tool_name, tool_result.success, cache_hit, execution_time, error
                )

            # Check continue_on_failure policy
            if not tool_result.success and not continue_on_failure:
                logger.warning(
                    f"Preflight tool {tool_name} failed and continue_on_failure is False, stopping execution"
                )
                break

        # Inject results into workflow context
        self._inject_preflight_results_into_context(workflow, preflight_results)

        # Log final summary with performance metrics
        successful_tools = len([r for r in preflight_results if r.success])
        total_execution_time = sum(r.execution_time for r in preflight_results)
        cache_hits = len([r for r in preflight_results if r.cache_hit])

        logger.info(
            f"Preflight execution completed: {successful_tools}/{len(preflight_results)} tools successful, "
            f"{cache_hits} cache hits, total time: {total_execution_time:.2f}s"
        )

        # Health check summary
        health_status = self._generate_health_summary(preflight_results)
        logger.info(f"Preflight health status: {health_status}")

    def _generate_health_summary(self, preflight_results: List[PreflightToolResult]) -> str:
        """Generate a health summary for monitoring."""
        if not preflight_results:
            return "no_tools_executed"

        successful = len([r for r in preflight_results if r.success])
        total = len(preflight_results)

        if successful == total:
            return "all_tools_healthy"
        elif successful > 0:
            return f"partial_success_{successful}/{total}"
        else:
            return "all_tools_failed"

    async def _execute_preflight_tool(
        self, tool_name: str, sql_query: str, table_names: List[str], catalog: str, database: str, schema: str
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
            # Extract potential table names from query first
            potential_tables = self._extract_potential_table_names(query)
            if potential_tables:
                # If we found potential table names, try to search for them specifically
                search_results = []
                for table_name in potential_tables[:3]:  # Limit to first 3 potential tables
                    try:
                        # Try to get table info
                        result = db_tool.describe_table(
                            table_name=table_name, catalog=catalog, database=database, schema_name=schema
                        )
                        if hasattr(result, "success") and result.success:
                            search_results.append(
                                {
                                    "table_name": table_name,
                                    "exists": True,
                                    "schema": result.result if hasattr(result, "result") else None,
                                }
                            )
                        else:
                            search_results.append(
                                {
                                    "table_name": table_name,
                                    "exists": False,
                                    "error": result.error if hasattr(result, "error") else "Table not found",
                                }
                            )
                    except Exception as e:
                        search_results.append({"table_name": table_name, "exists": False, "error": str(e)})

                return {
                    "success": len([r for r in search_results if r.get("exists")]) > 0,
                    "tables": search_results,
                    "count": len(search_results),
                    "search_method": "direct_table_lookup",
                }

            # Fallback to semantic search
            result = db_tool.search_table(query=query, catalog=catalog, database=database, schema_name=schema, top_n=10)
            return {
                "success": True,
                "tables": result.result if hasattr(result, "result") else [],
                "count": len(result.result) if hasattr(result, "result") else 0,
                "search_method": "semantic_search",
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
                    table_name=table_name, catalog=catalog, database=database, schema_name=schema
                )
                if hasattr(result, "result") and result.result:
                    results.append({"table_name": table_name, "schema": result.result})
            except Exception as e:
                logger.warning(f"Failed to describe table {table_name}: {e}")

        return {"success": len(results) > 0, "tables_described": results, "count": len(results)}

    async def _execute_search_reference_sql(self, query: str) -> Dict[str, Any]:
        """Execute reference SQL search."""
        search_tools = self._get_context_search_tools()
        try:
            result = search_tools.search_reference_sql(
                query_text=query, domain="general", layer1="queries", layer2="examples", top_n=5
            )
            return {
                "success": True,
                "reference_sqls": result.result if hasattr(result, "result") else [],
                "count": len(result.result) if hasattr(result, "result") else 0,
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
                "temporal_expressions": result.result if hasattr(result, "result") else [],
                "count": len(result.result) if hasattr(result, "result") else 0,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _extract_sql_from_workflow(self, workflow: "Workflow") -> str:
        """Extract SQL query from workflow task."""
        if hasattr(workflow, "task") and workflow.task:
            return getattr(workflow.task, "task", "")
        return ""

    def _generate_cache_key(
        self,
        tool_name: str,
        workflow: "Workflow",
        catalog: str,
        database: str,
        schema: str,
        table_names: List[str],
    ) -> Dict[str, Any]:
        """Generate consistent cache key parameters for a tool."""
        # Base parameters that are always included
        cache_key = {"catalog": catalog, "database": database, "schema": schema}

        task_content = ""
        if workflow.task:
            task_content = getattr(workflow.task, "task", "")

        # Add tool-specific parameters
        if tool_name == "search_table":
            # Use query content for search_table
            cache_key["query"] = task_content[:200]
        elif tool_name == "describe_table" and table_names:
            # Use first table name for describe_table
            cache_key["table_name"] = table_names[0]
        elif tool_name == "search_reference_sql":
            # Use query content for reference SQL search
            cache_key["query"] = task_content[:200]
        elif tool_name == "parse_temporal_expressions":
            # Use full text for temporal parsing
            cache_key["text"] = task_content

        return cache_key

    def _extract_potential_table_names(self, query: str) -> List[str]:
        """Extract potential table names from natural language query."""
        import re

        # Common table name patterns in Chinese business context
        table_patterns = [
            r"\b([a-zA-Z_][a-zA-Z0-9_]*_(?:fact|dim|dws|dwd|ads|dm|tmp))\b",  # Data warehouse patterns
            r"\b(订单|用户|客户|商品|销售|库存|物流|财务|报表)\b",  # Business entity names
            r"\b(table|tbl)_([a-zA-Z_][a-zA-Z0-9_]*)\b",  # Generic table prefixes
        ]

        potential_tables = []
        query_lower = query.lower()

        for pattern in table_patterns:
            matches = re.findall(pattern, query_lower, re.IGNORECASE)
            if matches:
                # Flatten matches if they're tuples (from capture groups)
                for match in matches:
                    if isinstance(match, tuple):
                        potential_tables.extend([m for m in match if m])
                    else:
                        potential_tables.append(match)

        # Remove duplicates and filter
        unique_tables = list(set(potential_tables))
        # Filter out very short or generic names
        filtered_tables = [t for t in unique_tables if len(t) > 2 and not t.isdigit()]

        return filtered_tables

    def _extract_table_names_from_workflow(self, workflow: "Workflow") -> List[str]:
        """Extract table names from workflow."""
        sql_query = self._extract_sql_from_workflow(workflow)
        if sql_query:
            return self._extract_potential_table_names(sql_query)
        return []

    def _inject_preflight_results_into_context(self, workflow: "Workflow", results: List[PreflightToolResult]) -> None:
        """Inject preflight results into workflow context for prompt access."""
        if not hasattr(workflow, "context"):
            # Create a basic context object if missing
            class Context:
                pass

            workflow.context = Context()  # type: ignore

        # Initialize preflight_results if not exists
        if not hasattr(workflow.context, "preflight_results"):
            workflow.context.preflight_results = []

        # Convert results to dict format for template access
        workflow.context.preflight_results = [result.to_dict() for result in results]

        # Extract structured data for Text2SQL template
        self._extract_structured_data_for_text2sql_template(workflow, results)

    def _extract_structured_data_for_text2sql_template(
        self, workflow: "Workflow", results: List[PreflightToolResult]
    ) -> None:
        """Extract structured data from preflight results for Text2SQL template access."""
        # 1. Available tables and schema information
        available_tables = []
        schema_info = []

        # From search_table results
        for result in results:
            if result.tool_name == "search_table" and result.success and result.result:
                tables = result.result.get("tables", [])
                if isinstance(tables, list):
                    for table in tables:
                        if isinstance(table, dict) and table.get("exists", True):
                            available_tables.append(table.get("table_name", ""))
                            if "schema" in table:
                                schema_info.append(f"Table: {table['table_name']}\n{table['schema']}")

        # From describe_table results (more detailed schema)
        for result in results:
            if result.tool_name == "describe_table" and result.success and result.result:
                tables = result.result.get("tables_described", [])
                for table in tables:
                    table_name = table.get("table_name", "")
                    if table_name and table_name not in available_tables:
                        available_tables.append(table_name)
                    schema_str = table.get("schema", "")
                    if schema_str:
                        schema_info.append(f"Table: {table_name}\n{schema_str}")

        # Store available tables for template
        if available_tables:
            workflow.context.available_tables = available_tables
            workflow.context.schema_info = "\n\n".join(schema_info) if schema_info else ""

        # 2. Reference SQL examples
        reference_sqls = []
        for result in results:
            if result.tool_name == "search_reference_sql" and result.success and result.result:
                sqls = result.result.get("reference_sqls", [])
                if isinstance(sqls, list):
                    reference_sqls.extend(sqls)

        if reference_sqls:
            workflow.context.reference_sql_examples = reference_sqls

        # 3. Temporal expressions and date parsing results
        temporal_info = {}
        for result in results:
            if result.tool_name == "parse_temporal_expressions" and result.success and result.result:
                expressions = result.result.get("temporal_expressions", [])
                if expressions:
                    temporal_info["expressions"] = expressions
                    # Extract any parsed dates
                    parsed_dates = []
                    for expr in expressions:
                        if isinstance(expr, dict) and "parsed_date" in expr:
                            parsed_dates.append(expr["parsed_date"])
                    if parsed_dates:
                        temporal_info["parsed_dates"] = parsed_dates

        if temporal_info:
            workflow.context.temporal_info = temporal_info

        # 4. Query intent hints from search results
        query_intent_hints = []
        for result in results:
            if result.tool_name == "search_table" and result.success and result.result:
                search_method = result.result.get("search_method", "")
                if search_method == "direct_table_lookup":
                    query_intent_hints.append("Direct table references found in query")
                elif search_method == "semantic_search":
                    query_intent_hints.append("Semantic search used to find relevant tables")

        if query_intent_hints:
            workflow.context.query_intent_hints = query_intent_hints

        # 5. Success/failure summary for template
        success_summary = {
            "total_tools": len(results),
            "successful_tools": len([r for r in results if r.success]),
            "failed_tools": len([r for r in results if not r.success]),
            "cache_hits": len([r for r in results if r.cache_hit]),
        }
        workflow.context.preflight_summary = success_summary

        logger.info(
            f"Injected preflight context: {success_summary['successful_tools']}/{success_summary['total_tools']} tools successful"
        )

    async def _send_tool_call_event(
        self, tool_name: str, tool_call_id: str, input_data: Dict[str, Any], execution_id: Optional[str] = None
    ) -> None:
        """Send tool call start event."""
        if self.execution_event_manager and execution_id:
            # Send tool call event through execution event manager
            await self.execution_event_manager.record_tool_execution(
                execution_id=execution_id,
                tool_name=f"preflight_{tool_name}",
                tool_call_id=tool_call_id,
                input_data=input_data,
                execution_time=0.0,  # Will be updated on completion
            )
        else:
            logger.debug(f"Tool call started: {tool_name} ({tool_call_id})")

    async def _send_tool_call_result_event(
        self,
        tool_call_id: str,
        result: Dict[str, Any],
        execution_time: float,
        cache_hit: bool,
        execution_id: Optional[str] = None,
    ) -> None:
        """Send tool call result event."""
        if self.execution_event_manager and execution_id:
            # Update the tool execution record with results
            tool_name = result.get("tool_name", "unknown_tool")

            # Send completion event
            await self.execution_event_manager.record_tool_execution(
                execution_id=execution_id,
                tool_name=f"preflight_{tool_name}_result",
                tool_call_id=f"{tool_call_id}_result",
                input_data={"tool_call_id": tool_call_id, "cache_hit": cache_hit, "execution_time": execution_time},
                result=result,
                execution_time=execution_time,
            )
        else:
            logger.debug(f"Tool call completed: {tool_call_id} (cache_hit: {cache_hit}, time: {execution_time:.2f}s)")
