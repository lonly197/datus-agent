# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
SQL Review Preflight Orchestrator (v2.4)

Unified scheduler for SQL review preflight tools, coordinating execution of
both legacy and enhanced tools with caching, batching, and event streaming.
"""

import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

from datus.agent.node.execution_event_manager import ExecutionEventManager
from datus.agent.node.preflight_orchestrator import PreflightToolResult
from datus.cli.plan_hooks import PlanModeHooks
from datus.configuration.agent_config import AgentConfig
from datus.schemas.action_history import (ActionHistory, ActionHistoryManager,
                                          ActionRole, ActionStatus)
from datus.tools.func_tool.context_search import ContextSearchTools
from datus.tools.func_tool.database import DBFuncTool
from datus.tools.func_tool.enhanced_preflight_tools import \
    EnhancedPreflightTools
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class SQLReviewPreflightOrchestrator:
    """
    Unified orchestrator for SQL review preflight tools (v2.4).

    Coordinates execution of 7 preflight tools:
    - Legacy: describe_table, search_external_knowledge, read_query, get_table_ddl
    - Enhanced: analyze_query_plan, check_table_conflicts, validate_partitioning
    """

    def __init__(
        self,
        agent_config: AgentConfig,
        plan_hooks: Optional[PlanModeHooks] = None,
        execution_event_manager: Optional[ExecutionEventManager] = None,
    ):
        """Initialize the SQL review preflight orchestrator."""
        self.agent_config = agent_config
        self.plan_hooks = plan_hooks
        self.execution_event_manager = execution_event_manager

        # Initialize tool instances
        self.db_func_tool = DBFuncTool(agent_config)
        self.context_search_tools = ContextSearchTools(agent_config)
        self.enhanced_tools = EnhancedPreflightTools(agent_config)

        # Tool classification
        self.legacy_tools = {"describe_table", "search_external_knowledge", "read_query", "get_table_ddl"}
        self.enhanced_tools_set = {"analyze_query_plan", "check_table_conflicts", "validate_partitioning"}

        # Critical tools must succeed for review to continue; auxiliary tools are optional
        self.critical_tools = {"describe_table", "search_external_knowledge"}
        self.auxiliary_tools = {
            "read_query",
            "get_table_ddl",
            "analyze_query_plan",
            "check_table_conflicts",
            "validate_partitioning",
        }

        # Tool execution order (mandatory sequence)
        self.required_tool_sequence = [
            "describe_table",  # 1. Table structure analysis
            "search_external_knowledge",  # 2. StarRocks rules retrieval
            "read_query",  # 3. SQL syntax validation
            "get_table_ddl",  # 4. DDL definition retrieval
            "analyze_query_plan",  # 5. Query plan analysis (enhanced)
            "check_table_conflicts",  # 6. Table conflict detection (enhanced)
            "validate_partitioning",  # 7. Partition validation (enhanced)
        ]

    async def run_preflight_tools(
        self,
        workflow,
        action_history_manager: ActionHistoryManager,
        execution_id: str = None,
        required_tools: List[str] = None,
    ) -> AsyncGenerator[ActionHistory, None]:
        """
        Execute SQL review preflight tools with enhanced orchestration.

        Args:
            workflow: Workflow instance
            action_history_manager: Manager for action history
            execution_id: Execution ID for tracking
            required_tools: List of tools to execute (uses default sequence if None)

        Yields:
            ActionHistory: Events for tool calls and results
        """
        if not required_tools:
            required_tools = self.required_tool_sequence.copy()

        if not execution_id:
            execution_id = f"sql_review_preflight_{int(time.time() * 1000)}"

        logger.info(f"Starting SQL review preflight execution for {len(required_tools)} tools: {required_tools}")

        # Extract context from workflow
        sql_query = self._extract_sql_from_workflow(workflow)
        table_names = self._extract_table_names_from_workflow(workflow)
        catalog = workflow.task.catalog_name or "default_catalog"
        database = workflow.task.database_name or ""
        schema = workflow.task.schema_name or ""

        # ========== SQL Pre-Validation ==========
        # Validate SQL before executing tools to catch parsing errors early
        from datus.utils.sql_utils import validate_and_suggest_sql_fixes

        logger.info(f"Validating SQL query before preflight tool execution")
        sql_validation = validate_and_suggest_sql_fixes(sql_query, dialect=workflow.task.database_type or "starrocks")

        # Inject validation results into context
        if not hasattr(workflow.context, "preflight_results") or workflow.context.preflight_results is None:
            workflow.context.preflight_results = {}
        workflow.context.preflight_results["sql_validation"] = sql_validation

        # Log validation results
        if not sql_validation["is_valid"] or not sql_validation["can_parse"]:
            logger.warning(
                f"SQL validation failed - can_parse={sql_validation['can_parse']}, "
                f"is_valid={sql_validation['is_valid']}. "
                f"Errors: {sql_validation['errors']}, "
                f"Warnings: {len(sql_validation['warnings'])}, "
                f"Fix suggestions: {len(sql_validation['fix_suggestions'])}"
            )

        # Update execution status with validation results
        execution_status = {
            "execution_id": execution_id,
            "tools_executed": [],
            "tools_failed": [],
            "start_time": time.time(),
            "cache_hits": 0,
            "total_execution_time": 0.0,
            "sql_validation_passed": sql_validation["is_valid"] and sql_validation["can_parse"],
            "sql_errors": sql_validation["errors"],
            "sql_warnings_count": len(sql_validation["warnings"]),
            "sql_fix_suggestions_count": len(sql_validation["fix_suggestions"]),
        }

        # Send SQL validation event if there are issues
        if sql_validation["fix_suggestions"]:
            logger.info(f"Generated {len(sql_validation['fix_suggestions'])} SQL fix suggestions")
            # TODO: Create and send SQL validation event here if needed

        # Initialize execution tracking (continued)

        # Execute tools in sequence
        for tool_name in required_tools:
            tool_start_time = time.time()
            tool_call_id = str(uuid.uuid4())

            try:
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

                # Execute tool based on type
                result = await self._execute_tool(tool_name, sql_query, table_names, catalog, database, schema)

                execution_time = time.time() - tool_start_time
                cache_hit = result.get("cache_hit", False) if isinstance(result, dict) else False

                # Track execution
                execution_status["tools_executed"].append(
                    {
                        "tool_name": tool_name,
                        "success": result.success if hasattr(result, "success") else False,
                        "execution_time": execution_time,
                        "cache_hit": cache_hit,
                    }
                )

                if cache_hit:
                    execution_status["cache_hits"] += 1

                execution_status["total_execution_time"] += execution_time

                # Send tool result event
                await self._send_tool_result_event(
                    tool_call_id=tool_call_id,
                    result=result,
                    execution_time=execution_time,
                    cache_hit=cache_hit,
                    execution_id=execution_id,
                )

                # Inject result into workflow context
                self._inject_tool_result_into_context(workflow, tool_name, result)

                # Yield action history
                yield self._create_tool_action(tool_name, result, execution_time, tool_call_id)

            except Exception as e:
                execution_time = time.time() - tool_start_time
                logger.error(f"Preflight tool '{tool_name}' failed: {e}")

                # Check if this is a critical tool failure
                is_critical = tool_name in self.critical_tools

                execution_status["tools_failed"].append(
                    {
                        "tool_name": tool_name,
                        "error": str(e),
                        "execution_time": execution_time,
                        "is_critical": is_critical,
                    }
                )

                # Send error event
                await self._send_tool_error_event(tool_name, str(e), execution_id, tool_call_id)

                # For critical tools, mark execution as degraded
                if is_critical:
                    execution_status["critical_failure"] = True
                    execution_status["critical_failure_reason"] = f"{tool_name}: {str(e)}"
                    logger.error(f"Critical tool {tool_name} failed, this will impact review quality")

                # Yield error action
                error_result = PreflightToolResult(
                    tool_name=tool_name, success=False, error=str(e), execution_time=execution_time
                )
                yield self._create_tool_action(tool_name, error_result, execution_time, tool_call_id)

                # Critical tool failure - stop execution
                if is_critical:
                    logger.error(f"Critical tool {tool_name} failed, stopping remaining preflight tools")
                    break

                # Auxiliary tool failure - continue with remaining tools
                logger.warning(f"Auxiliary tool {tool_name} failed, continuing with remaining tools")
                continue

        # Log execution summary
        total_time = time.time() - execution_status["start_time"]
        success_count = sum(1 for t in execution_status["tools_executed"] if t["success"])
        cache_hit_rate = execution_status["cache_hits"] / len(required_tools) if required_tools else 0

        logger.info(
            f"SQL review preflight completed: {success_count}/{len(required_tools)} tools successful, "
            f"cache_hit_rate={cache_hit_rate:.1%}, total_time={total_time:.2f}s"
        )

    async def _execute_tool(
        self, tool_name: str, sql_query: str, table_names: List[str], catalog: str, database: str, schema: str
    ) -> Any:
        """Execute a specific preflight tool."""
        # Check cache first (if available)
        if self.plan_hooks and hasattr(self.plan_hooks, "query_cache"):
            cache_params = {
                "sql_query": sql_query,
                "table_names": table_names,
                "catalog": catalog,
                "database": database,
                "schema": schema,
            }
            cached_result = self.plan_hooks.query_cache.get(tool_name, **cache_params)
            if cached_result is not None:
                logger.debug(f"Cache hit for {tool_name}")
                if isinstance(cached_result, dict):
                    cached_result["cache_hit"] = True
                return cached_result

        # Execute tool based on type
        if tool_name in self.legacy_tools:
            result = await self._execute_legacy_tool(tool_name, sql_query, table_names, catalog, database, schema)
        elif tool_name in self.enhanced_tools_set:
            result = await self._execute_enhanced_tool(tool_name, sql_query, table_names, catalog, database, schema)
        else:
            raise ValueError(f"Unknown tool: {tool_name}")

        # Cache successful results
        if (
            self.plan_hooks
            and hasattr(self.plan_hooks, "query_cache")
            and result
            and (hasattr(result, "success") and result.success)
            or (isinstance(result, dict) and result.get("success"))
        ):
            try:
                self.plan_hooks.query_cache.set(tool_name, result, **cache_params)
            except Exception as e:
                logger.warning(f"Failed to cache result for {tool_name}: {e}")

        return result

    async def _execute_legacy_tool(
        self, tool_name: str, sql_query: str, table_names: List[str], catalog: str, database: str, schema: str
    ) -> Any:
        """Execute legacy preflight tools."""
        if tool_name == "describe_table":
            if table_names:
                table_name = table_names[0]
                return self.db_func_tool.describe_table(
                    table_name=table_name, catalog=catalog, database=database, schema_name=schema
                )
            else:
                # Try to extract from SQL
                extracted_tables = self._extract_table_names_from_sql(sql_query)
                if extracted_tables:
                    table_name = extracted_tables[0]
                    return self.db_func_tool.describe_table(
                        table_name=table_name, catalog=catalog, database=database, schema_name=schema
                    )
                return PreflightToolResult(tool_name=tool_name, success=False, error="No table name available")

        elif tool_name == "search_external_knowledge":
            query = "StarRocks 3.3 SQL审查规则和最佳实践"
            return self.context_search_tools.search_external_knowledge(
                query_text=query, domain="database", layer1="sql_review", layer2="starrocks", top_n=5
            )

        elif tool_name == "read_query":
            if sql_query:
                return self.db_func_tool.read_query(sql=sql_query)
            return PreflightToolResult(tool_name=tool_name, success=False, error="No SQL query available")

        elif tool_name == "get_table_ddl":
            if table_names:
                table_name = table_names[0]
                return self.db_func_tool.get_table_ddl(
                    table_name=table_name, catalog=catalog, database=database, schema_name=schema
                )
            else:
                extracted_tables = self._extract_table_names_from_sql(sql_query)
                if extracted_tables:
                    table_name = extracted_tables[0]
                    return self.db_func_tool.get_table_ddl(
                        table_name=table_name, catalog=catalog, database=database, schema_name=schema
                    )
                return PreflightToolResult(tool_name=tool_name, success=False, error="No table name available")

        else:
            raise ValueError(f"Unknown legacy tool: {tool_name}")

    async def _execute_enhanced_tool(
        self, tool_name: str, sql_query: str, table_names: List[str], catalog: str, database: str, schema: str
    ) -> Any:
        """Execute enhanced preflight tools."""
        if tool_name == "analyze_query_plan":
            return await self.enhanced_tools.analyze_query_plan(
                sql=sql_query, catalog=catalog, database=database, schema=schema
            )

        elif tool_name == "check_table_conflicts":
            if table_names:
                table_name = table_names[0]
                return await self.enhanced_tools.check_table_conflicts(
                    table_name=table_name, catalog=catalog, database=database, schema=schema
                )
            else:
                extracted_tables = self._extract_table_names_from_sql(sql_query)
                if extracted_tables:
                    table_name = extracted_tables[0]
                    return await self.enhanced_tools.check_table_conflicts(
                        table_name=table_name, catalog=catalog, database=database, schema=schema
                    )
                return PreflightToolResult(tool_name=tool_name, success=False, error="No table name available")

        elif tool_name == "validate_partitioning":
            if table_names:
                table_name = table_names[0]
                return await self.enhanced_tools.validate_partitioning(
                    table_name=table_name, catalog=catalog, database=database, schema=schema
                )
            else:
                extracted_tables = self._extract_table_names_from_sql(sql_query)
                if extracted_tables:
                    table_name = extracted_tables[0]
                    return await self.enhanced_tools.validate_partitioning(
                        table_name=table_name, catalog=catalog, database=database, schema=schema
                    )
                return PreflightToolResult(tool_name=tool_name, success=False, error="No table name available")

        else:
            raise ValueError(f"Unknown enhanced tool: {tool_name}")

    async def _send_tool_call_event(
        self, tool_name: str, tool_call_id: str, input_data: Dict[str, Any], execution_id: str
    ) -> None:
        """Send tool call start event."""
        if self.execution_event_manager:
            await self.execution_event_manager.send_tool_call_event(
                tool_name=tool_name, tool_call_id=tool_call_id, input_data=input_data, execution_id=execution_id
            )

    async def _send_tool_result_event(
        self, tool_call_id: str, result: Any, execution_time: float, cache_hit: bool, execution_id: str
    ) -> None:
        """Send tool result event."""
        if self.execution_event_manager:
            await self.execution_event_manager.send_tool_result_event(
                tool_call_id=tool_call_id,
                result=result,
                execution_time=execution_time,
                cache_hit=cache_hit,
                execution_id=execution_id,
            )

    async def _send_tool_error_event(self, tool_name: str, error: str, execution_id: str, tool_call_id: str) -> None:
        """Send tool error event."""
        if self.execution_event_manager:
            await self.execution_event_manager.send_tool_error_event(
                tool_name=tool_name, error=error, execution_id=execution_id, tool_call_id=tool_call_id
            )

    def _inject_tool_result_into_context(self, workflow, tool_name: str, result: Any) -> None:
        """Inject tool result into workflow context."""
        if not hasattr(workflow, "context"):
            return

        # Initialize preflight results if not exists
        if not hasattr(workflow.context, "preflight_results"):
            workflow.context.preflight_results = {}

        # Convert result to serializable format
        if hasattr(result, "to_dict"):
            result_data = result.to_dict()
        elif hasattr(result, "result") and hasattr(result, "success"):
            # FuncToolResult format
            result_data = {"success": result.success, "result": result.result, "error": getattr(result, "error", None)}
        elif isinstance(result, dict):
            result_data = result
        else:
            result_data = {"raw_result": str(result)}

        # Inject based on tool type
        workflow.context.preflight_results[tool_name] = result_data

        logger.debug(f"Injected {tool_name} result into workflow context")

    def _extract_sql_from_workflow(self, workflow) -> str:
        """Extract SQL query from workflow."""
        if hasattr(workflow, "task") and hasattr(workflow.task, "task"):
            return workflow.task.task
        return ""

    def _extract_table_names_from_workflow(self, workflow) -> List[str]:
        """Extract table names from workflow."""
        table_names = []

        # From task tables
        if hasattr(workflow, "task") and hasattr(workflow.task, "tables"):
            if workflow.task.tables:
                table_names.extend(workflow.task.tables)

        # Try to extract from SQL query
        sql_query = self._extract_sql_from_workflow(workflow)
        if sql_query:
            extracted = self._extract_table_names_from_sql(sql_query)
            table_names.extend(extracted)

        return list(set(table_names))  # Remove duplicates

    def _extract_table_names_from_sql(self, sql: str) -> List[str]:
        """Extract table names from SQL query."""
        import re

        if not sql:
            return []

        # Simple regex to find table names (basic implementation)
        # This is a simplified version - production would need more sophisticated parsing
        table_patterns = [
            r"\bFROM\s+([`\w\.-]+)",
            r"\bJOIN\s+([`\w\.-]+)",
            r"\bUPDATE\s+([`\w\.-]+)",
            r"\bINSERT\s+INTO\s+([`\w\.-]+)",
            r"\bDELETE\s+FROM\s+([`\w\.-]+)",
        ]

        tables = []
        sql_upper = sql.upper()

        for pattern in table_patterns:
            matches = re.findall(pattern, sql_upper)
            for match in matches:
                # Clean up table name
                table_name = match.strip("`").split(".")[-1]  # Get last part after dot
                if table_name and table_name not in tables:
                    tables.append(table_name)

        return tables

    def _create_tool_action(
        self, tool_name: str, result: Any, execution_time: float, tool_call_id: str
    ) -> ActionHistory:
        """Create ActionHistory for tool execution."""
        success = result.success if hasattr(result, "success") else False
        error = getattr(result, "error", None) if hasattr(result, "error") else None

        return ActionHistory.create_action(
            role=ActionRole.TOOL,
            action_type=f"preflight_{tool_name}",
            messages=f"Executed preflight tool: {tool_name}",
            input_data={"tool_name": tool_name},
            output={
                "success": success,
                "result": getattr(result, "result", None),
                "error": error,
                "execution_time": execution_time,
            },
            status=ActionStatus.SUCCESS if success else ActionStatus.FAILED,
            action_id=tool_call_id,
        )
