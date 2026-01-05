# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
ChatAgenticNode implementation for flexible CLI chat interactions.

This module provides a concrete implementation of GenSQLAgenticNode specifically
designed for chat interactions with database and filesystem tool support.
"""
import asyncio
import re
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

try:
    from typing import override
except ImportError:
    from typing_extensions import override

from datus.agent.node.execution_event_manager import (
    ExecutionEventManager,
    create_execution_mode,
)
from datus.agent.node.gen_sql_agentic_node import GenSQLAgenticNode
from datus.agent.workflow import Workflow
from datus.configuration.agent_config import AgentConfig
from datus.schemas.action_history import (
    ActionHistory,
    ActionHistoryManager,
    ActionRole,
    ActionStatus,
)
from datus.schemas.chat_agentic_node_models import ChatNodeInput
from datus.schemas.node_models import SQLContext
from datus.tools.db_tools.db_manager import db_manager_instance
from datus.tools.func_tool import ContextSearchTools, DBFuncTool
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class PreflightOrchestrator:
    """
    Orchestrator for executing preflight tools with enhanced capabilities.

    Handles tool execution, caching, batching, and event emission for the
    new SQL review preflight tools.
    """

    def __init__(self, chat_node: "ChatAgenticNode"):
        self.chat_node = chat_node
        self.db_func_tool = chat_node.db_func_tool
        self.plan_hooks = getattr(chat_node, "plan_hooks", None)
        self.execution_event_manager = getattr(chat_node, "execution_event_manager", None)

    async def execute_enhanced_preflight_tools(
        self,
        workflow,
        action_history_manager: ActionHistoryManager,
        execution_id: Optional[str] = None,
    ) -> AsyncGenerator[ActionHistory, None]:
        """
        Execute enhanced preflight tools with proper orchestration.

        Args:
            workflow: The workflow instance
            action_history_manager: Action history manager
            execution_id: Execution ID for event tracking

        Yields:
            ActionHistory: Progress and result actions
        """
        required_tools = workflow.metadata.get("required_tool_sequence", [])
        if not required_tools:
            return

        # Extract task information
        task = workflow.task
        sql_query = self.chat_node._extract_sql_from_task(task.task)
        catalog = task.catalog_name or ""
        database = task.database_name or ""
        schema = task.schema_name or ""

        # Parse table names
        table_names = self.chat_node._parse_table_names_from_sql(sql_query)

        # Separate new enhanced tools from existing ones
        existing_tools = [
            "describe_table",
            "search_external_knowledge",
            "read_query",
            "get_table_ddl",
        ]
        enhanced_tools = [
            "analyze_query_plan",
            "check_table_conflicts",
            "validate_partitioning",
        ]

        all_tools = existing_tools + enhanced_tools
        tools_to_execute = [tool for tool in required_tools if tool in all_tools]

        if not tools_to_execute:
            return

        # Send plan update event
        await self._send_preflight_plan_update(workflow, tools_to_execute, execution_id)

        # Execute tools
        for tool_name in tools_to_execute:
            async for action in self._execute_single_tool(
                tool_name,
                sql_query,
                table_names,
                catalog,
                database,
                schema,
                workflow,
                action_history_manager,
                execution_id,
            ):
                yield action

    async def _execute_single_tool(
        self,
        tool_name: str,
        sql_query: str,
        table_names: List[str],
        catalog: str,
        database: str,
        schema: str,
        workflow,
        action_history_manager: ActionHistoryManager,
        execution_id: str = None,
    ) -> AsyncGenerator[ActionHistory, None]:
        """Execute a single preflight tool with full orchestration."""
        start_time = time.time()
        tool_call_id = f"preflight_{tool_name}_{start_time}"

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

            # Create and yield action for tool start
            tool_action = ActionHistory.create_action(
                role=ActionRole.TOOL,
                action_type=f"preflight_{tool_name}",
                messages=f"Executing enhanced preflight tool: {tool_name}",
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

            # Check enhanced cache first
            cache_hit = False
            if self.plan_hooks and self.plan_hooks.enable_query_caching:
                cache_key_params = {
                    "catalog": catalog,
                    "database": database,
                    "schema": schema,
                }

                if tool_name == "analyze_query_plan":
                    cache_key_params["sql_query"] = sql_query
                elif tool_name in ["check_table_conflicts", "validate_partitioning"]:
                    if table_names:
                        cache_key_params["table_name"] = table_names[0]

                cached_result = self.plan_hooks.query_cache.get_enhanced(tool_name, **cache_key_params)
                if cached_result is not None:
                    cache_hit = True
                    result = cached_result
                else:
                    # Check if batch processing is available for this tool
                    batch_processor = getattr(self.plan_hooks, "batch_processor", None)
                    if batch_processor and tool_name in ["check_table_conflicts", "validate_partitioning"]:
                        # Add to batch for later processing
                        batch_size = batch_processor.get_batch_size(tool_name)
                        if batch_size > 0:
                            # Process existing batch first
                            existing_batch = batch_processor.clear_batch(tool_name)
                            if existing_batch:
                                batch_results = await self._process_tool_batch(tool_name, existing_batch, workflow)
                                # Cache batch results
                                for batch_result in batch_results:
                                    if batch_result.get("success", False):
                                        batch_cache_params = cache_key_params.copy()
                                        batch_cache_params["table_name"] = batch_result.get("table_name", "")
                                        self.plan_hooks.query_cache.set_enhanced(
                                            tool_name, batch_result, **batch_cache_params
                                        )

                    # Execute the tool
                    result = await self._call_enhanced_tool(
                        tool_name, sql_query, table_names, catalog, database, schema
                    )

                    # Cache successful results using enhanced caching
                    if result.get("success", False):
                        self.plan_hooks.query_cache.set_enhanced(tool_name, result, **cache_key_params)

            execution_time = time.time() - start_time

            # Send tool result event
            await self._send_tool_call_result_event(
                tool_call_id=tool_call_id,
                result=result,
                execution_time=execution_time,
                cache_hit=cache_hit,
                execution_id=execution_id,
            )

            # Update action with results
            tool_action.output = result
            tool_action.status = ActionStatus.SUCCESS if result.get("success", False) else ActionStatus.FAILED
            tool_action.messages = f"Preflight tool {tool_name} completed"

            # Inject results into workflow context
            self._inject_tool_result_into_context(workflow, tool_name, result)

            # Update execution status
            if hasattr(self.chat_node, "execution_status") and self.chat_node.execution_status:
                success = result.get("success", False)
                error_type = None if success else self._classify_error_type(result.get("error", ""), tool_name)
                self.chat_node.execution_status.add_tool_execution(tool_name, success, execution_time, error_type)

                if not success:
                    error_desc = result.get("error", "Unknown error")
                    self.chat_node.execution_status.add_error(error_type, error_desc)
                    await self._dispatch_error_event(error_type, sql_query, error_desc, tool_name, table_names)

            # Record monitoring data
            if self.plan_hooks and hasattr(self.plan_hooks, "monitor"):
                self.plan_hooks.monitor.record_preflight_tool_call(
                    execution_id or f"preflight_{id(workflow)}",
                    tool_name,
                    result.get("success", False),
                    cache_hit,
                    execution_time,
                    result.get("error", "") if not result.get("success", False) else "",
                )

            yield tool_action

        except Exception as e:
            logger.error(f"Error executing enhanced preflight tool {tool_name}: {e}")

            # Send error event
            error_desc = str(e)
            await self._send_diagnostic_info_event(
                f"Tool execution failed: {tool_name}", {"error": error_desc, "tool_name": tool_name}, execution_id
            )

            # Create error action
            error_action = ActionHistory.create_action(
                role=ActionRole.SYSTEM,
                action_type=f"preflight_{tool_name}_error",
                messages=f"Preflight tool {tool_name} failed: {error_desc}",
                input_data={"tool_name": tool_name},
                output_data={"error": error_desc},
                status=ActionStatus.FAILED,
            )
            action_history_manager.add_action(error_action)
            yield error_action

    async def _call_enhanced_tool(
        self, tool_name: str, sql_query: str, table_names: List[str], catalog: str, database: str, schema: str
    ) -> Dict[str, Any]:
        """Call the appropriate enhanced preflight tool."""
        if self.db_func_tool is None:
            return {"success": False, "error": "DB function tool not initialized"}
        if tool_name == "analyze_query_plan":
            return self.db_func_tool.analyze_query_plan(sql_query, catalog, database, schema)

        elif tool_name == "check_table_conflicts":
            if not table_names:
                return {"success": False, "error": "No table names found in SQL query"}
            return self.db_func_tool.check_table_conflicts(table_names[0], catalog, database, schema)

        elif tool_name == "validate_partitioning":
            if not table_names:
                return {"success": False, "error": "No table names found in SQL query"}
            return self.db_func_tool.validate_partitioning(table_names[0], catalog, database, schema)

        else:
            # Fallback to existing tool execution
            return await self.chat_node._execute_preflight_tool(
                tool_name, sql_query, table_names, catalog, database, schema
            )

    def _inject_tool_result_into_context(self, workflow, tool_name: str, result: Dict[str, Any]) -> None:
        """Inject tool results into workflow context for LLM consumption."""
        if not hasattr(workflow, "context") or not workflow.context:
            return

        try:
            if tool_name == "analyze_query_plan" and result.get("success"):
                # Store query plan analysis results
                analysis_data = {
                    "plan_text": result.get("plan_text", ""),
                    "hotspots": result.get("hotspots", []),
                    "estimated_cost": result.get("estimated_cost", 0),
                    "join_analysis": result.get("join_analysis", {}),
                    "index_usage": result.get("index_usage", {}),
                    "warnings": result.get("warnings", []),
                }
                workflow.context.preflight_results = getattr(workflow.context, "preflight_results", {})
                workflow.context.preflight_results["query_plan_analysis"] = analysis_data

            elif tool_name == "check_table_conflicts" and result.get("success"):
                # Store conflict analysis results
                conflict_data = {
                    "exists_similar": result.get("exists_similar", False),
                    "matches": result.get("matches", []),
                    "duplicate_build_risk": result.get("duplicate_build_risk", "low"),
                    "layering_violations": result.get("layering_violations", []),
                }
                workflow.context.preflight_results = getattr(workflow.context, "preflight_results", {})
                workflow.context.preflight_results["table_conflicts"] = conflict_data

            elif tool_name == "validate_partitioning" and result.get("success"):
                # Store partitioning validation results
                partitioning_data = {
                    "partitioned": result.get("partitioned", False),
                    "partition_info": result.get("partition_info", {}),
                    "issues": result.get("issues", []),
                    "recommended_partition": result.get("recommended_partition", {}),
                    "performance_impact": result.get("performance_impact", {}),
                }
                workflow.context.preflight_results = getattr(workflow.context, "preflight_results", {})
                workflow.context.preflight_results["partitioning_validation"] = partitioning_data

        except Exception as e:
            logger.warning(f"Failed to inject tool result into context for {tool_name}: {e}")

    async def _send_preflight_plan_update(
        self, workflow, tools_to_execute: List[str], execution_id: Optional[str] = None
    ) -> None:
        """Send preflight plan update event."""
        if self.execution_event_manager and execution_id:
            await self.execution_event_manager.update_execution_status(
                execution_id, "planning", f"Executing enhanced preflight tools: {tools_to_execute}"
            )

    async def _send_tool_call_event(
        self, tool_name: str, tool_call_id: str, input_data: Dict[str, Any], execution_id: Optional[str] = None
    ) -> None:
        """Send tool call start event."""
        if self.execution_event_manager and execution_id:
            await self.execution_event_manager.record_tool_execution(execution_id, tool_name, tool_call_id, input_data)

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
            await self.execution_event_manager.record_tool_execution(
                execution_id, tool_call_id, tool_call_id, {}, result, None, execution_time
            )

    async def _send_diagnostic_info_event(
        self, message: str, data: Dict[str, Any], execution_id: Optional[str] = None
    ) -> None:
        """Send diagnostic information event."""
        if self.execution_event_manager and execution_id:
            await self.execution_event_manager.record_llm_interaction(
                execution_id, "diagnostic", message, additional_data=data
            )

    def _classify_error_type(self, error_msg: str, tool_name: str) -> str:
        """Classify error type for better error handling."""
        error_lower = error_msg.lower()

        if any(keyword in error_lower for keyword in ["permission", "access denied", "privilege"]):
            return "permission_error"
        elif any(keyword in error_lower for keyword in ["timeout", "timed out"]):
            return "timeout_error"
        elif any(keyword in error_lower for keyword in ["table", "relation", "does not exist"]):
            return "table_not_found"
        elif any(keyword in error_lower for keyword in ["connection", "network", "connect"]):
            return "connection_error"
        elif any(keyword in error_lower for keyword in ["syntax", "parse error"]):
            return "syntax_error"
        else:
            return "unknown_error"

    async def _dispatch_error_event(
        self, error_type: str, sql_query: str, error_desc: str, tool_name: str, table_names: List[str]
    ) -> None:
        """Dispatch error events based on error type."""
        try:
            if error_type == "permission_error":
                await self._send_permission_error_event(sql_query, error_desc)
            elif error_type == "timeout_error":
                await self._send_timeout_error_event(sql_query, error_desc)
            elif error_type == "table_not_found":
                if table_names:
                    await self._send_table_not_found_error_event(table_names[0], error_desc)
            elif error_type == "connection_error":
                await self._send_db_connection_error_event(sql_query, error_desc)
        except Exception as e:
            logger.warning(f"Failed to dispatch error event for {error_type}: {e}")

    async def _send_permission_error_event(self, sql_query: str, error_desc: str) -> None:
        """Send permission error event."""
        if self.execution_event_manager:
            await self.execution_event_manager.record_llm_interaction(
                "",
                "permission_error",
                f"Database permission error: {error_desc}",
                additional_data={"sql_query": sql_query},
            )

    async def _send_timeout_error_event(self, sql_query: str, error_desc: str) -> None:
        """Send timeout error event."""
        if self.execution_event_manager:
            await self.execution_event_manager.record_llm_interaction(
                "", "timeout_error", f"Query timeout error: {error_desc}", additional_data={"sql_query": sql_query}
            )

    async def _send_table_not_found_error_event(self, table_name: str, error_desc: str) -> None:
        """Send table not found error event."""
        if self.execution_event_manager:
            await self.execution_event_manager.record_llm_interaction(
                "",
                "table_not_found",
                f"Table not found: {table_name} - {error_desc}",
                additional_data={"table_name": table_name},
            )

    async def _send_db_connection_error_event(self, sql_query: str, error_desc: str) -> None:
        """Send database connection error event."""
        if self.execution_event_manager:
            await self.execution_event_manager.record_llm_interaction(
                "",
                "connection_error",
                f"Database connection error: {error_desc}",
                additional_data={"sql_query": sql_query},
            )

    async def _process_tool_batch(self, tool_name: str, batch: List[Tuple], workflow) -> List[Dict[str, Any]]:
        """Process a batch of tool calls efficiently."""
        if not batch:
            return []
        if self.db_func_tool is None:
            return []

        results = []

        if tool_name == "check_table_conflicts":
            # Batch process table conflict checks
            for todo_item, params in batch:
                table_name = params.get("table_name", "")
                catalog = params.get("catalog", "")
                database = params.get("database", "")
                schema = params.get("schema", "")

                result = self.db_func_tool.check_table_conflicts(table_name, catalog, database, schema)
                result["table_name"] = table_name  # Add table name to result for caching
                results.append(result)

        elif tool_name == "validate_partitioning":
            # Batch process partitioning validation
            for todo_item, params in batch:
                table_name = params.get("table_name", "")
                catalog = params.get("catalog", "")
                database = params.get("database", "")
                schema = params.get("schema", "")

                result = self.db_func_tool.validate_partitioning(table_name, catalog, database, schema)
                result["table_name"] = table_name  # Add table name to result for caching
                results.append(result)

        return results


class ChatAgenticNode(GenSQLAgenticNode):
    """
    Chat-focused agentic node with database and filesystem tool support.

    This node provides flexible chat capabilities with:
    - Namespace-based database MCP server selection
    - Default filesystem MCP server
    - Streaming response generation
    - Session-based conversation management
    """

    def __init__(
        self,
        node_id: str,
        description: str,
        node_type: str,
        input_data: Optional[ChatNodeInput] = None,
        agent_config: Optional[AgentConfig] = None,
        tools: Optional[list] = None,
    ):
        """
        Initialize the ChatAgenticNode as a specialized GenSQLAgenticNode.

        Args:
            node_id: Unique identifier for the node
            description: Human-readable description of the node
            node_type: Type of the node (should be 'chat')
            input_data: Chat input data
            agent_config: Agent configuration
            tools: List of tools (will be populated in setup_tools)
        """
        # Call parent constructor with node_name="chat"
        # This will initialize max_turns, tool attributes, plan mode attributes, and MCP servers
        super().__init__(
            node_id=node_id,
            description=description,
            node_type=node_type,
            input_data=input_data,
            agent_config=agent_config,
            tools=tools,
            node_name="chat",
        )

        # Initialize action_history_manager attribute for plan mode support
        self.action_history_manager = None

        # Initialize unified execution event manager
        self.execution_event_manager = None

        logger.debug(
            f"ChatAgenticNode initialized: {self.agent_config.current_namespace} {self.agent_config.current_database}"
        )

    def _validate_sql_syntax_comprehensive(self, sql: str) -> dict:
        """Comprehensive SQL syntax validation."""
        try:
            import sqlglot

            # Basic syntax check
            parsed = sqlglot.parse_one(sql)

            # Additional checks
            issues = []

            # Check for incomplete statements
            has_main_operation = any(
                isinstance(node, (sqlglot.exp.Select, sqlglot.exp.Insert, sqlglot.exp.Update, sqlglot.exp.Delete))
                for node in parsed.walk()
            )
            if not has_main_operation:
                issues.append("SQL statement appears incomplete - missing main operation (SELECT/INSERT/UPDATE/DELETE)")

            # Check for table references
            tables = [str(table) for table in parsed.find_all(sqlglot.exp.Table)]
            if not tables:
                issues.append("No table references found in SQL")

            # Check for potentially problematic patterns
            upper_sql = sql.upper()
            if "SELECT *" in upper_sql and len(tables) > 0:
                # This is actually allowed, just log it for review
                logger.info("SELECT * detected - will be flagged in review")

            return {
                "valid": len(issues) == 0,
                "error": "; ".join(issues) if issues else None,
                "tables_found": tables,
                "parsed_ast": str(parsed),
                "sql_length": len(sql),
            }

        except Exception as e:
            return {
                "valid": False,
                "error": f"SQL parsing failed: {str(e)}",
                "tables_found": [],
                "parsed_ast": None,
                "sql_length": len(sql),
            }

    def _analyze_tool_failure_impact(self, tool_name: str, result: dict) -> str:
        """Analyze the impact of a tool failure on SQL review quality."""
        if result.get("success", False):
            return "无影响 - 工具执行成功"

        error_msg = result.get("error", "").lower()

        if tool_name == "describe_table":
            return "表结构分析受限，可能无法准确评估字段类型、索引使用和分区策略"
        elif tool_name == "get_table_ddl":
            return "DDL信息缺失，表设计审查和约束检查能力下降"
        elif tool_name == "read_query":
            if "syntax" in error_msg:
                return "语法验证失败，SQL正确性检查受限"
            else:
                return "查询执行失败，性能评估依赖理论分析"
        elif tool_name == "search_external_knowledge":
            return "规则检索失败，使用通用最佳实践而非StarRocks 3.3特定规范"
        else:
            return f"工具{tool_name}执行失败，具体影响需进一步分析"

    def setup_input(self, workflow: Workflow) -> dict:
        """
        Setup chat input from workflow context.

        Creates ChatNodeInput with user message from task and context data.

        Args:
            workflow: Workflow instance containing context and task

        Returns:
            Dictionary with success status and message
        """
        # Store workflow reference for access to metadata
        self.workflow = workflow

        # Update database connection if task specifies a different database
        task_database = workflow.task.database_name
        if task_database and self.db_func_tool and task_database != self.db_func_tool.connector.database_name:
            logger.info(
                f"Updating database connection from '{self.db_func_tool.connector.database_name}' "
                f"to '{task_database}' based on workflow task"
            )
            self._update_database_connection(task_database)

        # Read plan_mode from workflow metadata
        plan_mode = workflow.metadata.get("plan_mode", False)
        auto_execute_plan = workflow.metadata.get("auto_execute_plan", False)

        # Create ChatNodeInput if not already set
        if not self.input:
            self.input = ChatNodeInput(
                user_message=workflow.task.task,
                external_knowledge=workflow.task.external_knowledge,
                catalog=workflow.task.catalog_name,
                database=workflow.task.database_name,
                db_schema=workflow.task.schema_name,
                schemas=workflow.context.table_schemas,
                metrics=workflow.context.metrics,
                reference_sql=None,
                plan_mode=plan_mode,
                auto_execute_plan=auto_execute_plan,
            )
        else:
            # Update existing input with workflow data
            self.input.user_message = workflow.task.task
            self.input.external_knowledge = workflow.task.external_knowledge
            self.input.catalog = workflow.task.catalog_name
            self.input.database = workflow.task.database_name
            self.input.db_schema = workflow.task.schema_name
            self.input.schemas = workflow.context.table_schemas
            self.input.metrics = workflow.context.metrics

        return {"success": True, "message": "Chat input prepared from workflow"}

    def update_context(self, workflow: Workflow) -> dict:
        """
        Update workflow context with chat results.

        Stores SQL to workflow context if present in result.
        In plan mode, SQL context is managed by PlanModeHooks, so skip here.

        Args:
            workflow: Workflow instance to update

        Returns:
            Dictionary with success status and message
        """
        # In plan mode, SQL context is managed by PlanModeHooks during tool execution
        if workflow.metadata.get("plan_mode"):
            logger.info("Plan mode: Skipping SQL context update in chat node, handled by PlanModeHooks")
            return {"success": True, "message": "Plan mode context update skipped, handled by hooks"}

        if not self.result:
            return {"success": False, "message": "No result to update context"}

        result = self.result

        try:
            sql_content = None
            sql_explanation = ""

            # First, try to get SQL from result.sql (regular chat mode)
            if hasattr(result, "sql") and result.sql:
                sql_content = result.sql
                sql_explanation = result.response if hasattr(result, "response") else ""

            # For plan mode or when result.sql is not available, try to extract from action history
            action_history_manager = getattr(self, "action_history_manager", None)
            if not sql_content and action_history_manager:
                try:
                    # Extract SQL directly from summary_report action if available (plan mode final result)
                    for stream_action in reversed(action_history_manager.get_actions()):
                        if stream_action.action_type == "summary_report" and stream_action.output:
                            if isinstance(stream_action.output, dict):
                                sql_content = stream_action.output.get("sql")
                                if sql_content:
                                    sql_explanation = stream_action.output.get(
                                        "content", ""
                                    ) or stream_action.output.get("response", "")
                                    break

                    # Fallback: try to extract SQL from any action that might contain it
                    if not sql_content:
                        response_content = result.response if hasattr(result, "response") and result.response else ""
                        extracted_sql, extracted_output = self._extract_sql_and_output_from_response(
                            {"content": response_content}
                        )
                        if extracted_sql:
                            sql_content = extracted_sql
                            sql_explanation = extracted_output or response_content
                except Exception as e:
                    logger.warning(f"Failed to extract SQL from action history: {e}")
                    # Continue with fallback extraction from response content
                    if not sql_content:
                        response_content = result.response if hasattr(result, "response") and result.response else ""
                        try:
                            extracted_sql, extracted_output = self._extract_sql_and_output_from_response(
                                {"content": response_content}
                            )
                            if extracted_sql:
                                sql_content = extracted_sql
                                sql_explanation = extracted_output or response_content
                        except Exception as e2:
                            logger.warning(f"Failed to extract SQL from response content: {e2}")

            # If we found SQL content, create and store the SQL context
            if sql_content:
                logger.info(f"Found SQL content in chat node, creating SQL context: {sql_content[:100]}...")
                from datus.schemas.node_models import SQLContext

                # Extract SQL result from the response if available
                sql_result = ""
                if sql_explanation:
                    # Try to extract SQL result from the explanation
                    _, sql_result = self._extract_sql_and_output_from_response({"content": sql_explanation})
                    sql_result = sql_result or ""

                new_record = SQLContext(
                    sql_query=sql_content,
                    explanation=sql_explanation,
                    sql_return=sql_result,
                )
                workflow.context.sql_contexts.append(new_record)
                logger.info(f"Added SQL context to workflow. Total contexts: {len(workflow.context.sql_contexts)}")
            else:
                logger.warning("No SQL content found in chat node result or action history")
                # Debug: log what we have available
                if hasattr(result, "sql"):
                    logger.warning(f"result.sql: {result.sql}")
                if hasattr(result, "response"):
                    logger.warning(f"result.response length: {len(result.response) if result.response else 0}")
                if action_history_manager:
                    logger.warning(f"action_history_manager has {len(action_history_manager.get_actions())} actions")

            return {"success": True, "message": "Updated chat context"}
        except Exception as e:
            logger.error(f"Failed to update chat context: {e}")
            return {"success": False, "message": str(e)}

    def _get_system_prompt(
        self, conversation_summary: Optional[str] = None, prompt_version: Optional[str] = None
    ) -> str:
        """
        Get the system prompt for this chat node, supporting task-specific prompts and plan mode.

        Args:
            conversation_summary: Optional summary from previous conversation compact
            prompt_version: Optional prompt version to use, overrides agent config version

        Returns:
            System prompt string loaded from the appropriate template
        """
        # First, check if a task-specific system prompt is specified in workflow metadata
        task_system_prompt = None
        if self.workflow and hasattr(self.workflow, "metadata"):
            task_system_prompt = self.workflow.metadata.get("system_prompt")

        if task_system_prompt:
            # Use task-specific system prompt from workflow metadata
            original_system_prompt = self.node_config.get("system_prompt")
            self.node_config["system_prompt"] = task_system_prompt
            try:
                return super()._get_system_prompt(conversation_summary, prompt_version)
            finally:
                # Restore original system prompt
                if original_system_prompt is not None:
                    self.node_config["system_prompt"] = original_system_prompt
                else:
                    self.node_config.pop("system_prompt", None)

        # Check if plan mode is active by looking at workflow metadata or node state
        is_plan_mode = getattr(self, "plan_mode_active", False)

        # Temporarily modify node_config to use plan_mode_system when in plan mode
        if is_plan_mode:
            original_system_prompt = self.node_config.get("system_prompt")
            self.node_config["system_prompt"] = "plan_mode"
            try:
                return super()._get_system_prompt(conversation_summary, prompt_version)
            finally:
                # Restore original system prompt
                if original_system_prompt is not None:
                    self.node_config["system_prompt"] = original_system_prompt
                else:
                    self.node_config.pop("system_prompt", None)
        else:
            return super()._get_system_prompt(conversation_summary, prompt_version)

    @override
    def setup_tools(self):
        """Initialize all tools with default database connection."""
        # Chat node uses all available tools by default
        db_manager = db_manager_instance(self.agent_config.namespaces)
        conn = db_manager.get_conn(self.agent_config.current_namespace, self.agent_config.current_database)
        self.db_func_tool = DBFuncTool(conn, agent_config=self.agent_config)
        self.context_search_tools = ContextSearchTools(self.agent_config)
        self._setup_date_parsing_tools()
        self._setup_filesystem_tools()
        self._rebuild_tools()

    async def _run_original_preflight_tools(
        self, workflow, action_history_manager: ActionHistoryManager = None, execution_id: str = None
    ) -> AsyncGenerator[ActionHistory, None]:
        """
        Execute preflight tools using the enhanced orchestrator for new tools,
        falling back to original implementation for existing tools.
        """
        required_tools = workflow.metadata.get("required_tool_sequence", [])
        if not required_tools:
            return

        # Separate enhanced tools from existing tools
        enhanced_tools = ["analyze_query_plan", "check_table_conflicts", "validate_partitioning"]
        existing_tools = [tool for tool in required_tools if tool not in enhanced_tools]
        enhanced_tools_in_sequence = [tool for tool in required_tools if tool in enhanced_tools]

        # Use enhanced orchestrator for new tools
        if enhanced_tools_in_sequence:
            orchestrator = PreflightOrchestrator(self)
            async for action in orchestrator.execute_enhanced_preflight_tools(
                workflow, action_history_manager, execution_id
            ):
                yield action

        # Use original implementation for existing tools
        if existing_tools:
            # Update workflow metadata to only include existing tools for original implementation
            original_metadata = workflow.metadata.copy()
            workflow.metadata["required_tool_sequence"] = existing_tools
            try:
                # Execute existing tools using the original implementation (lines below)
                # Use execution event manager if provided, otherwise fall back to action_history_manager
                event_manager = (
                    self.execution_event_manager
                    if hasattr(self, "execution_event_manager") and self.execution_event_manager
                    else None
                )

                required_tools = workflow.metadata.get("required_tool_sequence", [])
                if not required_tools:
                    logger.debug("No required tool sequence specified, skipping preflight")
                    return

                logger.info(f"Starting preflight tool execution: {required_tools}")

                # Send preflight plan update if using event manager
                if event_manager and execution_id:
                    await event_manager.update_execution_status(
                        execution_id, "planning", f"Executing preflight tools: {required_tools}"
                    )

                # Execute each tool in the sequence
                for tool_name in required_tools:
                    try:
                        logger.debug(f"Executing preflight tool: {tool_name}")

                        # Send tool start event if using event manager
                        if event_manager and execution_id:
                            await event_manager.update_execution_status(
                                execution_id, "executing", f"Running {tool_name}"
                            )

                        # Execute the tool based on its name
                        if tool_name == "describe_table":
                            result = await self._execute_describe_table_tool(workflow, action_history_manager)
                        elif tool_name == "search_external_knowledge":
                            result = await self._execute_search_external_knowledge_tool(
                                workflow, action_history_manager
                            )
                        elif tool_name == "read_query":
                            result = await self._execute_read_query_tool(workflow, action_history_manager)
                        elif tool_name == "get_table_ddl":
                            result = await self._execute_get_table_ddl_tool(workflow, action_history_manager)
                        else:
                            logger.warning(f"Unknown preflight tool: {tool_name}")
                            continue

                        # Yield the action if successful
                        if result:
                            yield result

                        # Send tool completion event if using event manager
                        if event_manager and execution_id:
                            await event_manager.update_execution_status(
                                execution_id, "executing", f"Completed {tool_name}"
                            )

                    except Exception as e:
                        logger.error(f"Error executing preflight tool {tool_name}: {e}")
                        # Send error event if using event manager
                        if event_manager and execution_id:
                            await event_manager.update_execution_status(
                                execution_id, "error", f"Failed {tool_name}: {str(e)}"
                            )
                        # Continue with next tool unless configured to fail fast
                        continue

            finally:
                # Restore original metadata
                workflow.metadata = original_metadata

    async def _execute_describe_table_tool(self, workflow, action_history_manager):
        """Execute describe_table preflight tool."""
        try:
            task = workflow.task
            sql_query = self._extract_sql_from_task(task.task)
            table_names = self._parse_table_names_from_sql(sql_query)

            result = await self._execute_preflight_tool(
                "describe_table", sql_query, table_names, task.catalog_name, task.database_name, task.schema_name
            )

            if result.get("success"):
                # Create action for successful tool execution
                action = ActionHistory.create_action(
                    role=ActionRole.ASSISTANT,
                    action_type="tool_call",
                    messages=f"Describe table structure for {table_names[0] if table_names else 'unknown table'}",
                    input_data={"table_names": table_names},
                    output_data=result,
                )
                if action_history_manager:
                    action_history_manager.add_action(action)
                return action

        except Exception as e:
            logger.error(f"Error in _execute_describe_table_tool: {e}")

        return None

    async def _execute_search_external_knowledge_tool(self, workflow, action_history_manager):
        """Execute search_external_knowledge preflight tool."""
        try:
            task = workflow.task

            result = await self._execute_preflight_tool(
                "search_external_knowledge",
                "",
                [],  # No SQL/table needed for knowledge search
                task.catalog_name,
                task.database_name,
                task.schema_name,
            )

            if result.get("success"):
                # Create action for successful tool execution
                action = ActionHistory.create_action(
                    role=ActionRole.ASSISTANT,
                    action_type="tool_call",
                    messages="Search StarRocks SQL review rules and best practices",
                    input_data={"query": "Search StarRocks SQL review rules and best practices"},
                    output_data=result,
                )
                if action_history_manager:
                    action_history_manager.add_action(action)
                return action

        except Exception as e:
            logger.error(f"Error in _execute_search_external_knowledge_tool: {e}")

        return None

    async def _execute_read_query_tool(self, workflow, action_history_manager):
        """Execute read_query preflight tool."""
        try:
            task = workflow.task
            sql_query = self._extract_sql_from_task(task.task)

            result = await self._execute_preflight_tool(
                "read_query", sql_query, [], task.catalog_name, task.database_name, task.schema_name
            )

            if result.get("success"):
                # Create action for successful tool execution
                action = ActionHistory.create_action(
                    role=ActionRole.ASSISTANT,
                    action_type="tool_call",
                    messages=f"Execute SQL query for validation: {sql_query[:100]}...",
                    input_data={"sql_query": sql_query[:100]},
                    output_data=result,
                )
                if action_history_manager:
                    action_history_manager.add_action(action)
                return action

        except Exception as e:
            logger.error(f"Error in _execute_read_query_tool: {e}")

        return None

    async def _execute_get_table_ddl_tool(self, workflow, action_history_manager):
        """Execute get_table_ddl preflight tool."""
        try:
            task = workflow.task
            sql_query = self._extract_sql_from_task(task.task)
            table_names = self._parse_table_names_from_sql(sql_query)

            result = await self._execute_preflight_tool(
                "get_table_ddl", sql_query, table_names, task.catalog_name, task.database_name, task.schema_name
            )

            if result.get("success"):
                # Create action for successful tool execution
                action = ActionHistory.create_action(
                    role=ActionRole.ASSISTANT,
                    action_type="tool_call",
                    messages=f"Get DDL for table {table_names[0] if table_names else 'unknown table'}",
                    input_data={"table_names": table_names},
                    output_data=result,
                )
                if action_history_manager:
                    action_history_manager.add_action(action)
                return action

        except Exception as e:
            logger.error(f"Error in _execute_get_table_ddl_tool: {e}")

        return None

    def _extract_sql_from_task(self, task_text: str) -> str:
        """Extract SQL query from task text with improved logic."""
        import re

        # 1. Try to find SQL in backticks first (highest priority)
        sql_match = re.search(r"```\s*sql\s*(.*?)\s*```", task_text, re.DOTALL | re.IGNORECASE)
        if sql_match:
            return sql_match.group(1).strip()

        # 2. Look for "待审查SQL：" or "SQL：" patterns (common in Chinese interfaces)
        sql_prefix_match = re.search(r"(?:待审查SQL|SQL)[:：]\s*(.+?)(?:\n|$)", task_text, re.IGNORECASE)
        if sql_prefix_match:
            candidate_sql = sql_prefix_match.group(1).strip()
            # Validate it's actually SQL by checking for SELECT keyword
            if candidate_sql.upper().startswith("SELECT"):
                return candidate_sql

        # 3. Try to find complete SQL statements with better boundary detection
        # Look for SELECT ... FROM pattern followed by semicolon or end of string
        select_pattern = r"SELECT\s+.*?(?:FROM\s+\w+.*?)(?:;|\s*$)"
        select_match = re.search(select_pattern, task_text, re.DOTALL | re.IGNORECASE)
        if select_match:
            sql_candidate = select_match.group(0).strip()
            # Additional validation: ensure it doesn't contain Chinese characters inappropriately
            if not self._contains_invalid_mixed_content(sql_candidate):
                return sql_candidate

        # 4. Fallback: original logic but with validation
        select_match = re.search(r"(SELECT\s+.*?);", task_text, re.DOTALL | re.IGNORECASE)
        if select_match:
            sql_candidate = select_match.group(1).strip()
            if not self._contains_invalid_mixed_content(sql_candidate):
                return sql_candidate

        # 5. Final fallback: return the entire task text
        return task_text.strip()

    def _contains_invalid_mixed_content(self, sql: str) -> bool:
        """Check if SQL contains invalid mixed content (Chinese explanations mixed with SQL)."""
        # Check for Chinese characters in SQL keywords area
        chinese_chars = re.findall(r"[\u4e00-\u9fff]", sql[:50])  # First 50 chars
        if len(chinese_chars) > 3:  # Too many Chinese chars suggest mixed content
            return True

        # Check for explanation patterns mixed with SQL
        invalid_patterns = [
            r"SELECT\s+.*禁止.*分区裁剪",  # Specific pattern from the error
            r"SELECT\s+.*等\)",  # Chinese closing parenthesis
            r"SELECT\s+.*规范等",  # Another pattern from logs
        ]

        for pattern in invalid_patterns:
            if re.search(pattern, sql, re.IGNORECASE):
                return True

        return False

    def _parse_table_names_from_sql(self, sql: str) -> list:
        """Parse table names from SQL query using sqlglot."""
        import sqlglot
        from sqlglot import exp

        try:
            # Use sqlglot to extract table names (sqlglot is already a dependency)
            parsed = sqlglot.parse_one(sql)
            tables = []

            for table in parsed.find_all(exp.Table):
                # Get the table name - sqlglot stores this in the table object
                # For simple table names, table.name works directly
                # For qualified names (schema.table), we need to get the last part
                if hasattr(table, "name") and table.name:
                    # Remove backticks and get the last part for qualified names
                    table_name = table.name.replace("`", "").split(".")[-1].strip()
                    if table_name and table_name not in tables:
                        tables.append(table_name)

            return tables

        except Exception as e:
            logger.warning(f"Failed to parse table names from SQL: {e}")
            # Fallback: simple regex extraction
            table_matches = re.findall(r"\bFROM\s+(\w+)", sql, re.IGNORECASE)
            return list(set(table_matches))

    def _validate_sql_syntax(self, sql: str, task_text: str) -> tuple[bool, Optional[str]]:
        """Validate SQL syntax using sqlglot.

        Args:
            sql: Extracted SQL query to validate
            task_text: Original task text for error context

        Returns:
            (is_valid, error_message): Tuple where is_valid is True if SQL is valid,
                                      error_message contains Chinese error description if invalid
        """
        import sqlglot

        from datus.utils.sql_utils import parse_read_dialect

        if not sql or not sql.strip():
            return False, "SQL语法错误：未找到SQL语句"

        # Get database dialect from agent config
        try:
            dialect = parse_read_dialect(self.agent_config.db_type)
        except Exception:
            dialect = "starrocks"  # Default fallback

        try:
            # Try to parse the SQL with error detection
            # Use error_level=RAISE to immediately catch syntax errors
            parsed = sqlglot.parse_one(sql.strip(), read=dialect, error_level=sqlglot.ErrorLevel.RAISE)

            # Additional validation: check for common SQL structure issues
            # Convert to string and re-parse to catch any hidden issues
            sql_str = str(parsed)
            sqlglot.parse_one(sql_str, read=dialect, error_level=sqlglot.ErrorLevel.RAISE)

            return True, None

        except Exception as e:
            # Extract meaningful error information
            error_desc = str(e)
            logger.warning(f"SQL syntax error detected: {error_desc}")

            # Generate user-friendly Chinese error message
            error_message = self._generate_sql_error_message(error_desc, sql)
            return False, error_message

    def _generate_sql_error_message(self, error_desc: str, sql: str) -> str:
        """Generate user-friendly Chinese error message from sqlglot error description.

        Args:
            error_desc: Raw error description from sqlglot
            sql: Original SQL that failed validation

        Returns:
            Chinese error message suitable for user display
        """
        import re

        error_lower = error_desc.lower()

        # Common error patterns with Chinese translations
        error_patterns = [
            (r"unexpected.*FROM", "SQL语法错误：缺少FROM子句"),
            (r"unexpected.*SELECT", "SQL语法错误：SELECT关键字位置错误"),
            (r"unexpected.*WHERE", "SQL语法错误：WHERE子句语法错误"),
            (r"unexpected.*JOIN", "SQL语法错误：JOIN连接语法错误"),
            (r"unexpected.*GROUP BY", "SQL语法错误：GROUP BY语法错误"),
            (r"unexpected.*ORDER BY", "SQL语法错误：ORDER BY语法错误"),
            (r"mismatched parentheses", "SQL语法错误：括号不匹配"),
            (r"unexpected.*\)", "SQL语法错误：多余的右括号"),
            (r"unexpected.*\(", "SQL语法错误：多余的左括号"),
            (r"unexpected end", "SQL语法错误：语句不完整"),
            (r"expected.*,", "SQL语法错误：缺少逗号分隔符"),
            (r"unexpected.*,", "SQL语法错误：多余的逗号"),
            (r"identifier.*too long", "SQL语法错误：标识符过长"),
            (r"invalid identifier", "SQL语法错误：无效的标识符"),
            (r"unknown column", "SQL语法错误：未知列名"),
            (r"unknown table", "SQL语法错误：未知表名"),
        ]

        for pattern, message in error_patterns:
            if re.search(pattern, error_lower, re.IGNORECASE):
                return message

        # Default generic error
        # Try to extract specific error type
        if "syntax" in error_lower:
            return "SQL语法错误：语法不正确"
        elif "parse" in error_lower:
            return "SQL语法错误：解析失败"
        else:
            return "SQL语法错误：无法解析SQL语句"

    def _classify_error_type(self, error_desc: str, tool_name: str) -> str:
        """Classify error type for better error handling and reporting.

        Args:
            error_desc: Raw error message
            tool_name: Tool that generated the error

        Returns:
            Error category string: syntax_error, table_not_found, permission_error,
            timeout_error, connection_error, or unknown_error
        """
        error_lower = error_desc.lower()

        # Check for specific error patterns
        if "syntax" in error_lower or "parse" in error_lower:
            return "syntax_error"
        elif "not found" in error_lower or "不存在" in error_lower or "unknown" in error_lower:
            if tool_name in ["describe_table", "get_table_ddl", "check_table_exists"]:
                return "table_not_found"
            return "resource_not_found"
        elif "permission" in error_lower or "access denied" in error_lower or "unauthorized" in error_lower:
            return "permission_error"
        elif "timeout" in error_lower or "timed out" in error_lower:
            return "timeout_error"
        elif "connection" in error_lower or "connect" in error_lower or "network" in error_lower:
            return "connection_error"
        else:
            return "unknown_error"

    def _get_recovery_suggestions(self, error_type: str) -> list[str]:
        """Get user-friendly recovery suggestions based on error type.

        Args:
            error_type: Classified error type

        Returns:
            List of recovery suggestion strings (Chinese)
        """
        suggestions_map = {
            "syntax_error": [
                "检查SQL语句是否包含必要的关键词(SELECT, FROM, WHERE等)",
                "验证括号和引号是否正确匹配",
                "确认表名和列名拼写正确",
            ],
            "table_not_found": [
                "验证表名拼写是否正确",
                "确认表存在于当前数据库/schema中",
                "检查数据库权限配置",
            ],
            "permission_error": [
                "检查数据库用户权限配置",
                "联系管理员确认访问权限",
                "确认namespace配置正确",
            ],
            "timeout_error": [
                "尝试简化SQL查询",
                "检查数据量是否过大",
                "考虑添加WHERE条件限制数据范围",
            ],
            "connection_error": [
                "检查数据库连接配置",
                "验证网络连接状态",
                "确认数据库服务是否运行",
            ],
            "unknown_error": [
                "查看详细错误信息",
                "检查系统日志",
                "联系技术支持",
            ],
        }
        return suggestions_map.get(error_type, suggestions_map["unknown_error"])

    def _adjust_tool_sequence_for_missing_tables(
        self, required_tools: list[str], table_names: list[str], table_existence: dict
    ) -> list[str]:
        """Adjust tool sequence based on table existence.

        Args:
            required_tools: Original tool sequence
            table_names: List of table names from SQL
            table_existence: Dict mapping table_name -> existence_result

        Returns:
            Adjusted tool sequence with schema-dependent tools removed if tables missing
        """
        if not table_names:
            return required_tools

        # Check if all tables exist
        all_exist = all(
            table_existence.get(name, {}).get("result", {}).get("table_exists", True) for name in table_names
        )

        if all_exist:
            return required_tools

        # Some tables missing - remove tools that depend on table schema
        schema_dependent_tools = {"describe_table", "get_table_ddl"}
        adjusted_sequence = [t for t in required_tools if t not in schema_dependent_tools]

        missing_count = len(
            [n for n in table_names if not table_existence.get(n, {}).get("result", {}).get("table_exists", True)]
        )
        logger.info(f"Tables missing ({missing_count}), skipping schema-dependent tools: {schema_dependent_tools}")

        return adjusted_sequence

    def _inject_tool_result_into_context(self, workflow, tool_name: str, result: dict):
        """Inject tool results into workflow context for LLM access."""
        if not result.get("success", False):
            return

        try:
            if tool_name == "describe_table":
                # Add table schema information
                if "columns" in result:
                    if not hasattr(workflow.context, "table_schemas") or not workflow.context.table_schemas:
                        workflow.context.table_schemas = []
                    # Avoid duplicates
                    existing_names = [s.get("table_name", "") for s in workflow.context.table_schemas]
                    if result.get("table_name", "") not in existing_names:
                        workflow.context.table_schemas.append(result)

            elif tool_name == "search_external_knowledge":
                # Add external knowledge
                if "result" in result and isinstance(result["result"], list):
                    knowledge_text = "\n".join(
                        [
                            f"- {item.get('terminology', '')}: {item.get('explanation', '')}"
                            for item in result["result"]
                            if item.get("terminology") and item.get("explanation")
                        ]
                    )
                    if knowledge_text:
                        workflow.task.external_knowledge = (
                            workflow.task.external_knowledge or ""
                        ) + f"\n\nStarRocks审查规则:\n{knowledge_text}"

            elif tool_name == "read_query":
                # Add SQL execution results to context
                if "result" in result:
                    sql_context = SQLContext(
                        sql_query=self._extract_sql_from_task(workflow.task.task),
                        explanation="Preflight SQL validation result",
                        sql_return=str(result["result"])[:1000],  # Limit size
                        row_count=result.get("row_count", 0),
                    )
                    workflow.context.sql_contexts.append(sql_context)

            elif tool_name == "get_table_ddl":
                # Add DDL information to external knowledge
                if "result" in result and isinstance(result.get("result"), dict):
                    ddl_info = result["result"].get("definition", "")
                    if ddl_info:
                        workflow.task.external_knowledge = (
                            workflow.task.external_knowledge or ""
                        ) + f"\n\n表DDL信息:\n```sql\n{ddl_info}\n```"

        except Exception as e:
            logger.warning(f"Failed to inject tool result into context for {tool_name}: {e}")

    async def _send_syntax_error_event(self, sql_query: str, error: str):
        """Send SQL syntax error event to frontend."""
        from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus

        error_action = ActionHistory.create_action(
            role=ActionRole.TOOL,
            action_type="sql_execution_error",
            messages=f"SQL syntax validation failed: {error}",
            input_data={"sql_query": sql_query, "error_type": "syntax"},
            output_data={
                "error": error,
                "error_type": "syntax",
                "suggestions": [
                    "检查SQL语句是否包含必要的关键词(SELECT, INSERT, UPDATE, DELETE等)",
                    "验证括号和引号是否匹配",
                    "确认表名和列名拼写是否正确",
                ],
                "can_retry": False,
            },
            status=ActionStatus.FAILED,
        )

        # Emit the error event if action_history_manager is available
        if hasattr(self, "action_history_manager") and self.action_history_manager:
            self.action_history_manager.add_action(error_action)

        return error_action

    async def _send_table_not_found_event(self, table_name: str, error: str):
        """Send table not found error event to frontend."""
        from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus

        error_action = ActionHistory.create_action(
            role=ActionRole.TOOL,
            action_type="table_not_found",
            messages=f"Table '{table_name}' not found: {error}",
            input_data={"table_name": table_name, "error_type": "table_not_found"},
            output_data={
                "error": error,
                "error_type": "table_not_found",
                "suggestions": ["检查表名拼写是否正确", "确认数据库权限", "使用search_table工具查找相似表名"],
                "can_retry": True,
            },
            status=ActionStatus.FAILED,
        )

        # Emit the error event if action_history_manager is available
        if hasattr(self, "action_history_manager") and self.action_history_manager:
            self.action_history_manager.add_action(error_action)

        return error_action

    async def _send_db_connection_error_event(self, sql_query: str, error: str):
        """Send database connection error event to frontend."""
        from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus

        error_action = ActionHistory.create_action(
            role=ActionRole.TOOL,
            action_type="db_connection_error",
            messages=f"Database connection failed: {error}",
            input_data={"sql_query": sql_query, "error_type": "connection"},
            output_data={
                "error": error,
                "error_type": "connection",
                "suggestions": ["检查数据库服务是否运行", "验证网络连接", "确认数据库连接配置"],
                "can_retry": True,
            },
            status=ActionStatus.FAILED,
        )

        # Emit the error event if action_history_manager is available
        if hasattr(self, "action_history_manager") and self.action_history_manager:
            self.action_history_manager.add_action(error_action)

        return error_action

    async def execute_stream(
        self, action_history_manager: Optional[ActionHistoryManager] = None
    ) -> AsyncGenerator[ActionHistory, None]:
        """
        Execute the chat interaction with streaming support.

        Input is accessed from self.input instead of parameters.

        Args:
            action_history_manager: Optional action history manager

        Yields:
            ActionHistory: Progress updates during execution
        """
        # Store action_history_manager for access in update_context
        self.action_history_manager = action_history_manager

        if not action_history_manager:
            action_history_manager = ActionHistoryManager()

        # Initialize unified execution event manager
        self.execution_event_manager = ExecutionEventManager(action_history_manager)

        # Get input from self.input (set by setup_input or directly)
        if not self.input:
            raise ValueError("Chat input not set. Call setup_input() first or set self.input directly.")

        user_input = self.input

        # Determine execution scenario
        scenario = self._determine_execution_scenario(user_input)
        logger.info(f"Detected execution scenario: {scenario}")

        is_plan_mode = getattr(user_input, "plan_mode", False)
        # emit_queue used to stream ActionHistory produced by PlanModeHooks back to node stream
        emit_queue: "asyncio.Queue[ActionHistory]" = asyncio.Queue()

        if is_plan_mode:
            self.plan_mode_active = True

            # Create plan mode hooks
            from rich.console import Console

            from datus.cli.plan_hooks import PlanModeHooks

            console = Console()
            session = self._get_or_create_session()[0]

            # Workflow sets 'auto_execute_plan' in metadata, CLI REPL does not
            auto_mode = getattr(user_input, "auto_execute_plan", False)
            logger.info(f"Plan mode auto_mode: {auto_mode} (from input)")

            # Check for auto-injected knowledge from workflow task
            auto_injected_knowledge = (
                getattr(self.workflow.task, "_auto_injected_knowledge", None) if self.workflow else None
            )

            self.plan_hooks = PlanModeHooks(
                console=console,
                session=session,
                auto_mode=auto_mode,
                action_history_manager=action_history_manager,
                agent_config=self.agent_config,
                emit_queue=emit_queue,
                model=self.model,
                auto_injected_knowledge=auto_injected_knowledge,
            )
            # Enable caching and batching for plan mode
            self.plan_hooks.enable_query_caching = True
            self.plan_hooks.enable_batch_processing = True
            # Set execution event manager if available
            if hasattr(self, "execution_event_manager") and self.execution_event_manager:
                execution_id = f"plan_{int(time.time() * 1000)}"
                self.plan_hooks.set_execution_event_manager(self.execution_event_manager, execution_id)

        # Execute preflight tools if required (for sql_review and text2sql tasks)
        if (scenario in ["sql_review", "text2sql"]) and hasattr(self, "workflow") and self.workflow:
            # Ensure plan_hooks are available for caching and batching
            if not hasattr(self, "plan_hooks") or not self.plan_hooks:
                # Create temporary plan hooks for preflight caching support
                from rich.console import Console

                from datus.cli.plan_hooks import PlanModeHooks

                console = Console()
                session = self._get_or_create_session()[0] if hasattr(self, "_get_or_create_session") else None
                if session:
                    self.plan_hooks = PlanModeHooks(
                        console=console,
                        session=session,
                        auto_mode=False,
                        action_history_manager=action_history_manager,
                        agent_config=self.agent_config,
                        emit_queue=asyncio.Queue(),
                        model=None,  # No LLM needed for preflight
                        auto_injected_knowledge=[],
                    )
                    # Enable caching and batching for preflight
                    self.plan_hooks.enable_query_caching = True
                    # Set execution event manager if available
                    if hasattr(self, "execution_event_manager") and self.execution_event_manager:
                        execution_id = f"preflight_{int(time.time() * 1000)}"
                        self.plan_hooks.set_execution_event_manager(self.execution_event_manager, execution_id)
                    self.plan_hooks.enable_batch_processing = True

            try:
                if scenario == "sql_review":
                    # Use new SQL Review Preflight Orchestrator (v2.4)
                    from datus.agent.node.sql_review_preflight_orchestrator import SQLReviewPreflightOrchestrator

                    sql_review_orchestrator = SQLReviewPreflightOrchestrator(
                        agent_config=self.agent_config,
                        plan_hooks=getattr(self, "plan_hooks", None),
                        execution_event_manager=getattr(self, "execution_event_manager", None),
                    )

                    # Execute all 7 preflight tools with enhanced orchestration
                    async for preflight_action in sql_review_orchestrator.run_preflight_tools(
                        workflow=self.workflow,
                        action_history_manager=action_history_manager,
                        execution_id=f"sql_review_{int(time.time() * 1000)}",
                    ):
                        yield preflight_action

                        # Check if this is a critical failure that should terminate execution
                        if (
                            preflight_action.action_type == "preflight_read_query"
                            and preflight_action.status == ActionStatus.FAILED
                        ):
                            # SQL syntax validation failed, terminate execution
                            logger.info("SQL syntax validation failed during preflight, terminating execution")
                            return
                elif scenario == "text2sql":
                    # Use PreflightOrchestrator for Text2SQL evidence gathering
                    from datus.agent.node.preflight_orchestrator import PreflightOrchestrator

                    preflight_orchestrator = PreflightOrchestrator(
                        agent_config=self.agent_config,
                        plan_hooks=getattr(self, "plan_hooks", None),
                        execution_event_manager=getattr(self, "execution_event_manager", None),
                    )

                    # Run preflight tools for Text2SQL
                    async for preflight_action in preflight_orchestrator.run_preflight_tools(
                        workflow=self.workflow,
                        action_history_manager=action_history_manager,
                        execution_id=f"text2sql_preflight_{int(time.time() * 1000)}",
                        required_tools=[
                            "search_table",
                            "describe_table",
                            "search_reference_sql",
                            "parse_temporal_expressions",
                        ],
                    ):
                        yield preflight_action

            except ValueError as e:
                # Preflight validation failed, already yielded error actions
                logger.info(f"Preflight terminated due to validation error: {e}")
                return
            except Exception as e:
                logger.warning(f"Preflight execution error: {e}")

        # Create execution context and mode
        from datus.agent.node.execution_event_manager import ExecutionContext

        execution_context = ExecutionContext(
            scenario=scenario,
            task_data={"task": user_input.user_message, "input": user_input},
            agent_config=self.agent_config,
            model=self.model,
            workflow_metadata=getattr(self.workflow, "metadata", {}) if self.workflow else {},
        )

        execution_mode = create_execution_mode(scenario, self.execution_event_manager, execution_context, self)

        # Register execution with event manager
        if hasattr(execution_mode, "start") and callable(getattr(execution_mode, "start")):
            await execution_mode.start()

        # Execute using unified execution mode
        execution_success = True
        execution_error = None
        try:
            await execution_mode.execute()
        except Exception as e:
            logger.error(f"Execution mode failed: {e}")
            execution_success = False
            execution_error = str(e)
            # Record the error in the event manager
            execution_id = getattr(execution_mode, "execution_id", f"error_{int(time.time())}")
            await self.execution_event_manager.complete_execution(execution_id, error=execution_error)

        # Create result object for update_context compatibility
        if execution_success:
            # Create a synthetic success result for update_context
            from datus.schemas.base import BaseResult

            self.result = BaseResult(success=True, message="Execution completed successfully")
        else:
            # Create error result for self.result
            from datus.agent.error_handling import NodeErrorResult

            self.result = NodeErrorResult(success=False, error_message=execution_error, error_code="execution_error")

        # Yield any remaining events from the event manager
        async for event in self.execution_event_manager.get_events_stream():
            yield event

    def _determine_execution_scenario(self, user_input) -> str:
        """
        Determine the execution scenario based on input content.

        Args:
        user_input: User input data

        Returns:
        str: Execution scenario ("text2sql", "sql_review", "data_analysis", "smart_query", "deep_analysis")
        """
        message = user_input.user_message.lower() if hasattr(user_input, "user_message") else str(user_input).lower()

        # Check for SQL review keywords
        sql_review_keywords = ["审查", "review", "检查", "check", "评估", "evaluate", "质量", "quality"]
        if any(keyword in message for keyword in sql_review_keywords):
            # Check for SQL content
            if "select" in message or "from" in message or "```sql" in message:
                return "sql_review"

        # Check for data analysis keywords
        analysis_keywords = ["分析", "analysis", "统计", "statistics", "趋势", "trend", "对比", "compare"]
        if any(keyword in message for keyword in analysis_keywords):
            return "data_analysis"

        # Default scenario for unmatched inputs
        return "text2sql"


class ExecutionStatus:
    """Execution status tracker for preflight and main execution (v2.4)."""

    def __init__(self):
        self.preflight_completed = False
        self.syntax_validation_passed = False
        self.tools_executed = []
        self.errors_encountered = []

    def mark_preflight_complete(self, success: bool):
        """Mark preflight execution as completed."""
        self.preflight_completed = True

    def mark_syntax_validation(self, passed: bool):
        """Mark syntax validation result."""
        self.syntax_validation_passed = passed

    def add_tool_execution(self, tool_name: str, success: bool, execution_time: float = None, error_type: str = None):
        """Add tool execution record with enhanced metadata."""
        record = {
            "tool": tool_name,
            "success": success,
            "timestamp": time.time(),
            "execution_time": execution_time,
            "error_type": error_type,
        }
        self.tools_executed.append(record)

    def add_error(self, error_type: str, error_msg: str):
        """Add error record."""
        self.errors_encountered.append({"type": error_type, "message": error_msg, "timestamp": time.time()})

    async def _dispatch_error_event(
        self, error_type: str, sql_query: str, error_desc: str, tool_name: str, table_names: list
    ):
        """根据错误类型分发相应的事件 (v2.4)"""
        try:
            if error_type == "permission_error":
                await self._send_permission_error_event(sql_query, error_desc)
            elif error_type == "timeout_error":
                await self._send_timeout_error_event(sql_query, error_desc)
            elif error_type == "table_not_found":
                table_name = table_names[0] if table_names else "unknown_table"
                await self._send_table_not_found_error_event(table_name, error_desc)
            elif error_type == "connection_error":
                await self._send_db_connection_error_event(sql_query, error_desc)
            # 对于其他错误类型，可以添加默认处理或忽略

        except Exception as e:
            # 错误事件发送失败不应该影响主要流程
            logger.warning(f"Failed to dispatch error event for {error_type}: {e}")

    async def _send_permission_error_event(self, sql_query: str, error_desc: str):
        """Send permission error event."""
        if hasattr(self, "execution_event_manager") and self.execution_event_manager:
            await self.execution_event_manager.send_error_event(
                error_type="permission_error", error_message=error_desc, sql_query=sql_query, severity="high"
            )

    async def _send_timeout_error_event(self, sql_query: str, error_desc: str):
        """Send timeout error event."""
        if hasattr(self, "execution_event_manager") and self.execution_event_manager:
            await self.execution_event_manager.send_error_event(
                error_type="timeout_error", error_message=error_desc, sql_query=sql_query, severity="medium"
            )

    async def _send_table_not_found_error_event(self, table_name: str, error_desc: str):
        """Send table not found error event."""
        if hasattr(self, "execution_event_manager") and self.execution_event_manager:
            await self.execution_event_manager.send_error_event(
                error_type="table_not_found", error_message=error_desc, table_name=table_name, severity="medium"
            )

    async def _send_db_connection_error_event(self, sql_query: str, error_desc: str):
        """Send database connection error event."""
        if hasattr(self, "execution_event_manager") and self.execution_event_manager:
            await self.execution_event_manager.send_error_event(
                error_type="connection_error", error_message=error_desc, sql_query=sql_query, severity="high"
            )
