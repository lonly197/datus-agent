# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from concurrent.futures import ThreadPoolExecutor
from typing import Any, AsyncGenerator, Dict, Optional, cast

from datus.agent.error_handling import unified_error_handler
from datus.agent.node import Node
from datus.agent.workflow import Workflow
from datus.schemas.action_history import (ActionHistory, ActionHistoryManager,
                                          ActionRole, ActionStatus)
from datus.schemas.node_models import ExecuteSQLInput, ExecuteSQLResult
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger
from datus.utils.text_utils import strip_markdown_code_block

logger = get_logger(__name__)


class ExecuteSQLNode(Node):
    """
    Node responsible for executing SQL queries against a database.

    Attributes:
        input (ExecuteSQLInput): Input data for the node.
        result (ExecuteSQLResult): Execution result.
    """

    @unified_error_handler("ExecuteSQLNode", "sql_execution")
    def execute(self) -> None:
        """
        Execute the SQL query.

        Updates self.result with the execution outcome.
        """
        self.result = self._execute_sql()

    async def execute_stream(
        self, action_history_manager: Optional[ActionHistoryManager] = None
    ) -> AsyncGenerator[ActionHistory, None]:
        """
        Execute SQL execution with streaming support.

        Args:
            action_history_manager: Manager for tracking action history.

        Yields:
            ActionHistory: Updates on execution progress.
        """
        async for action in self._execute_sql_stream(action_history_manager):
            yield action

    def setup_input(self, workflow: Workflow) -> Dict[str, Any]:
        """
        Prepare input data for SQL execution.

        Args:
            workflow: The workflow instance.

        Returns:
            Dict containing success status and suggestions.
        """
        if not workflow.context.sql_contexts:
            return {"success": False, "message": "No SQL context available"}

        task = workflow.task
        if not task:
            return {"success": False, "message": "No task available in workflow"}

        last_sql_context = workflow.get_last_sqlcontext()
        if not last_sql_context:
            return {"success": False, "message": "No valid SQL context found"}

        next_input = ExecuteSQLInput(
            sql_query=strip_markdown_code_block(last_sql_context.sql_query),
            database_name=task.database_name,
            permission_reason=None,
        )
        self.input = next_input
        return {"success": True, "message": "Node input appears valid", "suggestions": [next_input]}

    def update_context(self, workflow: Workflow) -> Dict[str, Any]:
        """
        Update SQL execution results to workflow context.

        Args:
            workflow: The workflow instance.

        Returns:
            Dict containing success status.
        """
        result = cast(Optional[ExecuteSQLResult], self.result)
        if not result:
            return {"success": False, "message": "No execution result available"}

        try:
            if not workflow.context.sql_contexts:
                return {"success": False, "message": "No SQL context found to update"}

            last_record = workflow.context.sql_contexts[-1]
            last_record.sql_return = result.sql_return
            last_record.row_count = result.row_count
            last_record.sql_error = result.error

            return {"success": True, "message": "Updated SQL execution context"}
        except Exception as e:
            logger.error(f"Failed to update SQL execution context: {str(e)}")
            return {"success": False, "message": f"SQL execution context update failed: {str(e)}"}

    def _execute_sql(self) -> ExecuteSQLResult:
        """
        Execute SQL query action to run the generated query.

        Returns:
            ExecuteSQLResult: Object containing execution results or errors.

        Raises:
            DatusException: If database connection fails or query times out.
        """
        input_data = cast(ExecuteSQLInput, self.input)
        if not input_data:
            raise DatusException(ErrorCode.COMMON_VALIDATION_FAILED, "Node input is missing")

        db_connector = self._sql_connector(input_data.database_name)
        if not db_connector:
            raise DatusException(
                ErrorCode.DB_CONNECTION_FAILED,
                "Database connection not initialized in workflow",
                {"database_name": input_data.database_name or "unknown"},
            )

        logger.debug(f"SQL execution input: {input_data}")

        # determine timeout: per-input -> task-level -> agent config
        timeout = None
        if input_data.query_timeout_seconds:
            timeout = int(input_data.query_timeout_seconds)
        elif getattr(self, "agent_config", None) and getattr(self.agent_config, "default_query_timeout_seconds", None):
            # agent_config is Optional, so check if not None
            if self.agent_config:
                timeout = int(self.agent_config.default_query_timeout_seconds)

        # Run blocking execute in thread and enforce timeout if provided
        def run_execute():
            return db_connector.execute(input_data)

        if timeout and timeout > 0:
            with ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(run_execute)
                try:
                    result = future.result(timeout=timeout)
                    logger.debug(f"SQL execution result: {result}")
                    return result
                except TimeoutError:
                    # Attempt best-effort cancellation/cleanup
                    try:
                        db_connector.close()
                    except (AttributeError, ConnectionError, Exception) as cleanup_error:
                        logger.warning(f"Failed to close database connection during timeout cleanup: {cleanup_error}")
                    raise DatusException(
                        ErrorCode.DB_EXECUTION_TIMEOUT,
                        f"Query timed out after {timeout} seconds",
                        {
                            "timeout_seconds": timeout,
                            "sql_preview": input_data.sql_query[:100] if input_data.sql_query else "",
                        },
                    )
        else:
            result = db_connector.execute(input_data)
            logger.debug(f"SQL execution result: {result}")
            return result

    async def _execute_sql_stream(
        self, action_history_manager: Optional[ActionHistoryManager] = None
    ) -> AsyncGenerator[ActionHistory, None]:
        """Execute SQL with streaming support and action history tracking."""
        try:
            input_data = cast(ExecuteSQLInput, self.input)
            if not input_data:
                # Should have been caught by validation, but safe check
                return

            # Database connection action
            connection_action = ActionHistory(
                action_id="database_connection",
                role=ActionRole.WORKFLOW,
                messages="Establishing database connection for SQL execution",
                action_type="database_connection",
                input={
                    "database_name": input_data.database_name or "",
                    "sql_query_preview": (
                        input_data.sql_query[:50] + "..."
                        if input_data.sql_query and len(input_data.sql_query) > 50
                        else (input_data.sql_query or "")
                    ),
                },
                status=ActionStatus.PROCESSING,
            )
            yield connection_action

            # Initialize database connection
            try:
                sql_connector = self._sql_connector(input_data.database_name)

                if not sql_connector:
                    connection_action.status = ActionStatus.FAILED
                    connection_action.output = {"error": "Database connection not initialized"}
                    logger.error("Database connection not initialized in workflow")
                    return

                connection_action.status = ActionStatus.SUCCESS
                connection_action.output = {
                    "connection_established": True,
                    "database_connected": True,
                }
            except Exception as e:
                connection_action.status = ActionStatus.FAILED
                connection_action.output = {"error": str(e)}
                logger.error(f"Database connection failed: {e}")
                raise

            # SQL execution action
            execution_action = ActionHistory(
                action_id="sql_execution",
                role=ActionRole.WORKFLOW,
                messages="Executing SQL query against the database",
                action_type="sql_execution",
                input={
                    "sql_query": input_data.sql_query or "",
                    "database_name": input_data.database_name or "",
                },
                status=ActionStatus.PROCESSING,
            )
            yield execution_action

            # Execute SQL - reuse existing logic
            try:
                result = self._execute_sql()

                execution_action.status = ActionStatus.SUCCESS if result.success else ActionStatus.FAILED
                execution_action.output = {
                    "success": result.success,
                    "row_count": result.row_count if result.row_count is not None else 0,
                    "has_results": bool(result.sql_return),
                    "sql_result": result.sql_return,
                    "error": result.error if result.error else None,
                }

                # Store result for later use
                self.result = result

            except Exception as e:
                execution_action.status = ActionStatus.FAILED
                execution_action.output = {"error": str(e)}
                logger.error(f"SQL execution error: {str(e)}")
                raise

            # Yield the updated execution action with final status
            yield execution_action

        except Exception as e:
            logger.error(f"SQL execution streaming error: {str(e)}")
            raise
