# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from typing import Any, AsyncGenerator, Dict, Optional, cast

from datus.agent.node import Node
from datus.agent.workflow import Workflow
from datus.schemas.action_history import (ActionHistory, ActionHistoryManager,
                                          ActionRole, ActionStatus)
from datus.schemas.node_models import OutputInput
from datus.tools.output_tools import OutputTool
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class OutputNode(Node):
    """
    Node responsible for generating the final output and benchmark reports.
    """

    def execute(self) -> None:
        """Execute the output generation logic."""
        self.result = self._execute_output()

    async def execute_stream(
        self, action_history_manager: Optional[ActionHistoryManager] = None
    ) -> AsyncGenerator[ActionHistory, None]:
        """
        Execute output generation with streaming support.

        Args:
            action_history_manager: Manager for tracking action history.

        Yields:
            ActionHistory: Updates on execution progress.
        """
        async for action in self._output_stream(action_history_manager):
            yield action

    def setup_input(self, workflow: Workflow) -> Dict[str, Any]:
        """
        Prepare input data for output generation.

        Args:
            workflow: The workflow instance.

        Returns:
            Dict containing success status and suggestions.
        """
        if not workflow.task:
            return {"success": False, "message": "No task available in workflow"}

        # Get SQL context if available, otherwise handle plan mode gracefully
        sql_context = None
        try:
            sql_context = workflow.get_last_sqlcontext()
        except Exception:
            # No SQL context - may be analysis/review task without SQL execution
            logger.info("No SQL context available, treating as non-SQL-execution task")

        # Handle both SQL execution and analysis/review tasks
        gen_sql = ""
        sql_result = ""
        row_count = 0
        error = None

        if sql_context:
            gen_sql = sql_context.sql_query or ""
            sql_result = sql_context.sql_return or ""
            row_count = sql_context.row_count or 0
            error = sql_context.sql_error
        # If no SQL context, check if this is plan mode (analysis task)
        elif workflow.metadata.get("plan_mode"):
            # For plan mode without SQL execution, the response is in action history
            logger.info("Plan mode without SQL execution - output will use chat response")

        # Collect workflow metadata for comprehensive report generation
        workflow_metadata = {
            "sql_validation": workflow.metadata.get("sql_validation"),
            "intent_clarification": workflow.metadata.get("intent_clarification"),
            "clarified_task": workflow.metadata.get("clarified_task"),
            "intent_analysis": workflow.metadata.get("intent_analysis"),
            "reflection_count": workflow.metadata.get("reflection_count") or None,  # Use None instead of 0
            "table_schemas": workflow.context.table_schemas,  # Pass table schemas for developer report
        }

        # Debug logging for metadata troubleshooting
        logger.debug(f"output_node metadata keys: {list(workflow_metadata.keys())}")
        logger.debug(
            f"table_schemas count: {len(workflow.context.table_schemas) if workflow.context.table_schemas else 0}"
        )
        logger.debug(f"reflection_count: {workflow_metadata['reflection_count']}")

        # normally last node of workflow
        next_input = OutputInput(
            finished=True,
            task_id=workflow.task.id,
            task=workflow.get_task(),
            database_name=workflow.task.database_name,
            output_dir=workflow.task.output_dir,
            gen_sql=gen_sql,
            sql_result=sql_result,
            row_count=row_count,
            table_schemas=workflow.context.table_schemas,
            metrics=workflow.context.metrics,
            external_knowledge=workflow.task.external_knowledge,
            error=error,
            metadata=workflow_metadata,
        )
        self.input = next_input
        return {"success": True, "message": "Output appears valid", "suggestions": [next_input]}

    def update_context(self, workflow: Workflow) -> Dict[str, Any]:
        """
        Update workflow context. Output node typically doesn't need to update context.

        Args:
            workflow: The workflow instance.

        Returns:
            Dict containing success status.
        """
        return {"success": True, "message": "Output node, no context update needed"}

    def _execute_output(self) -> Any:
        """
        Execute output action to present the results.

        Returns:
            Any: The result of the output tool execution.
        """
        input_data = cast(OutputInput, self.input)
        if not input_data:
            raise DatusException(ErrorCode.COMMON_VALIDATION_FAILED, "Node input is missing")

        tool = OutputTool()
        return tool.execute(input_data, sql_connector=self._sql_connector(input_data.database_name), model=self.model)

    async def _output_stream(
        self, action_history_manager: Optional[ActionHistoryManager] = None
    ) -> AsyncGenerator[ActionHistory, None]:
        """Execute output generation with streaming support and action history tracking."""
        input_data = cast(OutputInput, self.input)
        if not input_data:
            return

        try:
            # Output generation action
            generation_action = ActionHistory(
                action_id="output_generation",
                role=ActionRole.WORKFLOW,
                messages="Generating final output with results and benchmark data",
                action_type="output_generation",
                input={
                    "task_id": input_data.task_id if hasattr(input_data, "task_id") else "",
                    "has_sql_result": bool(getattr(input_data, "sql_result", None)),
                    "row_count": getattr(input_data, "row_count", 0),
                },
                status=ActionStatus.PROCESSING,
            )
            yield generation_action

            # Execute output generation
            result = self._execute_output()

            generation_action.status = ActionStatus.SUCCESS
            generation_action.output = {
                "output_generated": True,
                "has_benchmark_data": bool(result),
                "success": getattr(result, "success", True) if result else True,
                "sql_query": getattr(input_data, "gen_sql", "") if hasattr(input_data, "gen_sql") else "",
                "sql_result": getattr(input_data, "sql_result", "") if hasattr(input_data, "sql_result") else "",
                "metadata": getattr(input_data, "metadata", {}) if hasattr(input_data, "metadata") else {},
            }

            # Store result for later use
            self.result = result

            # Yield the updated action with final status
            yield generation_action

        except Exception as e:
            logger.error(f"Output generation streaming error: {str(e)}")
            raise
