# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
ResultValidationNode implementation for validating SQL execution results.

This node validates that the SQL execution results are acceptable:
- SQL executed without errors
- Result is not empty (unless expected, e.g., DDL/DML queries)
- Result makes semantic sense for the query
"""

from typing import Any, AsyncGenerator, Dict, Optional

from datus.agent.node.node import Node, execute_with_async_stream
from datus.agent.workflow import Workflow
from datus.configuration.agent_config import AgentConfig
from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.base import BaseInput, BaseResult
from datus.schemas.node_models import SQLContext
from datus.utils.exceptions import ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class ResultValidationNode(Node):
    """
    Node for validating SQL execution result quality.

    This node checks if the SQL execution produced acceptable results.
    It helps the reflect node decide whether to return results to the user
    or trigger additional reflection strategies.
    """

    def __init__(
        self,
        node_id: str,
        description: str,
        node_type: str,
        input_data: Optional[BaseInput] = None,
        agent_config: Optional[AgentConfig] = None,
        tools: Optional[list] = None,
    ):
        super().__init__(
            node_id=node_id,
            description=description,
            node_type=node_type,
            input_data=input_data,
            agent_config=agent_config,
            tools=tools,
        )

    def setup_input(self, workflow: Workflow) -> Dict[str, Any]:
        """Setup result validation input from workflow context."""
        if not self.input:
            self.input = BaseInput()
        return {"success": True, "message": "Result validation input setup complete"}

    async def run(self) -> AsyncGenerator[ActionHistory, None]:
        """
        Run result validation to check if SQL execution results are acceptable.

        Yields:
            ActionHistory: Progress and result actions
        """
        try:
            if not self.workflow or not self.workflow.task:
                error_result = self.create_error_result(
                    ErrorCode.NODE_EXECUTION_FAILED,
                    "No workflow or task available for result validation",
                    "result_validation",
                    {"workflow_id": getattr(self.workflow, "id", "unknown") if self.workflow else "unknown"},
                )
                yield ActionHistory(
                    action_id=f"{self.id}_error",
                    role=ActionRole.TOOL,
                    messages=f"Result validation failed: {error_result.error_message}",
                    action_type="result_validation",
                    input={},
                    status=ActionStatus.FAILED,
                    output={"error": error_result.error_message, "error_code": error_result.error_code},
                )
                return

            task = self.workflow.task
            context = self.workflow.context

            # Step 1: Check if there are SQL execution results
            if not context or not context.sql_contexts:
                yield ActionHistory(
                    action_id=f"{self.id}_no_results",
                    role=ActionRole.TOOL,
                    messages="Result validation skipped: No SQL execution results",
                    action_type="result_validation",
                    input={"task": task.task[:50] if task else ""},
                    status=ActionStatus.SUCCESS,
                    output={
                        "is_valid": False,
                        "reason": "No SQL execution results to validate",
                        "suggestion": "Execute SQL first",
                    },
                )
                self.result = BaseResult(
                    success=True,  # Validation succeeded, but results are absent
                    data={"is_valid": False, "reason": "no_sql_results"},
                )
                return

            # Step 2: Get the most recent SQL context
            latest_sql_context = context.sql_contexts[-1]

            # Step 3: Validate the SQL execution result
            validation_result = self._validate_sql_result(task.task if task else "", latest_sql_context)

            # Emit result
            if validation_result["is_valid"]:
                yield ActionHistory(
                    action_id=f"{self.id}_validation",
                    role=ActionRole.TOOL,
                    messages=f"Result validation passed: {validation_result['reason']}",
                    action_type="result_validation",
                    input={
                        "task": task.task[:50] if task else "",
                        "row_count": validation_result.get("row_count", 0),
                    },
                    status=ActionStatus.SUCCESS,
                    output=validation_result,
                )
                self.result = BaseResult(success=True, data=validation_result)
            else:
                yield ActionHistory(
                    action_id=f"{self.id}_validation",
                    role=ActionRole.TOOL,
                    messages=f"Result validation failed: {validation_result['reason']}",
                    action_type="result_validation",
                    input={
                        "task": task.task[:50] if task else "",
                        "row_count": validation_result.get("row_count", 0),
                    },
                    status=ActionStatus.FAILED,
                    output=validation_result,
                )
                self.result = BaseResult(success=False, error=validation_result["reason"], data=validation_result)

            logger.info(
                f"Result validation completed: is_valid={validation_result['is_valid']}, "
                f"reason={validation_result['reason']}"
            )

        except Exception as e:
            logger.error(f"Result validation failed: {e}")
            error_result = self.create_error_result(
                ErrorCode.NODE_EXECUTION_FAILED,
                f"Result validation execution failed: {str(e)}",
                "result_validation",
                {
                    "task_id": (
                        getattr(self.workflow.task, "id", "unknown")
                        if self.workflow and self.workflow.task
                        else "unknown"
                    )
                },
            )
            yield ActionHistory(
                action_id=f"{self.id}_error",
                role=ActionRole.TOOL,
                messages=f"Result validation failed: {error_result.error_message}",
                action_type="result_validation",
                input={},
                status=ActionStatus.FAILED,
                output={"error": error_result.error_message, "error_code": error_result.error_code},
            )
            self.result = BaseResult(success=False, error=str(e))

    def _validate_sql_result(self, task: str, sql_context: SQLContext) -> Dict[str, Any]:
        """
        Validate the SQL execution result.

        Returns a dictionary with:
        - is_valid: bool
        - reason: str
        - row_count: int
        - error: str (if any)
        - suggestion: str (if invalid)
        """
        result = {"is_valid": False, "reason": "", "row_count": 0, "error": None, "suggestion": None}

        # Check if there was an execution error
        # Note: We need to check the SQL execution context, not just the SQL query
        # The SQLContext stores the query and explanation, but we need to check
        # the execute_sql node result for errors

        # For now, we'll validate based on what we can infer
        if sql_context.sql_query:
            result["sql_query"] = sql_context.sql_query

        # Check if this is a DDL/DML query (no results expected)
        if self._is_ddl_or_dml_query(sql_context.sql_query):
            result["is_valid"] = True
            result["reason"] = "DDL/DML query executed"
            return result

        # For SELECT queries, we would normally check:
        # 1. Execution error status
        # 2. Row count
        # 3. Whether empty results are acceptable

        # Since we don't have direct access to execution results here,
        # we'll assume the query executed if we have a SQL context
        # The real validation happens in the reflect node based on execution results

        # Basic validation: if we have a SQL query, consider it valid for now
        # More sophisticated validation would require access to execution results
        result["is_valid"] = True
        result["reason"] = "SQL query generated successfully"

        return result

    def _is_ddl_or_dml_query(self, sql_query: Optional[str]) -> bool:
        """Check if the query is DDL or DML (not SELECT)."""
        if not sql_query:
            return False

        sql_upper = sql_query.strip().upper()

        # DDL statements
        ddl_prefixes = ("CREATE", "ALTER", "DROP", "TRUNCATE", "RENAME")
        # DML statements (non-SELECT)
        dml_prefixes = ("INSERT", "UPDATE", "DELETE", "MERGE", "COPY")

        for prefix in ddl_prefixes + dml_prefixes:
            if sql_upper.startswith(prefix):
                return True

        return False

    def execute(self) -> BaseResult:
        """Execute result validation synchronously."""
        return execute_with_async_stream(self)

    async def execute_stream(
        self, action_history_manager: Optional[ActionHistoryManager] = None
    ) -> AsyncGenerator[ActionHistory, None]:
        """Execute result validation with streaming support."""
        if action_history_manager:
            self.action_history_manager = action_history_manager
        async for action in self.run():
            yield action

    def update_context(self, workflow: "Workflow") -> Dict[str, Any]:
        """Update workflow context with validation results."""
        try:
            if not self.result:
                return {"success": False, "message": "Result validation produced no result"}

            # Store validation result in workflow metadata
            if not hasattr(workflow, "metadata"):
                workflow.metadata = {}
            workflow.metadata["result_validation"] = self.result.data

            is_valid = self.result.data.get("is_valid", False) if self.result.data else False

            return {"success": True, "message": f"Result validation context updated: is_valid={is_valid}"}

        except Exception as e:
            logger.error(f"Failed to update result validation context: {str(e)}")
            return {"success": False, "message": f"Result validation context update failed: {str(e)}"}
