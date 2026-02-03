# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details

"""
Workflow Termination and Error Handling.

Handles workflow termination logic, error reporting,
and ensures output node execution.
"""

import asyncio
from typing import Any, AsyncGenerator, Dict, List, Optional

from datus.agent.evaluate import setup_node_input
from datus.agent.node import Node
from datus.agent.runner.workflow_lifecycle import ActionHistoryManagerMixin
from datus.agent.workflow import Workflow
from datus.agent.workflow_status import WorkflowTerminationStatus
from datus.configuration.node_type import NodeType
from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionStatus
from datus.utils.error_handling import check_reflect_node_reachable
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class WorkflowTerminationManager:
    """Manages workflow termination and error handling."""

    def __init__(self, workflow: Optional[Workflow] = None, workflow_task: Optional[asyncio.Task] = None):
        self.workflow = workflow
        self.workflow_task = workflow_task

    def set_workflow(self, workflow: Workflow):
        """Set the workflow reference."""
        self.workflow = workflow

    def set_workflow_task(self, task: asyncio.Task):
        """Set the workflow task for cancellation."""
        self.workflow_task = task

    def handle_node_failure(
        self,
        current_node: Node,
        node_start_action: Optional[ActionHistory] = None,
    ) -> WorkflowTerminationStatus:
        """
        Handle node failure with unified soft/hard failure logic.

        Args:
            current_node: The failed node
            node_start_action: Optional action history entry for streaming mode

        Returns:
            WorkflowTerminationStatus: Clear termination status for workflow execution
        """
        # 1. Check ActionStatus first (SOFT_FAILED vs FAILED)
        if hasattr(current_node, "last_action_status"):
            last_status = current_node.last_action_status
            if last_status == ActionStatus.SOFT_FAILED:
                logger.info(f"Node returned SOFT_FAILED status: {current_node.description}")
                return WorkflowTerminationStatus.SKIP_TO_REFLECT

            elif last_status == ActionStatus.FAILED:
                if current_node.type in {"schema_discovery", "schema_validation"}:
                    if self.workflow and self.workflow.metadata is not None:
                        self.workflow.metadata["termination_status"] = WorkflowTerminationStatus.PROCEED_TO_OUTPUT
                        self.workflow.metadata["termination_reason"] = (
                            f"{current_node.type} failed; proceeding to output with SQL generation failure report"
                        )
                        self.workflow.metadata["sql_generation_failed"] = True
                        self.workflow.metadata["failure_stage"] = current_node.type
                    return WorkflowTerminationStatus.PROCEED_TO_OUTPUT
                # Special case: if the reflect node itself failed, proceed to output for report generation
                if current_node.type == NodeType.TYPE_REFLECT:
                    logger.info(
                        f"Reflect node failed, proceeding to output node for report generation: "
                        f"{current_node.description}"
                    )
                    # Set metadata for output node to generate proper failure report
                    if self.workflow and self.workflow.metadata is not None:
                        self.workflow.metadata["termination_status"] = WorkflowTerminationStatus.PROCEED_TO_OUTPUT
                        # Get termination_reason from reflect node result if available
                        termination_reason = getattr(current_node.result, "error", None) if hasattr(current_node, "result") and current_node.result else None
                        if not termination_reason and hasattr(current_node, "result") and current_node.result:
                            termination_reason = current_node.result.details.get("termination_reason") if hasattr(current_node.result, "details") and current_node.result.details else None
                        if not termination_reason:
                            termination_reason = f"Reflect node failed after {getattr(self.workflow, 'reflection_round', 0)} reflection rounds"
                        self.workflow.metadata["termination_reason"] = termination_reason
                        self.workflow.metadata["sql_generation_failed"] = True
                        self.workflow.metadata["failure_stage"] = "reflection"
                    return WorkflowTerminationStatus.PROCEED_TO_OUTPUT

                # Check if reflect node is reachable for recovery
                has_reflect = check_reflect_node_reachable(self.workflow)

                if has_reflect:
                    # Soft failure - continue to reflection for recovery
                    logger.info(
                        f"Node failed but reflect node is reachable. "
                        f"Continuing as Soft Failure: {current_node.description}"
                    )
                    return WorkflowTerminationStatus.SKIP_TO_REFLECT
                else:
                    # Hard failure - terminate workflow immediately
                    logger.error(
                        f"Node failed with no reachable reflect node. "
                        f"Terminating workflow: {current_node.description}"
                    )

                    # Update action history if in streaming mode
                    if node_start_action:
                        self._update_action_status(
                            node_start_action,
                            success=False,
                            error=f"Node execution failed (no recovery path): {current_node.description}",
                        )

                    return WorkflowTerminationStatus.TERMINATE_WITH_ERROR
            else:
                # Unknown status - default to checking for reflect
                # Special case: if the reflect node itself failed, proceed to output for report generation
                if current_node.type == NodeType.TYPE_REFLECT:
                    logger.info(
                        f"Reflect node failed with unknown status, proceeding to output node for report generation: "
                        f"{current_node.description}"
                    )
                    # Set metadata for output node to generate proper failure report
                    if self.workflow and self.workflow.metadata is not None:
                        self.workflow.metadata["termination_status"] = WorkflowTerminationStatus.PROCEED_TO_OUTPUT
                        self.workflow.metadata["termination_reason"] = f"Reflect node failed with unknown status"
                        self.workflow.metadata["sql_generation_failed"] = True
                        self.workflow.metadata["failure_stage"] = "reflection"
                    return WorkflowTerminationStatus.PROCEED_TO_OUTPUT

                has_reflect = check_reflect_node_reachable(self.workflow)
                if has_reflect:
                    return WorkflowTerminationStatus.SKIP_TO_REFLECT
                else:
                    return WorkflowTerminationStatus.TERMINATE_WITH_ERROR
        else:
            # Fallback to old logic if last_action_status not available
            # Special case: if the reflect node itself failed, proceed to output for report generation
            if current_node.type == NodeType.TYPE_REFLECT:
                logger.info(
                    f"Reflect node failed (no action status), proceeding to output node for report generation: "
                    f"{current_node.description}"
                )
                # Set metadata for output node to generate proper failure report
                if self.workflow and self.workflow.metadata is not None:
                    self.workflow.metadata["termination_status"] = WorkflowTerminationStatus.PROCEED_TO_OUTPUT
                    self.workflow.metadata["termination_reason"] = f"Reflect node failed (no action status)"
                    self.workflow.metadata["sql_generation_failed"] = True
                    self.workflow.metadata["failure_stage"] = "reflection"
                return WorkflowTerminationStatus.PROCEED_TO_OUTPUT

            has_reflect = check_reflect_node_reachable(self.workflow)
            if has_reflect:
                logger.info(
                    f"Node failed but workflow has reflection. "
                    f"Continuing as Soft Failure: {current_node.description}"
                )
                return WorkflowTerminationStatus.SKIP_TO_REFLECT
            else:
                # Hard failure - terminate workflow
                logger.error(f"Node failed: {current_node.description}")

                # Update action history if in streaming mode
                if node_start_action:
                    self._update_action_status(
                        node_start_action,
                        success=False,
                        error=f"Node execution failed: {current_node.description}",
                    )

                return WorkflowTerminationStatus.TERMINATE_WITH_ERROR

    def handle_node_failure_legacy(
        self,
        current_node: Node,
        node_start_action: Optional[ActionHistory] = None,
    ) -> tuple[bool, bool]:
        """
        Legacy method for backward compatibility.

        Returns:
            tuple: (is_soft_failure: bool, should_continue: bool)
        """
        termination_status = self.handle_node_failure(current_node, node_start_action)

        if termination_status == WorkflowTerminationStatus.SKIP_TO_REFLECT:
            return True, True
        elif termination_status == WorkflowTerminationStatus.TERMINATE_WITH_ERROR:
            return False, False
        else:
            return True, True

    def terminate_workflow(
        self,
        termination_status: WorkflowTerminationStatus,
        error_message: Optional[str] = None,
    ):
        """
        Terminate workflow with proper event sending and task cancellation.

        This method ensures that:
        1. ErrorEvent and CompletedEvent are both sent
        2. Background workflow task is cancelled to prevent continued execution
        3. Proper logging is performed

        Args:
            termination_status: The termination status (TERMINATE_WITH_ERROR or TERMINATE_SUCCESS)
            error_message: Optional error message for failure termination
        """
        if termination_status == WorkflowTerminationStatus.TERMINATE_WITH_ERROR:
            logger.error(f"Workflow terminated with error: {error_message}")

            # Note: ErrorEvent and CompletedEvent will be sent by run_stream's finally block
            # This method only ensures task cancellation

            # CRITICAL: Cancel workflow task to prevent background execution
            if self.workflow_task and not self.workflow_task.done():
                logger.info("Cancelling workflow_task to prevent background execution")
                self.workflow_task.cancel()
                # Verify cancellation
                try:
                    # Allow a brief moment for cancellation to take effect
                    if hasattr(asyncio, 'get_running_loop'):
                        loop = asyncio.get_running_loop()
                        # Small delay to allow cancellation processing
                        # Note: We can't await in synchronous context, but we can log if task isn't cancelled
                        if not self.workflow_task.cancelled():
                            logger.warning("Task may not have been cancelled successfully immediately")
                except Exception as e:
                    logger.error(f"Error during workflow task cancellation: {e}", exc_info=True)

        elif termination_status == WorkflowTerminationStatus.TERMINATE_SUCCESS:
            logger.info("Workflow completed successfully")

    def _update_action_status(
        self,
        action: ActionHistory,
        success: bool,
        error: Optional[str] = None,
    ):
        """Update action status (helper method)."""
        if success:
            action.status = ActionStatus.SUCCESS
            action.output = action.output or {}
        else:
            action.status = ActionStatus.FAILED
            action.output = {"error": error or "Unknown error"}


class ErrorMarkdownBuilder:
    """Builds error messages in markdown format."""

    def __init__(self, workflow: Optional[Workflow] = None):
        self.workflow = workflow

    def set_workflow(self, workflow: Workflow):
        """Set the workflow reference."""
        self.workflow = workflow

    def build_termination_error_markdown(self) -> Optional[str]:
        """Build markdown-formatted error message for ErrorEvent."""
        if not self.workflow or not self.workflow.metadata:
            return None

        helper = ActionHistoryManagerMixin()
        suggestions = helper.generate_failure_suggestions(self.workflow)

        lines = [
            "# SQL生成失败",
            "",
        ]

        # Error reason
        termination_reason = self.workflow.metadata.get("termination_reason", "Unknown error")
        lines.append(f"**错误原因**: {termination_reason}")
        lines.append("")

        # Reflection rounds
        reflection_round = getattr(self.workflow, "reflection_round", 0)
        lines.append(f"**总尝试次数**: {reflection_round}")
        lines.append("")

        # Strategies used
        strategy_counts = self.workflow.metadata.get("strategy_counts", {})
        if strategy_counts:
            lines.append("**使用的策略**:")
            for strategy, count in strategy_counts.items():
                lines.append(f"- {strategy}: {count}次")
            lines.append("")

        # Last SQL query and error
        sql_contexts = self.workflow.context.sql_contexts if self.workflow.context else []
        if sql_contexts:
            last_sql = sql_contexts[-1]
            if hasattr(last_sql, "sql_query") and last_sql.sql_query:
                lines.append("**最后执行的SQL**:")
                lines.append("```sql")
                lines.append(last_sql.sql_query)
                lines.append("```")
                lines.append("")

            if hasattr(last_sql, "sql_error") and last_sql.sql_error:
                lines.append("**错误信息**:")
                lines.append(f"```\n{last_sql.sql_error}\n```")
                lines.append("")

        # Suggestions
        lines.append("**建议**:")
        for i, suggestion in enumerate(suggestions, 1):
            lines.append(f"{i}. {suggestion}")

        return "\n".join(lines)


class OutputNodeExecutor:
    """Ensures output node execution even when workflow exits early."""

    def __init__(self, workflow: Optional[Workflow] = None):
        self.workflow = workflow

    def set_workflow(self, workflow: Workflow):
        """Set the workflow reference."""
        self.workflow = workflow

    async def run_stream(
        self,
        action_history_manager: Optional[ActionHistoryManager] = None,
    ) -> AsyncGenerator[ActionHistory, None]:
        """
        Execute the output node with streaming support for SSE scenarios.

        This method is used in /workflows/chat_research streaming mode.
        It yields ActionHistory events including output_generation.

        Args:
            action_history_manager: Manager for tracking action history (optional)

        Yields:
            ActionHistory: Streaming action history events
        """
        if not self.workflow or not self.workflow.nodes:
            return

        # Skip if output was already executed successfully
        if self.workflow and self.workflow.metadata is not None:
            if self.workflow.metadata.get("_output_executed"):
                return

        # Find the output node
        output_node = None
        for node in self.workflow.nodes.values():
            if node.type == "output":
                output_node = node
                break

        if not output_node:
            logger.debug("No output node found in workflow")
            return

        # Create action helper for generating ActionHistory
        action_helper = ActionHistoryManagerMixin()

        # Check if output node needs to be executed
        if output_node.status in ["completed", "skipped"]:
            logger.debug(f"Output node already executed (status: {output_node.status})")
            if self.workflow and self.workflow.metadata is not None:
                self.workflow.metadata["_output_executed"] = True
            return

        if output_node.status == "failed":
            logger.warning(f"Output node previously failed, attempting to re-execute")
            output_node.status = "pending"
            output_node.result = None

        # Create output generation start action (only when executing)
        output_start_action = action_helper.create_action_history(
            action_id="output_generation_start",
            messages="Starting output generation for SQL report",
            action_type="output_generation",
            input_data={
                "task_id": getattr(self.workflow, "task_id", None),
                "has_error": bool(output_node.input.error if hasattr(output_node, "input") and output_node.input else False),
            },
        )
        yield output_start_action

        # Find output node index in node_order
        output_idx = None
        for idx, node_id in enumerate(self.workflow.node_order):
            if node_id == output_node.id:
                output_idx = idx
                break

        if output_idx is None:
            output_idx = len(self.workflow.node_order)
            self.workflow.node_order.append(output_node.id)
            logger.info(f"Added output node to node_order at index {output_idx}")

        # Update current_node_index to output node position
        old_index = self.workflow.current_node_index
        self.workflow.current_node_index = output_idx
        logger.info(
            f"Forcing output node execution: "
            f"old_index={old_index}, new_index={output_idx}, "
            f"node_order_len={len(self.workflow.node_order)}"
        )

        # Setup input for output node
        setup_node_input(output_node, self.workflow)

        # Execute output node using run_stream to collect all actions
        logger.info(f"Executing pending output node (stream): {output_node.description}")

        # Use the node's run_stream method
        async for action in output_node.run_stream(action_history_manager):
            yield action

        # Create completion action
        output_status = "completed" if output_node.status == "completed" else "failed"
        output_helper = ActionHistoryManagerMixin()
        output_completion_action = output_helper.create_action_history(
            action_id="output_generation_completion",
            messages=f"Output generation {output_status}",
            action_type="output_generation_completion",
            input_data={
                "task_id": getattr(self.workflow, "task_id", None),
                "status": output_status,
                "has_result": bool(output_node.result),
            },
        )

        # Set result on completion action
        if output_node.result:
            if hasattr(output_node.result, "success"):
                metadata = {}
                if hasattr(output_node, "input") and output_node.input:
                    metadata = getattr(output_node.input, "metadata", {}) or {}
                output_completion_action.output = {
                    "success": output_node.result.success,
                    "sql_query": getattr(output_node.result, "sql_query", ""),
                    "sql_result": getattr(output_node.result, "sql_result", ""),
                    "sql_query_final": getattr(output_node.result, "sql_query_final", ""),
                    "sql_result_final": getattr(output_node.result, "sql_result_final", ""),
                    "row_count": getattr(output_node.input, "row_count", 0)
                    if hasattr(output_node, "input") and output_node.input
                    else 0,
                    "metadata": metadata,
                }
            elif isinstance(output_node.result, dict):
                output_completion_action.output = output_node.result
                if hasattr(output_node, "input") and output_node.input:
                    output_completion_action.output.setdefault(
                        "metadata", getattr(output_node.input, "metadata", {}) or {}
                    )

        output_completion_action.status = ActionStatus.SUCCESS if output_status == "completed" else ActionStatus.FAILED
        yield output_completion_action

        if output_status == "completed" and self.workflow and self.workflow.metadata is not None:
            self.workflow.metadata["_output_executed"] = True

    def run(self, metadata: Optional[Dict] = None) -> None:
        """
        Execute the output node synchronously for non-streaming scenarios.

        This method is used in sync mode and ensures ActionHistory is recorded.

        Args:
            metadata: The metadata dict that will be passed to _finalize_workflow
        """
        if not self.workflow or not self.workflow.nodes:
            return

        # Create a temporary ActionHistoryManager for recording actions
        action_history_manager = ActionHistoryManager()
        action_history_manager.initialize(
            task_id=getattr(self.workflow, "task_id", "unknown"),
            task=getattr(self.workflow, "task", None),
        )

        # Create action helper for generating ActionHistory
        action_helper = ActionHistoryManagerMixin()

        # Find the output node
        output_node = None
        for node in self.workflow.nodes.values():
            if node.type == "output":
                output_node = node
                break

        if not output_node:
            logger.debug("No output node found in workflow")
            return

        # Create output generation action
        output_action = action_helper.create_action_history(
            action_id="output_generation",
            messages="Generating final output with results and benchmark data",
            action_type="output_generation",
            input_data={
                "task_id": getattr(self.workflow, "task_id", None),
                "has_error": bool(
                    output_node.input.error
                    if hasattr(output_node, "input") and output_node.input
                    else False
                ),
            },
            role=ActionRole.WORKFLOW,
            status=ActionStatus.PROCESSING,
        )

        # Check if output node needs to be executed
        if output_node.status in ["completed", "skipped"]:
            logger.debug(f"Output node already executed (status: {output_node.status})")
            action_helper.update_action_status(
                output_action,
                success=True,
                output_data={"node_already_executed": True, "status": output_node.status},
            )
            return

        if output_node.status == "failed":
            logger.warning(f"Output node previously failed, attempting to re-execute")
            output_node.status = "pending"
            output_node.result = None

        # Find output node index in node_order
        output_idx = None
        for idx, node_id in enumerate(self.workflow.node_order):
            if node_id == output_node.id:
                output_idx = idx
                break

        if output_idx is None:
            output_idx = len(self.workflow.node_order)
            self.workflow.node_order.append(output_node.id)
            logger.info(f"Added output node to node_order at index {output_idx}")

        # Update current_node_index to output node position
        old_index = self.workflow.current_node_index
        self.workflow.current_node_index = output_idx
        logger.info(
            f"Forcing output node execution: "
            f"old_index={old_index}, new_index={output_idx}, "
            f"node_order_len={len(self.workflow.node_order)}"
        )

        # Setup input for output node
        setup_node_input(output_node, self.workflow)

        # Execute output node
        try:
            logger.info(f"Executing pending output node: {output_node.description}")
            output_node.run()

            if output_node.status == "completed":
                logger.info(f"Output node executed successfully")

                # Update final_result in metadata if available
                if hasattr(output_node, "result") and output_node.result:
                    if isinstance(output_node.result, dict):
                        metadata["output_result"] = output_node.result

                    # Record the output_generation action with result
                    if hasattr(output_node.result, "success"):
                        output_action.output = {
                            "success": output_node.result.success,
                            "sql_query": getattr(output_node.result, "sql_query", ""),
                            "sql_result": getattr(output_node.result, "sql_result", ""),
                            "sql_query_final": getattr(output_node.result, "sql_query_final", ""),
                            "sql_result_final": getattr(output_node.result, "sql_result_final", ""),
                            "row_count": getattr(output_node.input, "row_count", 0)
                            if hasattr(output_node, "input") and output_node.input
                            else 0,
                            "metadata": getattr(output_node.input, "metadata", {})
                            if hasattr(output_node, "input") and output_node.input
                            else {},
                        }
                    elif isinstance(output_node.result, dict):
                        output_action.output = output_node.result
                        if hasattr(output_node, "input") and output_node.input:
                            output_action.output.setdefault(
                                "metadata", getattr(output_node.input, "metadata", {}) or {}
                            )

                    output_action.status = ActionStatus.SUCCESS
                    action_helper.update_action_status(
                        output_action,
                        success=True,
                        output_data={
                            "output_generated": True,
                            "has_result": True,
                            "status": "completed",
                        },
                    )
                    if self.workflow and self.workflow.metadata is not None:
                        self.workflow.metadata["_output_executed"] = True
                else:
                    output_action.status = ActionStatus.SUCCESS
                    action_helper.update_action_status(
                        output_action,
                        success=True,
                        output_data={"output_generated": True, "has_result": False, "status": "completed"},
                    )
                    if self.workflow and self.workflow.metadata is not None:
                        self.workflow.metadata["_output_executed"] = True
            else:
                logger.warning(f"Output node execution returned status: {output_node.status}")
                output_action.status = ActionStatus.FAILED
                action_helper.update_action_status(
                    output_action,
                    success=False,
                    output_data={"status": output_node.status, "error": "Output node execution failed"},
                )
        except Exception as e:
            logger.error(f"Failed to execute output node: {e}", exc_info=True)
            output_action.status = ActionStatus.FAILED
            action_helper.update_action_status(
                output_action, success=False, output_data={"error": str(e), "status": "error"}
            )

    def ensure_output_node_execution(self, metadata: Dict) -> None:
        """
        Ensure the output node executes even when workflow exits early.

        This guarantees that the SQL generation report is always generated
        and returned to the user, regardless of whether the workflow
        completed all nodes or exited due to max_steps limit or other reasons.

        Args:
            metadata: The metadata dict that will be passed to _finalize_workflow
        """
        # Delegate to run() method which handles everything
        self.run(metadata)

    def check_parallel_node_success(self, node: Node) -> bool:
        """Check if any child of parallel node succeeded."""
        try:
            if node.result and hasattr(node.result, "child_results"):
                for v in node.result.child_results.values():
                    ok = v.get("success", False) if isinstance(v, dict) else getattr(v, "success", False)
                    if ok:
                        return True
            return False
        except Exception as e:
            logger.error(f"Error checking parallel node success: {e}", exc_info=True)
            return False  # Return False but log error for debugging
