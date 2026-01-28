# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details

"""
Workflow Execution Engine.

Core execution logic for synchronous and streaming workflow runs.
"""

import argparse
import asyncio
import os
import time
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, Optional

from datus.agent.evaluate import evaluate_result
from datus.agent.node import Node
from datus.agent.workflow import Workflow
from datus.agent.workflow_status import WorkflowTerminationStatus
from datus.configuration.agent_config import AgentConfig
from datus.schemas.action_history import ActionHistory, ActionHistoryManager
from datus.schemas.node_models import SqlTask
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class WorkflowExecutor:
    """Core workflow execution engine for synchronous and streaming modes."""

    def __init__(
        self,
        args: argparse.Namespace,
        agent_config: AgentConfig,
        workflow: Optional[Workflow] = None,
        pre_run_callable=None,
        run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.args = args
        self.global_config = agent_config
        self.workflow = workflow
        self.workflow_ready = False
        self._pre_run = pre_run_callable
        # Generate run_id if not provided (format: YYYYMMDD_HHMMSS)
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.initial_metadata = metadata or {}
        # Performance optimization: track completed nodes incrementally
        self._completed_nodes_count = 0
        # Track workflow task for cancellation (async mode only)
        self.workflow_task: Optional[asyncio.Task] = None

    def _finalize_workflow(self, step_count: int) -> Dict:
        """Persist workflow state and return final result metadata."""
        if not self.workflow:
            return {}

        self.workflow.display()
        file_name = self.workflow.task.id
        timestamp = int(time.time())

        # Use new hierarchical directory structure: {trajectory_dir}/{namespace}/{run_id}/
        trajectory_dir = self.global_config.get_trajectory_run_dir(self.run_id)
        os.makedirs(trajectory_dir, exist_ok=True)

        save_path = f"{trajectory_dir}/{file_name}_{timestamp}.yaml"
        self.workflow.save(save_path)
        logger.info(f"Workflow saved to {save_path}")
        final_result = self.workflow.get_final_result()

        # Use incremental count instead of scanning all nodes (O(1) vs O(n))
        logger.info(
            f"Workflow execution completed. "
            f"StepsAttempted:{step_count} "
            f"CompletedNodes:{self._completed_nodes_count}"
        )

        return {
            "final_result": final_result,
            "save_path": save_path,
            "steps": step_count,
            "completed_nodes": self._completed_nodes_count,
            "run_id": self.run_id,
        }

    def _increment_completed_count(self):
        """Increment the completed nodes counter."""
        self._completed_nodes_count += 1

    def _handle_termination_status(
        self,
        current_node: Node,
        termination_status: Optional[WorkflowTerminationStatus],
        is_soft_failure: bool,
        jump_to_reflect: bool,
    ) -> tuple[bool, bool]:
        """
        Handle workflow termination status from metadata.

        Returns:
            tuple: (is_soft_failure, jump_to_reflect)
        """
        if termination_status == WorkflowTerminationStatus.SKIP_TO_REFLECT:
            is_soft_failure = True
            jump_to_reflect = True
            # Clear termination status to avoid repeated jumps
            self.workflow.metadata.pop("termination_status", None)
        elif termination_status == WorkflowTerminationStatus.PROCEED_TO_OUTPUT:
            # Strategies exhausted - continue to output node for report generation
            logger.info("Strategies exhausted, proceeding to output node for report generation")
            # Clear termination status to allow normal continuation
            self.workflow.metadata.pop("termination_status", None)
        elif termination_status == WorkflowTerminationStatus.TERMINATE_WITH_ERROR:
            from datus.agent.runner.workflow_termination import WorkflowTerminationManager

            termination_mgr = WorkflowTerminationManager(self.workflow)
            termination_mgr.terminate_workflow(
                termination_status=WorkflowTerminationStatus.TERMINATE_WITH_ERROR,
                error_message=f"Node {current_node.description} requested termination"
            )
            return True, False  # is_soft_failure, should_break
        elif termination_status == WorkflowTerminationStatus.TERMINATE_SUCCESS:
            from datus.agent.runner.workflow_termination import WorkflowTerminationManager

            termination_mgr = WorkflowTerminationManager(self.workflow)
            termination_mgr.terminate_workflow(
                termination_status=WorkflowTerminationStatus.TERMINATE_SUCCESS
            )
            return True, False  # is_soft_failure, should_break

        return is_soft_failure, jump_to_reflect

    def _check_and_handle_termination(
        self,
        current_node: Node,
        is_soft_failure: bool,
        jump_to_reflect: bool,
    ) -> tuple[bool, bool, bool]:
        """
        Check and handle termination status from workflow metadata.

        Returns:
            tuple: (is_soft_failure, jump_to_reflect, should_break)
        """
        from datus.schemas.action_history import ActionStatus
        from datus.agent.runner.workflow_termination import WorkflowTerminationManager

        termination_mgr = WorkflowTerminationManager(self.workflow)

        # Check for workflow metadata termination status (set by nodes like SQLValidateNode)
        metadata_termination_status = self.workflow.metadata.get("termination_status") if self.workflow.metadata else None
        if metadata_termination_status:
            logger.info(f"Node requested termination via metadata: {metadata_termination_status}")
            is_soft_failure, jump_to_reflect = self._handle_termination_status(
                current_node, metadata_termination_status, is_soft_failure, jump_to_reflect
            )

        # Handle node failure status
        last_action_status = getattr(current_node, "last_action_status", None)
        if last_action_status == ActionStatus.SOFT_FAILED:
            is_soft_failure = True
            jump_to_reflect = True

        if current_node.status == "failed":
            # Use new termination status API
            termination_status = termination_mgr.handle_node_failure(current_node)

            if termination_status == WorkflowTerminationStatus.TERMINATE_WITH_ERROR:
                # Terminate workflow immediately
                termination_mgr.terminate_workflow(
                    termination_status=termination_status,
                    error_message=f"Node {current_node.description} failed with no recovery path"
                )
                return is_soft_failure, jump_to_reflect, True  # should_break
            elif termination_status == WorkflowTerminationStatus.SKIP_TO_REFLECT:
                is_soft_failure = True
                jump_to_reflect = True
            elif termination_status == WorkflowTerminationStatus.PROCEED_TO_OUTPUT:
                # Reflect node or other critical node failed, jump to output for report generation
                logger.info(f"Proceeding to output node for report generation after {current_node.description} failure")
                navigator = WorkflowNavigator(self.workflow)
                navigator.jump_to_output_node()

        return is_soft_failure, jump_to_reflect, False

    def run(self, sql_task: Optional[SqlTask] = None, check_storage: bool = False) -> Dict:
        """Execute the workflow synchronously."""
        from datus.agent.runner.workflow_lifecycle import WorkflowLifecycle
        from datus.agent.runner.workflow_navigator import WorkflowNavigator
        from datus.agent.runner.workflow_termination import WorkflowTerminationManager, OutputNodeExecutor

        logger.info("Starting agent execution")

        # Initialize lifecycle manager
        lifecycle = WorkflowLifecycle(self.args, self.global_config, self._pre_run)
        prerequisites_ok, self.workflow = lifecycle.check_prerequisites(sql_task, self.workflow, check_storage)
        if not prerequisites_ok:
            return {}

        step_count = 0
        max_steps = self.args.max_steps or 20

        # Initialize helpers
        navigator = WorkflowNavigator(self.workflow)
        termination_mgr = WorkflowTerminationManager(self.workflow)
        output_executor = OutputNodeExecutor(self.workflow)

        lifecycle.prepare_first_node(self.workflow)

        while self.workflow and not self.workflow.is_complete() and step_count < max_steps:
            current_node = self.workflow.get_current_node()
            if not current_node:
                logger.warning("No more tasks to execute. Exiting.")
                break

            logger.info(f"Executing task: {current_node.description}")
            current_node.run()

            # Track completed nodes incrementally
            if current_node.status == "completed":
                self._increment_completed_count()

            is_soft_failure = False
            jump_to_reflect = False

            # Check and handle termination
            is_soft_failure, jump_to_reflect, should_break = self._check_and_handle_termination(
                current_node, is_soft_failure, jump_to_reflect
            )
            if should_break:
                break

            if jump_to_reflect:
                jumped = navigator.jump_to_reflect_node()
                if jumped:
                    logger.info("Skipping to reflect node due to termination request")
                    continue

            evaluation = evaluate_result(current_node, self.workflow)
            logger.debug(f"Evaluation result for {current_node.type}: {evaluation}")
            if not evaluation["success"]:
                if is_soft_failure:
                    logger.warning(
                        f"Evaluation failed for {current_node.type}, but continuing due to Soft Failure mode."
                    )
                    # Jump directly to reflect node for recovery instead of advancing sequentially
                    jumped = navigator.jump_to_reflect_node()
                    if not jumped:
                        # No reflect node found, fall back to normal advancement
                        self.workflow.advance_to_next_node()
                else:
                    logger.error(f"Setting {current_node.type} status to failed due to evaluation failure")
                    current_node.status = "failed"
                    break
            else:
                self.workflow.advance_to_next_node()

            step_count += 1

        if step_count >= max_steps:
            logger.warning(f"Workflow execution stopped after reaching max steps: {max_steps}")

        # CRITICAL: Ensure output node executes even when workflow exits early
        output_executor.ensure_output_node_execution({})

        metadata = self._finalize_workflow(step_count)
        return metadata.get("final_result", {})

    async def run_stream(
        self,
        sql_task: Optional[SqlTask] = None,
        check_storage: bool = False,
        action_history_manager: Optional[ActionHistoryManager] = None,
        task_id: Optional[str] = None,
    ) -> AsyncGenerator[ActionHistory, None]:
        """Stream workflow execution progress."""
        from datus.agent.runner.workflow_lifecycle import ActionHistoryManagerMixin, WorkflowLifecycle
        from datus.agent.runner.workflow_navigator import WorkflowNavigator
        from datus.agent.runner.workflow_termination import (
            ErrorMarkdownBuilder,
            OutputNodeExecutor,
            WorkflowTerminationManager,
        )
        from datus.schemas.action_history import ActionStatus

        logger.info("Starting agent execution with streaming")
        max_steps = getattr(self.args, "max_steps", 100)

        # Initialize lifecycle manager
        lifecycle = WorkflowLifecycle(self.args, self.global_config, self._pre_run)

        # Log workflow details for observability
        if self.workflow:
            workflow_name = getattr(self.workflow, "name", "unknown")
            node_count = len(self.workflow.nodes) if self.workflow.nodes else 0
            logger.info(f"Workflow '{workflow_name}' initialized with {node_count} nodes, max_steps={max_steps}")

            # Store task_id in workflow metadata for cancellation checks
            if task_id:
                self.workflow.metadata["task_id"] = task_id

            if sql_task:
                logger.info(f"SQL task: {sql_task.task[:100]}... (id: {sql_task.id})")

        # Performance tracking
        start_time = time.time()

        # Initialize helpers
        navigator = WorkflowNavigator(self.workflow)
        termination_mgr = WorkflowTerminationManager(self.workflow)
        output_executor = OutputNodeExecutor(self.workflow)
        action_helper = ActionHistoryManagerMixin()
        error_builder = ErrorMarkdownBuilder(self.workflow)

        init_action = action_helper.create_action_history(
            action_id="workflow_initialization",
            messages="Initializing workflow and checking prerequisites",
            action_type="workflow_init",
            input_data={
                "has_sql_task": bool(sql_task),
                "check_storage": check_storage,
                "load_from_checkpoint": bool(self.args.load_cp),
            },
        )
        yield init_action

        # Track workflow execution state for completion action
        workflow_succeeded = False
        workflow_error = None
        step_count = 0
        metadata = {}

        try:
            if not lifecycle.check_prerequisites(sql_task, self.workflow, check_storage)[0]:
                action_helper.update_action_status(init_action, success=False, error="Failed to initialize workflow")
                workflow_error = "Failed to initialize workflow prerequisites"
                # Don't return here - let finally block handle completion action
            else:
                # Update workflow reference after initialization
                self.workflow = lifecycle.init_or_load_workflow(sql_task, self.workflow, self.initial_metadata)[0]
                navigator.set_workflow(self.workflow)
                termination_mgr.set_workflow(self.workflow)
                output_executor.set_workflow(self.workflow)
                error_builder.set_workflow(self.workflow)

                action_helper.update_action_status(
                    init_action,
                    success=True,
                    output_data={
                        "workflow_ready": True,
                        "total_nodes": len(self.workflow.nodes) if self.workflow else 0,
                        "current_node_index": self.workflow.current_node_index if self.workflow else 0,
                    },
                )

                lifecycle.prepare_first_node(self.workflow)

                while self.workflow and not self.workflow.is_complete() and step_count < max_steps:
                    # Check for cancellation at loop start
                    self.workflow.ensure_not_cancelled()

                    current_node = self.workflow.get_current_node()
                    if not current_node:
                        logger.warning("No more tasks to execute. Exiting.")
                        break

                    node_start_action = action_helper.create_action_history(
                        action_id=f"node_execution_{current_node.id}",
                        messages=f"Executing node: {current_node.description}",
                        action_type="node_execution",
                        input_data={
                            "node_id": current_node.id,
                            "node_type": current_node.type,
                            "description": current_node.description,
                            "step_count": step_count,
                        },
                    )
                    yield node_start_action

                    try:
                        logger.info(f"Executing task: {current_node.description}")

                        # Check for cancellation before node execution
                        self.workflow.ensure_not_cancelled()

                        async for node_action in current_node.run_stream(action_history_manager):
                            yield node_action

                            # Check for cancellation during node execution
                            self.workflow.ensure_not_cancelled()

                        # Track completed nodes incrementally
                        if current_node.status == "completed":
                            self._increment_completed_count()

                        is_soft_failure = False
                        jump_to_reflect = False
                        last_action_status = getattr(current_node, "last_action_status", None)
                        if last_action_status == ActionStatus.SOFT_FAILED:
                            is_soft_failure = True
                            jump_to_reflect = True

                        # Check and handle termination
                        is_soft_failure, jump_to_reflect, should_break = self._check_and_handle_termination(
                            current_node, is_soft_failure, jump_to_reflect
                        )
                        if should_break:
                            break

                        action_helper.update_action_status(
                            node_start_action,
                            success=True,
                            output_data={
                                "node_completed": True,
                                "execution_successful": True,
                            },
                        )

                        if jump_to_reflect:
                            jumped = navigator.jump_to_reflect_node()
                            if jumped:
                                logger.info("Skipping to reflect node due to termination request")
                                continue

                    except Exception as e:
                        # Regular exception (CancelledError will propagate naturally)
                        action_helper.update_action_status(node_start_action, success=False, error=str(e))
                        logger.error(f"Node execution error: {e}")
                        break

                    try:
                        evaluation = evaluate_result(current_node, self.workflow)
                        logger.debug(f"Evaluation result: {evaluation}")

                        if evaluation.get("success"):
                            self.workflow.advance_to_next_node()
                        elif is_soft_failure:
                            logger.warning(f"Node evaluation failed but continuing due to Soft Failure mode: {evaluation}")
                            # Jump directly to reflect node for recovery instead of advancing sequentially
                            jumped = navigator.jump_to_reflect_node()
                            if not jumped:
                                # No reflect node found, fall back to normal advancement
                                self.workflow.advance_to_next_node()
                        else:
                            logger.warning(f"Node evaluation failed: {evaluation}")
                            current_node.status = "failed"
                            break

                    except Exception as e:
                        logger.error(f"Evaluation error: {e}")
                        break

                    step_count += 1

                if step_count >= max_steps:
                    logger.warning(f"Workflow execution stopped after reaching max steps: {max_steps}")

                # CRITICAL: Ensure output node executes even when workflow exits early
                output_executor.ensure_output_node_execution(metadata)

                metadata = self._finalize_workflow(step_count)
                workflow_succeeded = True

        except Exception as e:
            # Regular exception (CancelledError will propagate naturally)
            logger.error(f"Workflow execution failed: {e}")
            workflow_error = str(e)
            # Yield error action for error event conversion
            error_action = action_helper.create_action_history(
                action_id="workflow_error",
                messages=f"Workflow execution failed: {str(e)}",
                action_type="error",
                input_data={"error_type": type(e).__name__},
            )
            error_action.status = ActionStatus.FAILED
            yield error_action
            raise
        # Note: CancelledError (Python 3.8+) or CancelledError from concurrent.futures (Python 3.7)
        # will propagate naturally and finally block will execute
        finally:
            # ALWAYS emit a completion action, regardless of how the workflow terminated
            # This ensures the frontend receives a CompleteEvent for all workflows
            try:
                output_executor.ensure_output_node_execution({})
            except Exception as e:
                logger.error(f"Failed to execute output node during finalization: {e}", exc_info=True)

            # Check for termination status and yield ErrorEvent if needed
            if self.workflow and self.workflow.metadata:
                termination_status = self.workflow.metadata.get("termination_status")
                if termination_status == WorkflowTerminationStatus.TERMINATE_WITH_ERROR:
                    # Build markdown-formatted error message
                    error_md = error_builder.build_termination_error_markdown()
                    if error_md:
                        # Create error action for ErrorEvent conversion
                        error_action = action_helper.create_action_history(
                            action_id="workflow_error",
                            messages=error_md,
                            action_type="error",
                            input_data={
                                "termination_status": termination_status,
                                "termination_reason": self.workflow.metadata.get("termination_reason", "Unknown"),
                            },
                        )
                        error_action.status = ActionStatus.FAILED
                        yield error_action

            # Calculate and log performance metrics
            execution_time = time.time() - start_time
            nodes_executed = self._completed_nodes_count if self.workflow else 0
            nodes_failed = (
                len([n for n in self.workflow.nodes.values() if n.status == "failed"]) if self.workflow else 0
            )

            # Determine completion status
            is_success = workflow_succeeded and workflow_error is None

            # Create appropriate completion message
            if is_success:
                completion_message = "Workflow execution completed successfully"
            else:
                completion_message = f"Workflow execution terminated: {workflow_error or 'Unknown error'}"

            logger.info(
                f"Workflow {('succeeded' if is_success else 'failed')} in {execution_time:.2f}s: "
                f"{nodes_executed} nodes completed, {nodes_failed} nodes failed"
            )

            # Create completion action with appropriate status
            completion_action = action_helper.create_action_history(
                action_id="workflow_completion",
                messages=completion_message,
                action_type="workflow_completion",
                input_data={
                    "steps_completed": step_count,
                    "workflow_saved": bool(metadata.get("save_path")),
                    "save_path": metadata.get("save_path", ""),
                    "terminated_early": not is_success,
                },
            )

            # Set completion action status based on workflow outcome
            completion_action.status = ActionStatus.SUCCESS if is_success else ActionStatus.FAILED

            # Update action status with execution details
            action_helper.update_action_status(
                completion_action,
                success=is_success,
                output_data={
                    "workflow_saved": bool(metadata.get("save_path")),
                    "execution_time_seconds": execution_time,
                    "nodes_completed": nodes_executed,
                    "nodes_failed": nodes_failed,
                    "save_path": metadata.get("save_path", ""),
                    "steps_completed": step_count,
                    "final_result_available": bool(metadata.get("final_result")),
                    "terminated_early": not is_success,
                    "error": workflow_error,
                },
            )

            # Always yield the completion action so it can be converted to a CompleteEvent
            yield completion_action
