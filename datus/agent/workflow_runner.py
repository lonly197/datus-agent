import argparse
import asyncio
import os
import time
from datetime import datetime
from enum import Enum
from typing import Any, AsyncGenerator, Callable, Dict, Optional

from datus.agent.evaluate import evaluate_result, setup_node_input
from datus.agent.node import Node
from datus.agent.plan import generate_workflow
from datus.agent.workflow import Workflow
from datus.configuration.agent_config import AgentConfig
from datus.configuration.node_type import NodeType
from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.base import BaseResult
from datus.schemas.node_models import SqlTask
from datus.utils.async_utils import ensure_not_cancelled
from datus.utils.error_handler import check_reflect_node_reachable
from datus.utils.loggings import get_logger
from datus.utils.traceable_utils import optional_traceable

logger = get_logger(__name__)


class WorkflowTerminationStatus(str, Enum):
    """工作流终止状态"""
    CONTINUE = "continue"  # 继续执行
    SKIP_TO_REFLECT = "skip_to_reflect"  # 跳转到反思节点
    TERMINATE_WITH_ERROR = "terminate_with_error"  # 终止并报错
    TERMINATE_SUCCESS = "terminate_success"  # 成功终止


class WorkflowRunner:
    """Encapsulates workflow lifecycle management so each runner can execute independently."""

    def __init__(
        self,
        args: argparse.Namespace,
        agent_config: AgentConfig,
        *,
        pre_run_callable: Optional[Callable[[], Dict]] = None,
        run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.args = args
        self.global_config = agent_config
        self.workflow: Optional[Workflow] = None
        self.workflow_ready = False
        self._pre_run = pre_run_callable
        # Generate run_id if not provided (format: YYYYMMDD_HHMMSS)
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.initial_metadata = metadata or {}
        # Performance optimization: track completed nodes incrementally
        self._completed_nodes_count = 0
        # Track workflow task for cancellation (async mode only)
        self.workflow_task: Optional[asyncio.Task] = None

    def initialize_workflow(self, sql_task: SqlTask):
        """Generate a new workflow plan."""
        plan_type = (
            self.initial_metadata.get("workflow")
            or getattr(self.args, "workflow", None)  # 优先检查metadata中的workflow
            or self.global_config.workflow_plan
        )

        # Check if plan_mode is specified in initial metadata (API context)
        plan_mode = False
        if self.initial_metadata and "plan_mode" in self.initial_metadata:
            plan_mode = self.initial_metadata["plan_mode"]
        elif hasattr(self.args, "plan_mode"):
            plan_mode = self.args.plan_mode

        logger.info(
            f"Workflow initialization: plan_type={plan_type}, plan_mode={plan_mode}, metadata={self.initial_metadata}"
        )

        # Override workflow type for plan mode
        if plan_mode:
            if plan_type == "chat_agentic":
                plan_type = "chat_agentic_plan"
            elif plan_type.endswith("_agentic") and not plan_type.endswith("_plan"):
                # Handle other agentic workflows that might have plan variants
                plan_type = plan_type + "_plan"
            logger.info(f"Plan mode enabled, using workflow type: {plan_type}")

        self.workflow = generate_workflow(
            task=sql_task,
            plan_type=plan_type,
            agent_config=self.global_config,
        )

        if plan_mode:
            self.workflow.metadata["plan_mode"] = plan_mode
            self.workflow.metadata["auto_execute_plan"] = True

        self.workflow.display()
        logger.info("Initial workflow generated")

    def resume_workflow(self, config: argparse.Namespace):
        """Resume a workflow from a checkpoint file."""
        logger.info(f"Resuming workflow from config: {config}")

        try:
            self.workflow = Workflow.load(config.load_cp)
            self.workflow.global_config = self.global_config
            self.workflow.resume()
            self.workflow.display()
            logger.info(f"Resume workflow from {config.load_cp} successfully")
        except Exception as exc:
            logger.error(f"Failed to resume workflow from {config.load_cp}: {exc}")
            raise

    def is_complete(self):
        if self.workflow is None:
            return True
        return self.workflow.is_complete()

    def init_or_load_workflow(self, sql_task: Optional[SqlTask]):
        if self.args.load_cp:
            self.workflow_ready = False
            self.workflow = None
            self.resume_workflow(self.args)
            # Set initial metadata after workflow is loaded
            if self.workflow and self.initial_metadata:
                self.workflow.metadata.update(self.initial_metadata)
        elif sql_task:
            self.workflow_ready = False
            self.workflow = None
            self.initialize_workflow(sql_task)
            # Set initial metadata after workflow is initialized
            if self.workflow and self.initial_metadata:
                self.workflow.metadata.update(self.initial_metadata)
        elif not self.workflow_ready:
            logger.error("Failed to initialize workflow. need a sql_task or to load from checkpoint.")
            return None

        if not self.workflow:
            logger.error("Failed to initialize workflow. Exiting.")
            return None

        self.workflow_ready = True
        return True

    def _prepare_first_node(self):
        if not self.workflow:
            return
        if self.workflow.current_node_index == 0:
            self.workflow.get_current_node().complete(BaseResult(success=True))
            next_node = self.workflow.advance_to_next_node()
            setup_node_input(next_node, self.workflow)

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

    def _ensure_prerequisites(self, sql_task: Optional[SqlTask], check_storage: bool) -> bool:
        if check_storage:
            self.global_config.check_init_storage_config("database")
            self.global_config.check_init_storage_config("metrics")

        if not self.init_or_load_workflow(sql_task):
            return False

        if self._pre_run:
            self._pre_run()

        return True

    def _create_action_history(
        self, action_id: str, messages: str, action_type: str, input_data: dict = None
    ) -> ActionHistory:
        return ActionHistory(
            action_id=action_id,
            role=ActionRole.WORKFLOW,
            messages=messages,
            action_type=action_type,
            input=input_data or {},
            status=ActionStatus.PROCESSING,
        )

    def _update_action_status(self, action: ActionHistory, success: bool, output_data: dict = None, error: str = None):
        if success:
            action.status = ActionStatus.SUCCESS
            action.output = output_data or {}
        else:
            action.status = ActionStatus.FAILED
            action.output = {"error": error or "Unknown error"}
            if output_data:
                action.output.update(output_data)

    def _handle_node_failure(
        self, current_node: Node, node_start_action: Optional[ActionHistory] = None
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
                has_reflect = check_reflect_node_reachable(self.workflow)
                if has_reflect:
                    return WorkflowTerminationStatus.SKIP_TO_REFLECT
                else:
                    return WorkflowTerminationStatus.TERMINATE_WITH_ERROR
        else:
            # Fallback to old logic if last_action_status not available
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

    def _handle_node_failure_legacy(
        self, current_node: Node, node_start_action: Optional[ActionHistory] = None
    ) -> tuple[bool, bool]:
        """
        Legacy method for backward compatibility.

        Returns:
            tuple: (is_soft_failure: bool, should_continue: bool)
        """
        termination_status = self._handle_node_failure(current_node, node_start_action)

        if termination_status == WorkflowTerminationStatus.SKIP_TO_REFLECT:
            return True, True
        elif termination_status == WorkflowTerminationStatus.TERMINATE_WITH_ERROR:
            return False, False
        else:
            return True, True

    def _check_parallel_node_success(self, node: Node) -> bool:
        """Check if any child of parallel node succeeded."""
        try:
            if node.result and hasattr(node.result, "child_results"):
                for v in node.result.child_results.values():
                    ok = v.get("success", False) if isinstance(v, dict) else getattr(v, "success", False)
                    if ok:
                        return True
            return False
        except Exception as e:
            logger.warning(f"Error checking parallel node success: {e}")
            return False

    def _jump_to_reflect_node(self) -> bool:
        """
        Jump directly to the next reflect node in the workflow execution path.

        When a node soft-fails, we want to skip all intermediate nodes and go
        directly to the reflect node for recovery.

        Returns:
            True if successfully jumped to a reflect node, False otherwise
        """
        if not self.workflow or not self.workflow.nodes:
            return False

        # Find the next reflect node in the execution order
        current_idx = self.workflow.current_node_index
        if current_idx is None:
            return False

        # Search forward in node_order for the first reflect node
        # Limit search range to prevent excessive iteration
        max_search_range = min(len(self.workflow.node_order), current_idx + 100)
        nodes_checked = 0

        for i in range(current_idx + 1, max_search_range):
            node_id = self.workflow.node_order[i]
            node = self.workflow.nodes.get(node_id)

            # Validate node_order and nodes are in sync
            if not node:
                logger.warning(f"Node {node_id} in node_order but not in nodes dict, skipping")
                nodes_checked += 1
                continue

            if node.type == "reflect" and node.status not in ["completed", "skipped"]:
                # Jump directly to this reflect node
                self.workflow.current_node_index = i
                logger.info(
                    f"Jumping to reflect node '{node_id}' at index {i} for recovery"
                )
                return True

            nodes_checked += 1

        logger.warning(
            f"No reflect node found in execution path for recovery (checked {nodes_checked} nodes)"
        )
        return False

    def _terminate_workflow(
        self,
        termination_status: WorkflowTerminationStatus,
        error_message: str = None
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
                    logger.warning(f"Error during workflow task cancellation: {e}")

        elif termination_status == WorkflowTerminationStatus.TERMINATE_SUCCESS:
            logger.info("Workflow completed successfully")

    @optional_traceable(name="agent")
    def run(self, sql_task: Optional[SqlTask] = None, check_storage: bool = False) -> Dict:
        """Execute the workflow synchronously."""
        logger.info("Starting agent execution")
        if not self._ensure_prerequisites(sql_task, check_storage):
            return {}

        step_count = 0
        max_steps = self.args.max_steps or 20
        self._prepare_first_node()

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
            if current_node.status == "failed":
                # Use new termination status API
                termination_status = self._handle_node_failure(current_node)

                if termination_status == WorkflowTerminationStatus.TERMINATE_WITH_ERROR:
                    # Terminate workflow immediately
                    self._terminate_workflow(
                        termination_status=termination_status,
                        error_message=f"Node {current_node.description} failed with no recovery path"
                    )
                    break
                elif termination_status == WorkflowTerminationStatus.SKIP_TO_REFLECT:
                    is_soft_failure = True

            # Check for workflow metadata termination status (set by nodes like SQLValidateNode)
            metadata_termination_status = self.workflow.metadata.get("termination_status") if self.workflow.metadata else None
            if metadata_termination_status:
                logger.info(f"Node requested termination via metadata: {metadata_termination_status}")
                if metadata_termination_status == WorkflowTerminationStatus.SKIP_TO_REFLECT:
                    is_soft_failure = True
                elif metadata_termination_status == WorkflowTerminationStatus.TERMINATE_WITH_ERROR:
                    self._terminate_workflow(
                        termination_status=WorkflowTerminationStatus.TERMINATE_WITH_ERROR,
                        error_message=f"Node {current_node.description} requested termination"
                    )
                    break
                elif metadata_termination_status == WorkflowTerminationStatus.TERMINATE_SUCCESS:
                    self._terminate_workflow(
                        termination_status=WorkflowTerminationStatus.TERMINATE_SUCCESS
                    )
                    break

            evaluation = evaluate_result(current_node, self.workflow)
            logger.debug(f"Evaluation result for {current_node.type}: {evaluation}")
            if not evaluation["success"]:
                if is_soft_failure:
                    logger.warning(
                        f"Evaluation failed for {current_node.type}, but continuing due to Soft Failure mode."
                    )
                    # Jump directly to reflect node for recovery instead of advancing sequentially
                    jumped = self._jump_to_reflect_node()
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

        metadata = self._finalize_workflow(step_count)
        return metadata.get("final_result", {})

    async def run_stream(
        self,
        sql_task: Optional[SqlTask] = None,
        check_storage: bool = False,
        action_history_manager: Optional[ActionHistoryManager] = None,
    ) -> AsyncGenerator[ActionHistory, None]:
        """Stream workflow execution progress."""
        logger.info("Starting agent execution with streaming")
        max_steps = getattr(self.args, "max_steps", 100)

        # Log workflow details for observability
        if self.workflow:
            workflow_name = getattr(self.workflow, "name", "unknown")
            node_count = len(self.workflow.nodes) if self.workflow.nodes else 0
            logger.info(f"Workflow '{workflow_name}' initialized with {node_count} nodes, max_steps={max_steps}")

            if sql_task:
                logger.info(f"SQL task: {sql_task.task[:100]}... (id: {sql_task.id})")

        # Performance tracking
        import time

        start_time = time.time()
        init_action = self._create_action_history(
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
            if not self._ensure_prerequisites(sql_task, check_storage):
                self._update_action_status(init_action, success=False, error="Failed to initialize workflow")
                workflow_error = "Failed to initialize workflow prerequisites"
                # Don't return here - let finally block handle completion action
            else:
                self._update_action_status(
                    init_action,
                    success=True,
                    output_data={
                        "workflow_ready": True,
                        "total_nodes": len(self.workflow.nodes) if self.workflow else 0,
                        "current_node_index": self.workflow.current_node_index if self.workflow else 0,
                    },
                )

                self._prepare_first_node()

                while self.workflow and not self.workflow.is_complete() and step_count < max_steps:
                    # Check for cancellation at loop start
                    ensure_not_cancelled()

                    current_node = self.workflow.get_current_node()
                    if not current_node:
                        logger.warning("No more tasks to execute. Exiting.")
                        break

                    node_start_action = self._create_action_history(
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
                        ensure_not_cancelled()

                        async for node_action in current_node.run_stream(action_history_manager):
                            yield node_action

                            # Check for cancellation during node execution
                            ensure_not_cancelled()

                        # Track completed nodes incrementally
                        if current_node.status == "completed":
                            self._increment_completed_count()

                        is_soft_failure = False
                        if current_node.status == "failed":
                            # Use new termination status API
                            termination_status = self._handle_node_failure(current_node, node_start_action)

                            if termination_status == WorkflowTerminationStatus.TERMINATE_WITH_ERROR:
                                # Terminate workflow immediately
                                self._terminate_workflow(
                                    termination_status=termination_status,
                                    error_message=f"Node {current_node.description} failed with no recovery path"
                                )
                                break
                            elif termination_status == WorkflowTerminationStatus.SKIP_TO_REFLECT:
                                is_soft_failure = True

                        # Check for workflow metadata termination status (set by nodes like SQLValidateNode)
                        metadata_termination_status = self.workflow.metadata.get("termination_status") if self.workflow.metadata else None
                        if metadata_termination_status:
                            logger.info(f"Node requested termination via metadata: {metadata_termination_status}")
                            if metadata_termination_status == WorkflowTerminationStatus.SKIP_TO_REFLECT:
                                is_soft_failure = True
                            elif metadata_termination_status == WorkflowTerminationStatus.TERMINATE_WITH_ERROR:
                                self._terminate_workflow(
                                    termination_status=WorkflowTerminationStatus.TERMINATE_WITH_ERROR,
                                    error_message=f"Node {current_node.description} requested termination"
                                )
                                break
                            elif metadata_termination_status == WorkflowTerminationStatus.TERMINATE_SUCCESS:
                                self._terminate_workflow(
                                    termination_status=WorkflowTerminationStatus.TERMINATE_SUCCESS
                                )
                                break

                        self._update_action_status(
                            node_start_action,
                            success=True,
                            output_data={
                                "node_completed": True,
                                "execution_successful": True,
                            },
                        )

                    except Exception as e:
                        # Regular exception (CancelledError will propagate naturally)
                        self._update_action_status(node_start_action, success=False, error=str(e))
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
                            jumped = self._jump_to_reflect_node()
                            if not jumped:
                                # No reflect node found, fall back to normal advancement
                                self.workflow.advance_to_next_node()
                        else:
                            logger.warning(f"Node evaluation failed: {evaluation}")
                            break

                    except Exception as e:
                        logger.error(f"Evaluation error: {e}")
                        break

                    step_count += 1

                if step_count >= max_steps:
                    logger.warning(f"Workflow execution stopped after reaching max steps: {max_steps}")

                metadata = self._finalize_workflow(step_count)
                workflow_succeeded = True

        except Exception as e:
            # Regular exception (CancelledError will propagate naturally)
            logger.error(f"Workflow execution failed: {e}")
            workflow_error = str(e)
            # Yield error action for error event conversion
            error_action = self._create_action_history(
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
            completion_action = self._create_action_history(
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
            self._update_action_status(
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
