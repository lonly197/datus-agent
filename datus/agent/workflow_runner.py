import argparse
import asyncio
import os
import time
from datetime import datetime
from typing import Any, AsyncGenerator, Callable, Dict, Optional

from datus.agent.evaluate import evaluate_result, setup_node_input
from datus.agent.plan import generate_workflow
from datus.agent.workflow import Workflow
from datus.configuration.agent_config import AgentConfig
from datus.configuration.node_type import NodeType
from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.base import BaseResult
from datus.schemas.node_models import SqlTask
from datus.utils.loggings import get_logger
from datus.utils.traceable_utils import optional_traceable

logger = get_logger(__name__)


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

    def initialize_workflow(self, sql_task: SqlTask):
        """Generate a new workflow plan."""
        plan_type = getattr(self.args, "workflow", None) or self.global_config.workflow_plan

        # Check if plan_mode is specified in initial metadata (API context)
        plan_mode = False
        if self.initial_metadata and "plan_mode" in self.initial_metadata:
            plan_mode = self.initial_metadata["plan_mode"]
        elif hasattr(self.args, "plan_mode"):
            plan_mode = self.args.plan_mode

        logger.info(f"Workflow initialization: plan_type={plan_type}, plan_mode={plan_mode}, metadata={self.initial_metadata}")

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

        # compute completed nodes instead of trusting step_count
        completed_nodes = sum(1 for n in self.workflow.nodes.values() if n.status == "completed")
        logger.info(f"Workflow execution completed. StepsAttempted:{step_count} CompletedNodes:{completed_nodes}")

        return {
            "final_result": final_result,
            "save_path": save_path,
            "steps": step_count,
            "completed_nodes": completed_nodes,
            "run_id": self.run_id,
        }

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

            if current_node.status == "failed":
                if current_node.type == NodeType.TYPE_PARALLEL:
                    try:
                        has_any_success = False
                        if current_node.result and hasattr(current_node.result, "child_results"):
                            for v in current_node.result.child_results.values():
                                ok = v.get("success", False) if isinstance(v, dict) else getattr(v, "success", False)
                                if ok:
                                    has_any_success = True
                                    break
                        if has_any_success:
                            logger.warning("Parallel node partial failure, continue to selection")
                        else:
                            logger.warning(f"Parallel node all failed: {current_node.description}")
                            break
                    except Exception:
                        logger.warning(f"Node failed: {current_node.description}")
                        break
                else:
                    logger.warning(f"Node failed: {current_node.description}")
                    break

            evaluation = evaluate_result(current_node, self.workflow)
            logger.debug(f"Evaluation result for {current_node.type}: {evaluation}")
            if not evaluation["success"]:
                logger.error(f"Setting {current_node.type} status to failed due to evaluation failure")
                current_node.status = "failed"
                break

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

        try:
            if not self._ensure_prerequisites(sql_task, check_storage):
                self._update_action_status(init_action, success=False, error="Failed to initialize workflow")
                return

            self._update_action_status(
                init_action,
                success=True,
                output_data={
                    "workflow_ready": True,
                    "total_nodes": len(self.workflow.nodes) if self.workflow else 0,
                    "current_node_index": self.workflow.current_node_index if self.workflow else 0,
                },
            )

            step_count = 0
            self._prepare_first_node()

            while self.workflow and not self.workflow.is_complete() and step_count < max_steps:
                # 检查是否被取消
                current_task = asyncio.current_task()
                if current_task and current_task.cancelled():
                    logger.info("Workflow execution was cancelled")
                    break

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

                    async for node_action in current_node.run_stream(action_history_manager):
                        yield node_action

                    if current_node.status == "failed":
                        self._update_action_status(
                            node_start_action, success=False, error=f"Node execution failed: {current_node.description}"
                        )
                        logger.warning(f"Node failed: {current_node.description}")
                        break

                    self._update_action_status(
                        node_start_action,
                        success=True,
                        output_data={
                            "node_completed": True,
                            "execution_successful": True,
                        },
                    )

                except asyncio.CancelledError:
                    # 重新抛出取消异常
                    self._update_action_status(node_start_action, success=False, error="Node execution was cancelled")
                    logger.info(f"Node execution cancelled: {current_node.description}")
                    raise
                except Exception as e:
                    self._update_action_status(node_start_action, success=False, error=str(e))
                    logger.error(f"Node execution error: {e}")
                    break

                try:
                    evaluation = evaluate_result(current_node, self.workflow)
                    logger.debug(f"Evaluation result: {evaluation}")

                    if evaluation.get("success"):
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

            # Create completion action for workflow completion event
            completion_action = self._create_action_history(
                action_id="workflow_completion",
                messages="Workflow execution completed successfully",
                action_type="workflow_completion",
                input_data={
                    "steps_completed": step_count,
                    "workflow_saved": True,
                    "save_path": metadata.get("save_path"),
                },
            )

            # Yield the completion action so it can be converted to a CompleteEvent
            yield completion_action

            self._update_action_status(
                completion_action,
                success=True,
                output_data={
                    "workflow_saved": True,
                    "save_path": metadata.get("save_path"),
                    "steps_completed": step_count,
                    "final_result_available": bool(metadata.get("final_result")),
                },
            )

        except asyncio.CancelledError:
            logger.info("Workflow stream execution was cancelled")
            raise
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
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
