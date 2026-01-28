# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details

"""
Workflow Lifecycle Management.

Handles workflow initialization, resumption, node preparation,
and action history management.
"""

import argparse
from typing import Any, Callable, Dict, List, Optional

from datus.agent.evaluate import setup_node_input
from datus.agent.workflow import Workflow
from datus.configuration.agent_config import AgentConfig
from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.node_models import SqlTask
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class WorkflowLifecycle:
    """Manages workflow lifecycle: initialization, resumption, and preparation."""

    def __init__(
        self,
        args: argparse.Namespace,
        global_config: AgentConfig,
        pre_run_callable: Optional[Callable[[], Dict]] = None,
    ):
        self.args = args
        self.global_config = global_config
        self._pre_run = pre_run_callable
        self._completed_nodes_count = 0

    def initialize_workflow(self, sql_task: SqlTask, initial_metadata: Optional[Dict] = None) -> Workflow:
        """Generate a new workflow plan."""
        from datus.agent.plan import generate_workflow

        plan_type = (
            initial_metadata.get("workflow")
            or getattr(self.args, "workflow", None)
            or self.global_config.workflow_plan
        )

        # Check if plan_mode is specified in initial metadata
        plan_mode = False
        if initial_metadata and "plan_mode" in initial_metadata:
            plan_mode = initial_metadata["plan_mode"]
        elif hasattr(self.args, "plan_mode"):
            plan_mode = self.args.plan_mode

        logger.info(
            f"Workflow initialization: plan_type={plan_type}, plan_mode={plan_mode}, metadata={initial_metadata}"
        )

        # Override workflow type for plan mode
        if plan_mode:
            if plan_type == "chat_agentic":
                plan_type = "chat_agentic_plan"
            elif plan_type.endswith("_agentic") and not plan_type.endswith("_plan"):
                plan_type = plan_type + "_plan"
            logger.info(f"Plan mode enabled, using workflow type: {plan_type}")

        workflow = generate_workflow(
            task=sql_task,
            plan_type=plan_type,
            agent_config=self.global_config,
        )

        if plan_mode:
            workflow.metadata["plan_mode"] = plan_mode
            workflow.metadata["auto_execute_plan"] = True

        if initial_metadata:
            workflow.metadata.update(initial_metadata)

        workflow.display()
        logger.info("Initial workflow generated")
        return workflow

    def resume_workflow(self, config: argparse.Namespace, global_config: AgentConfig) -> Workflow:
        """Resume a workflow from a checkpoint file."""
        logger.info(f"Resuming workflow from config: {config}")

        try:
            workflow = Workflow.load(config.load_cp)
            workflow.global_config = global_config
            workflow.resume()
            workflow.display()
            logger.info(f"Resume workflow from {config.load_cp} successfully")
            return workflow
        except Exception as exc:
            logger.error(f"Failed to resume workflow from {config.load_cp}: {exc}")
            raise

    def init_or_load_workflow(
        self,
        sql_task: Optional[SqlTask],
        workflow: Optional[Workflow],
        initial_metadata: Optional[Dict],
    ) -> tuple[Optional[Workflow], bool]:
        """Initialize new workflow or load from checkpoint."""
        workflow_ready = False

        if self.args.load_cp:
            workflow = self.resume_workflow(self.args, self.global_config)
            if workflow and initial_metadata:
                workflow.metadata.update(initial_metadata)
        elif sql_task:
            workflow = self.initialize_workflow(sql_task, initial_metadata or {})
        elif not workflow:
            logger.error("Failed to initialize workflow. need a sql_task or to load from checkpoint.")
            return None, False

        if not workflow:
            logger.error("Failed to initialize workflow. Exiting.")
            return None, False

        workflow_ready = True
        return workflow, workflow_ready

    def prepare_first_node(self, workflow: Workflow):
        """Prepare the first node for execution."""
        if workflow.current_node_index == 0:
            from datus.schemas.base import BaseResult

            workflow.get_current_node().complete(BaseResult(success=True))
            next_node = workflow.advance_to_next_node()
            setup_node_input(next_node, workflow)

    def check_prerequisites(
        self,
        sql_task: Optional[SqlTask],
        workflow: Optional[Workflow],
        check_storage: bool,
    ) -> tuple[bool, Optional[Workflow]]:
        """Check prerequisites and initialize workflow if needed."""
        if check_storage:
            self.global_config.check_init_storage_config("database")
            self.global_config.check_init_storage_config("metrics")

        workflow, workflow_ready = self.init_or_load_workflow(sql_task, workflow, {})
        if not workflow_ready:
            return False, workflow

        if self._pre_run:
            self._pre_run()

        return True, workflow

    def increment_completed_count(self):
        """Increment the completed nodes counter."""
        self._completed_nodes_count += 1

    def get_completed_count(self) -> int:
        """Get the completed nodes count."""
        return self._completed_nodes_count

    def set_completed_count(self, count: int):
        """Set the completed nodes count."""
        self._completed_nodes_count = count


class ActionHistoryManagerMixin:
    """Mixin for managing action history in workflow execution."""

    def create_action_history(
        self,
        action_id: str,
        messages: str,
        action_type: str,
        input_data: Optional[Dict] = None,
    ) -> ActionHistory:
        """Create an action history entry."""
        return ActionHistory(
            action_id=action_id,
            role=ActionRole.WORKFLOW,
            messages=messages,
            action_type=action_type,
            input=input_data or {},
            status=ActionStatus.PROCESSING,
        )

    def update_action_status(
        self,
        action: ActionHistory,
        success: bool,
        output_data: Optional[Dict] = None,
        error: Optional[str] = None,
    ):
        """Update action history status."""
        if success:
            action.status = ActionStatus.SUCCESS
            action.output = output_data or {}
        else:
            action.status = ActionStatus.FAILED
            action.output = {"error": error or "Unknown error"}
            if output_data:
                action.output.update(output_data)

    def generate_failure_suggestions(self, workflow: Optional[Workflow]) -> List[str]:
        """Generate actionable suggestions based on failure context."""
        suggestions = []

        if not workflow or not workflow.context:
            return ["无法获取详细错误信息，请检查系统日志"]

        # Check if schema was found
        if not hasattr(workflow.context, "table_schemas") or not workflow.context.table_schemas:
            suggestions.append("未找到表 - 请验证数据库连接和表是否存在")

        # Check specific error patterns
        sql_contexts = workflow.context.sql_contexts if workflow.context else []
        for ctx in sql_contexts:
            if hasattr(ctx, "sql_error") and ctx.sql_error:
                error_msg = ctx.sql_error
                if "Column" in error_msg and "cannot be resolved" in error_msg:
                    suggestions.append("列名不匹配 - 请检查表结构中的正确列名")
                elif "Table" in error_msg and "doesn't exist" in error_msg:
                    suggestions.append("表不存在 - 请验证数据库中是否存在该表")
                elif "syntax" in error_msg.lower():
                    suggestions.append("SQL语法错误 - 生成的SQL有无效语法")

        # Add default suggestions
        if not suggestions:
            suggestions = [
                "尝试简化您的查询需求",
                "检查数据库中是否存在所需数据",
                "验证表名和列名是否正确",
            ]

        return suggestions
