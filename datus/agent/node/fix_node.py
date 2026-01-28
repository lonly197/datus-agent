# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from typing import AsyncGenerator, Dict, Optional

from datus.agent.node import Node
from datus.agent.workflow import Workflow
from datus.schemas.action_history import (ActionHistory, ActionHistoryManager,
                                          ActionRole, ActionStatus)
from datus.schemas.fix_node_models import FixInput, FixResult
from datus.schemas.node_models import SQLContext
from datus.tools.llms_tools import autofix_sql
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class FixNode(Node):
    def execute(self):
        self.result = self._execute_fix()

    async def execute_stream(
        self, action_history_manager: Optional[ActionHistoryManager] = None
    ) -> AsyncGenerator[ActionHistory, None]:
        """Execute SQL fix with streaming support."""
        async for action in self._fix_stream(action_history_manager):
            yield action

    def setup_input(self, workflow: Workflow) -> Dict:
        # irrelevant to current node
        # Safely access workflow.task and workflow.context
        task = getattr(workflow, 'task', None)
        context = getattr(workflow, 'context', None)
        
        if not task:
            return {"success": False, "message": "No task available in workflow"}
        
        next_input = FixInput(
            sql_task=task,
            sql_context=workflow.get_last_sqlcontext() if hasattr(workflow, 'get_last_sqlcontext') else None,
            schemas=getattr(context, 'table_schemas', None) if context else None,
        )
        self.input = next_input
        return {"success": True, "message": "Schema appears valid", "suggestions": [next_input]}

    def update_context(self, workflow: Workflow) -> Dict:
        """Update fix SQL results to workflow context."""
        result = self.result
        
        # Check if result is None
        if not result:
            error_msg = "Fix result is not available"
            logger.error(error_msg)
            return {"success": False, "message": error_msg}
        
        workflow_context = getattr(workflow, 'context', None)
        if not workflow_context:
            error_msg = "Workflow context is not available"
            logger.error(error_msg)
            return {"success": False, "message": error_msg}
        
        try:
            sql_query = getattr(result, 'sql_query', None)
            if not sql_query:
                error_msg = "Fixed SQL query is not available"
                logger.error(error_msg)
                return {"success": False, "message": error_msg}
            
            new_record = SQLContext(sql_query=sql_query, explanation=getattr(result, 'explanation', None) or "")
            if not hasattr(workflow_context, 'sql_contexts'):
                workflow_context.sql_contexts = []
            workflow_context.sql_contexts.append(new_record)
            return {"success": True, "message": "Updated fix SQL context"}
        except Exception as e:
            logger.error(f"Failed to update fix SQL context: {str(e)}")
            return {"success": False, "message": f"Fix SQL context update failed: {str(e)}"}

    def _execute_fix(self) -> FixResult:
        """Execute fix action to fix the SQL query."""

        if not self.model:
            return FixResult(
                success=False,
                error="SQL fix model not provided",
                sql_query="",
                explanation="",
            )

        try:
            logger.debug(f"Fix SQL input: {type(self.input)} {self.input}")

            # ToDo: add docs from search tools
            return autofix_sql(self.model, self.input, docs=[])
        except Exception as e:
            logger.error(f"SQL fix execution error: {str(e)}")
            return FixResult(success=False, error=str(e), sql_query="", explanation="")

    async def _fix_stream(
        self, action_history_manager: Optional[ActionHistoryManager] = None
    ) -> AsyncGenerator[ActionHistory, None]:
        """Execute SQL fix with streaming support and action history tracking."""
        if not self.model:
            logger.error("Model not available for SQL fix")
            return

        try:
            # SQL fix action
            fix_action = ActionHistory(
                action_id="sql_fix",
                role=ActionRole.WORKFLOW,
                messages="Analyzing and fixing SQL query issues",
                action_type="sql_fix",
                input={
                    "original_sql": self.input.sql_context.sql_query if hasattr(self.input, "sql_context") else "",
                    "has_schemas": bool(hasattr(self.input, "schemas") and self.input.schemas),
                },
                status=ActionStatus.PROCESSING,
            )
            yield fix_action

            # Execute SQL fix
            result = self._execute_fix()

            fix_action.status = ActionStatus.SUCCESS if result.success else ActionStatus.FAILED
            fix_action.output = {
                "success": result.success,
                "fixed_sql": result.sql_query,
                "has_explanation": bool(result.explanation),
                "error": result.error if hasattr(result, "error") and result.error else None,
            }

            # Store result for later use
            self.result = result

            # Yield the updated action with final status
            yield fix_action

        except Exception as e:
            logger.error(f"SQL fix streaming error: {str(e)}")
            raise
