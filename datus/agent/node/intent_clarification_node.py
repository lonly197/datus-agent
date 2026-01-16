# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
IntentClarificationNode implementation for clarifying user analytical intent.

This node processes the user's task to:
- Clarify unclear expressions
- Correct typos and errors
- Extract business terms and entities
- Normalize the query for better schema discovery

This is NOT for task type classification (that's IntentAnalysisNode's job).
Instead, it helps "理清用户真实分析意图" (clarify user's true analytical intent).
"""

import json
from typing import Any, AsyncGenerator, Dict, List, Optional

from datus.agent.node.node import Node, execute_with_async_stream
from datus.agent.workflow import Workflow
from datus.configuration.agent_config import AgentConfig
from datus.schemas.action_history import (
    ActionHistory,
    ActionHistoryManager,
    ActionRole,
    ActionStatus,
)
from datus.schemas.base import BaseInput, BaseResult
from datus.utils.error_handler import LLMMixin
from datus.utils.exceptions import ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class IntentClarificationNode(Node, LLMMixin):
    """
    Node for clarifying user's true analytical intent.

    This node processes the user's task to:
    1. Clarify unclear expressions (e.g., "最近的销售" → "最近30天的销售数据")
    2. Correct typos (e.g., "华山" → "华南")
    3. Extract business terms, time ranges, data dimensions
    4. Normalize the query for better schema discovery

    This node should ONLY be used when execution_mode is explicitly set to "text2sql".
    When execution_mode is empty, use IntentAnalysisNode instead to detect task type.
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
        Node.__init__(
            self,
            node_id=node_id,
            description=description,
            node_type=node_type,
            input_data=input_data,
            agent_config=agent_config,
            tools=tools,
        )
        LLMMixin.__init__(self)

    def setup_input(self, workflow: Workflow) -> Dict[str, Any]:
        """Setup intent clarification input from workflow context."""
        if not self.input:
            self.input = BaseInput()
        return {"success": True, "message": "Intent clarification input setup complete"}

    async def run(self) -> AsyncGenerator[ActionHistory, None]:
        """
        Run intent clarification to normalize and clarify user query.

        Yields:
            ActionHistory: Progress and result actions
        """
        try:
            if not self.workflow or not self.workflow.task:
                error_result = self.create_error_result(
                    ErrorCode.NODE_EXECUTION_FAILED,
                    "No workflow or task available for intent clarification",
                    "intent_clarification",
                    {"workflow_id": getattr(self.workflow, "id", "unknown") if self.workflow else "unknown"},
                )
                yield ActionHistory(
                    action_id=f"{self.id}_error",
                    role=ActionRole.TOOL,
                    messages=f"Intent clarification failed: {error_result.error}",
                    action_type="intent_clarification",
                    input={},
                    status=ActionStatus.FAILED,
                    output={"error": error_result.error, "error_code": error_result.error_code},
                )
                return

            task = self.workflow.task
            context = self.workflow.context

            # Step 1: Clarify intent using LLM
            clarification_result = await self._clarify_intent(
                task_text=task.task,
                ext_knowledge=getattr(task, "external_knowledge", None),
            )

            # Step 2: Store in workflow.metadata (not context.metadata) for downstream nodes
            if not hasattr(self.workflow, "metadata") or self.workflow.metadata is None:
                self.workflow.metadata = {}

            # Store clarification results
            self.workflow.metadata["intent_clarification"] = clarification_result

            # Step 3: Store clarified task for downstream nodes
            original_task = task.task
            clarified_task = clarification_result.get("clarified_task", original_task)

            # Store both original and clarified tasks in workflow.metadata
            self.workflow.metadata["original_task"] = original_task
            self.workflow.metadata["clarified_task"] = clarified_task

            # Emit success action
            yield ActionHistory(
                action_id=f"{self.id}_clarification",
                role=ActionRole.TOOL,
                messages=f"Intent clarification completed: {original_task[:50]}... → {clarified_task[:50]}...",
                action_type="intent_clarification",
                input={
                    "original_task": original_task[:100],
                    "has_entities": bool(clarification_result.get("entities")),
                    "has_corrections": bool(clarification_result.get("corrections")),
                },
                status=ActionStatus.SUCCESS,
                output=clarification_result,
            )

            self.result = BaseResult(
                success=True,
                data=clarification_result,
            )

            logger.info(
                f"Intent clarification completed: "
                f"original='{original_task[:50]}...', "
                f"clarified='{clarified_task[:50]}...', "
                f"confidence={clarification_result.get('confidence', 0.0)}"
            )

        except Exception as e:
            logger.error(f"Intent clarification failed: {e}")
            error_result = self.create_error_result(
                ErrorCode.NODE_EXECUTION_FAILED,
                f"Intent clarification execution failed: {str(e)}",
                "intent_clarification",
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
                messages=f"Intent clarification failed: {error_result.error}",
                action_type="intent_clarification",
                input={},
                status=ActionStatus.FAILED,
                output={"error": error_result.error, "error_code": error_result.error_code},
            )
            self.result = BaseResult(success=False, error=str(e))

    async def _clarify_intent(
        self,
        task_text: str,
        ext_knowledge: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Use LLM to clarify user intent and extract structured information.

        Args:
            task_text: The original user task/query
            ext_knowledge: Optional external business knowledge

        Returns:
            Dict with clarification results including:
            - clarified_task: Normalized and clarified query
            - entities: Extracted business terms, time ranges, dimensions, metrics
            - corrections: Typos fixed, ambiguities resolved
            - confidence: Confidence score (0-1)
            - need_clarification: Questions if more info needed
        """
        prompt = f"""你是一个专业的数据分析助手。请分析用户的查询意图并输出结构化信息。

用户查询：{task_text}

业务知识：
{ext_knowledge or "无"}

请按以下 JSON 格式输出（必须是有效的 JSON，不要包含其他内容）：
{{
    "clarified_task": "澄清和规范化后的查询，保持简洁",
    "entities": {{
        "business_terms": ["提取的业务术语"],
        "time_range": "时间范围（如果有）",
        "dimensions": ["数据维度"],
        "metrics": ["指标名称"]
    }},
    "corrections": {{
        "typos_fixed": ["纠正的错别字（原词→纠正词）"],
        "ambiguities_resolved": ["澄清的模糊表述"]
    }},
    "confidence": 0.95,
    "need_clarification": "如果需要更多信息才能理解，列出问题（可选）"
}}

重要：
1. 仅输出 JSON，不要包含其他解释
2. 如果没有错别字或模糊表述，corrections 可以为空数组
3. confidence 应该基于查询的清晰度和完整性（0-1之间）
4. 如果查询已经很清晰，clarified_task 可以与原查询相同或稍作调整
"""

        try:
            # Use LLMMixin with retry and caching
            cache_key = f"intent_clarification:{hash(task_text)}"
            response = await self.llm_call_with_retry(
                prompt=prompt,
                operation_name="intent_clarification",
                cache_key=cache_key,
                max_retries=3,
            )

            # Parse LLM response
            response_text = response.get("text", "")
            # Extract JSON from response (handle potential markdown code blocks)
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()

            clarification_result = json.loads(response_text)

            # Validate required fields
            if "clarified_task" not in clarification_result:
                clarification_result["clarified_task"] = task_text

            if "entities" not in clarification_result:
                clarification_result["entities"] = {}

            if "corrections" not in clarification_result:
                clarification_result["corrections"] = {}

            if "confidence" not in clarification_result:
                clarification_result["confidence"] = 0.5

            logger.info(
                f"LLM intent clarification completed: "
                f"clarified='{clarification_result.get('clarified_task', '')[:50]}...', "
                f"confidence={clarification_result.get('confidence', 0.0)}"
            )

            return clarification_result

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response as JSON: {e}. Using fallback.")
            # Fallback: return minimal clarification result
            return {
                "clarified_task": task_text,
                "entities": {},
                "corrections": {},
                "confidence": 0.3,
                "error": f"JSON parse error: {str(e)}",
            }
        except Exception as e:
            logger.error(f"Intent clarification LLM call failed: {e}")
            # Fallback: return minimal clarification result
            return {
                "clarified_task": task_text,
                "entities": {},
                "corrections": {},
                "confidence": 0.0,
                "error": str(e),
            }

    def execute(self) -> BaseResult:
        """Execute intent clarification synchronously."""
        return execute_with_async_stream(self)

    async def execute_stream(
        self, action_history_manager: Optional[ActionHistoryManager] = None
    ) -> AsyncGenerator[ActionHistory, None]:
        """Execute intent clarification with streaming support."""
        if action_history_manager:
            self.action_history_manager = action_history_manager
        async for action in self.run():
            yield action

    def update_context(self, workflow: "Workflow") -> Dict[str, Any]:
        """Update workflow context with clarification results."""
        try:
            if not self.result or not self.result.success:
                return {"success": False, "message": "Intent clarification failed, cannot update context"}

            # Store clarification result in workflow metadata
            if not hasattr(workflow, "metadata") or workflow.metadata is None:
                workflow.metadata = {}
            workflow.metadata["intent_clarification"] = self.result.data

            return {
                "success": True,
                "message": f"Intent clarification context updated: {self.result.data.get('clarified_task', 'N/A')[:50]}...",
            }

        except Exception as e:
            logger.error(f"Failed to update intent clarification context: {str(e)}")
            return {"success": False, "message": f"Intent clarification context update failed: {str(e)}"}
