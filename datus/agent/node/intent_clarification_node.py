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
from datus.schemas.action_history import (ActionHistory, ActionHistoryManager,
                                          ActionRole, ActionStatus)
from datus.schemas.base import BaseInput, BaseResult
from datus.utils.error_handler import LLMMixin
from datus.utils.exceptions import ErrorCode
from datus.utils.json_utils import llm_result2json
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
                    messages=f"Intent clarification failed: {error_result.error_message}",
                    action_type="intent_clarification",
                    input={},
                    status=ActionStatus.FAILED,
                    output={"error": error_result.error_message, "error_code": error_result.error_code},
                )
                return

            task = self.workflow.task
            task_text = getattr(task, "task", "") or ""
            if not task_text:
                error_result = self.create_error_result(
                    ErrorCode.COMMON_VALIDATION_FAILED,
                    "Task content is empty",
                    "intent_clarification",
                    {"task_id": getattr(task, "id", "unknown")},
                )
                yield ActionHistory(
                    action_id=f"{self.id}_error",
                    role=ActionRole.TOOL,
                    messages="Intent clarification failed: Task content is empty",
                    action_type="intent_clarification",
                    input={},
                    status=ActionStatus.FAILED,
                    output={"error": error_result.error_message, "error_code": error_result.error_code},
                )
                return

            context = self.workflow.context

            # Step 1: Clarify intent using LLM
            clarification_result = await self._clarify_intent(
                task_text=task_text,
                ext_knowledge=getattr(task, "external_knowledge", None),
            )

            # Step 2: Store in workflow.metadata (not context.metadata) for downstream nodes
            if not hasattr(self.workflow, "metadata") or self.workflow.metadata is None:
                self.workflow.metadata = {}

            # Store clarification results
            self.workflow.metadata["intent_clarification"] = clarification_result

            # Step 3: Store clarified task for downstream nodes
            original_task = task_text
            clarified_task = clarification_result.get("clarified_task", original_task)

            # Store both original and clarified tasks in workflow.metadata
            self.workflow.metadata["original_task"] = original_task
            self.workflow.metadata["clarified_task"] = clarified_task

            has_entities = self._has_entity_content(clarification_result.get("entities", {}))
            has_corrections = self._has_correction_content(clarification_result.get("corrections", {}))

            # Emit success action
            yield ActionHistory(
                action_id=f"{self.id}_clarification",
                role=ActionRole.TOOL,
                messages=f"Intent clarification completed: {original_task[:50]}... → {clarified_task[:50]}...",
                action_type="intent_clarification",
                input={
                    "original_task": original_task[:100],
                    "has_entities": has_entities,
                    "has_corrections": has_corrections,
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
                messages=f"Intent clarification failed: {error_result.error_message}",
                action_type="intent_clarification",
                input={},
                status=ActionStatus.FAILED,
                output={"error": error_result.error_message, "error_code": error_result.error_code},
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
        """
        # Try up to 2 attempts with different prompt strictness
        max_attempts = 2

        for attempt in range(max_attempts):
            try:
                # Build prompt with increasing strictness for retries
                prompt = self._build_clarification_prompt(task_text, ext_knowledge, strict=(attempt > 0))

                if attempt > 0:
                    logger.info(f"Retry {attempt + 1}: Using stricter prompt for JSON parsing")

                # Use different cache key for each attempt to avoid cache poisoning
                cache_key = f"intent_clarification:{hash(task_text)}_{attempt}"
                response = await self.llm_call_with_retry(
                    prompt=prompt,
                    operation_name="intent_clarification",
                    cache_key=cache_key,
                    max_retries=1,  # Reduce retries on each attempt
                )

                clarification_result = None
                response_text = ""
                if isinstance(response, dict):
                    if "clarified_task" in response:
                        clarification_result = response
                    else:
                        response_text = (
                            response.get("text")
                            or response.get("result")
                            or response.get("raw_response")
                            or ""
                        )
                else:
                    response_text = str(response)

                if not clarification_result:
                    if not response_text.strip():
                        logger.warning(
                            "Intent clarification received empty response (attempt %d of %d)",
                            attempt + 1,
                            max_attempts,
                        )
                        continue

                    # Parse LLM response using robust JSON utility
                    clarification_result = llm_result2json(response_text, expected_type=dict)

                if clarification_result is not None:
                    # Validate and normalize required fields
                    if "clarified_task" not in clarification_result:
                        clarification_result["clarified_task"] = task_text

                    if "entities" not in clarification_result:
                        clarification_result["entities"] = {}

                    if "corrections" not in clarification_result:
                        clarification_result["corrections"] = {}

                    if "confidence" not in clarification_result:
                        clarification_result["confidence"] = 0.7  # Better default than 0.3

                    # Ensure entities structure is complete
                    entities = clarification_result["entities"]
                    if "business_terms" not in entities:
                        entities["business_terms"] = []
                    if "time_range" not in entities:
                        entities["time_range"] = None
                    if "dimensions" not in entities:
                        entities["dimensions"] = []
                    if "metrics" not in entities:
                        entities["metrics"] = []

                    # Ensure corrections structure is complete
                    corrections = clarification_result["corrections"]
                    if "typos_fixed" not in corrections:
                        corrections["typos_fixed"] = []
                    if "ambiguities_resolved" not in corrections:
                        corrections["ambiguities_resolved"] = []

                    logger.info(
                        f"Intent clarification succeeded on attempt {attempt + 1}: "
                        f"clarified='{clarification_result.get('clarified_task', '')[:50]}...', "
                        f"confidence={clarification_result.get('confidence', 0.0)}"
                    )

                    return clarification_result

                logger.warning(
                    "Intent clarification JSON parsing failed on attempt %s. Raw response: %s",
                    attempt + 1,
                    response_text,
                )

            except Exception as e:
                logger.warning(f"Intent clarification attempt {attempt + 1} failed: {e}")

        # All attempts exhausted - use enhanced fallback with basic entity extraction
        logger.warning("All intent clarification attempts failed, using enhanced fallback")

        # Extract basic entities using regex patterns
        entities = self._extract_entities_fallback(task_text)

        return {
            "clarified_task": task_text,
            "entities": entities,
            "corrections": {"typos_fixed": [], "ambiguities_resolved": []},
            "confidence": 0.5,  # Higher than 0.3, signals partial success
            "fallback": True,
            "error": "JSON parsing failed after retries",
        }

    def _build_clarification_prompt(self, task_text: str, ext_knowledge: Optional[str], strict: bool) -> str:
        """Build prompt for intent clarification with configurable strictness."""
        if strict:
            # Stricter prompt for retries
            return f"""你是一个专业的数据分析助手。请分析用户的查询意图并输出结构化信息。

用户查询：{task_text}

业务知识：
{ext_knowledge or "无"}

【重要】请严格按照以下JSON格式输出，不要包含任何其他文字、解释或标记：
{{
    "clarified_task": "澄清和规范化后的查询",
    "entities": {{
        "business_terms": ["业务术语1", "业务术语2"],
        "time_range": "时间范围",
        "dimensions": ["维度1", "维度2"],
        "metrics": ["指标1"]
    }},
    "corrections": {{
        "typos_fixed": ["原词→纠正词"],
        "ambiguities_resolved": ["澄清的模糊表述"]
    }},
    "confidence": 0.95
}}

示例：
输入："华山地区最近的销售额"
输出：{{"clarified_task": "华南地区最近30天的销售额", "entities": {{"business_terms": ["华南", "销售额"], "time_range": "最近30天"}}, "corrections": {{"typos_fixed": ["华山→华南"]}}, "confidence": 0.9}}

要求：
1. 必须只输出JSON对象，不要包含markdown标记（如```json）
2. 不要输出任何解释文字
3. 确保JSON格式完全正确，所有括号和引号都匹配
"""
        else:
            # Normal prompt for first attempt
            return f"""你是一个专业的数据分析助手。请分析用户的查询意图并输出结构化信息。

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
    "confidence": 0.95
}}

重要：
1. 仅输出 JSON，不要包含其他解释
2. 如果没有错别字或模糊表述，corrections 可以为空对象
3. confidence 应该基于查询的清晰度和完整性（0-1之间）
4. 如果查询已经很清晰，clarified_task 可以与原查询相同或稍作调整
"""

    def _extract_entities_fallback(self, task_text: str) -> Dict[str, Any]:
        """
        Extract basic entities using simple regex patterns when LLM fails.

        This is a fallback method to provide at least some entity extraction
        when JSON parsing completely fails.
        """
        import re

        entities = {"business_terms": [], "time_range": None, "dimensions": [], "metrics": []}

        # Extract time-related keywords
        time_patterns = [
            r"最近\d+天",
            r"最近\d+月",
            r"最近\d+周",
            r"\d+月份",
            r"\d+月",
            r"今年",
            r"去年",
            r"本月",
            r"上月",
            r"每天",
            r"每周",
            r"每月",
            r"每个?月",
            r"每年",
        ]
        for pattern in time_patterns:
            match = re.search(pattern, task_text)
            if match:
                entities["time_range"] = match.group(0)
                break

        # Extract business terms (quoted content in single or double quotes, or Chinese quotes)
        quoted_terms = re.findall(r"""['"‘’“”]([^'"‘’“”]*?)['"‘’“”]""", task_text)
        if quoted_terms:
            entities["business_terms"] = quoted_terms

        # Extract common metric keywords
        metric_keywords = ["销售额", "利润", "转化率", "平均", "总计", "数量", "金额", "周期"]
        for keyword in metric_keywords:
            if keyword in task_text:
                entities["metrics"].append(keyword)

        return entities

    @staticmethod
    def _has_entity_content(entities: Dict[str, Any]) -> bool:
        if not isinstance(entities, dict):
            return False
        if entities.get("time_range"):
            return True
        for key in ("business_terms", "dimensions", "metrics"):
            value = entities.get(key) or []
            if isinstance(value, list) and len(value) > 0:
                return True
        return False

    @staticmethod
    def _has_correction_content(corrections: Dict[str, Any]) -> bool:
        if not isinstance(corrections, dict):
            return False
        for key in ("typos_fixed", "ambiguities_resolved"):
            value = corrections.get(key) or []
            if isinstance(value, list) and len(value) > 0:
                return True
        return False

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
