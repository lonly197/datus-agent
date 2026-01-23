# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
IntentAnalysisNode implementation for analyzing query intent.
"""

from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, Optional

from datus.agent.node.node import Node, execute_with_async_stream
from datus.agent.workflow import Workflow
from datus.configuration.agent_config import AgentConfig
from datus.schemas.action_history import (ActionHistory, ActionHistoryManager,
                                          ActionRole, ActionStatus)
from datus.schemas.base import BaseResult
from datus.schemas.node_models import BaseInput
from datus.utils.exceptions import ErrorCode
from datus.utils.loggings import get_logger

if TYPE_CHECKING:
    from datus.agent.intent_detection import IntentResult

logger = get_logger(__name__)


class IntentAnalysisNode(Node):
    """
    Node for analyzing query intent using heuristics and optional LLM fallback.

    This node determines the type of query (text2sql, sql_review, data_analysis)
    and populates workflow metadata for downstream nodes to use.
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

    def should_skip(self, workflow: Workflow) -> bool:
        """
        Determine if this node should be skipped based on execution_mode.

        When execution_mode is explicitly specified (e.g., "text2sql"), the task type
        is already known, so we skip intent classification. IntentClarificationNode will
        handle query clarification instead (typos, ambiguities, entity extraction).

        Args:
            workflow: Workflow instance containing metadata

        Returns:
            True if execution_mode is specified, False otherwise
        """
        if not workflow or not hasattr(workflow, "metadata") or not workflow.metadata:
            return False

        execution_mode = workflow.metadata.get("execution_mode")
        should_skip_node = execution_mode is not None and execution_mode != ""

        if should_skip_node:
            logger.info(
                f"IntentAnalysisNode skipped: execution_mode='{execution_mode}' is specified. "
                f"IntentClarificationNode will handle query clarification instead."
            )

        return should_skip_node

    def setup_input(self, workflow: Workflow) -> Dict[str, Any]:
        """
        Setup intent analysis input from workflow task.

        Args:
            workflow: Workflow instance containing the task to analyze

        Returns:
            Dictionary with success status
        """
        # Store workflow reference for access to metadata
        self.workflow = workflow

        if not self.input:
            self.input = BaseInput()

        return {"success": True, "message": "Intent analysis input setup complete"}

    def update_context(self, workflow: Workflow) -> Dict[str, Any]:
        """
        Update workflow context with intent analysis results.

        Args:
            workflow: Workflow instance to update

        Returns:
            Dictionary with success status
        """
        # Intent analysis doesn't modify context directly
        return {"success": True, "message": "Intent analysis context update complete"}

    def execute(self) -> BaseResult:
        """
        Execute intent analysis synchronously.

        Returns:
            BaseResult: Execution result
        """
        # For synchronous execution, we use the async stream approach
        # execute_with_async_stream ensures self.result is properly set
        return execute_with_async_stream(self)

    async def execute_stream(
        self, action_history_manager: Optional[ActionHistoryManager] = None
    ) -> AsyncGenerator[ActionHistory, None]:
        """
        Execute intent analysis with streaming support.

        Args:
            action_history_manager: Optional action history manager

        Yields:
            ActionHistory: Progress updates during execution
        """
        # Store action history manager for use in run method
        self.action_history_manager = action_history_manager

        # Delegate to the existing run method
        async for action in self.run():
            yield action

    async def run(self) -> AsyncGenerator[ActionHistory, None]:
        """
        Run intent analysis using heuristics and optional LLM fallback.

        Yields:
            ActionHistory: Progress and result actions
        """
        try:
            # Extract task text from workflow
            task_text = ""
            if self.workflow and hasattr(self.workflow, "task") and self.workflow.task:
                task_text = self.workflow.task.task or ""

            if not task_text.strip():
                logger.warning("No task text provided for intent analysis")
                error_result = self.create_error_result(
                    ErrorCode.COMMON_VALIDATION_FAILED,
                    "No task text provided for intent analysis",
                    "intent_analysis",
                    {"task_id": getattr(self.workflow, "id", "unknown") if self.workflow else "unknown"},
                )
                yield ActionHistory(
                    action_id=f"{self.id}_error",
                    role=ActionRole.TOOL,
                    messages=f"Intent analysis failed: {error_result.error_message}",
                    action_type="intent_analysis",
                    input={"task_text": ""},
                    status=ActionStatus.FAILED,
                    output={"error": error_result.error_message, "error_code": error_result.error_code},
                )
                return

            # Perform intent detection
            intent_result = await self._detect_intent(task_text)

            # Populate workflow metadata with validation
            if self.workflow:
                try:
                    # Ensure metadata dict exists
                    if not hasattr(self.workflow, "metadata") or self.workflow.metadata is None:
                        self.workflow.metadata = {}

                    # Set intent analysis results
                    self.workflow.metadata["detected_intent"] = intent_result.intent
                    self.workflow.metadata["intent_confidence"] = intent_result.confidence
                    self.workflow.metadata["intent_metadata"] = intent_result.metadata

                    logger.debug(
                        f"Workflow metadata updated: intent={intent_result.intent}, confidence={intent_result.confidence}"
                    )

                except Exception as e:
                    logger.warning(f"Failed to update workflow metadata: {e}")
                    # Set fallback values to ensure downstream nodes don't fail
                    self.workflow.metadata["detected_intent"] = "text2sql"
                    self.workflow.metadata["intent_confidence"] = 0.5
                    self.workflow.metadata["intent_metadata"] = {"fallback": True, "error": str(e)}

            # ✅ Set self.result for successful execution
            self.result = BaseResult(
                success=True,
                data={
                    "intent": intent_result.intent,
                    "confidence": intent_result.confidence,
                    "metadata": intent_result.metadata,
                },
            )

            # Emit success action with results
            yield ActionHistory(
                action_id=f"{self.id}_intent_analysis",
                role=ActionRole.TOOL,
                messages=f"Intent analysis completed: {intent_result.intent} (confidence: {intent_result.confidence:.2f})",
                action_type="intent_analysis",
                input={"task_text": task_text},
                status=ActionStatus.SUCCESS,
                output={
                    "intent": intent_result.intent,
                    "confidence": intent_result.confidence,
                    "metadata": intent_result.metadata,
                },
            )

            logger.info(f"Intent analysis completed: {intent_result.intent} with confidence {intent_result.confidence}")

        except Exception as e:
            logger.error(f"Intent analysis failed: {e}")
            error_result = self.create_error_result(
                ErrorCode.NODE_EXECUTION_FAILED,
                f"Intent analysis execution failed: {str(e)}",
                "intent_analysis",
                {"task_id": getattr(self.workflow, "id", "unknown") if self.workflow else "unknown"},
            )

            # ✅ Set self.result for failed execution
            self.result = error_result

            yield ActionHistory(
                action_id=f"{self.id}_error",
                role=ActionRole.TOOL,
                messages=f"Intent analysis failed: {error_result.error_message}",
                action_type="intent_analysis",
                input={"task_text": ""},
                status=ActionStatus.FAILED,
                output={"error": error_result.error_message, "error_code": error_result.error_code},
            )

    async def _detect_intent(self, task_text: str) -> "IntentResult":
        """
        Detect intent using heuristics with optional LLM fallback.

        Args:
            task_text: The task text to analyze

        Returns:
            IntentResult: Detected intent with confidence and metadata
        """
        from datus.agent.intent_detection import IntentDetector, IntentResult

        # Use the existing intent detector
        detector = IntentDetector()

        # First try heuristic detection (fast)
        heuristic_result = detector.detect_sql_intent_by_keyword(task_text)

        is_sql_intent, metadata = heuristic_result

        # Derive intent from boolean flag
        intent = "sql" if is_sql_intent else "unknown"

        # Calculate confidence based on matches
        total_matches = metadata.get("total_matches", 0)
        has_patterns = metadata.get("has_sql_patterns", False)

        # Base confidence on total matches, with bonus for pattern matches
        base_confidence = min(total_matches * 0.2, 0.8)  # Max 0.8 from keyword matches
        if has_patterns:
            base_confidence += 0.2  # Bonus for pattern matches

        confidence = min(base_confidence, 1.0)

        intent_result = IntentResult(
            intent=intent,
            confidence=confidence,
            metadata=metadata,
        )

        # If confidence is low and LLM fallback is enabled, try LLM
        use_llm_fallback = (
            getattr(self.agent_config, "intent_detector_llm_fallback", False) if self.agent_config else False
        )

        if intent_result.confidence < 0.7 and use_llm_fallback:
            try:
                # Get model for LLM fallback
                model = None
                if self.agent_config:
                    from datus.models.base import LLMBaseModel

                    model = LLMBaseModel.create_model(model_name="default", agent_config=self.agent_config)

                if model:
                    llm_result = await detector.classify_intent_with_llm(task_text, model)
                    # Convert LLM result to IntentResult format
                    llm_confidence = float(llm_result[1])
                    if llm_confidence > intent_result.confidence:
                        intent_result = IntentResult(
                            intent=llm_result[0], confidence=llm_confidence, metadata={"llm_fallback": True}
                        )
                        logger.info("LLM fallback improved intent detection confidence")
            except Exception as e:
                logger.warning(f"LLM fallback for intent detection failed: {e}")

        return intent_result
