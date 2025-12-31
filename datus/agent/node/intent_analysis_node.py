# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
IntentAnalysisNode implementation for analyzing query intent.
"""

from typing import Any, AsyncGenerator, Dict, Optional

from datus.agent.node.node import Node
from datus.agent.workflow import Workflow
from datus.configuration.agent_config import AgentConfig
from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus
from datus.schemas.node_models import BaseInput
from datus.utils.loggings import get_logger

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

    def setup_input(self, workflow: Workflow) -> Dict[str, Any]:
        """
        Setup intent analysis input from workflow task.

        Args:
            workflow: Workflow instance containing the task to analyze

        Returns:
            Dictionary with success status
        """
        if not self.input:
            self.input = BaseInput()

        return {"success": True, "message": "Intent analysis input setup complete"}

    async def run(self) -> AsyncGenerator[ActionHistory, None]:
        """
        Run intent analysis using heuristics and optional LLM fallback.

        Yields:
            ActionHistory: Progress and result actions
        """
        try:
            # Extract task text from workflow
            task_text = ""
            if hasattr(self.workflow, 'task') and self.workflow.task:
                task_text = self.workflow.task.task or ""

            if not task_text.strip():
                logger.warning("No task text provided for intent analysis")
                yield self._create_error_action("No task text provided for analysis")
                return

            # Perform intent detection
            intent_result = await self._detect_intent(task_text)

            # Populate workflow metadata
            if self.workflow:
                self.workflow.metadata['detected_intent'] = intent_result.intent
                self.workflow.metadata['intent_confidence'] = intent_result.confidence
                self.workflow.metadata['intent_metadata'] = intent_result.metadata

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
            yield self._create_error_action(str(e))

    async def _detect_intent(self, task_text: str) -> 'IntentResult':
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
        intent_result = detector.detect_intent_heuristic(task_text)

        # If confidence is low and LLM fallback is enabled, try LLM
        use_llm_fallback = getattr(self.agent_config, 'intent_detector_llm_fallback', False) if self.agent_config else False

        if intent_result.confidence < 0.7 and use_llm_fallback:
            try:
                # Get model for LLM fallback
                model = None
                if self.agent_config:
                    model = self.agent_config.get_model()

                if model:
                    llm_result = await detector.detect_intent_llm(task_text, model)
                    if llm_result.confidence > intent_result.confidence:
                        intent_result = llm_result
                        logger.info("LLM fallback improved intent detection confidence")
            except Exception as e:
                logger.warning(f"LLM fallback for intent detection failed: {e}")

        return intent_result

    def _create_error_action(self, error_message: str) -> ActionHistory:
        """Create an error action for intent analysis failures."""
        return ActionHistory(
            action_id=f"{self.id}_error",
            role=ActionRole.TOOL,
            messages=f"Intent analysis failed: {error_message}",
            action_type="intent_analysis",
            input={},
            status=ActionStatus.FAILED,
            output={"error": error_message},
        )
