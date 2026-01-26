# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""MiniMax model implementation for Datus.

MiniMax API is OpenAI-compatible, so this implementation inherits from OpenAICompatibleModel.
See https://platform.minimaxi.com/docs/guides/models-intro for API documentation.
"""

from typing import Dict, Optional

from datus.configuration.agent_config import ModelConfig
from datus.models.openai_compatible import OpenAICompatibleModel
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class MiniMaxModel(OpenAICompatibleModel):
    """
    MiniMax language model implementation.

    MiniMax provides OpenAI-compatible API. This class extends OpenAICompatibleModel
    to provide MiniMax-specific configuration and defaults.

    Supported models (from official docs):
    - MiniMax-M2.1: 多语言编程SOTA，为真实世界复杂任务而生
    - MiniMax-M2.1-lightning: 与 M2.1 同等效果，速度大幅提升
    - MiniMax-M2: 专为高效编码与Agent工作流而生
    """

    # MiniMax model specifications
    # Based on official MiniMax documentation:
    # - M1 series supports up to 1 million token context length
    # - M2 series is optimized for coding and agent workflows
    MODEL_SPECS = {
        # MiniMax-M2.1 Series (Current Flagship)
        "MiniMax-M2.1": {"context_length": 1048576, "max_tokens": 65536},
        "MiniMax-M2.1-lightning": {"context_length": 1048576, "max_tokens": 65536},
        # MiniMax-M2 Series (Coding & Agent Optimized)
        "MiniMax-M2": {"context_length": 1048576, "max_tokens": 65536},
        # Legacy M1 Series (for backward compatibility)
        "MiniMax-M1": {"context_length": 1048576, "max_tokens": 65536},
        "MiniMax-Text-01": {"context_length": 1048576, "max_tokens": 65536},
    }

    def __init__(self, model_config: ModelConfig, **kwargs):
        """Initialize MiniMax model with configuration."""
        super().__init__(model_config, **kwargs)
        logger.info(f"Initialized MiniMax model: {self.model_name}")

    def _get_api_key(self) -> str:
        """Get MiniMax API key from config or environment variable.

        MiniMax API key can be configured via:
        1. model_config.api_key
        2. MINIMAX_API_KEY environment variable
        """
        # First try config
        if self.model_config.api_key:
            return self.model_config.api_key

        # Fallback to environment variable
        import os

        api_key = os.environ.get("MINIMAX_API_KEY", "")
        if not api_key:
            logger.warning("MiniMax API key not found in config or MINIMAX_API_KEY env var")
        return api_key

    def _get_base_url(self) -> Optional[str]:
        """Get MiniMax API base URL.

        MiniMax default API endpoint: https://api.minimax.chat/v1/text/chatcompletion_v2
        """
        if self.model_config.base_url:
            return self.model_config.base_url

        # Default MiniMax API endpoint
        return "https://api.minimax.chat/v1/text/chatcompletion_v2"

    @property
    def model_specs(self) -> Dict[str, Dict[str, int]]:
        """MiniMax model specifications (context_length and max_tokens)."""
        return self.MODEL_SPECS

    def _uses_completion_tokens_parameter(self) -> bool:
        """
        Check if model uses max_completion_tokens instead of max_tokens.

        MiniMax follows OpenAI's parameter conventions.

        Returns:
            False - uses standard max_tokens parameter
        """
        return False
