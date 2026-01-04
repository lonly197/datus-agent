# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import os
from typing import Optional

from datus.configuration.agent_config import ModelConfig
from datus.models.openai_compatible import OpenAICompatibleModel
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class GlmModel(OpenAICompatibleModel):
    """
    Implementation of the BaseModel for ZhipuAI's GLM API.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        **kwargs,
    ):
        super().__init__(model_config, **kwargs)
        logger.debug(f"Using GLM model: {self.model_name} base Url: {self.base_url}")

    def _get_api_key(self) -> str:
        """Get GLM API key from config or environment."""
        api_key = self.model_config.api_key or os.environ.get("GLM_API_KEY") or os.environ.get("ZHIPUAI_API_KEY")
        if not api_key:
            raise ValueError("GLM API key must be provided or set as GLM_API_KEY environment variable")
        return api_key

    def _get_base_url(self) -> str:
        """Get GLM base URL from config or environment."""
        return self.model_config.base_url or os.environ.get("GLM_API_BASE", "https://open.bigmodel.cn/api/paas/v4/")

    def token_count(self, prompt: str) -> int:
        """
        Estimate the number of tokens in a text using the GLM tokenizer approximation.
        """
        # GLM models generally have a token-to-char ratio similar to other models
        return int(len(prompt) * 0.3 + 0.5)

    def max_tokens(self) -> int:
        """
        Get the maximum number of tokens for the model.
        Most GLM-4 models support 128k context.
        """
        # Default to 128k for GLM-4 if not specified in specs
        return 128000
