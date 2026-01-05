# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import os
from typing import Any

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

    def generate(self, prompt: Any, enable_thinking: bool = False, **kwargs) -> str:
        """
        Generate a response from the GLM model with thinking support.

        Args:
            prompt: The input prompt to send to the model
            enable_thinking: Enable thinking mode (default: False)
            **kwargs: Additional generation parameters

        Returns:
            The generated text response
        """
        # Merge default parameters with any provided kwargs
        params = {
            "model": self.model_name,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 1000),
            "top_p": kwargs.get("top_p", 1.0),
            **{k: v for k, v in kwargs.items() if k not in ["temperature", "max_tokens", "top_p"]},
        }

        # Add thinking parameter for GLM API when enabled
        if enable_thinking:
            params["thinking"] = {"type": "enabled"}

        # Convert prompt to messages format
        if isinstance(prompt, list):
            messages = prompt
        else:
            messages = [{"role": "user", "content": str(prompt)}]

        response = self.client.chat.completions.create(messages=messages, **params)
        return response.choices[0].message.content

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
