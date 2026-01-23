# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import os
from typing import Dict, List, Union

import google.genai as genai

from datus.configuration.agent_config import ModelConfig
from datus.models.openai_compatible import OpenAICompatibleModel
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class GeminiModel(OpenAICompatibleModel):
    """Google Gemini model implementation

    Note: This class inherits from OpenAICompatibleModel for code reuse but
    uses Google's native genai client instead of OpenAI-compatible API.
    """

    def __init__(self, model_config: ModelConfig, **kwargs):
        # Initialize only what's needed, skip OpenAI client creation
        self.model_config = model_config
        self.model_name = model_config.model
        self.api_key = self._get_api_key()
        self.base_url = self._get_base_url()
        self.default_headers = self.model_config.default_headers

        # Initialize Gemini-specific client (skip OpenAI clients)
        import google.genai as genai

        self.client = genai.Client(api_key=self.api_key)
        self.gemini_model = self.client.models.generate_content

        # Context for tracing
        self.current_node = None

        # Cache for model info
        self._model_info = None

    def _get_api_key(self) -> str:
        """Get Gemini API key from config or environment."""
        api_key = self.model_config.api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Gemini API key must be provided or set as GEMINI_API_KEY environment variable")
        return api_key

    def generate(self, prompt: Union[str, List[Dict[str, str]]], **kwargs) -> str:
        try:
            # Convert prompt to string if it's a list
            if isinstance(prompt, list):
                prompt_text = "\n".join([msg.get("content", "") for msg in prompt if isinstance(msg, dict)])
            else:
                prompt_text = str(prompt)

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt_text,
                config={
                    "temperature": kwargs.get("temperature", 0.7),
                    "max_output_tokens": kwargs.get("max_tokens", 10000),
                    "top_p": kwargs.get("top_p", 1.0),
                    "top_k": kwargs.get("top_k", 40),
                },
            )

            if response.candidates:
                return response.candidates[0].content.parts[0].text
            else:
                logger.warning("No candidates returned from Gemini model")
                return ""

        except Exception as e:
            logger.error(f"Error generating content with Gemini: {str(e)}")
            raise

    def _get_base_url(self) -> str:
        """Get Gemini base URL for OpenAI compatibility."""
        return "https://generativelanguage.googleapis.com/v1beta/openai"

    def token_count(self, prompt: str) -> int:
        try:
            response = self.client.models.count_tokens(model=self.model_name, contents=prompt)
            return getattr(response, "total_tokens", len(prompt) // 4)
        except Exception as e:
            logger.warning(f"Error counting tokens with Gemini: {str(e)}")
            return len(prompt) // 4
