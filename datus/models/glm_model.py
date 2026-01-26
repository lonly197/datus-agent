# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
GLM (智谱 AI) 模型实现。

支持智谱 AI 的 GLM 系列文本模型，API 兼容 OpenAI 格式。
官方文档: https://docs.bigmodel.cn/cn/guide/start/model-overview
"""

import os
from typing import Any, Optional

from datus.configuration.agent_config import ModelConfig
from datus.models.openai_compatible import OpenAICompatibleModel
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class GlmModel(OpenAICompatibleModel):
    """
    智谱 AI GLM 系列语言模型实现。

    支持的模型（基于官方文档 https://docs.bigmodel.cn/cn/guide/start/model-overview）:

    文本模型:
    - GLM-4.7: 高智能旗舰模型，通用对话、推理与智能体能力全面升级
    - GLM-4.7-FlashX: 轻量高速模型，适用于中文写作、翻译、长文本等通用场景
    - GLM-4.6: 超强性能模型，上下文提升至 200K，高级编码和推理能力
    - GLM-4.5-Air: 高性价比模型，在推理、编码和智能体任务上表现强劲
    - GLM-4.5-AirX: 高性价比-极速版，推理速度快，适用于时效性要求场景
    - GLM-4-Long: 超长输入模型，支持高达 1M 上下文长度
    - GLM-4.7-Flash: 免费模型，最新基座模型的普惠版本
    - GLM-4-Flash-250414: 免费模型，超长上下文处理能力，多语言支持
    """

    # GLM 模型规格（基于官方文档）
    MODEL_SPECS = {
        # GLM-4.7 系列（最新旗舰）
        "GLM-4.7": {"context_length": 200000, "max_output": 128000},
        "GLM-4.7-FlashX": {"context_length": 200000, "max_output": 128000},
        # GLM-4.6 系列
        "GLM-4.6": {"context_length": 200000, "max_output": 128000},
        # GLM-4.5 系列
        "GLM-4.5-Air": {"context_length": 128000, "max_output": 96000},
        "GLM-4.5-AirX": {"context_length": 128000, "max_output": 96000},
        # GLM-4 系列
        "GLM-4-Long": {"context_length": 1000000, "max_output": 4000},
        "GLM-4-FlashX-250414": {"context_length": 128000, "max_output": 16000},
        # 免费模型
        "GLM-4.7-Flash": {"context_length": 200000, "max_output": 128000},
        "GLM-4-Flash-250414": {"context_length": 128000, "max_output": 16000},
        # 旧版模型（兼容）
        "glm-4": {"context_length": 128000, "max_output": 4096},
        "glm-4-plus": {"context_length": 128000, "max_output": 4096},
        "glm-4-flash": {"context_length": 128000, "max_output": 4096},
        "glm-4-air": {"context_length": 128000, "max_output": 4096},
    }

    def __init__(self, model_config: ModelConfig, **kwargs):
        """初始化 GLM 模型。"""
        super().__init__(model_config, **kwargs)
        logger.info(f"Initialized GLM model: {self.model_name}")

    def generate(self, prompt: Any, enable_thinking: bool = False, **kwargs) -> str:
        """
        生成 GLM 模型响应。

        Args:
            prompt: 输入提示词
            enable_thinking: 启用深度思考模式（部分模型支持）
            **kwargs: 其他生成参数

        Returns:
            生成的文本响应
        """
        params = {
            "model": self.model_name,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", self.max_output_tokens()),
            "top_p": kwargs.get("top_p", 1.0),
            **{k: v for k, v in kwargs.items() if k not in ["temperature", "max_tokens", "top_p"]},
        }

        # 深度思考模式（GLM-4.5-Flash 及部分模型支持）
        if enable_thinking:
            params["thinking"] = {"type": "enabled"}

        # 转换为 messages 格式
        if isinstance(prompt, list):
            messages = prompt
        else:
            messages = [{"role": "user", "content": str(prompt)}]

        response = self.client.chat.completions.create(messages=messages, **params)
        return response.choices[0].message.content

    def _get_api_key(self) -> str:
        """获取 GLM API Key。"""
        api_key = (
            self.model_config.api_key
            or os.environ.get("GLM_API_KEY")
            or os.environ.get("ZHIPUAI_API_KEY")
        )
        if not api_key:
            raise ValueError(
                "GLM API key must be provided in config or set as GLM_API_KEY/ZHIPUAI_API_KEY environment variable"
            )
        return api_key

    def _get_base_url(self) -> str:
        """获取 GLM API 基础 URL。"""
        return self.model_config.base_url or os.environ.get(
            "GLM_API_BASE", "https://open.bigmodel.cn/api/paas/v4/"
        )

    def context_tokens(self) -> int:
        """
        获取模型支持的最大上下文长度。

        Returns:
            上下文 token 数量
        """
        return self.MODEL_SPECS.get(self.model_name, {}).get("context_length", 128000)

    def max_output_tokens(self) -> int:
        """
        获取模型支持的最大输出 token 数。

        Returns:
            最大输出 token 数量
        """
        return self.MODEL_SPECS.get(self.model_name, {}).get("max_output", 4096)

    def token_count(self, prompt: str) -> int:
        """
        估算文本的 token 数量。

        Args:
            prompt: 输入文本

        Returns:
            估算的 token 数量
        """
        return int(len(prompt) * 0.3 + 0.5)

    def _uses_completion_tokens_parameter(self) -> bool:
        """
        检查模型是否使用 max_completion_tokens 而非 max_tokens。

        Returns:
            False - 使用标准 max_tokens 参数
        """
        return False
