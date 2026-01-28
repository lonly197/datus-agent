# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Shared utilities for FTS (Full-Text Search) query processing.

This module provides common functions for:
- Query rewriting with LLM for Chinese keyword extraction
- FTS query simplification and tokenization
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datus.configuration.agent_config import AgentConfig


# ============================================================================
# FTS Query Configuration
# ============================================================================

# Maximum number of tokens to return from query simplification
FTS_MAX_TOKENS: int = 6

# Common business terms to preserve during Chinese tokenization
# These are automotive/dealership specific terms commonly used in queries
FTS_BUSINESS_TERMS: list[str] = [
    "线索",
    "试驾",
    "订单",
    "漏斗",
    "到店",
    "转化",
    "统计",
    "渠道",
    "有效",
    "意向",
    "车型",
    "车种",
]

# Chinese stop characters to skip during bigram generation
FTS_STOP_CHARS: set[str] = {
    "的",
    "和",
    "与",
    "及",
    "或",
    "按",
    "对",
    "在",
    "中",
    "以",
    "于",
    "为",
}


# ============================================================================
# LLM Query Rewrite
# ============================================================================

def rewrite_fts_query_with_llm(
    query: str,
    agent_config: "AgentConfig",
    model_name: str = "",
) -> str:
    """
    Rewrite a user query into concise Chinese keywords for FTS schema search.

    This uses an LLM to extract and simplify the key search terms from a natural
    language query, improving FTS recall for schema discovery.

    Args:
        query: The original user query in natural language
        agent_config: The agent configuration containing LLM settings
        model_name: Optional override for the LLM model name

    Returns:
        Simplified keywords string suitable for FTS search.
        Returns original query if rewriting fails or produces empty result.
    """
    if not query or not agent_config:
        return query

    from datus.models.base import LLMBaseModel

    try:
        llm_model = LLMBaseModel.create_model(agent_config=agent_config, model_name=model_name or None)
    except Exception:
        return query

    prompt = (
        "You are a data analyst. Rewrite the user query into short Chinese keywords for schema search.\n"
        "Return ONLY JSON: {{\"query\": \"<keywords>\"}}.\n"
        "Constraints:\n"
        "- Use 3-8 concise keywords\n"
        "- Keep product or model names as-is (e.g., 铂智3X)\n"
        "- Remove filler words\n"
        f"User query: {query}\n"
    )

    try:
        response = llm_model.generate_with_json_output(prompt)
    except Exception:
        return query

    if isinstance(response, dict):
        rewritten = response.get("query", "")
        if isinstance(rewritten, str) and rewritten.strip():
            return rewritten.strip()

    return query
