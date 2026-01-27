# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Node configuration constants.

This module centralizes hardcoded default values used across the codebase,
making configuration management easier and improving maintainability.
"""

# =============================================================================
# Timeout Configuration
# =============================================================================

DEFAULT_NODE_TIMEOUT = 60  # Default timeout for node execution (seconds)
DEFAULT_PREFLIGHT_TOOL_TIMEOUT = 30.0  # Default timeout for preflight tools (seconds)

# =============================================================================
# LLM Parameters
# =============================================================================

DEFAULT_SUMMARIZATION_TEMPERATURE = 0.3  # Default temperature for summarization tasks
DEFAULT_SUMMARIZATION_MAX_TOKENS = 2000  # Default max tokens for summarization tasks
DEFAULT_SUMMARIZATION_MAX_TURNS = 1  # Default max turns for conversational summarization

# =============================================================================
# Retry Configuration
# =============================================================================

DEFAULT_LLM_MAX_RETRIES = 3  # Default max retries for LLM calls
DEFAULT_INTENT_CLARIFICATION_RETRIES = 1  # Max retries for intent clarification
DEFAULT_KNOWLEDGE_ENHANCEMENT_RETRIES = 2  # Max retries for knowledge enhancement

# =============================================================================
# Schema Discovery Configuration
# =============================================================================

# Hybrid search weights and thresholds
DEFAULT_HYBRID_COMMENT_BONUS = 0.05  # Bonus for matching table comments
DEFAULT_HYBRID_RERANK_WEIGHT = 0.2  # Weight for reranking in hybrid search
DEFAULT_HYBRID_RERANK_MIN_TABLES = 20  # Minimum tables to trigger reranking
DEFAULT_HYBRID_RERANK_TOP_N = 50  # Top N tables to consider for reranking

# Reranking system requirements
DEFAULT_HYBRID_RERANK_MIN_CPU_COUNT = 4  # Minimum CPU cores for reranking
DEFAULT_HYBRID_RERANK_MIN_MEMORY_GB = 8.0  # Minimum memory (GB) for reranking

# Default rerank model
DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-large"

# =============================================================================
# Suggester Configuration
# =============================================================================

DEFAULT_MAX_CORRELATIONS = 10  # Default max correlations to suggest
DEFAULT_JOIN_DEPTH = 3  # Default max depth for join suggestions
DEFAULT_VECTOR_SEARCH_TOP_N = 5  # Default top N for vector search
DEFAULT_MAX_KNOWLEDGE_LENGTH = 5000  # Default max knowledge length (characters)

# =============================================================================
# Summary Configuration
# =============================================================================

DEFAULT_SQL_SUMMARY_TOP_N = 5  # Default top N for SQL summary results
