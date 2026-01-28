# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details

"""
LLM-driven table discovery for schema discovery.

This module provides LLM-based table selection from actual database tables,
preventing hallucination of non-existent table names.
"""

from typing import Any, List

from datus.configuration.business_term_config import LLM_TABLE_DISCOVERY_CONFIG
from datus.utils.error_handler import NodeExecutionResult
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


async def llm_based_table_discovery(
    query: str,
    all_tables: List[str],
    llm_call_func,
) -> List[str]:
    """
    Stage 1.5: Use LLM to select relevant tables from actual database tables.

    Fixed: LLM selects from real DB tables instead of hallucinating.
    The LLM is given the actual list of database tables and asked to select
    the most relevant ones for the query. This prevents hallucination of
    non-existent table names.

    Args:
        query: User query text
        all_tables: List of actual database tables
        llm_call_func: Async function for LLM calls with retry

    Returns:
        List of validated table names that exist in the database
    """
    if not query:
        return []

    if not all_tables:
        logger.debug("No database tables available for LLM-based discovery")
        return []

    # Use configured prompt template with actual table list
    prompt_template = LLM_TABLE_DISCOVERY_CONFIG.get("prompt_template", "")
    # Format table list for prompt (limit to prevent token overflow)
    tables_list = "\n".join(f"- {t}" for t in all_tables[:100])
    prompt = prompt_template.format(tables_list=tables_list, query=query)

    try:
        # Use LLMMixin with retry and caching
        cache_key = f"llm_table_discovery:{hash(query)}:{hash(tuple(all_tables))}"
        max_retries = LLM_TABLE_DISCOVERY_CONFIG.get("max_retries", 3)
        cache_enabled = LLM_TABLE_DISCOVERY_CONFIG.get("cache_enabled", True)
        cache_ttl_seconds = LLM_TABLE_DISCOVERY_CONFIG.get("cache_ttl_seconds") if cache_enabled else None

        response = await llm_call_func(
            prompt=prompt,
            operation_name="table_discovery",
            cache_key=cache_key if cache_enabled else None,
            max_retries=max_retries,
            cache_ttl_seconds=cache_ttl_seconds,
        )

        tables = response.get("tables", [])
        # Validate that returned tables actually exist in database
        all_tables_set = set(all_tables)
        validated_tables = [str(t) for t in tables if isinstance(t, (str, int, float)) and str(t) in all_tables_set]
        unique_tables = list(set(validated_tables))

        if len(tables) != len(validated_tables):
            invalid_tables = set(str(t) for t in tables if isinstance(t, (str, int, float))) - all_tables_set
            logger.warning(
                f"LLM returned {len(invalid_tables)} invalid table(s) that don't exist: {invalid_tables}"
            )

        logger.info(f"LLM selected {len(unique_tables)} tables from actual database: {unique_tables}")
        return unique_tables

    except NodeExecutionResult as e:
        logger.warning(f"LLM table discovery failed after retries: {e.error_message}")
        return []
    except Exception as e:
        logger.warning(f"LLM table discovery failed with unexpected error: {e}")
        return []
