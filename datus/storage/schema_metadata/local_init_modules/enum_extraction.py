# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Enum extraction module for local schema initialization.

This module provides functionality for extracting enum values from column comments,
with optional LLM-enhanced extraction for better accuracy.
"""

from typing import Any, Dict, List, Optional

from datus.models.base import LLMBaseModel
from datus.utils.loggings import get_logger
from datus.utils.sql_utils import extract_enum_values_from_comment

from ..llm_enhanced_extract import EnhancedEnumExtractor

logger = get_logger(__name__)

# Global enum extractor instance
_enum_extractor: Optional[EnhancedEnumExtractor] = None


def _get_enum_extractor(llm_model: Optional[LLMBaseModel], llm_enum_extraction: bool) -> Optional[EnhancedEnumExtractor]:
    """
    Get or create the enhanced enum extractor.

    Args:
        llm_model: LLM model for enhanced extraction
        llm_enum_extraction: Whether to use LLM-enhanced extraction

    Returns:
        EnhancedEnumExtractor instance if enabled, None otherwise
    """
    global _enum_extractor
    if llm_enum_extraction and llm_model:
        if _enum_extractor is None or _enum_extractor.llm_model != llm_model:
            _enum_extractor = EnhancedEnumExtractor(
                llm_model=llm_model,
                use_regex_first=True,
                regex_confidence_threshold=0.6,
            )
        return _enum_extractor
    return None


def _extract_enums(
    comment: str,
    column_name: str = "",
    llm_extractor: Optional[EnhancedEnumExtractor] = None,
) -> List[Dict[str, str]]:
    """
    Extract enum values from a comment, optionally using LLM enhancement.

    Args:
        comment: The comment text to parse
        column_name: Optional column name for context
        llm_extractor: Optional LLM-enhanced extractor

    Returns:
        List of {"value": code, "label": label} dicts
    """
    if not comment:
        return []

    if llm_extractor:
        enums, is_enum, confidence = llm_extractor.extract(comment, column_name)
        if enums:
            logger.debug(
                f"LLM enum extraction for {column_name}: {len(enums)} enums, "
                f"confidence: {confidence:.2f}"
            )
            return [{"value": e["code"], "label": e["label"]} for e in enums]

    # Fallback to regex
    pairs = extract_enum_values_from_comment(comment)
    if pairs:
        return [{"value": code, "label": label} for code, label in pairs]

    return []


__all__ = [
    "_get_enum_extractor",
    "_extract_enums",
]
