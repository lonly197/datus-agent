# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Business Metadata Extraction Module

This module provides LLM-based extraction of business metadata from DDL comments,
including field types, business meanings, and related concepts.
"""

from typing import Any, Dict, List

from datus.utils.loggings import get_logger

logger = get_logger(__name__)

# Default LLM prompt for business metadata extraction
DEFAULT_METADATA_EXTRACT_PROMPT = """You are a data warehouse schema analyst. Extract business metadata from DDL comments.

## Input
Column Comment: {comment}
Column Name: {column_name}
Table Name: {table_name}

## Task
Analyze the comment and extract business metadata.

## Output Format
Return ONLY a JSON object:
{{
  "field_type": "enum|categorical|text|date|number|boolean|unknown",
  "is_key_field": true/false,
  "business_meaning": "concise description of business purpose",
  "related_concepts": ["concept1", "concept2"],
  "enum_values": [
    {{"code": "0", "label": "description", "description": "detailed explanation"}}
  ],
  "calculation_hint": "how this field is typically calculated (if applicable)",
  "confidence": 0.0-1.0
}}

## Rules
- field_type: classify the data type based on comment
- is_key_field: true if this is an ID, code, or key identifier
- business_meaning: 1-2 sentence description
- related_concepts: extract business terms and concepts
- enum_values: extract any enumeration values mentioned
- calculation_hint: for derived/calculated fields
"""


def extract_business_metadata_with_llm(
    comment: str,
    column_name: str,
    table_name: str,
    llm_model: Any,
    prompt_template: str = DEFAULT_METADATA_EXTRACT_PROMPT,
) -> Dict[str, Any]:
    """
    Use LLM to extract comprehensive business metadata from a column comment.

    Args:
        comment: The column comment text
        column_name: Name of the column
        table_name: Name of the table
        llm_model: LLM model instance
        prompt_template: Custom prompt template (optional)

    Returns:
        Dictionary with extracted business metadata
    """
    if not comment or not llm_model:
        return {}

    prompt = prompt_template.format(
        comment=comment,
        column_name=column_name,
        table_name=table_name
    )

    try:
        response = llm_model.generate_with_json_output(prompt)

        if not isinstance(response, dict):
            logger.debug(f"LLM metadata response is not a dict: {type(response)}")
            return {}

        metadata = {
            "field_type": response.get("field_type", "unknown"),
            "is_key_field": response.get("is_key_field", False),
            "business_meaning": response.get("business_meaning", "")[:500],
            "related_concepts": response.get("related_concepts", [])[:10],
            "enum_values": response.get("enum_values", []),
            "calculation_hint": response.get("calculation_hint", ""),
            "confidence": response.get("confidence", 0.0),
        }

        logger.debug(f"LLM extracted metadata for {table_name}.{column_name}")
        return metadata

    except Exception as e:
        logger.warning(f"LLM metadata extraction failed: {e}")
        return {}
