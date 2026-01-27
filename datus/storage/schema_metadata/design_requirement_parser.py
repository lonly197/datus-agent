# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Design Requirement Parser Module

This module provides LLM-based parsing of design requirement specifications
from structured data rows.
"""

import json
from typing import Any, Dict

from datus.utils.loggings import get_logger

logger = get_logger(__name__)

# Default LLM prompt for design requirement parsing
DEFAULT_DESIGN_REQ_PROMPT = """You are a data modeling expert. Parse design requirement specifications.

## Input
Requirement Row: {row_data}

## Task
Extract structured information from design requirement specifications.

## Expected Columns in Source
- 分析对象编码 (Analysis Object Code)
- 分析对象（中文） (Analysis Object Chinese)
- 属性（中文） (Attribute Chinese)
- 属性业务定义 (Business Definition)
- 数据Owner (Data Owner)
- 安全分类 (Security Classification)
- 安全等级 (Security Level)

## Output Format
Return ONLY a JSON object:
{{
  "object_code": "MSD-01-001-03",
  "object_name_cn": "客户",
  "attribute_name_cn": "客户类型",
  "business_definition": "描述客户类型的业务含义",
  "data_owner": "责任人名称",
  "security_classification": "数据安全分类",
  "security_level": "1/2/3",
  "confidence": 0.0-1.0
}}

If field is empty/missing, use empty string.
"""


def parse_design_requirement_with_llm(
    row_data: Dict[str, Any],
    llm_model: Any,
    prompt_template: str = DEFAULT_DESIGN_REQ_PROMPT,
) -> Dict[str, Any]:
    """
    Use LLM to parse design requirement specification rows.

    Args:
        row_data: Dictionary containing requirement row data
        llm_model: LLM model instance
        prompt_template: Custom prompt template (optional)

    Returns:
        Dictionary with parsed requirement fields
    """
    if not row_data or not llm_model:
        return {}

    row_str = json.dumps(row_data, ensure_ascii=False, indent=2)
    prompt = prompt_template.format(row_data=row_str)

    try:
        response = llm_model.generate_with_json_output(prompt)

        if not isinstance(response, dict):
            logger.debug(f"LLM design req response is not a dict: {type(response)}")
            return {}

        parsed = {
            "object_code": response.get("object_code", ""),
            "object_name_cn": response.get("object_name_cn", ""),
            "attribute_name_cn": response.get("attribute_name_cn", ""),
            "business_definition": response.get("business_definition", ""),
            "data_owner": response.get("data_owner", ""),
            "security_classification": response.get("security_classification", ""),
            "security_level": response.get("security_level", ""),
            "confidence": response.get("confidence", 0.0),
        }

        logger.debug(
            f"LLM parsed design requirement: {parsed.get('object_code', 'unknown')}"
        )
        return parsed

    except Exception as e:
        logger.warning(f"LLM design requirement parsing failed: {e}")
        return {}
