# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
LLM-Enhanced Enum and Metadata Extraction Module

This module provides LLM-based extraction for complex metadata patterns
that are difficult to parse with regex, such as:
- Nested enumeration in comments
- Enum values with complex descriptions
- Business terminology extraction
- Multi-language enum mappings
"""

import json
from typing import Any, Dict, List, Optional, Tuple

from datus.utils.loggings import get_logger

logger = get_logger(__name__)

# Default LLM prompt for enum extraction
DEFAULT_ENUM_EXTRACT_PROMPT = """You are a SQL comment parser. Extract all enum-like code/label pairs from the given comment.

## Input
Comment: {comment}

## Task
Extract all enumeration values in the format: code -> label

## Rules
1. Look for patterns like: "（1:选项1,2:选项2）" or "(0:option1,1:option2)"
2. Codes can be: numbers (0,1,2), letters (active,inactive), or boolean (true,false)
3. Labels are the descriptions following the code
4. Preserve Chinese characters exactly as written
5. Ignore false positives like dates, IDs, or simple numbers

## Output Format
Return ONLY a JSON object with this schema:
{{
  "enums": [
    {{"code": "0", "label": "录入"}},
    {{"code": "1", "label": "生产"}}
  ],
  "is_enum": true/false,
  "confidence": 0.0-1.0
}}

If no enums found or not an enum field, return:
{{"enums": [], "is_enum": false, "confidence": 0.0}}
"""


def extract_enums_with_llm(
    comment: str,
    llm_model: Any,
    prompt_template: str = DEFAULT_ENUM_EXTRACT_PROMPT,
    cache_ttl: int = 3600,
) -> Tuple[List[Dict[str, str]], bool, float]:
    """
    Use LLM to extract enum values from a comment string.

    Args:
        comment: The comment text to parse
        llm_model: LLM model instance with generate_with_json_output method
        prompt_template: Custom prompt template (optional)
        cache_ttl: Cache TTL in seconds (default 1 hour)

    Returns:
        Tuple of (enum_list, is_enum_flag, confidence_score)
    """
    if not comment or not llm_model:
        return [], False, 0.0

    # Build the prompt
    prompt = prompt_template.format(comment=comment)

    try:
        response = llm_model.generate_with_json_output(prompt)

        if not isinstance(response, dict):
            logger.debug(f"LLM response is not a dict: {type(response)}")
            return [], False, 0.0

        enums = response.get("enums", [])
        is_enum = response.get("is_enum", False)
        confidence = response.get("confidence", 0.0)

        # Validate enum structure
        validated_enums = []
        for item in enums:
            if isinstance(item, dict) and "code" in item and "label" in item:
                validated_enums.append({
                    "code": str(item["code"]),
                    "label": str(item["label"])
                })

        # Filter by confidence threshold
        if confidence < 0.5:
            return [], False, confidence

        logger.debug(f"LLM extracted {len(validated_enums)} enums from comment (confidence: {confidence:.2f})")
        return validated_enums, is_enum, confidence

    except Exception as e:
        logger.warning(f"LLM enum extraction failed: {e}")
        return [], False, 0.0


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

        # Validate and clean response
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

    # Convert row to string representation for prompt
    row_str = json.dumps(row_data, ensure_ascii=False, indent=2)
    prompt = prompt_template.format(row_data=row_str)

    try:
        response = llm_model.generate_with_json_output(prompt)

        if not isinstance(response, dict):
            logger.debug(f"LLM design req response is not a dict: {type(response)}")
            return {}

        # Map response to standardized fields
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

        logger.debug(f"LLM parsed design requirement: {parsed.get('object_code', 'unknown')}")
        return parsed

    except Exception as e:
        logger.warning(f"LLM design requirement parsing failed: {e}")
        return {}


def create_enhanced_enum_extractor(
    llm_model: Optional[Any] = None,
    use_regex_first: bool = True,
    regex_confidence_threshold: float = 0.7,
) -> "EnhancedEnumExtractor":
    """
    Factory function to create an enhanced enum extractor.

    Args:
        llm_model: LLM model instance (required for LLM enhancement)
        use_regex_first: If True, try regex first, fall back to LLM
        regex_confidence_threshold: Min confidence for regex results

    Returns:
        EnhancedEnumExtractor instance
    """
    return EnhancedEnumExtractor(
        llm_model=llm_model,
        use_regex_first=use_regex_first,
        regex_confidence_threshold=regex_confidence_threshold,
    )


class EnhancedEnumExtractor:
    """
    Enhanced enum extractor that combines regex and LLM approaches.

    Usage:
        extractor = EnhancedEnumExtractor(llm_model=model)
        enums, is_enum, confidence = extractor.extract("订单状态（0:录入、1:生产）")
    """

    def __init__(
        self,
        llm_model: Optional[Any] = None,
        use_regex_first: bool = True,
        regex_confidence_threshold: float = 0.7,
    ):
        self.llm_model = llm_model
        self.use_regex_first = use_regex_first
        self.regex_confidence_threshold = regex_confidence_threshold

    def extract(
        self,
        comment: str,
        column_name: str = "",
        table_name: str = "",
    ) -> Tuple[List[Dict[str, str]], bool, float]:
        """
        Extract enum values from a comment.

        Strategy:
        1. Try regex extraction first (fast)
        2. If regex confidence low, use LLM (if available)
        3. Combine results if both succeed

        Args:
            comment: Comment text to parse
            column_name: Optional column name for context
            table_name: Optional table name for context

        Returns:
            Tuple of (enum_list, is_enum_flag, confidence_score)
        """
        from datus.utils.sql_utils import extract_enum_values_from_comment

        # Strategy 1: Regex extraction
        if self.use_regex_first:
            regex_enums = extract_enum_values_from_comment(comment)

            # Estimate regex confidence based on result quality
            if len(regex_enums) >= 2:
                regex_confidence = min(0.9, 0.5 + len(regex_enums) * 0.1)
                is_enum = True
            elif len(regex_enums) == 1:
                regex_confidence = 0.4  # Single result might be false positive
                is_enum = False
            else:
                regex_confidence = 0.0
                is_enum = False

            # If regex confidence is high enough, return results
            if regex_confidence >= self.regex_confidence_threshold:
                logger.debug(
                    f"Regex extraction succeeded: {len(regex_enums)} enums, "
                    f"confidence: {regex_confidence:.2f}"
                )
                return (
                    [{"code": code, "label": label} for code, label in regex_enums],
                    is_enum,
                    regex_confidence,
                )

            # If regex confidence is low and LLM is available, try LLM
            if self.llm_model:
                llm_enums, llm_is_enum, llm_confidence = extract_enums_with_llm(
                    comment, self.llm_model
                )

                # If LLM succeeded, return LLM results
                if llm_confidence > regex_confidence:
                    logger.debug(
                        f"LLM extraction succeeded: {len(llm_enums)} enums, "
                        f"confidence: {llm_confidence:.2f}"
                    )
                    return llm_enums, llm_is_enum, llm_confidence

                # If both have results, combine (LLM takes priority for labels)
                if llm_enums and regex_enums:
                    combined = self._combine_enum_results(regex_enums, llm_enums)
                    return combined, True, max(regex_confidence, llm_confidence)

            # Fall back to regex results
            return (
                [{"code": code, "label": label} for code, label in regex_enums],
                is_enum,
                regex_confidence,
            )

        # Strategy 2: LLM-only (if regex is disabled)
        if self.llm_model:
            return extract_enums_with_llm(comment, self.llm_model)

        # No extraction method available
        logger.warning("No extraction method available (no LLM and regex disabled)")
        return [], False, 0.0

    def _combine_enum_results(
        self,
        regex_enums: List[Tuple[str, str]],
        llm_enums: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        """
        Combine regex and LLM enum results.

        Priority: LLM labels > regex labels
        Codes from both are merged and deduplicated
        """
        code_to_label = {}

        # Add regex results
        for code, label in regex_enums:
            if code not in code_to_label:
                code_to_label[code] = label

        # Override/add LLM results (LLM labels are considered more accurate)
        for item in llm_enums:
            code = item.get("code", "")
            label = item.get("label", "")
            if code:
                code_to_label[code] = label

        return [
            {"code": code, "label": label}
            for code, label in code_to_label.items()
        ]
