# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
LLM-Enhanced Enum Extraction Module

This module provides LLM-based extraction for enum values from comments,
combining regex and LLM approaches for optimal results.
"""

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

    prompt = prompt_template.format(comment=comment)

    try:
        response = llm_model.generate_with_json_output(prompt)

        if not isinstance(response, dict):
            logger.debug(f"LLM response is not a dict: {type(response)}")
            return [], False, 0.0

        enums = response.get("enums", [])
        is_enum = response.get("is_enum", False)
        confidence = response.get("confidence", 0.0)

        validated_enums = []
        for item in enums:
            if isinstance(item, dict) and "code" in item and "label" in item:
                validated_enums.append({
                    "code": str(item["code"]),
                    "label": str(item["label"])
                })

        if confidence < 0.5:
            return [], False, confidence

        logger.debug(
            f"LLM extracted {len(validated_enums)} enums from comment "
            f"(confidence: {confidence:.2f})"
        )
        return validated_enums, is_enum, confidence

    except Exception as e:
        logger.warning(f"LLM enum extraction failed: {e}")
        return [], False, 0.0


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

        if self.use_regex_first:
            return self._extract_with_regex_fallback(comment)
        if self.llm_model:
            return extract_enums_with_llm(comment, self.llm_model)

        logger.warning("No extraction method available (no LLM and regex disabled)")
        return [], False, 0.0

    def _extract_with_regex_fallback(
        self, comment: str
    ) -> Tuple[List[Dict[str, str]], bool, float]:
        """Extract with regex first, then LLM fallback."""
        from datus.utils.sql_utils import extract_enum_values_from_comment

        regex_enums = extract_enum_values_from_comment(comment)

        if len(regex_enums) >= 2:
            regex_confidence = min(0.9, 0.5 + len(regex_enums) * 0.1)
            is_enum = True
        elif len(regex_enums) == 1:
            regex_confidence = 0.4
            is_enum = False
        else:
            regex_confidence = 0.0
            is_enum = False

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

        if not self.llm_model:
            return (
                [{"code": code, "label": label} for code, label in regex_enums],
                is_enum,
                regex_confidence,
            )

        llm_enums, llm_is_enum, llm_confidence = extract_enums_with_llm(
            comment, self.llm_model
        )

        if llm_confidence > regex_confidence:
            logger.debug(
                f"LLM extraction succeeded: {len(llm_enums)} enums, "
                f"confidence: {llm_confidence:.2f}"
            )
            return llm_enums, llm_is_enum, llm_confidence

        if llm_enums and regex_enums:
            combined = self._combine_enum_results(regex_enums, llm_enums)
            return combined, True, max(regex_confidence, llm_confidence)

        return (
            [{"code": code, "label": label} for code, label in regex_enums],
            is_enum,
            regex_confidence,
        )

    def _combine_enum_results(
        self,
        regex_enums: List[Tuple[str, str]],
        llm_enums: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        """Combine regex and LLM enum results. LLM labels take priority."""
        code_to_label = {}

        for code, label in regex_enums:
            if code not in code_to_label:
                code_to_label[code] = label

        for item in llm_enums:
            code = item.get("code", "")
            label = item.get("label", "")
            if code:
                code_to_label[code] = label

        return [
            {"code": code, "label": label}
            for code, label in code_to_label.items()
        ]


def create_enhanced_enum_extractor(
    llm_model: Optional[Any] = None,
    use_regex_first: bool = True,
    regex_confidence_threshold: float = 0.7,
) -> EnhancedEnumExtractor:
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
