# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Intent detection module for automatically identifying SQL-related tasks.

This module provides hybrid intent detection (keyword heuristics + LLM fallback)
to automatically inject relevant external knowledge when front-end omits ext_knowledge.
"""

import re
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass

from datus.utils.loggings import get_logger

logger = get_logger(__name__)


@dataclass
class IntentResult:
    """Result of intent detection."""
    intent: str
    confidence: float
    metadata: Dict[str, Any]


class IntentDetector:
    """
    Hybrid intent detector for SQL-related tasks.

    Uses keyword heuristics first (fast), then falls back to LLM classification
    for ambiguous cases.
    """

    # SQL-related keywords in both English and Chinese
    SQL_KEYWORDS = {
        # English keywords
        "sql", "select", "from", "where", "join", "group", "order", "limit",
        "partition", "starrocks", "hive", "spark", "presto", "clickhouse",
        "snowflake", "bigquery", "redshift", "athena", "drill", "impala",
        "kudu", "hbase", "cassandra", "mongodb", "elasticsearch",

        # Chinese keywords
        "查询", "审查", "审核", "检查", "分析", "统计", "报表", "数据",
        "表", "字段", "索引", "分区", "分桶", "物化视图", "视图",
        "主键", "外键", "约束", "优化", "性能", "执行计划",
        "试驾", "线索", "转化", "下定", "用户", "订单", "商品",
        "销售额", "收入", "利润", "成本", "库存", "物流"
    }

    # Patterns that strongly indicate SQL intent
    SQL_PATTERNS = [
        r'\bSELECT\b.*\bFROM\b',  # SELECT ... FROM
        r'\bINSERT\b.*\bINTO\b',  # INSERT ... INTO
        r'\bUPDATE\b.*\bSET\b',   # UPDATE ... SET
        r'\bDELETE\b.*\bFROM\b',  # DELETE ... FROM
        r'\bCREATE\b.*\bTABLE\b', # CREATE TABLE
        r'\bALTER\b.*\bTABLE\b',  # ALTER TABLE
        r'\bDROP\b.*\bTABLE\b',   # DROP TABLE
        r'\b\d+\s+(days?|hours?|minutes?|seconds?|weeks?|months?|years?)\s+ago\b',  # Time expressions
        r'\bdate_format\b|\bdatediff\b|\bdate_add\b|\bdate_sub\b',  # Date functions
        r'\bpartition\s+by\b|\bpartitioned\s+by\b',  # Partition keywords
        r'\bdistribute\s+by\b|\bcluster\s+by\b',     # Distribution keywords
    ]

    def __init__(self, keyword_threshold: int = 1, llm_confidence_threshold: float = 0.7):
        """
        Initialize the intent detector.

        Args:
            keyword_threshold: Minimum number of keyword matches to consider SQL intent
            llm_confidence_threshold: Minimum confidence for LLM classification to be accepted
        """
        self.keyword_threshold = keyword_threshold
        self.llm_confidence_threshold = llm_confidence_threshold

    def detect_sql_intent_by_keyword(self, text: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Fast keyword-based intent detection.

        Args:
            text: Input text to analyze

        Returns:
            Tuple of (is_sql_intent, metadata_dict)
        """
        if not text:
            return False, {}

        text_lower = text.lower()
        keyword_matches = []
        pattern_matches = []

        # Check for keyword matches
        for keyword in self.SQL_KEYWORDS:
            if keyword.lower() in text_lower:
                keyword_matches.append(keyword)

        # Check for pattern matches
        for pattern in self.SQL_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                pattern_matches.append(pattern)

        # Determine intent based on matches
        total_matches = len(keyword_matches) + len(pattern_matches)

        metadata = {
            "keyword_matches": keyword_matches,
            "pattern_matches": pattern_matches,
            "total_matches": total_matches,
            "has_sql_patterns": len(pattern_matches) > 0,
            "method": "keyword"
        }

        is_sql_intent = total_matches >= self.keyword_threshold
        return is_sql_intent, metadata

    async def classify_intent_with_llm(self, text: str, model) -> Tuple[str, float]:
        """
        LLM-based intent classification as fallback.

        Args:
            text: Input text to classify
            model: LLM model instance to use for classification

        Returns:
            Tuple of (intent_label, confidence_score)
        """
        try:
            prompt = f"""
Classify the following user request into one of these categories:
- sql_generation: Request to generate or write SQL queries
- sql_review: Request to review, audit, or analyze existing SQL
- metadata_query: Request for database schema, table structure, or metadata
- other: Any other type of request

Return only a JSON object with "intent" and "confidence" fields.

User request: {text}
"""

            # Use the model's generate method
            response = await model.generate(prompt, max_tokens=100, temperature=0.1)

            # Parse the response (assuming JSON format)
            import json
            try:
                result = json.loads(response.strip())
                intent = result.get("intent", "other")
                confidence = float(result.get("confidence", 0.0))
                return intent, confidence
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                logger.warning(f"Failed to parse LLM response: {e}, response: {response}")
                return "other", 0.0

        except Exception as e:
            logger.error(f"LLM classification failed: {e}")
            return "other", 0.0

    async def detect_sql_intent(
        self,
        text: str,
        model=None,
        use_llm_fallback: bool = True
    ) -> IntentResult:
        """
        Hybrid intent detection: keyword first, LLM fallback.

        Args:
            text: Input text to analyze
            model: LLM model instance (optional, for fallback)
            use_llm_fallback: Whether to use LLM fallback if keyword detection is uncertain

        Returns:
            IntentResult with intent, confidence, and metadata
        """
        # Step 1: Fast keyword-based detection
        is_sql_keyword, keyword_metadata = self.detect_sql_intent_by_keyword(text)

        # Step 2: If keyword detection is confident, return result
        if is_sql_keyword and keyword_metadata.get("total_matches", 0) >= 2:
            intent = "sql_related"  # Generic SQL intent
            confidence = min(0.9, 0.5 + (keyword_metadata["total_matches"] * 0.1))
            return IntentResult(intent=intent, confidence=confidence, metadata=keyword_metadata)

        # Step 3: If keyword detection is uncertain and LLM fallback enabled, use LLM
        if use_llm_fallback and model and keyword_metadata.get("total_matches", 0) < self.keyword_threshold:
            llm_intent, llm_confidence = await self.classify_intent_with_llm(text, model)

            # Only accept LLM result if confidence is high enough
            if llm_confidence >= self.llm_confidence_threshold:
                metadata = keyword_metadata.copy()
                metadata.update({
                    "llm_intent": llm_intent,
                    "llm_confidence": llm_confidence,
                    "method": "llm_fallback"
                })
                return IntentResult(intent=llm_intent, confidence=llm_confidence, metadata=metadata)

        # Step 4: Default to no SQL intent if both methods are uncertain
        metadata = keyword_metadata.copy()
        metadata["method"] = "no_match"
        return IntentResult(intent="other", confidence=0.0, metadata=metadata)


# Global instance for convenience
default_intent_detector = IntentDetector()


async def detect_sql_intent(
    text: str,
    model=None,
    keyword_threshold: int = 1,
    llm_confidence_threshold: float = 0.7,
    use_llm_fallback: bool = True
) -> IntentResult:
    """
    Convenience function for intent detection.

    Args:
        text: Input text to analyze
        model: LLM model instance (optional)
        keyword_threshold: Minimum keyword matches for SQL intent
        llm_confidence_threshold: Minimum LLM confidence to accept
        use_llm_fallback: Whether to use LLM fallback

    Returns:
        IntentResult with detected intent
    """
    detector = IntentDetector(
        keyword_threshold=keyword_threshold,
        llm_confidence_threshold=llm_confidence_threshold
    )
    return await detector.detect_sql_intent(text, model, use_llm_fallback)
