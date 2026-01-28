# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details

"""
Term processing utilities for schema validation.

This module provides utilities for extracting, filtering, and processing
query terms for schema matching and coverage analysis.
"""

import re
from typing import Any, Dict, List, Optional

from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class TermProcessor:
    """
    Utility class for processing query and schema terms.

    Provides methods for term extraction, normalization, filtering,
    and set operations (merge, intersection, difference) used in
    schema validation and coverage analysis.
    """

    def fallback_term_extraction(self, query: str, protected_terms: Optional[List[str]] = None) -> List[str]:
        """
        Fallback to simple regex-based term extraction if LLM fails.

        Args:
            query: Query text to extract terms from
            protected_terms: Terms that should never be filtered out

        Returns:
            List of extracted terms
        """
        words = re.findall(r"\b\w+\b", query)
        # Simple stop word filtering for fallback
        stop_words = {"the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "of", "with", "is", "are"}
        terms = [w for w in words if w.lower() not in stop_words and len(w) > 1]
        return self.filter_generic_terms(terms, protected_terms=protected_terms)

    @staticmethod
    def filter_generic_terms(terms: List[str], protected_terms: Optional[List[str]] = None) -> List[str]:
        """
        Filter out generic terms that don't add semantic value.

        Args:
            terms: List of terms to filter
            protected_terms: Terms that should never be filtered

        Returns:
            Filtered list of terms
        """
        if not terms:
            return []
        protected = {term for term in (protected_terms or []) if isinstance(term, str)}
        generic_terms = {
            "统计",
            "查询",
            "分析",
            "对比",
            "趋势",
            "汇总",
            "报表",
            "平均",
            "总计",
            "数量",
            "金额",
            "指标",
            "维度",
            "周期",
            "天数",
            "月份",
            "每月",
            "每个月",
            "每天",
            "每周",
            "每年",
        }
        return [term for term in terms if term not in generic_terms or term in protected]

    @staticmethod
    def tokenize_schema_term(term: str) -> List[str]:
        """
        Tokenize a schema term into individual components.

        Splits on non-alphanumeric characters to handle compound terms.

        Args:
            term: Schema term to tokenize

        Returns:
            List of tokens
        """
        if not term:
            return []
        tokens = re.split(r"[^a-zA-Z0-9]+", term.lower())
        return [t for t in tokens if t]

    @staticmethod
    def normalize_terms(terms: List[str]) -> List[str]:
        """
        Normalize terms to lowercase for case-insensitive comparison.

        Args:
            terms: List of terms to normalize

        Returns:
            Normalized list of terms
        """
        return [term.lower() for term in terms if isinstance(term, str)]

    @staticmethod
    def dedupe_terms(terms: List[str]) -> List[str]:
        """
        Remove duplicate terms while preserving order.

        Args:
            terms: List of terms that may contain duplicates

        Returns:
            Deduplicated list of terms
        """
        seen = set()
        deduped = []
        for term in terms:
            normalized = str(term).strip()
            if not normalized:
                continue
            if normalized not in seen:
                deduped.append(normalized)
                seen.add(normalized)
        return deduped

    def merge_terms(self, primary_terms: List[str], secondary_terms: List[str]) -> List[str]:
        """
        Merge two lists of terms, removing duplicates.

        Args:
            primary_terms: Primary term list (takes precedence)
            secondary_terms: Secondary term list to merge

        Returns:
            Merged, deduplicated list of terms
        """
        merged = []
        merged.extend(primary_terms or [])
        merged.extend(secondary_terms or [])
        return self.dedupe_terms(merged)

    def intersection_terms(self, terms_a: List[str], terms_b: List[str]) -> List[str]:
        """
        Find the intersection of two term lists.

        Args:
            terms_a: First term list
            terms_b: Second term list

        Returns:
            List of terms present in both lists (from terms_a)
        """
        normalized_b = set(self.normalize_terms(terms_b))
        return [term for term in terms_a if term.lower() in normalized_b]

    def difference_terms(self, terms_a: List[str], terms_b: List[str]) -> List[str]:
        """
        Find terms in terms_a that are not in terms_b.

        Args:
            terms_a: First term list
            terms_b: Second term list

        Returns:
            List of terms from terms_a not in terms_b
        """
        normalized_b = set(self.normalize_terms(terms_b))
        return [term for term in terms_a if term.lower() not in normalized_b]

    @staticmethod
    def calculate_critical_threshold(critical_terms: List[str]) -> float:
        """
        Calculate critical coverage threshold based on term count.

        Fewer critical terms require higher coverage threshold.

        Args:
            critical_terms: List of critical terms

        Returns:
            Coverage threshold (0.0 to 1.0)
        """
        if not critical_terms:
            return 0.0
        return 0.5 if len(critical_terms) <= 3 else 0.3

    def calculate_coverage_threshold(self, query_terms: List[str], thresholds_config: Dict[str, Any]) -> float:
        """
        Calculate dynamic coverage threshold based on query complexity.

        Simple queries (≤3 terms) require higher coverage.
        Complex queries (>6 terms) allow lower coverage.

        Args:
            query_terms: List of query terms
            thresholds_config: Configuration dict with thresholds

        Returns:
            Coverage threshold (0.0 to 1.0)
        """
        term_count = len(query_terms)
        thresholds = thresholds_config.get("coverage_thresholds", {})

        # Simple queries (≤3 terms) require higher coverage
        if term_count <= thresholds.get("simple", {}).get("max_terms", 3):
            return thresholds.get("simple", {}).get("threshold", 0.5)
        # Medium queries (4-6 terms) use standard coverage
        elif term_count <= thresholds.get("medium", {}).get("max_terms", 6):
            return thresholds.get("medium", {}).get("threshold", 0.3)
        # Complex queries (>6 terms) allow lower coverage
        else:
            return thresholds.get("complex", {}).get("threshold", 0.2)
