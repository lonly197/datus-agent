# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Correlation Suggester for correlation analysis support.

This module provides functionality to suggest potential correlations for analysis
based on sample statistics and business domain context.
"""

import json
from typing import Any, Dict, List

from datus.storage.schema_metadata import SchemaStorage
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


async def suggest_correlations(
    storage: SchemaStorage,
    table_name: str,
    max_correlations: int = 10,
    catalog_name: str = "",
    database_name: str = "",
    schema_name: str = "",
) -> List[Dict[str, Any]]:
    """
    Suggest potential correlations for analysis based on metadata.

    Analyzes numeric columns and their statistics to suggest likely correlation candidates.
    Uses business domain context from business_tags and column comments.

    Args:
        storage: SchemaStorage instance
        table_name: Table name to analyze
        max_correlations: Maximum number of correlation suggestions to return
        catalog_name: Optional catalog filter
        database_name: Optional database filter
        schema_name: Optional schema filter

    Returns:
        List of correlation suggestions:
        [
            {
                "column1": "price",
                "column2": "volume",
                "correlation_type": "statistical",
                "strength": "strong",
                "reason": "Both numeric columns in finance domain with similar value ranges",
                "column1_stats": {"min": 0, "max": 1000, "mean": 250},
                "column2_stats": {"min": 1, "max": 500, "mean": 125}
            }
        ]
    """
    try:
        # Get table schema
        schemas = storage.get_table_schemas(
            table_names=[table_name], catalog_name=catalog_name, database_name=database_name, schema_name=schema_name
        )

        if not schemas or len(schemas) == 0:
            logger.warning(f"No schema found for table: {table_name}")
            return []

        schema = schemas[0]

        # Parse sample statistics
        column_stats = {}
        if schema.sample_statistics:
            try:
                column_stats = json.loads(schema.sample_statistics)
            except (json.JSONDecodeError, TypeError):
                logger.debug(f"No sample statistics available for {table_name}")

        # Parse column comments
        column_comments = {}
        if schema.column_comments:
            try:
                column_comments = json.loads(schema.column_comments)
            except (json.JSONDecodeError, TypeError):
                logger.debug(f"No column comments available for {table_name}")

        # Get business tags for domain context
        business_tags = []
        if hasattr(schema, "business_tags") and schema.business_tags:
            if isinstance(schema.business_tags, list):
                business_tags = schema.business_tags

        # Find numeric columns with statistics
        numeric_cols = []
        for col, stats in column_stats.items():
            if "min" in stats and "max" in stats:
                numeric_cols.append((col, stats))

        if len(numeric_cols) < 2:
            logger.info(f"Not enough numeric columns for correlation analysis in {table_name}")
            return []

        # Generate correlation suggestions
        correlations = []
        for i, (col1, stats1) in enumerate(numeric_cols):
            for col2, stats2 in numeric_cols[i + 1 :]:
                # Calculate overlap in value ranges (simple heuristic)
                range1 = stats1["max"] - stats1["min"]
                range2 = stats2["max"] - stats2["min"]
                mean1, mean2 = stats1["mean"], stats2["mean"]

                # Check if columns are related
                if _columns_related(col1, col2, column_comments, business_tags):
                    correlation_type = "statistical"
                    strength = _estimate_strength(stats1, stats2)
                    reason = _infer_correlation_reason(col1, col2, column_comments, business_tags, stats1, stats2)

                    correlations.append(
                        {
                            "column1": col1,
                            "column2": col2,
                            "correlation_type": correlation_type,
                            "strength": strength,
                            "reason": reason,
                            "column1_stats": stats1,
                            "column2_stats": stats2,
                        }
                    )

                # Stop if we've reached max suggestions
                if len(correlations) >= max_correlations:
                    break

            if len(correlations) >= max_correlations:
                break

        # Sort by strength (strong, medium, weak)
        correlations.sort(key=lambda c: (0 if c["strength"] == "strong" else (1 if c["strength"] == "medium" else 2)))

        return correlations[:max_correlations]

    except Exception as e:
        logger.error(f"Failed to suggest correlations for {table_name}: {e}")
        return []


def _columns_related(col1: str, col2: str, column_comments: Dict[str, str], business_tags: List[str]) -> bool:
    """
    Determine if two columns are likely related for correlation analysis.

    Args:
        col1: First column name
        col2: Second column name
        column_comments: Dictionary of column comments
        business_tags: Business domain tags

    Returns:
        True if columns should be analyzed for correlation
    """
    # Check comment semantic similarity
    comment1 = column_comments.get(col1, "").lower()
    comment2 = column_comments.get(col2, "").lower()

    # Extract keywords from comments
    keywords1 = set(comment1.split())
    keywords2 = set(comment2.split())

    # If comments share keywords, they might be related
    if keywords1 & keywords2:
        return True

    # Check business domain patterns
    domain_patterns = {
        "finance": ["price", "cost", "revenue", "profit", "volume", "amount", "quantity"],
        "sales": ["order", "customer", "product", "sales", "invoice"],
        "temporal": ["date", "time", "year", "month", "day"],
    }

    # Check if both columns belong to same domain pattern
    col1_lower = col1.lower()
    col2_lower = col2.lower()

    for domain, patterns in domain_patterns.items():
        if any(p in col1_lower for p in patterns) and any(p in col2_lower for p in patterns):
            return True

    # Check if both columns have similar prefixes (e.g., "price_usd", "price_eur")
    if "_" in col1 and "_" in col2:
        prefix1 = col1.split("_")[0]
        prefix2 = col2.split("_")[0]
        if prefix1 == prefix2:
            return True

    return False


def _estimate_strength(stats1: Dict, stats2: Dict) -> str:
    """
    Estimate correlation strength based on value ranges and means.

    This is a heuristic - actual correlation would require data analysis.

    Args:
        stats1: Statistics for column 1
        stats2: Statistics for column 2

    Returns:
        Strength estimate: "strong", "medium", or "weak"
    """
    # Check if value ranges are similar (normalized correlation)
    range1 = stats1["max"] - stats1["min"]
    range2 = stats2["max"] - stats2["min"]

    if range1 == 0 or range2 == 0:
        return "weak"  # Constant values

    # Check if means are within same order of magnitude
    mean1, mean2 = abs(stats1["mean"]), abs(stats2["mean"])
    if mean1 == 0 and mean2 == 0:
        return "weak"  # Both means are zero
    elif mean1 == 0 or mean2 == 0:
        return "weak"  # One mean is zero, incomparable scales
    else:
        ratio = max(mean1, mean2) / min(mean1, mean2)

    if ratio < 2:
        return "strong"  # Similar magnitude
    elif ratio < 10:
        return "medium"  # Different but comparable
    else:
        return "weak"  # Very different scales


def _infer_correlation_reason(
    col1: str, col2: str, column_comments: Dict[str, str], business_tags: List[str], stats1: Dict, stats2: Dict
) -> str:
    """
    Generate human-readable reason for correlation suggestion.

    Args:
        col1: First column name
        col2: Second column name
        column_comments: Column comments
        business_tags: Business domain tags
        stats1: Statistics for column 1
        stats2: Statistics for column 2

    Returns:
        Human-readable reason string
    """
    reasons = []

    # Business domain context
    if business_tags:
        domain = business_tags[0]
        reasons.append(f"Both columns in {domain} domain")

    # Comment similarity
    comment1 = column_comments.get(col1, "")
    comment2 = column_comments.get(col2, "")

    if comment1 and comment2:
        # Extract keywords
        words1 = set(comment1.lower().split())
        words2 = set(comment2.lower().split())
        common = words1 & words2

        if common:
            reasons.append(f"Share keywords: {', '.join(list(common)[:3])}")

    # Naming patterns
    if "_" in col1 and "_" in col2:
        prefix1 = col1.split("_")[0]
        prefix2 = col2.split("_")[0]
        if prefix1 == prefix2:
            reasons.append(f"Both start with '{prefix1}'")

    # Statistical properties
    range1 = stats1["max"] - stats1["min"]
    range2 = stats2["max"] - stats2["min"]
    mean1, mean2 = stats1["mean"], stats2["mean"]

    reasons.append(f"Value ranges: {range1:.0f} vs {range2:.0f}")
    reasons.append(f"Means: {mean1:.2f} vs {mean2:.2f}")

    # Combine reasons
    if reasons:
        return "; ".join(reasons)
    else:
        return "Numeric columns available for correlation analysis"
