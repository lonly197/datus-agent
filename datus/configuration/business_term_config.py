# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Business term mapping configuration.

This file contains configurable mappings between business terminology
(including Chinese terms) and database schema elements.

These mappings can be customized per deployment without modifying core code.
"""

# Schema Discovery: Business term to table name mappings
BUSINESS_TERM_TO_TABLE_MAPPING = {
    # Test drive related
    "试驾": ["dwd_assign_dlr_clue_fact_di", "test_drive", "trial_drive", "testdrive"],
    "首次": ["first", "initial", "first_time"],
    "线索": ["dwd_assign_dlr_clue_fact_di", "clue", "clue_id", "lead", "lead_id"],
    # Order related
    "下定": ["order", "order_date", "booking", "book"],
    "订单": ["order", "order_id", "orders"],
    "转化": ["conversion", "convert", "transform", "conversions"],
    # Time related
    "周期": ["cycle", "period", "duration"],
    "天数": ["days", "date_diff", "day_count"],
    "月份": ["month", "monthly", "mth"],
    # Statistics
    "统计": ["count", "sum", "avg", "calculate", "compute"],
    "平均": ["avg", "average", "mean"],
    # Common business tables (default mappings)
    "用户": ["users", "user"],
    "客户": ["customers", "customer"],
    "产品": ["products", "product"],
    "销售": ["sales", "sale"],
    "交易": ["transactions", "transaction"],
    "员工": ["employees", "employee"],
    "部门": ["departments", "department"],
}


# Schema Validation: Chinese to English schema term mappings
BUSINESS_TERM_TO_SCHEMA_MAPPING = {
    # Test drive related
    "试驾": ["test_drive", "test_drive_date", "trial_drive", "testdrive"],
    "首次": ["first", "initial", "first_time"],
    "线索": ["clue", "clue_id", "lead", "lead_id"],
    # Order related
    "下定": ["order", "order_date", "booking", "book"],
    "订单": ["order", "order_id", "orders"],
    "转化": ["conversion", "convert", "transform"],
    # Time related
    "周期": ["cycle", "period", "duration"],
    "天数": ["days", "date_diff", "day_count"],
    "月份": ["month", "monthly", "mth"],
    # Statistics
    "统计": ["count", "sum", "avg", "calculate", "compute"],
    "平均": ["avg", "average", "mean"],
}


# Keyword extraction patterns for table discovery
TABLE_KEYWORD_PATTERNS = {
    # Direct mappings
    "用户": "users",
    "user": "users",
    "客户": "customers",
    "customer": "customers",
    "订单": "orders",
    "order": "orders",
    "产品": "products",
    "product": "products",
    "销售": "sales",
    "sale": "sales",
    "交易": "transactions",
    "transaction": "transactions",
    "员工": "employees",
    "employee": "employees",
    "部门": "departments",
    "department": "departments",
    "试驾": "dwd_assign_dlr_clue_fact_di",
    "线索": "dwd_assign_dlr_clue_fact_di",
    "转化": "conversions",
}


# LLM fallback table discovery configuration
LLM_TABLE_DISCOVERY_CONFIG = {
    "max_retries": 3,
    "cache_enabled": True,
    "cache_ttl_seconds": 3600,  # 1 hour
    "prompt_template": """
Analyze the user query and list potential database table names or business entities.
The query might be in Chinese, but the database schema uses English table names.
Translate business concepts into English database terms (e.g., "试驾" -> "test_drive", "lead", "clue").
Return a JSON object with a single key "tables" containing a list of strings (potential English table names).

Query: {query}
""",
}


# LLM keyword extraction configuration for schema validation
LLM_TERM_EXTRACTION_CONFIG = {
    "max_retries": 3,
    "cache_enabled": True,
    "cache_ttl_seconds": 1800,  # 30 minutes
    "prompt_template": """
Analyze the user query and extract key database search terms for schema matching.

Rules:
1. Extract potential table names, column names, and business concepts.
2. Break down compound business terms into atomic concepts (e.g., "首次试驾" -> "首次", "试驾").
3. Ignore common stop words and grammatical particles.
4. Return a JSON object with a single key "terms" containing a list of strings.

Examples:
Input: "统计每个月首次试驾的平均转化周期"
Output: {{"terms": ["统计", "每个月", "首次", "试驾", "平均", "转化", "周期"]}}

Input: "查询最近一周的下定订单数"
Output: {{"terms": ["查询", "最近", "一周", "下定", "订单", "数"]}}

Query: {query}
""",
}


# Schema validation configuration
SCHEMA_VALIDATION_CONFIG = {
    # Dynamic coverage thresholds based on query complexity
    "coverage_thresholds": {
        "simple": {"max_terms": 3, "threshold": 0.5},  # Simple queries need 50% coverage
        "medium": {"max_terms": 6, "threshold": 0.3},  # Medium queries need 30% coverage
        "complex": {"min_terms": 7, "threshold": 0.2},  # Complex queries need 20% coverage
    },
    "enable_semantic_matching": True,
    "enable_partial_matching": True,
}


# Table discovery stage configuration
TABLE_DISCOVERY_STAGES = {
    "stage1_semantic": {
        "enabled": True,
        "priority": 1,
        "top_n": 5,
    },
    "stage1_keyword": {
        "enabled": True,
        "priority": 2,
    },
    "stage1_5_llm": {
        "enabled": True,
        "priority": 3,
        "config": LLM_TABLE_DISCOVERY_CONFIG,
    },
    "stage2_context": {
        "enabled": True,
        "priority": 4,
        "min_table_threshold": 3,  # Trigger if < 3 tables found
        "metrics_top_n": 3,
        "ref_sql_top_n": 3,
    },
    "stage3_fallback": {
        "enabled": True,
        "priority": 5,
        "max_tables": 50,  # Limit to prevent context overflow
    },
}


def get_business_term_mapping(term: str) -> list:
    """
    Get table/schema mappings for a business term.

    Args:
        term: Business term (can be Chinese or English)

    Returns:
        List of possible table/schema names
    """
    return BUSINESS_TERM_TO_TABLE_MAPPING.get(term, [])


def get_schema_term_mapping(term: str) -> list:
    """
    Get schema term mappings for validation.

    Args:
        term: Business term

    Returns:
        List of English schema terms
    """
    return BUSINESS_TERM_TO_SCHEMA_MAPPING.get(term, [])


def get_table_keyword_pattern(keyword: str) -> str:
    """
    Get direct table name mapping for a keyword.

    Args:
        keyword: Keyword from user query

    Returns:
        Table name or empty string if no mapping
    """
    return TABLE_KEYWORD_PATTERNS.get(keyword, "")
