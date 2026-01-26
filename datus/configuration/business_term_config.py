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
    "试驾": ["dwd_testdrv_test_drive_fact", "dwd_assign_dlr_clue_fact_di", "test_drive", "trial_drive", "testdrive", "testdrive"],
    "首次": ["first", "initial", "first_time"],
    "首次试驾": ["dwd_testdrv_test_drive_fact", "first_test_drive", "test_drive_fact"],
    "试乘试驾": ["dwd_testdrv_test_drive_fact", "test_drive", "trial_drive", "testdrive"],
    # Order related
    "下定": ["dwd_testdrv_tsd_order_relation_fact", "order", "order_date", "booking", "book"],
    "订单": ["dwd_testdrv_tsd_order_relation_fact", "order", "order_id", "orders"],
    "转化": ["dwd_testdrv_tsd_order_relation_fact", "conversion", "convert", "transform", "conversions"],
    "转化订单": ["dwd_testdrv_tsd_order_relation_fact", "conversion_order", "tsd_order_relation"],
    "转化周期": ["conversion_cycle", "conversion_period", "conversion_duration"],
    # Clue related (production tables)
    "线索": ["dim_cust_clue_customer", "dwd_clue_clue_fact", "dwd_assign_dlr_clue_fact_di", "clue", "clue_id", "lead", "lead_id"],
    "客户线索": ["dim_cust_clue_customer", "customer_clue", "clue_customer"],
    # Time related
    "周期": ["cycle", "period", "duration"],
    "天数": ["days", "date_diff", "day_count"],
    "月份": ["month", "monthly", "mth"],
    # Statistics
    "统计": ["count", "sum", "avg", "calculate", "compute"],
    "平均": ["avg", "average", "mean"],
    # Common business tables (default mappings)
    "用户": ["users", "user"],
    "客户": ["customers", "customer", "dim_customer"],
    "产品": ["products", "product"],
    "销售": ["sales", "sale"],
    "交易": ["transactions", "transaction"],
    "员工": ["employees", "employee"],
    "部门": ["departments", "department"],
}


# Schema Validation: Chinese to English schema term mappings
BUSINESS_TERM_TO_SCHEMA_MAPPING = {
    # Test drive related
    "试驾": ["test_drive", "test_drive_date", "trial_drive", "testdrive", "first_test_drive_finish_time"],
    "首次": ["first", "initial", "first_time"],
    "首次试驾": ["first_test_drive", "test_drive_finish_time", "first_test_drive_finish_time"],
    "试乘试驾": ["test_drive", "trial_drive", "testdrive"],
    # Order related
    "下定": ["order", "order_date", "booking", "book", "sales_date", "order_dealer_code"],
    "订单": ["order", "order_id", "orders", "tsd_order_relation_flag"],
    "转化": ["conversion", "convert", "transform", "tsd_order_relation"],
    "转化订单": ["conversion_order", "tsd_order_relation", "tsd_order_relation_flag"],
    "转化周期": ["conversion_cycle", "conversion_period", "conversion_duration"],
    # Clue related (production columns)
    "线索": ["clue", "clue_id", "lead", "lead_id", "clue_create_time", "customer_phone"],
    "客户线索": ["customer_clue", "clue_customer", "customer_phone"],
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
    # Direct mappings (general)
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
    # Production-specific mappings (automotive domain)
    "试驾": "dwd_testdrv_test_drive_fact",
    "首次试驾": "dwd_testdrv_test_drive_fact",
    "试乘试驾": "dwd_testdrv_test_drive_fact",
    "线索": "dim_cust_clue_customer",
    "客户线索": "dim_cust_clue_customer",
    "下定": "dwd_testdrv_tsd_order_relation_fact",
    "转化": "dwd_testdrv_tsd_order_relation_fact",
    "转化订单": "dwd_testdrv_tsd_order_relation_fact",
}


# LLM fallback table discovery configuration
LLM_TABLE_DISCOVERY_CONFIG = {
    "max_retries": 3,
    "cache_enabled": True,
    "cache_ttl_seconds": 3600,  # 1 hour
    "prompt_template": """
Given these actual database tables:
{tables_list}

Select the most relevant tables for this query: {query}

Rules:
1. Return ONLY table names that exist in the list above
2. Do not hallucinate or invent table names
3. The query might be in Chinese, but the database schema uses English table names
4. Translate business concepts into English database terms (e.g., "试驾" -> "test_drive", "lead", "clue")

Return a JSON object with a single key "tables" containing a list of strings (table names from the list above).
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
2. Keep compound business terms AND include their atomic components (e.g., "首次试驾" -> ["首次试驾", "首次", "试驾"]).
3. Ignore common stop words and grammatical particles.
4. Return a JSON object with a single key "terms" containing a list of strings.

Examples:
Input: "统计每个月首次试驾的平均转化周期"
Output: {{"terms": ["统计", "每个月", "首次试驾", "首次", "试驾", "平均", "转化", "周期"]}}

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


# Business tag patterns for automatic domain detection
BUSINESS_TAG_PATTERNS = {
    "finance": ["fact_", "transaction", "payment", "revenue", "expense", "budget", "cost", "profit", "margin", "invoice", "billing"],
    "sales": ["sales_", "order", "customer", "product_", "invoice", "quote", "deal", "opportunity", "lead"],
    "inventory": ["inventory_", "stock", "warehouse", "supply_chain", "product_", "item", "sku", "quantity"],
    "analytics": ["agg_", "summary", "metrics", "kpi_", "report_", "dashboard", "analytics", "statistics"],
    "temporal": ["date_", "time_", "calendar", "fiscal_", "quarter", "month_", "day_", "year_", "period", "date_dim", "time_dim"],
    "user": ["user_", "customer", "account", "profile", "person", "employee", "staff"],
    "location": ["geo_", "location", "address", "region", "country", "city", "state", "province", "zip"],
    "dimension": ["dim_", "dimension", "lookup", "ref_", "reference"],
    "fact": ["fact_", "fct_", "metrics", "measures", "transactions"]
}


def infer_business_tags(table_name: str, column_names: list) -> list:
    """
    Infer business domain tags from table and column naming patterns.

    Args:
        table_name: Name of the table
        column_names: List of column names in the table

    Returns:
        List of inferred business tags (e.g., ["finance", "fact_table", "temporal"])
    """
    import re
    from typing import List

    tags = []
    table_lower = table_name.lower()
    columns_lower = [col.lower() for col in column_names]

    # Check table-level patterns
    for tag, patterns in BUSINESS_TAG_PATTERNS.items():
        for pattern in patterns:
            if pattern in table_lower:
                if tag not in tags:
                    tags.append(tag)
                break

    # Check column-level patterns (aggregate evidence)
    column_tag_scores = {}
    for col in columns_lower:
        for tag, patterns in BUSINESS_TAG_PATTERNS.items():
            for pattern in patterns:
                if pattern in col:
                    column_tag_scores[tag] = column_tag_scores.get(tag, 0) + 1

    # Add tags that appear in multiple columns (stronger signal)
    for tag, score in column_tag_scores.items():
        if score >= 2 and tag not in tags:  # Appears in 2+ columns
            tags.append(tag)

    # Detect fact vs dimension tables
    if table_lower.startswith("fact_") or table_lower.startswith("fct_"):
        tags.append("fact_table")
    elif table_lower.startswith("dim_") or any(col.endswith("_id") or col.endswith("_key") for col in columns_lower):
        tags.append("dimension_table")

    # Detect aggregate/summary tables
    if any(kw in table_lower for kw in ["agg_", "summary", "metrics", "kpi_"]):
        tags.append("aggregate_table")

    # Detect temporal tables
    if any(col in columns_lower for col in ["date", "time", "year", "month", "day", "quarter", "fiscal"]):
        tags.append("temporal_table")

    return tags


def detect_temporal_granularity(column_name: str, comment: str = "") -> str:
    """
    Infer date/time granularity from naming/comment patterns.

    Args:
        column_name: Name of the temporal column
        comment: Optional column comment

    Returns:
        Granularity level: "daily", "weekly", "monthly", "quarterly", "yearly", "unknown"
    """
    patterns = {
        "daily": ["daily", "_day", "_d_", "date"],
        "weekly": ["weekly", "_week", "_w_", "week"],
        "monthly": ["monthly", "_month", "_m_", "month"],
        "quarterly": ["quarter", "_q_", "fiscal_quarter"],
        "yearly": ["yearly", "_year", "_y_", "annual", "fiscal_year"]
    }

    col_lower = column_name.lower() + " " + comment.lower()
    for granularity, keywords in patterns.items():
        if any(kw in col_lower for kw in keywords):
            return granularity
    return "unknown"
