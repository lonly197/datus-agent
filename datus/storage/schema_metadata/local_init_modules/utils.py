# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Utility functions for local schema initialization.

This module provides helper functions for relationship inference,
DDL parsing, and sample data collection.
"""

import json
import re
from typing import Any, Dict, List, Optional

from datus.models.base import LLMBaseModel
from datus.tools.db_tools.base import BaseSqlConnector
from datus.utils.constants import DBType
from datus.utils.loggings import get_logger
from datus.utils.sql_utils import (
    extract_metadata_from_ddl_regex_only,
    sanitize_ddl_for_storage,
    validate_comment,
    validate_table_name,
)

logger = get_logger(__name__)

# Relationship inference constants
_RELATION_PREFIXES = (
    "ods_",
    "dwd_",
    "dws_",
    "dim_",
    "ads_",
    "tmp_",
    "stg_",
    "stage_",
    "fact_",
)
_RELATION_SUFFIXES = (
    "_di",
    "_df",
    "_tmp",
    "_temp",
    "_bak",
    "_backup",
)
_RELATION_DATE_SUFFIX_RE = re.compile(r"_(?:19|20)\d{6,8}$")


def _normalize_relationship_name(name: str) -> str:
    """
    Normalize a table name for relationship inference.

    Removes common prefixes, suffixes, and date suffixes.

    Args:
        name: Table name to normalize

    Returns:
        Normalized table name
    """
    value = (name or "").strip().strip("`").lower()
    value = _RELATION_DATE_SUFFIX_RE.sub("", value)
    for suffix in _RELATION_SUFFIXES:
        if value.endswith(suffix):
            value = value[: -len(suffix)]
            break
    for prefix in _RELATION_PREFIXES:
        if value.startswith(prefix):
            value = value[len(prefix):]
            break
    return value


def _build_table_name_map(table_names: List[str]) -> Dict[str, List[str]]:
    """
    Build a mapping from normalized names to actual table names.

    Args:
        table_names: List of table names to map

    Returns:
        Dictionary mapping normalized names to lists of actual names
    """
    mapping: Dict[str, List[str]] = {}
    for name in table_names:
        normalized = _normalize_relationship_name(name)
        if not normalized:
            continue
        mapping.setdefault(normalized, []).append(name)
    return mapping


def _infer_relationships_from_names(
    table_name: str,
    column_names: List[str],
    table_name_map: Dict[str, List[str]],
) -> Dict[str, Any]:
    """
    Infer foreign key relationships from table and column naming patterns.

    Args:
        table_name: Source table name
        column_names: List of column names in the source table
        table_name_map: Mapping from normalized names to actual table names

    Returns:
        Dictionary with foreign_keys and join_paths, or empty dict if none found
    """
    foreign_keys = []
    for col in column_names:
        col_lower = (col or "").lower()
        if not col_lower or not col_lower.endswith("_id"):
            continue
        base = col_lower[:-3]
        normalized_base = _normalize_relationship_name(base)
        if len(normalized_base) < 3:
            continue
        candidates = list(table_name_map.get(normalized_base, []))
        if not candidates:
            for normalized, names in table_name_map.items():
                if normalized == normalized_base:
                    continue
                if normalized.endswith(f"_{normalized_base}") or normalized.endswith(normalized_base):
                    candidates.extend(names)
        if not candidates:
            continue
        to_table = sorted(candidates)[0]
        foreign_keys.append({
            "from_column": col,
            "to_table": to_table,
            "to_column": "id",
        })
    if not foreign_keys:
        return {}
    return {
        "foreign_keys": foreign_keys,
        "join_paths": [f"{fk['from_column']} -> {fk['to_table']}.{fk['to_column']}" for fk in foreign_keys],
    }


def _normalize_dialect(value: Any) -> str:
    """
    Normalize a database type to a string value.

    Args:
        value: Database type (DBType enum or string)

    Returns:
        Normalized database type string
    """
    if isinstance(value, DBType):
        return value.value
    return str(value or "").strip().lower()


def _llm_fallback_parse_ddl(ddl: str, llm_model: Optional[LLMBaseModel]) -> Optional[Dict[str, Any]]:
    """
    Parse DDL using LLM as a fallback when regex parsing fails.

    Args:
        ddl: DDL statement to parse
        llm_model: LLM model for parsing

    Returns:
        Parsed metadata dictionary or None if parsing fails
    """
    if not llm_model or not ddl or not isinstance(ddl, str):
        return None

    prompt = (
        "You are a SQL DDL parser. Extract metadata from the CREATE TABLE statement.\n"
        "Return ONLY a JSON object with this schema:\n"
        "{\n"
        '  "table": {"name": "", "comment": ""},\n'
        '  "columns": [{"name": "", "type": "", "comment": "", "nullable": true}],\n'
        '  "primary_keys": [],\n'
        '  "foreign_keys": [],\n'
        '  "indexes": []\n'
        "}\n"
        "Rules:\n"
        "- Only use information present in the DDL.\n"
        "- If a field is missing, use empty string or empty list.\n"
        "- Keep comments exactly as written (may contain Chinese and punctuation).\n"
        "- If NULL/NOT NULL is not specified, set nullable=true.\n"
        "DDL:\n"
        "```sql\n"
        f"{ddl}\n"
        "```\n"
    )

    try:
        response = llm_model.generate_with_json_output(prompt)
    except Exception as exc:
        logger.warning(f"LLM fallback parse failed: {exc}")
        return None

    if not isinstance(response, dict):
        return None

    table = response.get("table") if isinstance(response.get("table"), dict) else {}
    table_name = table.get("name", "") if isinstance(table.get("name"), str) else ""
    is_valid, _, table_name = validate_table_name(table_name)
    if not is_valid or not table_name:
        table_name = ""

    table_comment = table.get("comment", "") if isinstance(table.get("comment"), str) else ""
    is_valid, _, table_comment = validate_comment(table_comment)
    if not is_valid or table_comment is None:
        table_comment = ""

    columns = []
    raw_columns = response.get("columns", [])
    if isinstance(raw_columns, list):
        for col in raw_columns:
            if not isinstance(col, dict):
                continue
            col_name = col.get("name", "")
            if not isinstance(col_name, str) or not col_name:
                continue
            col_type = col.get("type", "")
            if not isinstance(col_type, str):
                col_type = ""
            col_comment = col.get("comment", "")
            if not isinstance(col_comment, str):
                col_comment = ""
            is_valid, _, col_comment = validate_comment(col_comment)
            if not is_valid or col_comment is None:
                col_comment = ""
            nullable = col.get("nullable", True)
            if not isinstance(nullable, bool):
                nullable = True
            columns.append(
                {
                    "name": col_name,
                    "type": col_type,
                    "comment": col_comment,
                    "nullable": nullable,
                }
            )

    if not columns:
        return None

    return {
        "table": {"name": table_name, "comment": table_comment},
        "columns": columns,
        "primary_keys": response.get("primary_keys", []) if isinstance(response.get("primary_keys"), list) else [],
        "foreign_keys": response.get("foreign_keys", []) if isinstance(response.get("foreign_keys"), list) else [],
        "indexes": response.get("indexes", []) if isinstance(response.get("indexes"), list) else [],
    }


def _fill_sample_rows(
    new_values: List[Dict[str, Any]], identifier: str, table_data: Dict[str, Any], connector: BaseSqlConnector
):
    """
    Fill sample rows for a table and add them to the new_values list.

    Args:
        new_values: List to append sample rows to
        identifier: Table identifier
        table_data: Table metadata dictionary
        connector: Database connector for fetching sample data
    """
    sample_rows = connector.get_sample_rows(
        tables=[table_data["table_name"]],
        top_n=5,
        catalog_name=table_data["catalog_name"],
        database_name=table_data["database_name"],
        schema_name=table_data["schema_name"],
    )
    if sample_rows:
        for row in sample_rows:
            if not row.get("identifier"):
                row["identifier"] = identifier
        new_values.extend(sample_rows)


# Note: extract_enhanced_metadata_from_ddl, parse_dialect, and sanitize_ddl_for_storage
# are imported here but re-exported for use by other modules in this package
from datus.utils.sql_utils import (
    extract_enhanced_metadata_from_ddl,
    parse_dialect,
    sanitize_ddl_for_storage,
)

__all__ = [
    "_normalize_relationship_name",
    "_build_table_name_map",
    "_infer_relationships_from_names",
    "_normalize_dialect",
    "_llm_fallback_parse_ddl",
    "_fill_sample_rows",
    "extract_enhanced_metadata_from_ddl",
    "parse_dialect",
    "sanitize_ddl_for_storage",
]
