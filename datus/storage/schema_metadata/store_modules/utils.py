# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details

"""
Utility functions for schema metadata storage modules.

This module contains helper functions used across multiple storage modules.
"""

import json
from typing import Any, Dict, List, Optional, Set

from datus.schemas.base import TABLE_TYPE
from datus.storage.lancedb_conditions import Node, and_, eq, or_


def _build_where_clause(
    catalog_name: str = "",
    database_name: str = "",
    schema_name: str = "",
    table_name: str = "",
    table_type: TABLE_TYPE = "table",
    available_fields: Optional[Set[str]] = None,
) -> Optional[Node]:
    """
    Build LanceDB WHERE clause from filter parameters.

    This function constructs a WHERE clause for filtering metadata records
    based on catalog, database, schema, table name, and table type.

    Args:
        catalog_name: Catalog name filter
        database_name: Database name filter
        schema_name: Schema name filter
        table_name: Table name filter
        table_type: Table type filter (table, view, mv, or 'full' for all)
        available_fields: Set of available field names (optional)

    Returns:
        LanceDB WHERE clause (Node), or None if no filters specified
    """
    def add_condition(field_name: str, value: str) -> None:
        """Add a condition if value is provided and field exists."""
        if not value:
            return
        if available_fields is not None and field_name not in available_fields:
            return
        conditions.append(eq(field_name, value))

    conditions = []
    add_condition("catalog_name", catalog_name)
    add_condition("database_name", database_name)
    add_condition("schema_name", schema_name)
    add_condition("table_name", table_name)
    if table_type and table_type != "full":
        add_condition("table_type", table_type)

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return and_(*conditions)


def _safe_json_dict(value: Any) -> Dict[str, Any]:
    """
    Safely parse a JSON string into a dictionary.

    Args:
        value: JSON string or dict

    Returns:
        Parsed dictionary, or empty dict if parsing fails
    """
    if isinstance(value, dict):
        return value
    if not value or not isinstance(value, str):
        return {}
    try:
        parsed = json.loads(value)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _safe_json_list(value: Any) -> List[Any]:
    """
    Safely parse a JSON string into a list.

    Args:
        value: JSON string or list

    Returns:
        Parsed list, or empty list if parsing fails
    """
    if isinstance(value, list):
        return value
    if not value or not isinstance(value, str):
        return []
    try:
        parsed = json.loads(value)
    except Exception:
        return []
    return parsed if isinstance(parsed, list) else []
