# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Database dialect-specific support utilities.

This module provides functions for handling database-specific syntax and features,
particularly for StarRocks and other dialects.
"""

import re
from typing import Any, Dict

from datus.utils.constants import DBType

# =============================================================================
# PRE-COMPILED REGEX PATTERNS FOR DIALECT SUPPORT
# =============================================================================

# StarRocks-specific patterns
_STARROCKS_PARTITION_RE = re.compile(
    r'PARTITION\s+BY\s+\w+\s*\(([^)]+)\)',
    re.IGNORECASE
)

_STARROCKS_DISTRIBUTED_RE = re.compile(
    r'DISTRIBUTED\s+BY\s+HASH\s*\(([^)]+)\)',
    re.IGNORECASE
)

_STARROCKS_PROPERTIES_RE = re.compile(
    r'PROPERTIES\s*\(\s*"([^"]+)"\s*=\s*"([^"]+)"',
    re.IGNORECASE
)


# =============================================================================
# STARROCKS-SPECIFIC PARSING FUNCTIONS
# =============================================================================

def extract_starrocks_properties(sql: str) -> Dict[str, Any]:
    """
    Extract StarRocks-specific table properties from DDL.

    StarRocks supports several proprietary features:
    - PARTITION BY: Define partition strategy
    - DISTRIBUTED BY HASH: Define distribution key
    - PROPERTIES: Table properties like replication, bloom filter, etc.

    Args:
        sql: StarRocks CREATE TABLE DDL

    Returns:
        Dictionary containing:
        - partitions: List of partition expressions
        - distributed_by: Distribution key columns
        - properties: Dict of table properties
    """
    result = {
        "partitions": [],
        "distributed_by": [],
        "properties": {}
    }

    # Extract partition information
    partition_match = _STARROCKS_PARTITION_RE.search(sql)
    if partition_match:
        partitions_str = partition_match.group(1)
        # Handle multiple partitions (e.g., RANGE MONTH (dt), RANGE MONTH (created_at))
        partitions = [p.strip() for p in partitions_str.split(',')]
        result["partitions"] = partitions

    # Extract distribution key
    distributed_match = _STARROCKS_DISTRIBUTED_RE.search(sql)
    if distributed_match:
        dist_str = distributed_match.group(1)
        result["distributed_by"] = [c.strip() for c in dist_str.split(',')]

    # Extract table properties
    for prop_match in _STARROCKS_PROPERTIES_RE.finditer(sql):
        key = prop_match.group(1)
        value = prop_match.group(2)
        result["properties"][key] = value

    return result


# =============================================================================
# DIALECT PARSING FUNCTIONS
# =============================================================================

def parse_read_dialect(dialect: str = DBType.SNOWFLAKE) -> str:
    """Map SQL dialect to the appropriate read dialect for sqlglot parsing."""
    db = (dialect or "").strip().lower()
    if db in (DBType.POSTGRES, DBType.POSTGRESQL, "redshift", "greenplum"):
        return DBType.POSTGRES
    if db in ("spark", "databricks", DBType.HIVE):
        return DBType.HIVE
    if db in (DBType.MSSQL, DBType.SQLSERVER):
        return "tsql"
    # StarRocks uses MySQL dialect in sqlglot
    if db == DBType.STARROCKS:
        return DBType.MYSQL
    return dialect


def parse_dialect(dialect: str = DBType.SNOWFLAKE) -> str:
    """Map SQL dialect to the dialect for sqlglot parsing."""
    normalized = (dialect or DBType.SNOWFLAKE).strip().lower()
    return parse_read_dialect(normalized)
