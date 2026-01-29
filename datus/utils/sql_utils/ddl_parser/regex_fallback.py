# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Fallback regex-based DDL parser.

This module provides regex-based DDL parsing as a fallback when
sqlglot fails to parse corrupted or dialect-specific DDL statements.
"""

import re
from typing import Any, Dict

from datus.utils.constants import DBType
from datus.utils.loggings import get_logger
from datus.utils.sql_utils.ddl_cleaner import _clean_ddl
from datus.utils.sql_utils.ddl_parser.core import (
    _COLUMN_COMMENT_RE,
    _COLUMN_DEF_RE,
    _FOREIGN_KEY_RE,
    _INDEX_RE,
    _PRIMARY_KEY_RE,
    _TABLE_COMMENT_LOOSE_RE,
    _TABLE_NAME_RE,
)
from datus.utils.sql_utils.ddl_cleaner import QuoteState
from datus.utils.sql_utils.dialect_support import extract_starrocks_properties, parse_dialect
from datus.utils.sql_utils.enum_utils import _extract_table_comment_after_columns
from datus.utils.sql_utils.validation import (
    MAX_COLUMN_NAME_LENGTH,
    MAX_TYPE_DEFINITION_LENGTH,
    validate_comment,
    validate_sql_input,
    validate_table_name,
)

logger = get_logger(__name__)


def _parse_ddl_with_regex(sql: str, dialect: str) -> Dict[str, Any]:
    """
    Fallback regex-based DDL parser for when sqlglot fails.

    This parser handles StarRocks and MySQL dialects with moderate complexity DDL.
    It uses pre-compiled regex patterns for performance and includes proper
    input validation.

    Args:
        sql: DDL statement
        dialect: SQL dialect (starrocks, mysql, etc.)

    Returns:
        Dict with parsed metadata
    """
    # Validate input
    is_valid, error_msg = validate_sql_input(sql)
    if not is_valid:
        if "Unbalanced parentheses" in error_msg:
            logger.debug(f"Continuing regex parsing despite: {error_msg}")
        else:
            logger.warning(f"Invalid SQL input: {error_msg}")
            return _empty_result()

    result = _init_result()

    # Keep original SQL for table comment extraction
    original_sql = sql

    # Extract table comment from ORIGINAL SQL before cleaning
    result["table"]["comment"] = _extract_table_comment(original_sql)

    # Clean the DDL
    sql = _clean_ddl(original_sql)
    if sql != original_sql:
        logger.debug(f"Cleaned corrupted DDL: {len(original_sql)} -> {len(sql)} chars")

    try:
        # Extract table name
        result["table"]["name"] = _extract_table_name(sql)

        # Extract table comment if not found
        if not result["table"]["comment"]:
            result["table"]["comment"] = _extract_table_comment(sql)

        # Extract column definitions
        columns_text = _extract_columns_text(sql)
        if columns_text:
            result["columns"] = _parse_columns(columns_text)

        # Extract constraints
        result["primary_keys"] = _extract_primary_keys(sql)
        result["foreign_keys"] = _extract_foreign_keys(sql)
        result["indexes"] = _extract_indexes(sql)

        # Extract StarRocks-specific properties
        if dialect == DBType.STARROCKS:
            _extract_starrocks_properties(result, sql)

    except Exception as e:
        logger.warning(f"Error in regex DDL parsing: {e}")

    return result


def _empty_result() -> Dict[str, Any]:
    """Return empty result structure."""
    return {
        "table": {"name": "", "comment": ""},
        "columns": [],
        "primary_keys": [],
        "foreign_keys": [],
        "indexes": []
    }


def _init_result() -> Dict[str, Any]:
    """Initialize result structure."""
    return {
        "table": {"name": "", "schema_name": "", "database_name": "", "comment": ""},
        "columns": [],
        "primary_keys": [],
        "foreign_keys": [],
        "indexes": []
    }


def _extract_table_comment(sql: str) -> str:
    """Extract table comment from SQL."""
    # Try from enum utils first
    original_comment = _extract_table_comment_after_columns(sql)
    if original_comment:
        is_valid, _, comment = validate_comment(original_comment)
        if is_valid:
            return comment

    # Try regex patterns
    comment_matches = re.findall(r"COMMENT\s*(?:=\s*)?['\"]([^'\"]+)['\"]\s*$", sql, re.IGNORECASE)
    if comment_matches:
        is_valid, _, comment = validate_comment(comment_matches[-1])
        if is_valid:
            return comment

    # Try loose pattern for corrupted DDL
    loose_matches = re.findall(r"COMMENT\s*(?:=\s*)?['\"]([^'\"]+)", sql, re.IGNORECASE)
    if loose_matches:
        comment = loose_matches[-1]
        # Clean up trailing error messages
        comment = re.sub(r"\s*contains.*$", "", comment)
        comment = re.sub(r"\s*is not valid.*$", "", comment)
        comment = re.sub(r"\s*unexpected.*$", "", comment)
        is_valid, _, comment = validate_comment(comment)
        if is_valid:
            return comment

    return ""


def _extract_table_name(sql: str) -> str:
    """Extract table name from SQL."""
    table_match = _TABLE_NAME_RE.search(sql)
    if table_match:
        table_name = table_match.group(1).strip('"`')
        is_valid, _, table_name = validate_table_name(table_name)
        if is_valid and table_name:
            return table_name
    return ""


def _extract_columns_text(sql: str) -> str:
    """Extract column definitions text between parentheses."""
    state = QuoteState()
    paren_count = 0
    start_idx = -1

    for i, char in enumerate(sql):
        if state.escaped:
            state.escaped = False
            continue

        if char == "\\":
            state.escaped = True
            continue

        if state.in_single or state.in_double or state.in_backtick:
            if state.in_single and char == "'":
                state.in_single = False
            elif state.in_double and char == '"':
                state.in_double = False
            elif state.in_backtick and char == "`":
                state.in_backtick = False
            continue

        # Not in quote
        if char == "'":
            state.in_single = True
            continue
        if char == '"':
            state.in_double = True
            continue
        if char == "`":
            state.in_backtick = True
            continue

        if char == '(':
            if paren_count == 0:
                start_idx = i + 1
            paren_count += 1
        elif char == ')':
            paren_count -= 1
            if paren_count == 0 and start_idx >= 0:
                return sql[start_idx:i]

    return ""


def _parse_columns(columns_text: str) -> list:
    """Parse column definitions from column text."""
    # Split by comma at top level
    column_defs = _split_columns(columns_text)
    columns = []

    for col_def in column_defs:
        # Skip constraint definitions
        if re.match(r'^(PRIMARY\s+KEY|FOREIGN\s+KEY|CONSTRAINT|INDEX|UNIQUE|KEY)\s', col_def, re.IGNORECASE):
            continue

        col_match = _COLUMN_DEF_RE.match(col_def)
        if not col_match:
            continue

        col_name = col_match.group(1).strip('"`')

        # Validate column name
        if len(col_name) > MAX_COLUMN_NAME_LENGTH:
            logger.warning(f"Column name too long: {col_name}")
            continue

        col_type = col_match.group(2).strip()

        # Validate type length
        if len(col_type) > MAX_TYPE_DEFINITION_LENGTH:
            logger.warning(f"Type definition too long: {col_type}")
            continue

        col_def_upper = col_def.upper()
        is_nullable = "NOT NULL" not in col_def_upper

        col_dict = {
            "name": col_name,
            "type": col_type,
            "nullable": is_nullable,
            "comment": ""
        }

        # Extract column comment
        col_comment_match = _COLUMN_COMMENT_RE.search(col_def)
        if col_comment_match:
            is_valid, _, comment = validate_comment(col_comment_match.group(1))
            if is_valid:
                col_dict["comment"] = comment

        columns.append(col_dict)

    return columns


def _split_columns(columns_text: str) -> list:
    """Split column text by commas at top level."""
    column_defs = []
    current_def = []
    paren_depth = 0
    state = QuoteState()

    for char in columns_text:
        if state.escaped:
            state.escaped = False
            current_def.append(char)
            continue

        if char == "\\":
            state.escaped = True
            current_def.append(char)
            continue

        if state.in_single:
            if char == "'":
                state.in_single = False
            current_def.append(char)
            continue

        if state.in_double:
            if char == '"':
                state.in_double = False
            current_def.append(char)
            continue

        if state.in_backtick:
            if char == "`":
                state.in_backtick = False
            current_def.append(char)
            continue

        # Not in quote
        if char == "'":
            state.in_single = True
            current_def.append(char)
            continue
        if char == '"':
            state.in_double = True
            current_def.append(char)
            continue
        if char == "`":
            state.in_backtick = True
            current_def.append(char)
            continue

        if char == '(':
            paren_depth += 1
            current_def.append(char)
        elif char == ')':
            paren_depth -= 1
            current_def.append(char)
        elif char == ',' and paren_depth == 0:
            if current_def.strip():
                column_defs.append(current_def.strip())
            current_def = []
        else:
            current_def.append(char)

    if current_def.strip():
        column_defs.append(current_def.strip())

    return column_defs


def _extract_primary_keys(sql: str) -> list:
    """Extract primary key columns."""
    pk_match = _PRIMARY_KEY_RE.search(sql)
    if pk_match:
        return [col.strip().strip('"`') for col in pk_match.group(1).split(',')]
    return []


def _extract_foreign_keys(sql: str) -> list:
    """Extract foreign key definitions."""
    fk_list = []
    for fk_match in _FOREIGN_KEY_RE.finditer(sql):
        fk_list.append({
            "from_column": fk_match.group(1).strip('"`'),
            "to_table": fk_match.group(2).strip('"`'),
            "to_column": fk_match.group(3).strip('"`')
        })
    return fk_list


def _extract_indexes(sql: str) -> list:
    """Extract index definitions."""
    index_list = []
    for idx_match in _INDEX_RE.finditer(sql):
        index_list.append({
            "name": idx_match.group(1),
            "columns": [col.strip().strip('"`') for col in idx_match.group(2).split(',')]
        })
    return index_list


def _extract_starrocks_properties(result: Dict[str, Any], sql: str):
    """Extract StarRocks-specific properties."""
    starrocks_props = extract_starrocks_properties(sql)
    if starrocks_props["properties"]:
        result["table"]["starrocks_properties"] = starrocks_props["properties"]
    if starrocks_props["distributed_by"]:
        result["table"]["distributed_by"] = starrocks_props["distributed_by"]
