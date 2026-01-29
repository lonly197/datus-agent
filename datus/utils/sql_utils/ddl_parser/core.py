# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Core DDL parsing functions and regex patterns.

This module provides the basic parsing utilities and pre-compiled
regex patterns used for DDL parsing.
"""

import re
from typing import Any, Dict, Optional

import sqlglot
from sqlglot.expressions import Table

from datus.utils.constants import DBType
from datus.utils.loggings import get_logger
from datus.utils.sql_utils.ddl_cleaner import _clean_ddl
from datus.utils.sql_utils.dialect_support import parse_dialect
from datus.utils.sql_utils.enum_utils import _extract_table_comment_after_columns
from datus.utils.sql_utils.validation import (
    MAX_COLUMN_NAME_LENGTH,
    validate_comment,
    validate_sql_input,
    validate_table_name,
)

logger = get_logger(__name__)

# =============================================================================
# PRE-COMPILED REGEX PATTERNS FOR DDL PARSING
# =============================================================================

_TABLE_NAME_RE = re.compile(
    r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?[`"]?([\w.]+)[`"]?\s*\(',
    re.IGNORECASE
)

_TABLE_COMMENT_RE = re.compile(
    r"COMMENT\s*(?:=\s*)?['\"]([^'\"]+)['\"]\s*$",
    re.IGNORECASE
)

_TABLE_COMMENT_LOOSE_RE = re.compile(
    r"COMMENT\s*(?:=\s*)?['\"]([^'\"]+)",
    re.IGNORECASE
)

_COLUMN_COMMENT_RE = re.compile(
    r"COMMENT\s*(?:=\s*)?['\"]([^'\"]+)['\"]",
    re.IGNORECASE
)

_COLUMN_DEF_RE = re.compile(
    r'^`?([\w]+)`?\s+(.+?)(?=\s+COMMENT|\s+NOT\s+NULL|\s+NULL|\s+DEFAULT|\s+PRIMARY|\s+KEY|\s+UNIQUE|\s+AUTO_INCREMENT|\s+ENCODING|\s+COLLATE|\s+AS|\s+COMPUTED|\s+MATERIALIZED|\s*$)',
    re.IGNORECASE | re.DOTALL
)

_PRIMARY_KEY_RE = re.compile(
    r'PRIMARY\s+KEY\s*\(([^)]+)\)',
    re.IGNORECASE
)

_FOREIGN_KEY_RE = re.compile(
    r'FOREIGN\s+KEY\s*\(`?([^`)]+)`?\)\s*REFERENCES\s+[`"]?([\w.]+)[`"]?\s*\(([^)]+)\)',
    re.IGNORECASE
)

_INDEX_RE = re.compile(
    r'(?:UNIQUE\s+)?(?:KEY|INDEX)\s+[`"]?([\w]+)[`"]?\s*\(([^)]+)\)',
    re.IGNORECASE
)


# =============================================================================
# CORE PARSING FUNCTIONS
# =============================================================================

def parse_metadata_from_ddl(
    sql: str,
    dialect: str = DBType.SNOWFLAKE,
    warn_on_invalid: bool = True,
) -> Dict[str, Any]:
    """
    Parse SQL CREATE TABLE statement and return structured table and column information.

    Args:
        sql: SQL CREATE TABLE statement
        dialect: SQL dialect (mysql, oracle, postgre, snowflake, bigquery, starrocks...)

    Returns:
        Dict containing table name, schema, database, and column list with types and comments.
    """
    # Validate input
    is_valid, error_msg = validate_sql_input(sql)
    if not is_valid:
        if warn_on_invalid:
            logger.warning(f"Invalid SQL input: {error_msg}")
        else:
            logger.debug(f"Invalid SQL input: {error_msg}")
        return {"table": {"name": ""}, "columns": []}

    dialect = parse_dialect(dialect)

    try:
        result = {"table": {"name": "", "schema_name": "", "database_name": ""}, "columns": []}

        # Parse SQL using sqlglot with error handling
        parsed = sqlglot.parse_one(sql.strip(), dialect=dialect, error_level=sqlglot.ErrorLevel.IGNORE)

        if isinstance(parsed, sqlglot.exp.Create):
            tb_info = parsed.find_all(Table).__next__()

            # Get and validate table name
            table_name = tb_info.name
            if isinstance(table_name, str):
                table_name = table_name.strip('"').strip("`").strip("[]")

            # Validate table name
            is_valid, _, table_name = validate_table_name(table_name)
            if not is_valid or not table_name:
                logger.warning(f"Invalid table name: {tb_info.name}")
                return {"table": {"name": ""}, "columns": []}

            result["table"]["name"] = table_name
            result["table"]["schema_name"] = tb_info.db
            result["table"]["database_name"] = tb_info.catalog

            # Handle table comment
            if tb_info.comments:
                is_valid, _, comment = validate_comment(tb_info.comments)
                if is_valid:
                    result["table"]["comment"] = comment

            # Fallback: extract table comment using pre-compiled regex
            if not result["table"].get("comment"):
                table_comment_matches = _TABLE_COMMENT_RE.findall(sql)
                if table_comment_matches:
                    is_valid, _, comment = validate_comment(table_comment_matches[-1])
                    if is_valid:
                        result["table"]["comment"] = comment
                else:
                    table_comment = _extract_table_comment_after_columns(sql)
                    if table_comment:
                        is_valid, _, comment = validate_comment(table_comment)
                        if is_valid:
                            result["table"]["comment"] = comment

            # Get column definitions
            for column in parsed.this.expressions:
                if isinstance(column, sqlglot.exp.ColumnDef):
                    col_name = column.name
                    if isinstance(col_name, str):
                        col_name = col_name.strip('"').strip("`").strip("[]")

                    col_dict = {"name": col_name, "type": str(column.kind)}

                    # Validate column name
                    if len(col_name) > MAX_COLUMN_NAME_LENGTH:
                        logger.warning(f"Column name too long: {col_name}")
                        continue

                    # Get column comment
                    if hasattr(column, "comments") and column.comments:
                        col_dict["comment"] = column.comments
                    elif hasattr(column, "comment") and column.comment:
                        col_dict["comment"] = column.comment

                    result["columns"].append(col_dict)

        return result

    except Exception as e:
        logger.error(f"Error parsing SQL: {e}")
        return {"table": {"name": ""}, "columns": []}


def _extract_table_info(
    parsed,
    sql: str,
) -> Dict[str, Any]:
    """Extract table information from parsed sqlglot result."""
    result = {"name": "", "schema_name": "", "database_name": "", "comment": ""}

    if not isinstance(parsed, sqlglot.exp.Create):
        return result

    tb_info = parsed.find_all(Table).__next__()

    # Get and validate table name
    table_name = tb_info.name
    if isinstance(table_name, str):
        table_name = table_name.strip('"').strip("`").strip("[]")

    is_valid, _, table_name = validate_table_name(table_name)
    if is_valid and table_name:
        result["name"] = table_name

    result["schema_name"] = tb_info.db
    result["database_name"] = tb_info.catalog

    # Handle table comment
    if tb_info.comments:
        is_valid, _, comment = validate_comment(tb_info.comments)
        if is_valid:
            result["comment"] = comment

    return result


def _extract_column_info(column) -> Optional[Dict[str, Any]]:
    """Extract column information from parsed sqlglot column definition."""
    if not isinstance(column, sqlglot.exp.ColumnDef):
        return None

    col_name = column.name
    if isinstance(col_name, str):
        col_name = col_name.strip('"').strip("`").strip("[]")

    # Validate column name length
    if len(col_name) > MAX_COLUMN_NAME_LENGTH:
        logger.warning(f"Column name too long: {col_name}")
        return None

    col_dict = {"name": col_name, "type": str(column.kind), "nullable": True}

    # Check NOT NULL constraint
    if hasattr(column, "constraints") and column.constraints:
        for constraint in column.constraints:
            if isinstance(constraint, sqlglot.exp.NotNullColumnConstraint):
                col_dict["nullable"] = False

    # Get column comment
    if hasattr(column, "comments") and column.comments:
        col_dict["comment"] = column.comments
    elif hasattr(column, "comment") and column.comment:
        col_dict["comment"] = column.comment

    return col_dict


def _extract_primary_keys(parsed) -> list:
    """Extract primary key columns from parsed DDL."""
    pk_columns = []
    for constraint in parsed.find_all(sqlglot.exp.PrimaryKey):
        pk_columns.extend([col.name for col in constraint.expressions if hasattr(col, 'name')])
    return pk_columns


def _extract_foreign_keys(parsed) -> list:
    """Extract foreign key definitions from parsed DDL."""
    fk_list = []
    for constraint in parsed.find_all(sqlglot.exp.ForeignKey):
        fk_dict = {"from_column": "", "to_table": "", "to_column": ""}

        if constraint.expressions:
            fk_dict["from_column"] = constraint.expressions[0].name if hasattr(constraint.expressions[0], 'name') else str(constraint.expressions[0])

        if constraint.ref:
            ref_table = constraint.ref.name if hasattr(constraint.ref, 'name') else str(constraint.ref)
            fk_dict["to_table"] = ref_table
            if constraint.ref.expressions:
                fk_dict["to_column"] = constraint.ref.expressions[0].name if hasattr(constraint.ref.expressions[0], 'name') else str(constraint.ref.expressions[0])

        fk_list.append(fk_dict)
    return fk_list


def _extract_indexes(parsed) -> list:
    """Extract index definitions from parsed DDL."""
    index_list = []
    for constraint in parsed.find_all(sqlglot.exp.Index):
        index_dict = {"name": "", "columns": []}

        if hasattr(constraint, 'name'):
            index_dict["name"] = constraint.name

        if constraint.expressions:
            index_dict["columns"] = [
                expr.name if hasattr(expr, 'name') else str(expr)
                for expr in constraint.expressions
            ]

        index_list.append(index_dict)
    return index_list


def _fallback_extract_table_comment(sql: str) -> str:
    """Fallback: extract table comment using regex when sqlglot fails."""
    table_comment_matches = _TABLE_COMMENT_RE.findall(sql)
    if table_comment_matches:
        is_valid, _, comment = validate_comment(table_comment_matches[-1])
        if is_valid:
            return comment

    table_comment = _extract_table_comment_after_columns(sql)
    if table_comment:
        is_valid, _, comment = validate_comment(table_comment)
        if is_valid:
            return comment
    return ""
