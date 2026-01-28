# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
DDL parsing and metadata extraction utilities.

This module provides comprehensive functions for parsing SQL DDL statements
and extracting structured metadata including tables, columns, constraints, etc.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

import sqlglot
from sqlglot.expressions import Table

from datus.utils.constants import DBType
from datus.utils.loggings import get_logger

from .ddl_cleaner import (
    _clean_ddl,
    ddl_has_missing_commas,
    fix_missing_commas_in_ddl,
    sanitize_ddl_for_storage,
)
from .dialect_support import extract_starrocks_properties, parse_dialect
from .enum_utils import _extract_table_comment_after_columns
from .validation import MAX_COLUMN_NAME_LENGTH, MAX_TYPE_DEFINITION_LENGTH, validate_comment, validate_sql_input, validate_table_name

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

# Fallback pattern for corrupted DDL where closing quote might be missing
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
# MAIN DDL PARSING FUNCTIONS
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


def extract_enhanced_metadata_from_ddl(
    sql: str,
    dialect: str = DBType.SNOWFLAKE,
    warn_on_invalid: bool = True,
) -> Dict[str, Any]:
    """
    Extract comprehensive metadata from DDL including comments, constraints, and relationships.

    This function uses a two-stage parsing strategy:
    1. Primary: sqlglot for accurate parsing
    2. Fallback: regex-based parsing for databases with limited sqlglot support

    Args:
        sql: SQL CREATE TABLE statement
        dialect: SQL dialect (mysql, oracle, postgre, snowflake, bigquery, starrocks...)

    Returns:
        Dict containing table info, columns, primary keys, foreign keys, and indexes.
    """
    # Validate input
    sql = fix_missing_commas_in_ddl(sql)
    is_valid, error_msg = validate_sql_input(sql)
    if not is_valid:
        cleaned_sql = sanitize_ddl_for_storage(sql)
        if cleaned_sql and cleaned_sql != sql:
            retry_valid, retry_error = validate_sql_input(cleaned_sql)
            if retry_valid:
                sql = cleaned_sql
                is_valid = True
            else:
                error_msg = retry_error or error_msg
                sql = cleaned_sql
        if not is_valid:
            if "Unbalanced parentheses" in error_msg:
                logger.debug(f"Continuing with best-effort parsing despite: {error_msg}")
            else:
                if warn_on_invalid:
                    logger.warning(f"Invalid SQL input: {error_msg}")
                else:
                    logger.debug(f"Invalid SQL input: {error_msg}")
                return {
                    "table": {"name": "", "comment": ""},
                    "columns": [],
                    "primary_keys": [],
                    "foreign_keys": [],
                    "indexes": []
                }

    dialect = parse_dialect(dialect)

    # First try sqlglot parsing
    result = {
        "table": {"name": "", "schema_name": "", "database_name": "", "comment": ""},
        "columns": [],
        "primary_keys": [],
        "foreign_keys": [],
        "indexes": []
    }

    # Try multiple parsing strategies
    parsing_attempts = []

    # Strategy 1: Try original SQL
    parsing_attempts.append(("original", sql.strip()))

    # Strategy 2: Try cleaned SQL
    cleaned_sql = _clean_ddl(sql)
    if cleaned_sql != sql:
        parsing_attempts.append(("cleaned", cleaned_sql))

    # Strategy 3: Try with different dialects
    if dialect == DBType.MYSQL:
        # StarRocks uses MySQL dialect but might have different syntax
        parsing_attempts.append(("mysql_dialect", cleaned_sql))
    else:
        # Try MySQL dialect for broader compatibility
        parsing_attempts.append(("mysql_dialect", cleaned_sql))

    # Try each parsing strategy
    for strategy_name, sql_to_parse in parsing_attempts:
        try:
            # Parse SQL using sqlglot with error handling
            parsed = sqlglot.parse_one(sql_to_parse, dialect=dialect, error_level=sqlglot.ErrorLevel.IGNORE)

            if isinstance(parsed, sqlglot.exp.Create):
                tb_info = parsed.find_all(Table).__next__()

                # Get and validate table name
                table_name = tb_info.name
                if isinstance(table_name, str):
                    table_name = table_name.strip('"').strip("`").strip("[]")

                is_valid, _, table_name = validate_table_name(table_name)
                if not is_valid or not table_name:
                    logger.warning(f"Invalid table name: {tb_info.name}")
                else:
                    result["table"]["name"] = table_name

                result["table"]["schema_name"] = tb_info.db
                result["table"]["database_name"] = tb_info.catalog

                # Handle table comment
                if tb_info.comments:
                    is_valid, _, comment = validate_comment(tb_info.comments)
                    if is_valid:
                        result["table"]["comment"] = comment

                # Get column definitions
                columns_parsed = 0
                for column in parsed.this.expressions:
                    if isinstance(column, sqlglot.exp.ColumnDef):
                        col_name = column.name
                        if isinstance(col_name, str):
                            col_name = col_name.strip('"').strip("`").strip("[]")

                        # Validate column name length
                        if len(col_name) > MAX_COLUMN_NAME_LENGTH:
                            logger.warning(f"Column name too long: {col_name}")
                            continue

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

                        result["columns"].append(col_dict)
                        columns_parsed += 1

                # Extract primary keys
                for constraint in parsed.find_all(sqlglot.exp.PrimaryKey):
                    pk_columns = [col.name for col in constraint.expressions if hasattr(col, 'name')]
                    result["primary_keys"].extend(pk_columns)

                # Extract foreign keys
                for constraint in parsed.find_all(sqlglot.exp.ForeignKey):
                    fk_dict = {"from_column": "", "to_table": "", "to_column": ""}

                    if constraint.expressions:
                        fk_dict["from_column"] = constraint.expressions[0].name if hasattr(constraint.expressions[0], 'name') else str(constraint.expressions[0])

                    if constraint.ref:
                        ref_table = constraint.ref.name if hasattr(constraint.ref, 'name') else str(constraint.ref)
                        fk_dict["to_table"] = ref_table
                        if constraint.ref.expressions:
                            fk_dict["to_column"] = constraint.ref.expressions[0].name if hasattr(constraint.ref.expressions[0], 'name') else str(constraint.ref.expressions[0])

                    result["foreign_keys"].append(fk_dict)

                # Extract indexes
                for constraint in parsed.find_all(sqlglot.exp.Index):
                    index_dict = {"name": "", "columns": []}

                    if hasattr(constraint, 'name'):
                        index_dict["name"] = constraint.name

                    if constraint.expressions:
                        index_dict["columns"] = [
                            expr.name if hasattr(expr, 'name') else str(expr)
                            for expr in constraint.expressions
                        ]

                    result["indexes"].append(index_dict)

                # Fallback: extract table comment using pre-compiled regex
                if not result["table"].get("comment"):
                    table_comment_matches = _TABLE_COMMENT_RE.findall(sql_to_parse)
                    if table_comment_matches:
                        is_valid, _, comment = validate_comment(table_comment_matches[-1])
                        if is_valid:
                            result["table"]["comment"] = comment
                    else:
                        table_comment = _extract_table_comment_after_columns(sql_to_parse)
                        if table_comment:
                            is_valid, _, comment = validate_comment(table_comment)
                            if is_valid:
                                result["table"]["comment"] = comment

                # Success criteria: at least have table name and some columns
                if result["table"]["name"] and columns_parsed > 0:
                    logger.debug(f"Successfully parsed DDL using {strategy_name} strategy")
                    return result

        except sqlglot.TokenError as e:
            # TokenError: This means the SQL is malformed (e.g., unclosed quotes, truncated)
            logger.debug(f"TokenError with {strategy_name} strategy: {e}")
            continue
        except Exception as e:
            # Other exceptions (parsing errors, etc.)
            logger.debug(f"Error with {strategy_name} strategy: {e}")
            continue

    # If all sqlglot strategies failed, try regex-based parsing
    logger.info(f"All sqlglot parsing strategies failed, falling back to regex parsing for dialect: {dialect}")
    regex_result = _parse_ddl_with_regex(sql, dialect)

    # Merge results
    if regex_result["table"]["name"]:
        result["table"]["name"] = regex_result["table"]["name"]
    if regex_result["columns"]:
        result["columns"] = regex_result["columns"]
    if regex_result["table"].get("comment"):
        result["table"]["comment"] = regex_result["table"]["comment"]
    if regex_result["primary_keys"]:
        result["primary_keys"] = regex_result["primary_keys"]
    if regex_result["foreign_keys"]:
        result["foreign_keys"] = regex_result["foreign_keys"]
    if regex_result["indexes"]:
        result["indexes"] = regex_result["indexes"]

    # Log the final attempt
    ddl_preview = sql[:200] if len(sql) > 200 else sql
    if result["table"]["name"]:
        logger.info(f"Successfully parsed DDL using regex fallback - Table: {result['table']['name']}")
    else:
        logger.warning(f"Failed to parse DDL with all strategies - Preview: {ddl_preview}...")

    return result


def extract_metadata_from_ddl_regex_only(
    sql: str,
    dialect: str = DBType.SNOWFLAKE,
    warn_on_invalid: bool = True,
) -> Dict[str, Any]:
    """
    Parse DDL using regex-only strategy (no sqlglot).

    This is intended for dialects where sqlglot is noisy or fails frequently.
    """
    sql = fix_missing_commas_in_ddl(sql)
    is_valid, error_msg = validate_sql_input(sql)
    if not is_valid:
        cleaned_sql = sanitize_ddl_for_storage(sql)
        if cleaned_sql and cleaned_sql != sql:
            retry_valid, retry_error = validate_sql_input(cleaned_sql)
            if retry_valid:
                sql = cleaned_sql
                is_valid = True
            else:
                error_msg = retry_error or error_msg
                sql = cleaned_sql
        if not is_valid:
            if "Unbalanced parentheses" in error_msg:
                logger.debug(f"Continuing regex-only parsing despite: {error_msg}")
            else:
                if warn_on_invalid:
                    logger.warning(f"Invalid SQL input: {error_msg}")
                else:
                    logger.debug(f"Invalid SQL input: {error_msg}")
                return {
                    "table": {"name": "", "comment": ""},
                    "columns": [],
                    "primary_keys": [],
                    "foreign_keys": [],
                    "indexes": []
                }

    dialect = parse_dialect(dialect)
    return _parse_ddl_with_regex(sql, dialect)


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
            return {
                "table": {"name": "", "comment": ""},
                "columns": [],
                "primary_keys": [],
                "foreign_keys": [],
                "indexes": []
            }

    result = {
        "table": {"name": "", "schema_name": "", "database_name": "", "comment": ""},
        "columns": [],
        "primary_keys": [],
        "foreign_keys": [],
        "indexes": []
    }

    # Keep original SQL for table comment extraction
    original_sql = sql

    # Extract table comment from ORIGINAL SQL before cleaning
    # This preserves comments even if error messages were appended
    original_comment = _extract_table_comment_after_columns(original_sql)
    if original_comment:
        is_valid, _, comment = validate_comment(original_comment)
        if is_valid:
            result["table"]["comment"] = comment
    if not result["table"]["comment"]:
        original_comment_matches = _TABLE_COMMENT_RE.findall(original_sql)
        if not original_comment_matches:
            # Fallback: try loose pattern for corrupted DDL
            original_comment_matches = _TABLE_COMMENT_LOOSE_RE.findall(original_sql)
            if original_comment_matches:
                # Clean up trailing error messages from the extracted comment
                comment = original_comment_matches[-1]
                comment = re.sub(r"\s*contains.*$", "", comment)
                comment = re.sub(r"\s*is not valid.*$", "", comment)
                comment = re.sub(r"\s*unexpected.*$", "", comment)
                is_valid, _, comment = validate_comment(comment)
                if is_valid:
                    result["table"]["comment"] = comment
        else:
            is_valid, _, comment = validate_comment(original_comment_matches[-1])
            if is_valid:
                result["table"]["comment"] = comment

    # Clean the DDL
    sql = _clean_ddl(original_sql)
    if sql != original_sql:
        logger.debug(f"Cleaned corrupted DDL: {len(original_sql)} -> {len(sql)} chars")

    try:
        # Extract table name using pre-compiled regex
        table_match = _TABLE_NAME_RE.search(sql)
        if table_match:
            table_name = table_match.group(1).strip('"`')
            is_valid, _, table_name = validate_table_name(table_name)
            if is_valid and table_name:
                result["table"]["name"] = table_name

        # If we already have a comment from original SQL, skip extracting from cleaned
        if not result["table"]["comment"]:
            comment_matches = _TABLE_COMMENT_RE.findall(sql)
            if comment_matches:
                is_valid, _, comment = validate_comment(comment_matches[-1])
                if is_valid:
                    result["table"]["comment"] = comment
            else:
                table_comment = _extract_table_comment_after_columns(sql)
                if table_comment:
                    is_valid, _, comment = validate_comment(table_comment)
                    if is_valid:
                        result["table"]["comment"] = comment

        # Find the content between the first ( and the matching )
        paren_count = 0
        start_idx = -1
        end_idx = -1

        in_single_quote = False
        in_double_quote = False
        in_backtick = False
        escaped = False
        for i, char in enumerate(sql):
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if in_single_quote:
                if char == "'":
                    in_single_quote = False
                continue
            if in_double_quote:
                if char == '"':
                    in_double_quote = False
                continue
            if in_backtick:
                if char == "`":
                    in_backtick = False
                continue
            if char == "'":
                in_single_quote = True
                continue
            if char == '"':
                in_double_quote = True
                continue
            if char == "`":
                in_backtick = True
                continue
            if char == '(':
                if paren_count == 0:
                    start_idx = i + 1
                paren_count += 1
            elif char == ')':
                paren_count -= 1
                if paren_count == 0:
                    end_idx = i
                    break

        if start_idx > 0 and end_idx > start_idx:
            columns_text = sql[start_idx:end_idx]

            # Split by comma, handling nested parentheses and quoted comments
            column_defs = []
            current_def = ""
            paren_depth = 0
            in_single_quote = False
            in_double_quote = False
            in_backtick = False
            escaped = False

            for char in columns_text:
                if escaped:
                    escaped = False
                    current_def += char
                    continue
                if char == "\\":
                    escaped = True
                    current_def += char
                    continue
                if in_single_quote:
                    if char == "'":
                        in_single_quote = False
                    current_def += char
                    continue
                if in_double_quote:
                    if char == '"':
                        in_double_quote = False
                    current_def += char
                    continue
                if in_backtick:
                    if char == "`":
                        in_backtick = False
                    current_def += char
                    continue
                if char == "'":
                    in_single_quote = True
                    current_def += char
                    continue
                if char == '"':
                    in_double_quote = True
                    current_def += char
                    continue
                if char == "`":
                    in_backtick = True
                    current_def += char
                    continue
                if char == '(':
                    paren_depth += 1
                    current_def += char
                elif char == ')':
                    paren_depth -= 1
                    current_def += char
                elif char == ',' and paren_depth == 0:
                    if current_def.strip():
                        column_defs.append(current_def.strip())
                    current_def = ""
                else:
                    current_def += char

            if current_def.strip():
                column_defs.append(current_def.strip())

            # Parse each column definition using pre-compiled regex
            for col_def in column_defs:
                # Skip constraint definitions
                if re.match(r'^(PRIMARY\s+KEY|FOREIGN\s+KEY|CONSTRAINT|INDEX|UNIQUE|KEY)\s', col_def, re.IGNORECASE):
                    continue

                col_match = _COLUMN_DEF_RE.match(col_def)
                if col_match:
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

                    result["columns"].append(col_dict)

        # Extract primary key using pre-compiled regex
        pk_match = _PRIMARY_KEY_RE.search(sql)
        if pk_match:
            pk_columns = [col.strip().strip('"`') for col in pk_match.group(1).split(',')]
            result["primary_keys"] = pk_columns

        # Extract foreign keys using pre-compiled regex
        for fk_match in _FOREIGN_KEY_RE.finditer(sql):
            fk_dict = {
                "from_column": fk_match.group(1).strip('"`'),
                "to_table": fk_match.group(2).strip('"`'),
                "to_column": fk_match.group(3).strip('"`')
            }
            result["foreign_keys"].append(fk_dict)

        # Extract indexes using pre-compiled regex
        for idx_match in _INDEX_RE.finditer(sql):
            index_dict = {
                "name": idx_match.group(1),
                "columns": [col.strip().strip('"`') for col in idx_match.group(2).split(',')]
            }
            result["indexes"].append(index_dict)

        # Extract StarRocks-specific properties
        if dialect == DBType.STARROCKS:
            starrocks_props = extract_starrocks_properties(sql)
            # Merge StarRocks properties into result
            if starrocks_props["properties"]:
                result["table"]["starrocks_properties"] = starrocks_props["properties"]
            if starrocks_props["distributed_by"]:
                result["table"]["distributed_by"] = starrocks_props["distributed_by"]

    except Exception as e:
        logger.warning(f"Error in regex DDL parsing: {e}")

    return result
