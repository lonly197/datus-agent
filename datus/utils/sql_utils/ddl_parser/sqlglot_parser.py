# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
sqlglot-based DDL parsing strategies.

This module provides enhanced metadata extraction using sqlglot parser
with multiple fallback strategies for robust DDL parsing.
"""

from typing import Any, Dict, List, Tuple

import sqlglot

from datus.utils.constants import DBType
from datus.utils.loggings import get_logger
from datus.utils.sql_utils.ddl_cleaner import (
    _clean_ddl,
    fix_missing_commas_in_ddl,
    sanitize_ddl_for_storage,
)
from datus.utils.sql_utils.ddl_parser.core import (
    _extract_column_info,
    _extract_foreign_keys,
    _extract_indexes,
    _extract_primary_keys,
    _extract_table_info,
    _fallback_extract_table_comment,
    _TABLE_COMMENT_RE,
    parse_metadata_from_ddl,
)
from datus.utils.sql_utils.ddl_parser.regex_fallback import _parse_ddl_with_regex
from datus.utils.sql_utils.dialect_support import parse_dialect
from datus.utils.sql_utils.validation import validate_sql_input

logger = get_logger(__name__)


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
    # Validate and clean input
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
                return _empty_metadata_result()

    dialect = parse_dialect(dialect)

    # Initialize result structure
    result = {
        "table": {"name": "", "schema_name": "", "database_name": "", "comment": ""},
        "columns": [],
        "primary_keys": [],
        "foreign_keys": [],
        "indexes": []
    }

    # Try multiple parsing strategies
    parsing_attempts = _build_parsing_attempts(sql, dialect)

    # Try each parsing strategy
    for strategy_name, sql_to_parse in parsing_attempts:
        parsed_result = _try_sqlglot_parse(sql_to_parse, dialect, strategy_name, result)
        if parsed_result:
            return parsed_result

    # If all sqlglot strategies failed, try regex-based parsing
    logger.info(f"All sqlglot parsing strategies failed, falling back to regex parsing for dialect: {dialect}")
    return _merge_regex_result(sql, result, dialect)


def _empty_metadata_result() -> Dict[str, Any]:
    """Return empty metadata result structure."""
    return {
        "table": {"name": "", "comment": ""},
        "columns": [],
        "primary_keys": [],
        "foreign_keys": [],
        "indexes": []
    }


def _build_parsing_attempts(sql: str, dialect: str) -> List[Tuple[str, str]]:
    """Build list of parsing attempts with different strategies."""
    attempts = []

    # Strategy 1: Try original SQL
    attempts.append(("original", sql.strip()))

    # Strategy 2: Try cleaned SQL
    cleaned_sql = _clean_ddl(sql)
    if cleaned_sql != sql:
        attempts.append(("cleaned", cleaned_sql))

    # Strategy 3: Try with different dialects
    if dialect == DBType.MYSQL:
        attempts.append(("mysql_dialect", cleaned_sql))
    else:
        attempts.append(("mysql_dialect", cleaned_sql))

    return attempts


def _try_sqlglot_parse(
    sql_to_parse: str,
    dialect: str,
    strategy_name: str,
    result: Dict[str, Any],
) -> Dict[str, Any]:
    """Try to parse DDL using sqlglot with given strategy."""
    try:
        parsed = sqlglot.parse_one(sql_to_parse, dialect=dialect, error_level=sqlglot.ErrorLevel.IGNORE)

        if isinstance(parsed, sqlglot.exp.Create):
            # Extract table info
            result["table"] = _extract_table_info(parsed, sql_to_parse)

            # Extract column definitions
            columns_parsed = 0
            for column in parsed.this.expressions:
                col_info = _extract_column_info(column)
                if col_info:
                    result["columns"].append(col_info)
                    columns_parsed += 1

            # Extract constraints
            result["primary_keys"] = _extract_primary_keys(parsed)
            result["foreign_keys"] = _extract_foreign_keys(parsed)
            result["indexes"] = _extract_indexes(parsed)

            # Fallback: extract table comment if not found
            if not result["table"].get("comment"):
                result["table"]["comment"] = _fallback_extract_table_comment(sql_to_parse)

            # Success criteria: at least have table name and some columns
            if result["table"]["name"] and columns_parsed > 0:
                logger.debug(f"Successfully parsed DDL using {strategy_name} strategy")
                return result

    except sqlglot.TokenError as e:
        logger.debug(f"TokenError with {strategy_name} strategy: {e}")
    except Exception as e:
        logger.debug(f"Error with {strategy_name} strategy: {e}")

    return {}


def _merge_regex_result(
    sql: str,
    result: Dict[str, Any],
    dialect: str,
) -> Dict[str, Any]:
    """Merge regex parsing result into main result."""
    regex_result = _parse_ddl_with_regex(sql, dialect)

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
    from datus.utils.sql_utils.ddl_parser.regex_fallback import _parse_ddl_with_regex

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
                return _empty_metadata_result()

    dialect = parse_dialect(dialect)
    return _parse_ddl_with_regex(sql, dialect)
