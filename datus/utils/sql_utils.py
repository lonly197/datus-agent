# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import re
from typing import Any, Dict, List, Optional, Tuple

import sqlglot
from sqlglot import expressions
from sqlglot.expressions import CTE, Table

from datus.utils.constants import DBType, SQLType
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

# =============================================================================
# PRE-COMPILED REGEX PATTERNS FOR PERFORMANCE
# =============================================================================
# Pre-compiling regex patterns at module level avoids repeated compilation overhead.
# This is crucial for production systems processing large numbers of DDL statements.

# Maximum lengths for ReDoS prevention and input validation
MAX_SQL_LENGTH = 1000000  # 1MB max DDL size
MAX_COLUMN_NAME_LENGTH = 128
MAX_TABLE_NAME_LENGTH = 256
MAX_COMMENT_LENGTH = 4000
MAX_TYPE_DEFINITION_LENGTH = 256

# Pre-compiled regex patterns for DDL parsing
# Using raw strings (r'...') to avoid escape sequence issues
_TABLE_NAME_RE = re.compile(
    r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?[`"]?([\w.]+)[`"]?\s*\(',
    re.IGNORECASE
)

_TABLE_COMMENT_RE = re.compile(
    r"COMMENT\s*=\s*['\"]([^'\"]+)['\"]\s*$",
    re.IGNORECASE
)

# Fallback pattern for corrupted DDL where closing quote might be missing
_TABLE_COMMENT_LOOSE_RE = re.compile(
    r"COMMENT\s*=\s*['\"]([^'\"]+)",
    re.IGNORECASE
)

_COLUMN_COMMENT_RE = re.compile(
    r"COMMENT\s*=\s*['\"]([^'\"]+)['\"]",
    re.IGNORECASE
)

_COLUMN_DEF_RE = re.compile(
    r'`?([\w]+)`?\s+(\w+(?:\([^)]{0,100}\))?)\s*(NULL|NOT\s+NULL)?',
    re.IGNORECASE
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

# DDL cleanup patterns - safely remove error message fragments
_ERROR_MESSAGE_PATTERNS = [
    re.compile(r"\s*'?\s*contains unsupported syntax.*", re.IGNORECASE),
    re.compile(r"\s*'?\s*is not valid at this position.*", re.IGNORECASE),
    re.compile(r"\s*'?\s*unexpected token.*", re.IGNORECASE),
    re.compile(r"\s*'?\s*missing.*", re.IGNORECASE),
    re.compile(r"\s*'?\s*found.*", re.IGNORECASE),
    re.compile(r"\s*Falling back to parsing as.*", re.IGNORECASE),
]

_UNCLOSED_QUOTE_RE = re.compile(r'COMMENT\s*["\'][^"\']*$', re.IGNORECASE)

_TRAILING_CHARS_RE = re.compile(r'[,\s]+$')

_MULTI_SPACE_RE = re.compile(r'\s+')

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
# INPUT VALIDATION FUNCTIONS
# =============================================================================

def validate_sql_input(sql: Any, max_length: int = MAX_SQL_LENGTH) -> Tuple[bool, str]:
    """
    Validate SQL input for security and correctness.

    This function performs essential input validation to prevent:
    - ReDoS attacks via malicious regex patterns
    - Memory exhaustion from oversized inputs
    - Type confusion attacks

    Args:
        sql: Input to validate
        max_length: Maximum allowed length for SQL string

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Type validation
    if sql is None:
        return True, ""  # None is allowed, will be handled by caller

    if not isinstance(sql, str):
        return False, f"SQL must be a string, got {type(sql).__name__}"

    # Length validation
    if len(sql) > max_length:
        return False, f"SQL length ({len(sql)}) exceeds maximum allowed ({max_length})"

    if len(sql) == 0:
        return True, ""  # Empty string is valid

    # Check for potential ReDoS patterns (excessive nested parentheses)
    paren_depth = 0
    max_paren_depth = 100  # Reasonable limit for DDL
    for char in sql:
        if char == '(':
            paren_depth += 1
            if paren_depth > max_paren_depth:
                return False, f"Excessive nested parentheses (depth {paren_depth})"
        elif char == ')':
            paren_depth -= 1
            if paren_depth < 0:
                return False, "Unbalanced parentheses in SQL"

    # Check for NULL bytes (could indicate binary injection)
    if '\x00' in sql:
        return False, "NULL bytes not allowed in SQL"

    return True, ""


def validate_table_name(name: Any) -> Tuple[bool, str, Optional[str]]:
    """
    Validate table name for security.

    Args:
        name: Table name to validate

    Returns:
        Tuple of (is_valid, error_message, sanitized_name)
    """
    if not name:
        return True, "", None

    if not isinstance(name, str):
        return False, f"Table name must be a string, got {type(name).__name__}", None

    if len(name) > MAX_TABLE_NAME_LENGTH:
        return False, f"Table name too long ({len(name)} > {MAX_TABLE_NAME_LENGTH})", None

    # Allow alphanumeric, underscore, dot, and backtick/quotes for quoted identifiers
    # This prevents SQL injection while allowing valid identifiers
    if not re.match(r'^[\w.`"]+$', name):
        return False, "Invalid characters in table name", None

    return True, "", name


def validate_comment(comment: Any) -> Tuple[bool, str, Optional[str]]:
    """
    Validate comment text for security and length.

    Args:
        comment: Comment text to validate

    Returns:
        Tuple of (is_valid, error_message, sanitized_comment)
    """
    if not comment:
        return True, "", None

    if not isinstance(comment, str):
        return False, f"Comment must be a string, got {type(comment).__name__}", None

    if len(comment) > MAX_COMMENT_LENGTH:
        return False, f"Comment too long ({len(comment)} > {MAX_COMMENT_LENGTH})", None

    # Check for potentially dangerous patterns and sanitize
    sanitized = comment

    # Remove HTML/Script tags and their content
    sanitized = re.sub(r'<script[^>]*>.*?</script>', '', sanitized, flags=re.IGNORECASE | re.DOTALL)
    sanitized = re.sub(r'<[^>]+>', '', sanitized)  # Remove remaining HTML tags

    # Remove JavaScript protocol
    sanitized = re.sub(r'javascript\s*:', '', sanitized, flags=re.IGNORECASE)

    # Remove event handlers (onclick, onerror, etc.)
    sanitized = re.sub(r'\s*on\w+\s*=\s*[^>\s]*', '', sanitized, flags=re.IGNORECASE)

    # If significant content was removed, flag it as potentially unsafe
    # but still allow the sanitized version
    tag_count = len(re.findall(r'<[^>]*>', comment))
    if tag_count > 0:
        logger.debug(f"Comment contained {tag_count} HTML tags, sanitized version returned")

    return True, "", sanitized


# =============================================================================
# DDL CLEANUP FUNCTIONS
# =============================================================================

def _clean_ddl(sql: str) -> str:
    """
    Clean corrupted DDL by removing common error message fragments.

    This function handles DDL that may have been corrupted during storage or
    transfer, such as when error messages get appended to the DDL text.

    Args:
        sql: Potentially corrupted DDL

    Returns:
        Cleaned DDL string
    """
    if not sql or not isinstance(sql, str):
        return sql

    cleaned = sql

    # Step 1: First, try to fix unclosed quotes by finding COMMENT without closing quote
    # Pattern: COMMENT followed by quote but no closing quote before end of line or error message
    # This needs to be done BEFORE error message cleanup to preserve table comments
    unclosed_comment_pattern = re.compile(r"COMMENT\s*['\"]([^'\"]*)$", re.IGNORECASE | re.MULTILINE)
    cleaned = unclosed_comment_pattern.sub("", cleaned)

    # Step 2: Apply pre-compiled error message patterns
    for pattern in _ERROR_MESSAGE_PATTERNS:
        cleaned = pattern.sub("", cleaned)

    # Step 3: Fix any remaining unclosed quotes in comments
    cleaned = _UNCLOSED_QUOTE_RE.sub("", cleaned)

    # Step 4: Remove trailing incomplete column definitions
    cleaned = _TRAILING_CHARS_RE.sub("", cleaned)

    # Step 5: Clean up multiple spaces
    cleaned = _MULTI_SPACE_RE.sub(" ", cleaned)

    return cleaned.strip()


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
# MAIN DDL PARSING FUNCTIONS
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
    return (dialect or DBType.SNOWFLAKE).strip().lower()


def parse_metadata_from_ddl(sql: str, dialect: str = DBType.SNOWFLAKE) -> Dict[str, Any]:
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
        logger.warning(f"Invalid SQL input: {error_msg}")
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


def extract_enhanced_metadata_from_ddl(sql: str, dialect: str = DBType.SNOWFLAKE) -> Dict[str, Any]:
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
    is_valid, error_msg = validate_sql_input(sql)
    if not is_valid:
        logger.warning(f"Invalid SQL input: {error_msg}")
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

    try:
        # Parse SQL using sqlglot with error handling
        parsed = sqlglot.parse_one(sql.strip(), dialect=dialect, error_level=sqlglot.ErrorLevel.IGNORE)

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
                table_comment_matches = _TABLE_COMMENT_RE.findall(sql)
                if table_comment_matches:
                    is_valid, _, comment = validate_comment(table_comment_matches[-1])
                    if is_valid:
                        result["table"]["comment"] = comment

            # If we got here, sqlglot parsing was successful
            if result["table"]["name"] and result["columns"]:
                return result

    except Exception as e:
        ddl_preview = sql[:200] if len(sql) > 200 else sql
        logger.warning(
            f"Error parsing SQL with sqlglot (dialect={dialect}): {e}\n"
            f"DDL preview: {ddl_preview}..."
        )

    # Fallback: Use regex-based parsing for StarRocks/MySQL-style DDL
    logger.info(f"Falling back to regex parsing for dialect: {dialect}")
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

    return result


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

        # Find the content between the first ( and the matching )
        paren_count = 0
        start_idx = -1
        end_idx = -1

        for i, char in enumerate(sql):
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

            # Split by comma, handling nested parentheses
            column_defs = []
            current_def = ""
            paren_depth = 0

            for char in columns_text:
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

                    col_type = col_match.group(2)

                    # Validate type length
                    if len(col_type) > MAX_TYPE_DEFINITION_LENGTH:
                        logger.warning(f"Type definition too long: {col_type}")
                        continue

                    is_nullable = col_match.group(3) is None or (col_match.group(3) and 'NULL' in col_match.group(3).upper())

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


# =============================================================================
# ADDITIONAL UTILITY FUNCTIONS
# =============================================================================

def extract_table_names(sql, dialect=DBType.SNOWFLAKE, ignore_empty=False) -> List[str]:
    """Extract fully qualified table names from SQL."""
    is_valid, error_msg = validate_sql_input(sql)
    if not is_valid:
        logger.warning(f"Invalid SQL input: {error_msg}")
        return []

    read_dialect = parse_read_dialect(dialect)
    try:
        parsed = sqlglot.parse_one(sql, read=read_dialect, error_level=sqlglot.ErrorLevel.IGNORE)
        if parsed is None:
            return []
    except Exception as e:
        logger.warning(f"Error parsing SQL: {e}")
        return []

    table_names = []
    cte_names = set()

    for cte in parsed.find_all(CTE):
        if hasattr(cte, "alias") and cte.alias:
            cte_names.add(cte.alias.lower())

    for tb in parsed.find_all(Table):
        if tb.name.lower() in cte_names:
            continue

        db = tb.catalog
        schema = tb.db
        table_name = tb.name

        full_name = []
        if dialect in [DBType.MYSQL, DBType.ORACLE, DBType.POSTGRES, DBType.POSTGRESQL]:
            if not ignore_empty or schema:
                full_name.append(schema)
        elif dialect not in (DBType.SQLITE,):
            if not ignore_empty or db:
                full_name.append(db)
            if not ignore_empty or schema:
                full_name.append(schema)
        full_name.append(table_name)

        table_names.append(".".join(full_name))

    return list(set(table_names))
