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
# Production-optimized limits: 500KB for DDL is sufficient for typical table definitions
# (Most real-world DDLs are under 100KB; 500KB provides safety margin)
MAX_SQL_LENGTH = 512000  # 500KB max DDL size
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

# DDL cleanup patterns for _clean_ddl()
_UNCLOSED_COMMENT_RE = re.compile(r"COMMENT\s*['\"]([^'\"]*)$", re.IGNORECASE | re.MULTILINE)

# Comment sanitization patterns for validate_comment()
_SCRIPT_TAG_RE = re.compile(r'<script[^>]*>.*?</script>', re.IGNORECASE | re.DOTALL)
_HTML_TAG_RE = re.compile(r'<[^>]+>')
_JAVASCRIPT_PROTOCOL_RE = re.compile(r'javascript\s*:', re.IGNORECASE)
_EVENT_HANDLER_RE = re.compile(r'\s*on\w+\s*=\s*[^>\s]*', re.IGNORECASE)

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

    # Additional validation for CREATE TABLE statements
    # Check if it's a truncated CREATE TABLE statement
    sql_upper = sql.upper().strip()
    if sql_upper.startswith("CREATE TABLE"):
        # If the statement seems truncated (ends with incomplete comment or doesn't end with semicolon)
        if not sql.rstrip().endswith(';') and ' COMMENT ' in sql:
            # This might be a truncated DDL with incomplete comment
            # Don't reject it, just warn
            logger.debug(f"Potentially truncated DDL detected: {sql[:100]}...")

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

    # Remove HTML/Script tags and their content using pre-compiled patterns
    sanitized = _SCRIPT_TAG_RE.sub('', sanitized)
    sanitized = _HTML_TAG_RE.sub('', sanitized)  # Remove remaining HTML tags

    # Remove JavaScript protocol and event handlers
    sanitized = _JAVASCRIPT_PROTOCOL_RE.sub('', sanitized)
    sanitized = _EVENT_HANDLER_RE.sub('', sanitized)

    # If significant content was removed, flag it as potentially unsafe
    # but still allow the sanitized version
    tag_count = len(_HTML_TAG_RE.findall(comment))
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
    cleaned = _UNCLOSED_COMMENT_RE.sub("", cleaned)

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


def metadata_identifier(
    dialect: str,
    catalog_name: str = "",
    database_name: str = "",
    schema_name: str = "",
    table_name: str = ""
) -> str:
    """
    Create a unique identifier for a database table.

    This function creates a fully-qualified table identifier in the format:
    catalog.database.schema.table

    Empty components are represented by empty strings, which results in
    consecutive dots (e.g., "catalog.database..table").

    Args:
        dialect: SQL dialect (e.g., 'snowflake', 'mysql', 'postgresql')
        catalog_name: Catalog name (optional, depends on dialect)
        database_name: Database name (optional, depends on dialect)
        schema_name: Schema name (optional, depends on dialect)
        table_name: Table name (required)

    Returns:
        A fully-qualified table identifier string

    Examples:
        >>> metadata_identifier("snowflake", "", "db", "schema", "table")
        'db.schema.table'
        >>> metadata_identifier("starrocks", "cat", "db", "", "table")
        'cat.db..table'
    """
    # Ensure table_name is provided
    if not table_name:
        raise ValueError("table_name is required")

    # Build the identifier from right to left (table is always present)
    parts = []

    # Add table name
    parts.append(table_name)

    # Add schema name (if present)
    if schema_name:
        parts.insert(0, schema_name)

    # Add database name (if present)
    if database_name:
        parts.insert(0, database_name)

    # Add catalog name (if present)
    if catalog_name:
        parts.insert(0, catalog_name)

    # Join with dots - empty components will create consecutive dots
    return ".".join(parts)


def parse_sql_type(sql: str, dialect: str = DBType.SNOWFLAKE) -> SQLType:
    """
    Parse SQL statement and determine its type.

    This function analyzes a SQL query and returns its type based on the first keyword.
    It handles various SQL statement types including SELECT, INSERT, UPDATE, DELETE,
    DDL statements, metadata queries, and more.

    Args:
        sql: SQL query string to analyze
        dialect: SQL dialect (used for dialect-specific parsing if needed)

    Returns:
        SQLType enum value representing the statement type

    Examples:
        >>> parse_sql_type("SELECT * FROM users")
        <SQLType.SELECT: 'select'>
        >>> parse_sql_type("INSERT INTO users VALUES (1, 'John')")
        <SQLType.INSERT: 'insert'>
        >>> parse_sql_type("CREATE TABLE users (id INT)")
        <SQLType.DDL: 'ddl'>
        >>> parse_sql_type("SHOW TABLES")
        <SQLType.METADATA_SHOW: 'metadata'>
    """
    if not sql or not sql.strip():
        return SQLType.UNKNOWN

    # Clean the SQL: remove leading/trailing whitespace and comments
    sql_clean = sql.strip()

    # Remove leading SQL comments (-- and /* */ style)
    # Handle multi-line comments
    sql_clean = re.sub(r'/\*.*?\*/', '', sql_clean, flags=re.DOTALL)
    # Handle single-line comments
    sql_clean = re.sub(r'--.*$', '', sql_clean, flags=re.MULTILINE)
    # Clean up extra whitespace
    sql_clean = sql_clean.strip()

    if not sql_clean:
        return SQLType.UNKNOWN

    # Get the first meaningful keyword
    # Find the first word that is not whitespace or parentheses
    first_word_match = re.match(r'^\s*([A-Za-z]+)', sql_clean, re.IGNORECASE)
    if not first_word_match:
        return SQLType.UNKNOWN

    first_keyword = first_word_match.group(1).upper()

    # Map of SQL keywords to SQLType
    keyword_map = {
        # Data Query Language (DQL)
        'SELECT': SQLType.SELECT,
        'WITH': SQLType.SELECT,  # CTEs are part of SELECT

        # Data Manipulation Language (DML)
        'INSERT': SQLType.INSERT,
        'UPDATE': SQLType.UPDATE,
        'DELETE': SQLType.DELETE,
        'MERGE': SQLType.MERGE,
        'UPSERT': SQLType.INSERT,  # UPSERT is essentially INSERT with ON CONFLICT

        # Data Definition Language (DDL)
        'CREATE': SQLType.DDL,
        'ALTER': SQLType.DDL,
        'DROP': SQLType.DDL,
        'TRUNCATE': SQLType.DDL,
        'RENAME': SQLType.DDL,

        # Metadata/Utility Commands
        'SHOW': SQLType.METADATA_SHOW,
        'DESCRIBE': SQLType.METADATA_SHOW,
        'DESC': SQLType.METADATA_SHOW,
        'EXPLAIN': SQLType.EXPLAIN,
        'USE': SQLType.CONTENT_SET,  # Database/schema selection

        # Control Commands
        'GRANT': SQLType.DDL,
        'REVOKE': SQLType.DDL,
    }

    # Direct keyword match
    if first_keyword in keyword_map:
        return keyword_map[first_keyword]

    # Special handling for complex queries
    # Check for EXPLAIN with SELECT
    if first_keyword == 'EXPLAIN':
        # Check what comes after EXPLAIN
        after_explain = sql_clean[7:].strip()
        explain_match = re.match(r'^\s*([A-Za-z]+)', after_explain, re.IGNORECASE)
        if explain_match:
            next_keyword = explain_match.group(1).upper()
            if next_keyword == 'SELECT':
                return SQLType.EXPLAIN
            # Other EXPLAIN variants (ANALYZE, FORMAT, etc.) are still EXPLAIN type
            return SQLType.EXPLAIN

    # Check for SELECT after WITH (CTE)
    if first_keyword == 'WITH':
        # This is a CTE which is part of a SELECT statement
        return SQLType.SELECT

    # For Snowflake-specific MERGE syntax
    if 'MERGE' in sql_clean.upper():
        return SQLType.MERGE

    # If we can't determine the type, return UNKNOWN
    return SQLType.UNKNOWN


def parse_context_switch(sql: str, dialect: str = DBType.SNOWFLAKE) -> Optional[Dict[str, Any]]:
    """
    Parse database context switch commands (USE, SET) to extract database/schema/catalog information.

    This function parses SQL commands that switch database context, such as:
    - USE database_name (MySQL, DuckDB)
    - USE catalog.database (StarRocks)
    - USE database.schema (Snowflake)
    - SET catalog catalog_name (StarRocks)

    Args:
        sql: SQL command string to parse
        dialect: SQL dialect (affects how context switches are interpreted)

    Returns:
        Dictionary containing:
        - command: Command type ("USE" or "SET")
        - target: What is being switched ("database", "schema", or "catalog")
        - catalog_name: Catalog name (if applicable)
        - database_name: Database name (if applicable)
        - schema_name: Schema name (if applicable)
        - fuzzy: Whether the context switch is ambiguous
        - raw: Original SQL command
        Or None if the command is not a context switch

    Examples:
        >>> parse_context_switch("USE analytics", dialect=DBType.DUCKDB)
        {
            "command": "USE",
            "target": "schema",
            "catalog_name": "",
            "database_name": "",
            "schema_name": "analytics",
            "fuzzy": True,
            "raw": "USE analytics"
        }
        >>> parse_context_switch("USE lakehouse.sales", dialect=DBType.STARROCKS)
        {
            "command": "USE",
            "target": "database",
            "catalog_name": "lakehouse",
            "database_name": "sales",
            "schema_name": "",
            "fuzzy": False,
            "raw": "USE lakehouse.sales"
        }
    """
    if not sql or not sql.strip():
        return None

    # Clean the SQL: remove leading/trailing whitespace
    sql_clean = sql.strip()

    # Normalize case for parsing
    sql_upper = sql_clean.upper()

    # Check for context switch commands
    if sql_upper.startswith("USE "):
        return _parse_use_command(sql_clean, dialect)
    elif sql_upper.startswith("SET "):
        return _parse_set_command(sql_clean, dialect)

    return None


def _parse_use_command(sql: str, dialect: str) -> Dict[str, Any]:
    """Parse USE command to extract context information."""
    # Extract the part after "USE"
    # Supports: USE database, USE catalog.database, USE database.schema
    use_pattern = re.match(r'^\s*USE\s+(.+?)\s*$', sql, re.IGNORECASE)
    if not use_pattern:
        return None

    target_str = use_pattern.group(1).strip()

    # Remove quotes if present
    if (target_str.startswith('`') and target_str.endswith('`')) or \
       (target_str.startswith('"') and target_str.endswith('"')) or \
       (target_str.startswith("'") and target_str.endswith("'")):
        target_str = target_str[1:-1]

    # Parse based on dialect
    if dialect == DBType.MYSQL:
        # MySQL: USE database
        return {
            "command": "USE",
            "target": "database",
            "catalog_name": "",
            "database_name": target_str,
            "schema_name": "",
            "fuzzy": False,
            "raw": sql.strip()
        }

    elif dialect == DBType.DUCKDB:
        # DuckDB: USE schema or USE database.schema
        if "." in target_str:
            # database.schema format
            parts = target_str.split(".", 1)
            return {
                "command": "USE",
                "target": "schema",
                "catalog_name": "",
                "database_name": parts[0],
                "schema_name": parts[1],
                "fuzzy": False,
                "raw": sql.strip()
            }
        else:
            # Just schema name - fuzzy because DuckDB treats schemas like databases
            return {
                "command": "USE",
                "target": "schema",
                "catalog_name": "",
                "database_name": "",
                "schema_name": target_str,
                "fuzzy": True,
                "raw": sql.strip()
            }

    elif dialect == DBType.STARROCKS:
        # StarRocks: USE catalog.database or USE database
        if "." in target_str:
            # catalog.database format
            parts = target_str.split(".", 1)
            return {
                "command": "USE",
                "target": "database",
                "catalog_name": parts[0],
                "database_name": parts[1],
                "schema_name": "",
                "fuzzy": False,
                "raw": sql.strip()
            }
        else:
            # Just database name
            return {
                "command": "USE",
                "target": "database",
                "catalog_name": "",
                "database_name": target_str,
                "schema_name": "",
                "fuzzy": False,
                "raw": sql.strip()
            }

    elif dialect in [DBType.SNOWFLAKE, DBType.BIGQUERY]:
        # Snowflake/BigQuery: USE database.schema
        if "." in target_str:
            # database.schema format
            parts = target_str.split(".", 1)
            return {
                "command": "USE",
                "target": "schema",
                "catalog_name": "",
                "database_name": parts[0],
                "schema_name": parts[1],
                "fuzzy": False,
                "raw": sql.strip()
            }
        else:
            # Just database name
            return {
                "command": "USE",
                "target": "database",
                "catalog_name": "",
                "database_name": target_str,
                "schema_name": "",
                "fuzzy": False,
                "raw": sql.strip()
            }

    # Default fallback
    return {
        "command": "USE",
        "target": "database",
        "catalog_name": "",
        "database_name": target_str,
        "schema_name": "",
        "fuzzy": False,
        "raw": sql.strip()
    }


def _parse_set_command(sql: str, dialect: str) -> Optional[Dict[str, Any]]:
    """Parse SET command to extract context information."""
    # Extract the part after "SET"
    # StarRocks supports: SET CATALOG catalog_name
    set_pattern = re.match(r'^\s*SET\s+(.+?)\s*$', sql, re.IGNORECASE)
    if not set_pattern:
        return None

    set_parts = set_pattern.group(1).strip().split(None, 1)

    if len(set_parts) < 2:
        return None

    keyword = set_parts[0].upper()
    value = set_parts[1].strip()

    # Remove quotes if present
    if (value.startswith('`') and value.endswith('`')) or \
       (value.startswith('"') and value.endswith('"')) or \
       (value.startswith("'") and value.endswith("'")):
        value = value[1:-1]

    # Handle SET CATALOG for StarRocks
    if keyword == "CATALOG" and dialect == DBType.STARROCKS:
        return {
            "command": "SET",
            "target": "catalog",
            "catalog_name": value,
            "database_name": "",
            "schema_name": "",
            "fuzzy": False,
            "raw": sql.strip()
        }

    # Other SET commands are not context switches
    return None
