# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Core SQL utility functions.

This module provides essential utility functions for SQL analysis, including
table name extraction, identifier generation, SQL type parsing, and
context switching detection.
"""

import re
from typing import Any, Dict, List, Optional, Set

import sqlglot
from sqlglot import exp
from sqlglot.expressions import CTE, Subquery, Table

from datus.utils.constants import DBType, SQLType
from datus.utils.loggings import get_logger

from .dialect_support import parse_dialect, parse_read_dialect
from .validation import validate_sql_input

logger = get_logger(__name__)


# =============================================================================
# ADDITIONAL UTILITY FUNCTIONS
# =============================================================================

def strip_sql_comments(sql: str) -> str:
    """Remove /* ... */ and -- ... comments (simple but effective)."""
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    sql = re.sub(r"--.*?$", " ", sql, flags=re.MULTILINE)
    return sql


def _is_escaped(text: str, index: int) -> bool:
    """Return True if the character at index is escaped by an odd number of backslashes."""
    backslash_count = 0
    position = index - 1
    while position >= 0 and text[position] == "\\":
        backslash_count += 1
        position -= 1
    return backslash_count % 2 == 1


_DOLLAR_QUOTE_RE = re.compile(r"\$[A-Za-z_0-9]*\$")


def _match_dollar_tag(text: str, index: int) -> Optional[str]:
    """Return the dollar-quote tag starting at index, if any."""
    match = _DOLLAR_QUOTE_RE.match(text, index)
    if not match:
        return None
    return match.group(0)


def _first_statement(sql: str) -> str:
    """Return the first non-empty statement (before the first ';'), with comments removed."""
    s = strip_sql_comments(sql).strip()
    if not s:
        return ""

    in_single_quote = False
    in_double_quote = False
    in_backtick = False
    in_bracket = False
    dollar_tag: Optional[str] = None

    i = 0
    length = len(s)
    while i < length:
        ch = s[i]

        if dollar_tag:
            if s.startswith(dollar_tag, i):
                i += len(dollar_tag)
                dollar_tag = None
                continue
            i += 1
            continue

        if in_single_quote:
            if ch == "'":
                if i + 1 < length and s[i + 1] == "'":
                    i += 2
                    continue
                if not _is_escaped(s, i):
                    in_single_quote = False
            i += 1
            continue

        if in_double_quote:
            if ch == '"':
                if i + 1 < length and s[i + 1] == '"':
                    i += 2
                    continue
                if not _is_escaped(s, i):
                    in_double_quote = False
            i += 1
            continue

        if in_backtick:
            if ch == "`":
                if i + 1 < length and s[i + 1] == "`":
                    i += 2
                    continue
                in_backtick = False
            i += 1
            continue

        if in_bracket:
            if ch == "]":
                in_bracket = False
            i += 1
            continue

        if ch == "'":
            in_single_quote = True
            i += 1
            continue
        if ch == '"':
            in_double_quote = True
            i += 1
            continue
        if ch == "`":
            in_backtick = True
            i += 1
            continue
        if ch == "[":
            in_bracket = True
            i += 1
            continue
        if ch == "$":
            tag = _match_dollar_tag(s, i)
            if tag:
                dollar_tag = tag
                i += len(tag)
                continue

        if ch == ";":
            return s[:i].strip()

        i += 1

    return s.strip()


def normalize_sql(sql: str) -> str:
    """Normalize SQL text by removing extra whitespace and line breaks."""
    if not sql:
        return sql
    s = re.sub(r"[\r\n\t]+", " ", sql)
    s = re.sub(r" +", " ", s)
    return s.strip()


def format_sql_to_pretty(sql: str, dialect: str) -> str:
    """Pretty print SQL if possible, otherwise return the original text."""
    if not sql:
        return sql
    read_dialect = parse_read_dialect(dialect)
    try:
        formatted = sqlglot.transpile(sql, read=read_dialect, pretty=True)
        if formatted:
            return formatted[0]
    except Exception as exc:
        logger.debug(f"Failed to format SQL for download: {exc}")
    return sql


def parse_table_name_parts(full_table_name: str, dialect: str = DBType.SNOWFLAKE) -> Dict[str, str]:
    """
    Parse a full table name into its component parts (catalog, database, schema, table).
    """
    dialect = parse_dialect(dialect)
    parts = full_table_name.split(".") if full_table_name else []
    result = {"catalog_name": "", "database_name": "", "schema_name": "", "table_name": ""}

    if dialect == "duckdb":
        if len(parts) == 1:
            result["table_name"] = parts[0]
        elif len(parts) == 2:
            result["schema_name"] = parts[0]
            result["table_name"] = parts[1]
        elif len(parts) == 3:
            result["database_name"] = parts[0]
            result["schema_name"] = parts[1]
            result["table_name"] = parts[2]
        else:
            result["database_name"] = parts[-3] if len(parts) > 2 else ""
            result["schema_name"] = parts[-2] if len(parts) > 1 else ""
            result["table_name"] = parts[-1] if parts else ""

    elif dialect == "sqlite":
        if len(parts) == 1:
            result["table_name"] = parts[0]
        elif len(parts) == 2:
            result["database_name"] = parts[0]
            result["table_name"] = parts[1]
        else:
            result["database_name"] = parts[-2] if len(parts) > 1 else ""
            result["table_name"] = parts[-1] if parts else ""

    elif dialect == "starrocks":
        if len(parts) == 1:
            result["table_name"] = parts[0]
        elif len(parts) == 2:
            result["database_name"] = parts[0]
            result["table_name"] = parts[1]
        elif len(parts) == 3:
            result["catalog_name"] = parts[0]
            result["database_name"] = parts[1]
            result["table_name"] = parts[2]
        else:
            result["catalog_name"] = parts[-3] if len(parts) > 2 else ""
            result["database_name"] = parts[-2] if len(parts) > 1 else ""
            result["table_name"] = parts[-1] if parts else ""

    elif dialect == "snowflake":
        if len(parts) == 1:
            result["table_name"] = parts[0]
        elif len(parts) == 2:
            result["schema_name"] = parts[0]
            result["table_name"] = parts[1]
        elif len(parts) == 3:
            result["database_name"] = parts[0]
            result["schema_name"] = parts[1]
            result["table_name"] = parts[2]
        elif len(parts) == 4:
            result["catalog_name"] = parts[0]
            result["database_name"] = parts[1]
            result["schema_name"] = parts[2]
            result["table_name"] = parts[3]
        else:
            result["catalog_name"] = parts[-4] if len(parts) > 3 else ""
            result["database_name"] = parts[-3] if len(parts) > 2 else ""
            result["schema_name"] = parts[-2] if len(parts) > 1 else ""
            result["table_name"] = parts[-1] if parts else ""
    else:
        result["table_name"] = parts[-1] if parts else ""
        if len(parts) > 1:
            result["schema_name"] = parts[-2]
        if len(parts) > 2:
            result["database_name"] = parts[-3]
        if len(parts) > 3:
            result["catalog_name"] = parts[-4]

    return result


def parse_table_names_parts(full_table_names: List[str], dialect: str = DBType.SNOWFLAKE) -> List[Dict[str, str]]:
    """Parse a list of full table names into their component parts."""
    return [parse_table_name_parts(table_name, dialect) for table_name in full_table_names]

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


def extract_sql_symbols(sql: str, dialect: str = DBType.SNOWFLAKE) -> Dict[str, Any]:
    """Extract SQL symbols for validation/reporting.

    Returns:
        {
            "alias_to_table": Dict[str, str],
            "virtual_tables": Dict[str, Set[str]],
            "virtual_columns": Set[str],
        }
    """
    is_valid, error_msg = validate_sql_input(sql)
    if not is_valid:
        logger.warning(f"Invalid SQL input: {error_msg}")
        return {"alias_to_table": {}, "virtual_tables": {}, "virtual_columns": set()}

    read_dialect = parse_read_dialect(dialect)
    try:
        parsed = sqlglot.parse_one(sql, read=read_dialect, error_level=sqlglot.ErrorLevel.IGNORE)
        if parsed is None:
            return {"alias_to_table": {}, "virtual_tables": {}, "virtual_columns": set()}
    except Exception as e:
        logger.warning(f"Error parsing SQL: {e}")
        return {"alias_to_table": {}, "virtual_tables": {}, "virtual_columns": set()}

    def _alias_name(node: Any) -> Optional[str]:
        alias = getattr(node, "alias", None)
        if alias:
            name = getattr(alias, "name", None)
            if name:
                return str(name)
            alias_this = getattr(alias, "this", None)
            if alias_this is not None:
                return getattr(alias_this, "name", None) or str(alias_this)
        alias_or_name = getattr(node, "alias_or_name", None)
        if alias_or_name:
            return str(alias_or_name)
        return None

    def _output_columns_from_select(select_expr: Any) -> Set[str]:
        cols: Set[str] = set()
        if not select_expr:
            return cols
        for projection in getattr(select_expr, "expressions", []) or []:
            if isinstance(projection, exp.Alias):
                alias_name = projection.alias
                if alias_name:
                    cols.add(str(alias_name).lower())
                    continue
            alias_or_name = getattr(projection, "alias_or_name", None)
            if alias_or_name:
                cols.add(str(alias_or_name).lower())
                continue
            if isinstance(projection, exp.Column) and projection.name:
                cols.add(str(projection.name).lower())
        return cols

    alias_to_table: Dict[str, str] = {}
    virtual_tables: Dict[str, Set[str]] = {}

    for table in parsed.find_all(Table):
        table_name = table.name
        if not table_name:
            continue
        alias = _alias_name(table)
        if alias:
            alias_to_table[alias.lower()] = table_name.lower()

    for cte in parsed.find_all(CTE):
        cte_name = _alias_name(cte)
        if not cte_name:
            continue
        cte_cols: Set[str] = set()
        alias = getattr(cte, "alias", None)
        alias_columns = getattr(alias, "columns", None) if alias else None
        if alias_columns:
            for col in alias_columns:
                col_name = getattr(col, "name", None) or str(col)
                if col_name:
                    cte_cols.add(str(col_name).lower())
        if not cte_cols:
            cte_cols = _output_columns_from_select(getattr(cte, "this", None))
        if cte_cols:
            virtual_tables[cte_name.lower()] = cte_cols

    for subquery in parsed.find_all(Subquery):
        alias = _alias_name(subquery)
        if not alias:
            continue
        sub_cols = _output_columns_from_select(getattr(subquery, "this", None))
        if sub_cols:
            virtual_tables[alias.lower()] = sub_cols

    virtual_columns: Set[str] = set()
    for cols in virtual_tables.values():
        virtual_columns.update(cols)

    return {
        "alias_to_table": alias_to_table,
        "virtual_tables": virtual_tables,
        "virtual_columns": virtual_columns,
    }


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


def validate_and_suggest_sql_fixes(sql: str, dialect: str = DBType.SNOWFLAKE) -> Dict[str, Any]:
    """
    Validate SQL syntax using sqlglot and provide fix suggestions.

    Args:
        sql: SQL query string
        dialect: Database dialect

    Returns:
        Dict containing validation results:
        - can_parse: bool
        - errors: List[str]
        - warnings: List[str]
        - fix_suggestions: List[Dict]
    """
    result = {
        "can_parse": True,
        "errors": [],
        "warnings": [],
        "fix_suggestions": [],
    }

    if not sql or not sql.strip():
        result["can_parse"] = False
        result["errors"].append("Empty SQL query")
        return result

    # Validate input length and basic safety
    is_valid, error_msg = validate_sql_input(sql)
    if not is_valid:
        result["can_parse"] = False
        result["errors"].append(error_msg)
        return result

    read_dialect = parse_read_dialect(dialect)

    try:
        # Parse with sqlglot to check for syntax errors
        # parse returns a list of expressions. If it raises, it's invalid.
        sqlglot.parse(sql, read=read_dialect)

    except sqlglot.errors.ParseError as e:
        result["can_parse"] = False
        # Extract error details
        error_str = str(e)
        result["errors"].append(f"Syntax error: {error_str}")

        # Add basic fix suggestion based on error
        suggestion = {
            "type": "syntax_error",
            "description": "Check SQL syntax matches the target dialect",
            "original_error": error_str
        }

        # Try to transpile to standard SQL to see if it fixes it (simple suggestion)
        try:
            transpiled = sqlglot.transpile(sql, read=read_dialect, write=read_dialect)[0]
            if transpiled != sql:
                suggestion["suggested_fix"] = transpiled
        except:
            pass

        result["fix_suggestions"].append(suggestion)

    except Exception as e:
        result["can_parse"] = False
        result["errors"].append(f"Validation error: {str(e)}")

    return result
