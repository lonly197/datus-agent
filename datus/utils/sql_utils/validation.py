# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Input validation functions for SQL utilities.

This module provides validation functions for SQL inputs, table names,
identifiers, and comments to ensure security and correctness.
"""

import re
from typing import Any, Optional, Tuple

from datus.utils.loggings import get_logger

logger = get_logger(__name__)

# =============================================================================
# PRE-COMPILED REGEX PATTERNS FOR PERFORMANCE
# =============================================================================

# Maximum lengths for ReDoS prevention and input validation
# Production-optimized limits: 500KB for DDL is sufficient for typical table definitions
# (Most real-world DDLs are under 100KB; 500KB provides safety margin)
MAX_SQL_LENGTH = 512000  # 500KB max DDL size
MAX_COLUMN_NAME_LENGTH = 128
MAX_TABLE_NAME_LENGTH = 256
MAX_COMMENT_LENGTH = 4000
MAX_TYPE_DEFINITION_LENGTH = 256
MAX_PAREN_DEPTH = 100  # Reasonable limit for DDL nested parentheses

# Comment sanitization patterns
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
    in_single_quote = False
    in_double_quote = False
    in_backtick = False
    in_line_comment = False
    in_block_comment = False
    escaped = False
    prev_char = ""
    for char in sql:
        if in_line_comment:
            if char == "\n":
                in_line_comment = False
            prev_char = char
            continue
        if in_block_comment:
            if prev_char == "*" and char == "/":
                in_block_comment = False
            prev_char = char
            continue
        if escaped:
            escaped = False
            prev_char = char
            continue
        if char == "\\":
            escaped = True
            prev_char = char
            continue
        if in_single_quote:
            if char == "'":
                in_single_quote = False
            prev_char = char
            continue
        if in_double_quote:
            if char == '"':
                in_double_quote = False
            prev_char = char
            continue
        if in_backtick:
            if char == "`":
                in_backtick = False
            prev_char = char
            continue
        if prev_char == "-" and char == "-":
            in_line_comment = True
            prev_char = char
            continue
        if prev_char == "/" and char == "*":
            in_block_comment = True
            prev_char = char
            continue
        if char == "'":
            in_single_quote = True
            prev_char = char
            continue
        if char == '"':
            in_double_quote = True
            prev_char = char
            continue
        if char == "`":
            in_backtick = True
            prev_char = char
            continue
        if char == '(':
            paren_depth += 1
            if paren_depth > MAX_PAREN_DEPTH:
                return False, f"Excessive nested parentheses (depth {paren_depth})"
        elif char == ')':
            paren_depth -= 1
            if paren_depth < 0:
                return False, "Unbalanced parentheses in SQL"
        prev_char = char

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


def quote_identifier(identifier: Optional[str], dialect: Any) -> Optional[str]:
    """
    Safely quote an identifier for a given SQL dialect.

    This is a defense-in-depth helper used to prevent SQL injection when
    constructing SQL strings from untrusted input.
    """
    if identifier is None:
        return None

    if identifier == "":
        return ""

    from datus.utils.constants import DBType

    if isinstance(dialect, DBType):
        dialect_name = dialect.value
    else:
        dialect_name = str(dialect).lower() if dialect is not None else ""

    backtick_dialects = {DBType.MYSQL.value}
    bracket_dialects = {DBType.MSSQL.value, DBType.SQLSERVER.value}

    if dialect_name in backtick_dialects:
        escaped = identifier.replace("`", "``")
        return f"`{escaped}`"

    if dialect_name in bracket_dialects:
        escaped = identifier.replace("]", "]]")
        return f"[{escaped}]"

    escaped = identifier.replace('"', '""')
    return f'"{escaped}"'


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
