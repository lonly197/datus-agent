# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
DDL cleaning and repair utilities.

This module provides functions for cleaning corrupted DDL statements,
fixing truncation issues, and normalizing DDL format for parsing.
"""

import re
from typing import Dict, Optional, Tuple

from datus.utils.loggings import get_logger

logger = get_logger(__name__)

# =============================================================================
# PRE-COMPILED REGEX PATTERNS FOR DDL CLEANING
# =============================================================================

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

_UNCLOSED_COMMENT_RE = re.compile(r"COMMENT\s*(?:=\s*)?['\"]([^'\"]*)$", re.IGNORECASE | re.MULTILINE)


# =============================================================================
# DDL CLEANUP FUNCTIONS
# =============================================================================

def _preserve_chinese_comments(sql: str) -> tuple:
    """
    Temporarily protect Chinese comment text during DDL cleaning.

    StarRocks uses double quotes for COMMENT text with Chinese characters.
    This function replaces those comment values with placeholders to prevent
    them from being removed by aggressive regex patterns.

    Args:
        sql: DDL string with potential Chinese comments

    Returns:
        Tuple of (modified_sql, protected_dict) where protected_dict maps
        placeholders to original COMMENT values
    """
    protected = {}
    pattern = re.compile(r'COMMENT\s*(?:=\s*)?"([^"]*)"', re.IGNORECASE)

    def replace_with_placeholder(match):
        content = match.group(0)
        placeholder = f"__CHINESE_COMMENT_{len(protected)}__"
        protected[placeholder] = content
        return f'COMMENT "{placeholder}"'

    result = pattern.sub(replace_with_placeholder, sql)
    return result, protected


def _restore_chinese_comments(sql: str, protected: dict) -> str:
    """
    Restore protected Chinese comment text after DDL cleaning.

    Args:
        sql: DDL string with placeholders
        protected: Dictionary mapping placeholders to original values

    Returns:
        DDL string with restored Chinese comments
    """
    result = sql
    for placeholder, original in protected.items():
        result = result.replace(f'COMMENT "{placeholder}"', original)
    return result


def _normalize_comment_quotes(sql: str) -> str:
    """
    Normalize COMMENT "text" to COMMENT 'text' for sqlglot parsing.

    StarRocks allows double-quoted comments, but sqlglot expects string literals.
    """
    pattern = re.compile(r'(COMMENT\s*(?:=\s*)?)"([^"]*)"', re.IGNORECASE)

    def replace(match):
        prefix = match.group(1)
        content = match.group(2).replace("'", "''")
        return f"{prefix}'{content}'"

    return pattern.sub(replace, sql)


def _clean_ddl(sql: str) -> str:
    """
    Clean corrupted DDL by removing common error message fragments and fixing syntax issues.

    This function handles DDL that may have been corrupted during storage or
    transfer, such as when error messages get appended to the DDL text.
    It also fixes common syntax issues like:
    - Unclosed quotes in comments
    - Mismatched quotes (e.g., " for opening and ' for closing)
    - Truncated column definitions
    - Incomplete statements

    Args:
        sql: Potentially corrupted DDL

    Returns:
        Cleaned DDL string
    """
    if not sql or not isinstance(sql, str):
        return sql

    cleaned = sql

    # Step 0: Preserve Chinese characters in double-quoted strings (StarRocks-specific)
    # StarRocks uses double quotes for COMMENT text with Chinese characters
    # We need to protect these from aggressive cleaning
    protected_content = {}
    has_chinese = 'COMMENT' in cleaned and any(ord(c) > 127 for c in cleaned)
    if has_chinese:
        # Has non-ASCII characters, protect them during cleaning
        cleaned, protected_content = _preserve_chinese_comments(cleaned)

    # Step 1: Fix mismatched quotes in comments
    # Pattern: COMMENT with mismatched quotes (e.g., " opening and ' closing)
    # This handles cases like: COMMENT "text'
    # SKIP this step if we have protected Chinese content to avoid corruption
    if not protected_content:
        mismatched_quote_pattern = re.compile(
            r'COMMENT\s*(["\'])[^"\']*\1?([^"\']*)$',
            re.IGNORECASE | re.MULTILINE
        )
        def fix_mismatched_quotes(match):
            opening_quote = match.group(1)
            content = match.group(2)
            # Close with the same quote type as opening
            return f'COMMENT {opening_quote}{content}{opening_quote}'

        cleaned = mismatched_quote_pattern.sub(fix_mismatched_quotes, cleaned)

    # Step 2: REMOVED - This was removing valid Chinese comments
    # The _UNCLOSED_COMMENT_RE pattern was too aggressive and removed complete comments
    # We now skip this step to preserve valid DDL content

    # Step 3: REMOVED - This pattern was too aggressive and removed valid content
    # The pattern r'[`\w]+$' was removing Chinese characters from COMMENT fields
    # We now skip this step to preserve valid DDL content

    # Step 4: Apply pre-compiled error message patterns
    for pattern in _ERROR_MESSAGE_PATTERNS:
        cleaned = pattern.sub("", cleaned)

    # Step 5: REMOVED - Skip unclosed quote fixing when we have Chinese content
    # The _UNCLOSED_QUOTE_RE pattern was removing valid Chinese comments
    # Be careful not to remove valid Chinese comments
    # if not protected_content:
    #     cleaned = _UNCLOSED_QUOTE_RE.sub("", cleaned)

    # Step 6: Remove trailing incomplete column definitions (conservative approach)
    # Only apply if we detect actual truncation (trailing comma, missing semicolon, etc.)
    # This prevents removing valid Chinese text from COMMENT fields
    trailing_incomplete_patterns = []
    ddl_stripped = cleaned.rstrip()
    if ddl_stripped.endswith(',') or not ddl_stripped.endswith(';'):
        # Only apply these patterns when actual truncation is detected
        trailing_incomplete_patterns = [
            re.compile(r'[`\s]*,[`\s]*$', re.MULTILINE),  # Only trailing comma + backticks
            re.compile(r',[`\s]*$', re.MULTILINE),  # Just trailing comma
        ]
    for pattern in trailing_incomplete_patterns:
        cleaned = pattern.sub("", cleaned)

    # Step 7: Restore protected Chinese comment text BEFORE other processing
    if protected_content:
        cleaned = _restore_chinese_comments(cleaned, protected_content)

    # Step 8: Normalize COMMENT quotes for parsing compatibility
    cleaned = _normalize_comment_quotes(cleaned)

    # Step 9: Clean up multiple spaces
    cleaned = _MULTI_SPACE_RE.sub(" ", cleaned)

    # Step 10: Final validation - if the SQL still looks incomplete, try to complete it
    cleaned = _complete_incomplete_ddl(cleaned)

    return cleaned.strip()


def fix_truncated_ddl(ddl: str) -> str:
    """
    Fix truncated DDL statements by detecting common truncation patterns and attempting completion.

    This function handles DDL that may have been truncated during retrieval from the database,
    such as when SHOW CREATE TABLE returns incomplete results due to length limits.

    Args:
        ddl: Potentially truncated DDL statement

    Returns:
        Fixed DDL statement or original if not recognized as truncated
    """
    if not ddl or not isinstance(ddl, str):
        return ddl

    stripped_ddl = ddl.rstrip()
    # Patterns that indicate truncation
    truncation_indicators = [
        stripped_ddl.endswith(','),  # Ends with comma (incomplete column list)
        not stripped_ddl.endswith(')'),  # Missing closing paren
    ]

    # Check if missing closing paren for CREATE TABLE
    open_parens = ddl.count('(')
    close_parens = ddl.count(')')
    missing_closing_paren = open_parens > close_parens

    # If any truncation indicators or missing closing paren, try to fix
    if sum(truncation_indicators) >= 1 or missing_closing_paren:
        fixed_ddl = ddl

        # Remove trailing comma if present
        if fixed_ddl.rstrip().endswith(','):
            fixed_ddl = fixed_ddl.rstrip()[:-1].rstrip()

        stripped_fixed = fixed_ddl.rstrip()
        if missing_closing_paren or not stripped_fixed.endswith(')'):
            if stripped_fixed.endswith(';'):
                fixed_ddl = stripped_fixed.rstrip(';')
            fixed_ddl += '\n)'

        fixed_upper = fixed_ddl.upper()
        starrocks_indicators = [
            'DUPLICATE KEY' in fixed_upper,
            'AGGREGATE KEY' in fixed_upper,
            'UNIQUE KEY' in fixed_upper,
            'PRIMARY KEY' in fixed_upper,
            'DISTRIBUTED BY' in fixed_upper,
            'PARTITION BY' in fixed_upper,
            'PROPERTIES' in fixed_upper,
        ]

        # Add basic StarRocks table structure if missing
        if 'ENGINE=' not in fixed_upper and any(starrocks_indicators):
            if not fixed_ddl.rstrip().endswith(';'):
                fixed_ddl += ' ENGINE=OLAP;'
            else:
                fixed_ddl = fixed_ddl.rstrip(';') + ' ENGINE=OLAP;'
        elif not fixed_ddl.rstrip().endswith(';'):
            fixed_ddl += ';'

        fixed_ddl = fixed_ddl.strip()

        if fixed_ddl != ddl:
            return fixed_ddl

    return ddl


def _find_create_table_columns_span(sql: str) -> Optional[Tuple[int, int]]:
    if not sql or not isinstance(sql, str):
        return None
    sql_upper = sql.upper()
    create_idx = sql_upper.find("CREATE TABLE")
    if create_idx < 0:
        return None
    paren_start = sql.find("(", create_idx)
    if paren_start < 0:
        return None

    paren_count = 0
    in_single_quote = False
    in_double_quote = False
    in_backtick = False
    escaped = False
    for idx in range(paren_start, len(sql)):
        char = sql[idx]
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
        if char == "(":
            if paren_count == 0:
                start_idx = idx + 1
            paren_count += 1
        elif char == ")":
            paren_count -= 1
            if paren_count == 0:
                return (start_idx, idx)
    return None


def _count_top_level_commas_and_backticks(segment: str) -> Tuple[int, int]:
    comma_count = 0
    backtick_count = 0
    depth = 0
    in_single_quote = False
    in_double_quote = False
    in_backtick = False
    escaped = False
    for char in segment:
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
            if depth == 0:
                backtick_count += 1
            continue
        if char == "(":
            depth += 1
        elif char == ")":
            depth = max(depth - 1, 0)
        elif char == "," and depth == 0:
            comma_count += 1
    return comma_count, backtick_count


def ddl_has_missing_commas(ddl: str) -> bool:
    if not ddl or not isinstance(ddl, str):
        return False
    span = _find_create_table_columns_span(ddl)
    if not span:
        return False
    start_idx, end_idx = span
    columns_text = ddl[start_idx:end_idx]
    comma_count, backtick_count = _count_top_level_commas_and_backticks(columns_text)
    return backtick_count >= 2 and comma_count == 0


def fix_missing_commas_in_ddl(ddl: str) -> str:
    if not ddl or not isinstance(ddl, str):
        return ddl
    span = _find_create_table_columns_span(ddl)
    if not span:
        return ddl
    if not ddl_has_missing_commas(ddl):
        return ddl

    start_idx, end_idx = span
    columns_text = ddl[start_idx:end_idx]
    fixed_columns = []
    depth = 0
    in_single_quote = False
    in_double_quote = False
    in_backtick = False
    escaped = False
    prev_non_space: Optional[str] = None

    for char in columns_text:
        if escaped:
            escaped = False
            fixed_columns.append(char)
            if not char.isspace():
                prev_non_space = char
            continue
        if char == "\\":
            escaped = True
            fixed_columns.append(char)
            continue
        if in_single_quote:
            if char == "'":
                in_single_quote = False
            fixed_columns.append(char)
            if not char.isspace():
                prev_non_space = char
            continue
        if in_double_quote:
            if char == '"':
                in_double_quote = False
            fixed_columns.append(char)
            if not char.isspace():
                prev_non_space = char
            continue
        if in_backtick:
            if char == "`":
                in_backtick = False
            fixed_columns.append(char)
            if not char.isspace():
                prev_non_space = char
            continue
        if char == "'":
            in_single_quote = True
            fixed_columns.append(char)
            prev_non_space = char
            continue
        if char == '"':
            in_double_quote = True
            fixed_columns.append(char)
            prev_non_space = char
            continue
        if char == "`":
            if depth == 0 and prev_non_space and prev_non_space not in (",", "("):
                fixed_columns.append(", ")
            in_backtick = True
            fixed_columns.append(char)
            prev_non_space = char
            continue
        if char == "(":
            depth += 1
            fixed_columns.append(char)
            prev_non_space = char
            continue
        if char == ")":
            depth = max(depth - 1, 0)
            fixed_columns.append(char)
            prev_non_space = char
            continue
        fixed_columns.append(char)
        if not char.isspace():
            prev_non_space = char

    fixed = "".join(fixed_columns)
    if fixed != columns_text:
        return ddl[:start_idx] + fixed + ddl[end_idx:]
    return ddl


def is_likely_truncated_ddl(ddl: str) -> bool:
    """Detect likely truncation indicators without modifying the DDL."""
    if not ddl or not isinstance(ddl, str):
        return False

    stripped_ddl = ddl.rstrip()
    if stripped_ddl.endswith(','):
        return True

    open_parens = ddl.count('(')
    close_parens = ddl.count(')')
    if open_parens > close_parens:
        return True

    return False


def sanitize_ddl_for_storage(ddl: str) -> str:
    """Fix and clean DDL before storing to ensure consistent downstream parsing."""
    if not ddl or not isinstance(ddl, str):
        return ddl

    fixed = fix_truncated_ddl(ddl)
    fixed = fix_missing_commas_in_ddl(fixed)
    cleaned = _clean_ddl(fixed)
    return cleaned


def _complete_incomplete_ddl(sql: str) -> str:
    """
    Attempt to complete incomplete DDL statements by adding missing closing parentheses.

    This function accounts for type annotations like varchar(1024) which contain
    parentheses that should not be counted as CREATE TABLE delimiters.

    Args:
        sql: Potentially incomplete DDL

    Returns:
        Completed DDL string
    """
    if not sql or not isinstance(sql, str):
        return sql

    stripped_sql = sql.rstrip()
    ddl_upper = stripped_sql.upper()

    # Check if DDL ends properly (with ); or just ;)
    ends_properly = stripped_sql.endswith(');') or (
        stripped_sql.endswith(';') and 'CREATE TABLE' in ddl_upper
    )
    if ends_properly:
        return sql

    # Indicators of actual truncation
    indicators = [
        stripped_sql.endswith(','),  # Ends with comma (incomplete column list)
        not stripped_sql.endswith(')'),  # No closing paren near end
    ]

    # Only complete if indicators suggest truncation
    if sum(indicators) >= 1:
        # Add one closing paren for CREATE TABLE if clearly missing
        # (not for each varchar() - those are already balanced in valid DDL)
        if not stripped_sql.endswith(')'):
            if stripped_sql.endswith(';'):
                sql = stripped_sql.rstrip(';')
            sql += '\n)'

        # Add ENGINE clause if completely missing and this looks like StarRocks
        stripped_sql = sql.rstrip()
        ddl_upper = stripped_sql.upper()
        starrocks_indicators = [
            'DUPLICATE KEY' in ddl_upper,
            'AGGREGATE KEY' in ddl_upper,
            'UNIQUE KEY' in ddl_upper,
            'PRIMARY KEY' in ddl_upper,
            'DISTRIBUTED BY' in ddl_upper,
            'PARTITION BY' in ddl_upper,
            'PROPERTIES' in ddl_upper,
        ]
        if 'ENGINE=' not in ddl_upper and any(starrocks_indicators):
            if not stripped_sql.endswith(';'):
                sql += ' ENGINE=OLAP;'
            else:
                sql = sql.rstrip(';') + ' ENGINE=OLAP;'
        # Add semicolon if missing
        elif not stripped_sql.endswith(';'):
            sql += ';'

    return sql
