# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
SQL Utilities Module

This module provides comprehensive SQL parsing, validation, and manipulation utilities.
It has been split into submodules for better maintainability:

- validation: Input validation functions
- enum_utils: Enum value extraction from comments
- ddl_cleaner: DDL cleaning and repair functions
- dialect_support: Database dialect-specific support
- ddl_parser: DDL parsing and metadata extraction
- core: Core utility functions
"""

# Re-export all public functions for backward compatibility
from .validation import (
    validate_sql_input,
    validate_table_name,
    quote_identifier,
    validate_comment,
)

from .enum_utils import (
    extract_enum_values_from_comment,
)

from .ddl_cleaner import (
    fix_truncated_ddl,
    is_likely_truncated_ddl,
    sanitize_ddl_for_storage,
)

from .dialect_support import (
    extract_starrocks_properties,
    parse_read_dialect,
    parse_dialect,
)

from .ddl_parser import (
    parse_metadata_from_ddl,
    extract_enhanced_metadata_from_ddl,
    extract_metadata_from_ddl_regex_only,
)

from .core import (
    extract_table_names,
    metadata_identifier,
    parse_sql_type,
    parse_context_switch,
    validate_and_suggest_sql_fixes,
)

# Re-export constants
from .validation import (
    MAX_SQL_LENGTH,
    MAX_COLUMN_NAME_LENGTH,
    MAX_TABLE_NAME_LENGTH,
    MAX_COMMENT_LENGTH,
    MAX_TYPE_DEFINITION_LENGTH,
    MAX_PAREN_DEPTH,
)

__all__ = [
    # Validation functions
    "validate_sql_input",
    "validate_table_name",
    "quote_identifier",
    "validate_comment",
    # Enum utilities
    "extract_enum_values_from_comment",
    # DDL cleaner
    "fix_truncated_ddl",
    "is_likely_truncated_ddl",
    "sanitize_ddl_for_storage",
    # Dialect support
    "extract_starrocks_properties",
    "parse_read_dialect",
    "parse_dialect",
    # DDL parser
    "parse_metadata_from_ddl",
    "extract_enhanced_metadata_from_ddl",
    "extract_metadata_from_ddl_regex_only",
    # Core utilities
    "extract_table_names",
    "metadata_identifier",
    "parse_sql_type",
    "parse_context_switch",
    "validate_and_suggest_sql_fixes",
    # Constants
    "MAX_SQL_LENGTH",
    "MAX_COLUMN_NAME_LENGTH",
    "MAX_TABLE_NAME_LENGTH",
    "MAX_COMMENT_LENGTH",
    "MAX_TYPE_DEFINITION_LENGTH",
    "MAX_PAREN_DEPTH",
]
