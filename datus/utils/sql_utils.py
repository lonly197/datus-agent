# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
SQL Utilities - Backward Compatibility Shim

This module provides backward compatibility by re-exporting all functions
from the modularized sql_utils package.

The original sql_utils.py has been split into the following modules:
- sql_utils.validation: Input validation functions
- sql_utils.enum_utils: Enum value extraction
- sql_utils.ddl_cleaner: DDL cleaning and repair
- sql_utils.dialect_support: Database dialect support
- sql_utils.ddl_parser: DDL parsing and metadata extraction
- sql_utils.core: Core utility functions

All public functions and constants are re-exported for backward compatibility.
"""

# Re-export everything from the modular package
from .sql_utils import *  # noqa: F401, F403

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
]
