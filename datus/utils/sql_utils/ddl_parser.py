# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
DDL parsing and metadata extraction utilities.

This module has been refactored into a sub-package for better maintainability.
Please use the new imports:

    from datus.utils.sql_utils.ddl_parser import (
        parse_metadata_from_ddl,
        extract_enhanced_metadata_from_ddl,
        extract_metadata_from_ddl_regex_only,
    )

The implementation has been moved to:
- datus.utils.sql_utils.ddl_parser.core
- datus.utils.sql_utils.ddl_parser.sqlglot_parser
- datus.utils.sql_utils.ddl_parser.regex_fallback
"""

# Re-export all public APIs for backward compatibility
from .core import (
    parse_metadata_from_ddl,
)

from .sqlglot_parser import (
    extract_enhanced_metadata_from_ddl,
    extract_metadata_from_ddl_regex_only,
)

from .regex_fallback import (
    _parse_ddl_with_regex,
)

__all__ = [
    "parse_metadata_from_ddl",
    "extract_enhanced_metadata_from_ddl",
    "extract_metadata_from_ddl_regex_only",
    "_parse_ddl_with_regex",
]
