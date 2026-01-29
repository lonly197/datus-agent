# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
DDL Parser sub-package.

This package provides DDL parsing and metadata extraction utilities,
split into focused modules for better maintainability:

- core: Core parsing functions and regex patterns
- sqlglot_parser: sqlglot-based parsing strategies
- regex_fallback: Fallback regex-based parsing for corrupted DDL
"""

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
