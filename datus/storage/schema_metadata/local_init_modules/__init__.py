# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Local schema initialization modules.

This package contains refactored modules from local_init.py for better maintainability.

MODULE STRUCTURE:
    enum_extraction.py    - Enum extraction from column comments
    utils.py              - Helper functions for relationships and DDL parsing
    table_storage.py      - Table storage and metadata processing
    db_initializers.py    - Database-specific initialization functions
    initialization.py     - Main schema initialization orchestrator

PUBLIC API:
    All public functions and classes from the original local_init.py are re-exported
    here for backward compatibility.
"""

from .enum_extraction import (
    _get_enum_extractor,
    _extract_enums,
)
from .utils import (
    _normalize_relationship_name,
    _build_table_name_map,
    _infer_relationships_from_names,
    _normalize_dialect,
    _llm_fallback_parse_ddl,
    _fill_sample_rows,
    extract_enhanced_metadata_from_ddl,
    parse_dialect,
    sanitize_ddl_for_storage,
)
from .table_storage import (
    store_tables,
)
from .db_initializers import (
    init_sqlite_schema,
    init_duckdb_schema,
    init_mysql_schema,
    init_starrocks_schema,
    init_other_three_level_schema,
)
from .initialization import (
    init_local_schema,
)

__all__ = [
    # Enum extraction
    "_get_enum_extractor",
    "_extract_enums",
    # Utility functions
    "_normalize_relationship_name",
    "_build_table_name_map",
    "_infer_relationships_from_names",
    "_normalize_dialect",
    "_llm_fallback_parse_ddl",
    "_fill_sample_rows",
    "extract_enhanced_metadata_from_ddl",
    "parse_dialect",
    "sanitize_ddl_for_storage",
    # Table storage
    "store_tables",
    # Database-specific initializers
    "init_sqlite_schema",
    "init_duckdb_schema",
    "init_mysql_schema",
    "init_starrocks_schema",
    "init_other_three_level_schema",
    # Main initialization
    "init_local_schema",
]
