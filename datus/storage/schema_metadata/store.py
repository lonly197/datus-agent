# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details

"""
Schema metadata storage - Backward compatibility wrapper.

This file has been refactored into separate modules for better maintainability.
All classes are re-exported from the new modules for backward compatibility.

REFACTORED STRUCTURE:
    datus/storage/schema_metadata/store_modules/
    ├── __init__.py       # Public API exports
    ├── utils.py          # Helper functions
    ├── search.py         # Search functionality
    ├── base_storage.py   # Core storage infrastructure
    └── metadata.py       # Concrete storage implementations

MIGRATION GUIDE:
    Old imports (still work):
        from datus.storage.schema_metadata.store import BaseMetadataStorage
        from datus.storage.schema_metadata.store import SchemaStorage

    New imports (recommended):
        from datus.storage.schema_metadata.store_modules import BaseMetadataStorage
        from datus.storage.schema_metadata.store_modules import SchemaStorage

For detailed refactoring information, see REFACTORING_GUIDE.md
"""

# Re-export everything from the new modules for backward compatibility
from datus.storage.schema_metadata.store_modules import (
    BaseMetadataStorage,
    SchemaStorage,
    SchemaValueStorage,
    SchemaWithValueRAG,
    SearchMixin,
    _build_where_clause,
    _safe_json_dict,
    _safe_json_list,
)

__all__ = [
    "BaseMetadataStorage",
    "SchemaStorage",
    "SchemaValueStorage",
    "SchemaWithValueRAG",
    "SearchMixin",
    "_build_where_clause",
    "_safe_json_dict",
    "_safe_json_list",
]
