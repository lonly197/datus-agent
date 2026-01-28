# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details

"""
Schema metadata store modules.

This package contains the refactored components of the original store.py,
split into focused modules for better maintainability.

Modules:
    - utils: Helper functions for WHERE clauses and JSON parsing
    - search: Full-text search and vector similarity search
    - base_storage: Core storage infrastructure
    - metadata: Concrete storage implementations (SchemaStorage, SchemaValueStorage, SchemaWithValueRAG)
"""

# Public API - re-export all storage classes
from datus.storage.schema_metadata.store_modules.base_storage import BaseMetadataStorage
from datus.storage.schema_metadata.store_modules.metadata import (
    SchemaStorage,
    SchemaValueStorage,
    SchemaWithValueRAG,
)
from datus.storage.schema_metadata.store_modules.search import SearchMixin
from datus.storage.schema_metadata.store_modules.utils import (
    _build_where_clause,
    _safe_json_dict,
    _safe_json_list,
)

__all__ = [
    # Core storage classes
    "BaseMetadataStorage",
    "SchemaStorage",
    "SchemaValueStorage",
    "SchemaWithValueRAG",
    # Mixins
    "SearchMixin",
    # Utilities (exported for testing)
    "_build_where_clause",
    "_safe_json_dict",
    "_safe_json_list",
]
