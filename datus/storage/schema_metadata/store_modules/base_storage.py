# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details

"""
Base metadata storage class.

Provides the core storage infrastructure including schema definition,
index management, and basic CRUD operations.
"""

import pyarrow as pa
from lancedb.rerankers import Reranker

from datus.schemas.base import TABLE_TYPE
from datus.storage.base import BaseEmbeddingStore, WhereExpr
from datus.storage.embedding_models import EmbeddingModel
from datus.utils.loggings import get_logger

# Import search mixin and utilities
from datus.storage.schema_metadata.store_modules.search import SearchMixin
from datus.storage.schema_metadata.store_modules.utils import _build_where_clause

logger = get_logger(__name__)


class BaseMetadataStorage(BaseEmbeddingStore, SearchMixin):
    """
    Base class for metadata storage with enhanced v1 schema support.

    Stores table, view, and materialized view metadata with:
    - Vector embeddings for semantic search
    - Full-text search (FTS) for keyword search
    - Enhanced metadata (comments, enums, business tags, statistics)

    Schema Fields:
        - identifier: Unique identifier (catalog.database.schema.table.type)
        - catalog_name: Catalog name
        - database_name: Database name
        - schema_name: Schema name
        - table_name: Table name (required)
        - table_type: Table type (table, view, mv)
        - definition: Raw DDL for SQL generation
        - vector_source_name: Enhanced search text for embedding (default: "search_text")
        - vector: Embedding vector (required)
        - Enhanced fields (v1):
            - table_comment: Table comment
            - column_comments: JSON of column comments
            - column_enums: JSON of column enum values
            - business_tags: List of business domain tags
            - row_count: Table row count
            - sample_statistics: JSON of column statistics
            - relationship_metadata: JSON of relationships
            - metadata_version: Schema version (0=legacy, 1=enhanced)
            - last_updated: Unix timestamp

    Note:
        - 'definition' stores the original DDL for SQL generation
        - 'search_text' (vector_source_name) contains enhanced content with comments for semantic search
        - Both fields are always present in the schema
    """

    def __init__(
        self,
        db_path: str,
        embedding_model: EmbeddingModel,
        table_name: str,
        vector_source_name: str,
    ):
        """
        Initialize base metadata storage.

        Args:
            db_path: Path to LanceDB database
            embedding_model: Embedding model for vector generation
            table_name: Name of the metadata table
            vector_source_name: Field name containing text to embed
        """
        # Build schema fields
        schema_fields = [
            # Core identification fields
            pa.field("identifier", pa.string()),
            pa.field("catalog_name", pa.string()),
            pa.field("database_name", pa.string()),
            pa.field("schema_name", pa.string()),
            pa.field("table_name", pa.string()),
            pa.field("table_type", pa.string()),
            # DDL fields: definition (raw DDL) + search_text (enhanced content for embedding)
            # definition: Stores original DDL for SQL generation
            pa.field("definition", pa.string()),
            # search_text: Enhanced content with comments, used as vector source
            pa.field(vector_source_name, pa.string()),
        ]

        schema_fields.extend([
            pa.field("vector", pa.list_(pa.float32(), list_size=embedding_model.dim_size)),
            # Enhanced fields (v1) - Business Semantics (HIGH PRIORITY)
            pa.field("table_comment", pa.string()),  # Extracted from DDL COMMENT
            pa.field("column_comments", pa.string()),  # JSON: {"col1": "comment1", ...}
            pa.field("column_enums", pa.string()),  # JSON: {"col1": [{"value": "0", "label": "foo"}], ...}
            pa.field("business_tags", pa.list_(pa.string())),  # ["finance", "fact_table", "revenue"]
            # Statistics (MEDIUM PRIORITY)
            pa.field("row_count", pa.int64()),  # Table row count
            pa.field("sample_statistics", pa.string()),  # JSON: {"col1": {"min": 0, "max": 100, ...}}
            # Relationships (MEDIUM PRIORITY)
            pa.field("relationship_metadata", pa.string()),  # JSON: {"foreign_keys": [...], "join_paths": [...]}
            # Metadata Management
            pa.field("metadata_version", pa.int32()),  # 0=legacy, 1=enhanced
            pa.field("last_updated", pa.int64()),  # Unix timestamp
        ])

        super().__init__(
            db_path=db_path,
            table_name=table_name,
            embedding_model=embedding_model,
            schema=pa.schema(schema_fields),
            vector_source_name=vector_source_name,
        )
        self.reranker = None

    def create_indices(self):
        """
        Create scalar and FTS indices for efficient querying.

        Creates:
        - Scalar indices on catalog, database, schema, table_name, table_type
        - Full-text search index on searchable text fields
        """
        # Ensure table is ready before creating indices
        self._ensure_table_ready()

        # Create scalar indices
        try:
            self.table.create_scalar_index("database_name", replace=True)
            self.table.create_scalar_index("catalog_name", replace=True)
            self.table.create_scalar_index("schema_name", replace=True)
            self.table.create_scalar_index("table_name", replace=True)
            self.table.create_scalar_index("table_type", replace=True)
        except Exception as e:
            logger.warning(f"Failed to create scalar index for {self.table_name} table: {str(e)}")

        # Create FTS index
        self.create_fts_index(
            [
                "catalog_name",
                "database_name",
                "schema_name",
                "table_name",
                "table_type",
                "definition",  # Raw DDL for keyword search
                self.vector_source_name,  # Enhanced content for semantic search
                "table_comment",
                "column_comments",
            ]
        )
