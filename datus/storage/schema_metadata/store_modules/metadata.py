# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details

"""
Metadata storage implementations.

This module provides concrete storage implementations for schemas,
values, and combined RAG (Retrieval-Augmented Generation) storage.
"""

import json
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import pyarrow as pa

from datus.configuration.agent_config import AgentConfig
from datus.schemas.base import TABLE_TYPE
from datus.schemas.node_models import TableSchema, TableValue
from datus.storage.lancedb_conditions import build_where, eq, or_
from datus.utils.constants import DBType
from datus.utils.json_utils import json2csv
from datus.utils.loggings import get_logger
from datus.utils.sql_utils import extract_enum_values_from_comment, is_likely_truncated_ddl, sanitize_ddl_for_storage

# Import base storage and utilities
from datus.storage.embedding_models import EmbeddingModel
from datus.storage.schema_metadata.store_modules.base_storage import BaseMetadataStorage
from datus.storage.schema_metadata.store_modules.utils import _build_where_clause, _safe_json_dict, _safe_json_list

logger = get_logger(__name__)


class SchemaStorage(BaseMetadataStorage):
    """
    Store and manage schema metadata in LanceDB.

    Extends BaseMetadataStorage with schema-specific operations including
    DDL sanitization, Chinese comment enhancement, and search text building.
    """

    def __init__(self, db_path: str, embedding_model: EmbeddingModel):
        """
        Initialize the schema store.

        Args:
            db_path: Path to the LanceDB database directory
            embedding_model: Embedding model for vector generation
        """
        super().__init__(
            db_path=db_path,
            table_name="schema_metadata",
            embedding_model=embedding_model,
            vector_source_name="definition",
        )
        self.reranker = None
        # Optional: Configure reranker
        # self.reranker = CrossEncoderReranker(
        #     model_name="BAAI/bge-reranker-large", device=get_device(), column="definition"
        # )

    def _extract_table_name(self, schema_text: str) -> str:
        """
        Extract table name from CREATE TABLE statement.

        Args:
            schema_text: DDL statement

        Returns:
            Extracted table name or empty string
        """
        words = schema_text.split()
        if len(words) >= 3 and words[0].upper() == "CREATE" and words[1].upper() == "TABLE":
            return words[2].strip("()").strip()
        return ""

    def _enhance_definition_with_comments(
        self,
        definition: str,
        table_comment: str = "",
        column_comments: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Enhance DDL definition with Chinese comments for better semantic search.

        This prepends Chinese comments to the DDL so that when embeddings are
        created, the Chinese terminology is included in the vector representation.
        This improves semantic search for Chinese queries.

        Args:
            definition: Original DDL statement
            table_comment: Table-level Chinese comment
            column_comments: Dict mapping column names to Chinese comments

        Returns:
            Enhanced DDL with Chinese comments prepended
        """
        enhanced_parts = []

        # Add table comment if available
        if table_comment and table_comment.strip():
            enhanced_parts.append(f"-- 表注释: {table_comment}")

        # Add column comments if available
        if column_comments:
            for col_name, col_comment in column_comments.items():
                if col_comment and col_comment.strip():
                    enhanced_parts.append(f"-- 列 {col_name}: {col_comment}")
                    enum_pairs = extract_enum_values_from_comment(col_comment)
                    if enum_pairs:
                        formatted = "; ".join([f"{code}={label}" for code, label in enum_pairs])
                        enhanced_parts.append(f"-- 列 {col_name} 枚举: {formatted}")

        # Add original DDL
        enhanced_parts.append(definition)

        return "\n".join(enhanced_parts)

    def _build_search_text(self, item: Dict[str, Any]) -> str:
        """
        Build search text from metadata fields.

        Combines table name, comments, database, schema, business tags,
        and column information into a searchable text string.

        Args:
            item: Metadata dictionary

        Returns:
            Search text string
        """
        parts: List[str] = []
        seen: Set[str] = set()

        def _add(value: Optional[str]) -> None:
            """Add unique non-empty value to parts."""
            if not value or not isinstance(value, str):
                return
            cleaned = value.strip()
            if not cleaned or cleaned in seen:
                return
            seen.add(cleaned)
            parts.append(cleaned)

        _add(item.get("table_name"))
        _add(item.get("table_comment"))
        _add(item.get("database_name"))
        _add(item.get("schema_name"))

        # Add business tags
        business_tags = item.get("business_tags") or []
        if isinstance(business_tags, list):
            for tag in business_tags:
                _add(tag if isinstance(tag, str) else "")

        # Add column comments
        column_comments = _safe_json_dict(item.get("column_comments"))
        for col_name, col_comment in column_comments.items():
            _add(col_name)
            _add(col_comment if isinstance(col_comment, str) else "")

        # Add column enum values
        column_enums = _safe_json_dict(item.get("column_enums"))
        for col_name, enum_values in column_enums.items():
            _add(col_name)
            for enum_item in _safe_json_list(enum_values):
                if not isinstance(enum_item, dict):
                    continue
                _add(str(enum_item.get("value", "")))
                _add(str(enum_item.get("label", "")))

        return " ".join(parts)

    def store_batch(self, data: List[Dict[str, Any]]):
        """
        Store a batch of schema metadata with DDL sanitization.

        Args:
            data: List of schema dictionaries to store
        """
        if not data:
            return

        self._ensure_table_ready()
        available_fields = set(self.table.schema.names)
        cleaned: List[Dict[str, Any]] = []

        for item in data:
            # Sanitize DDL definition
            definition = item.get("definition")
            if isinstance(definition, str) and definition:
                truncated_before = is_likely_truncated_ddl(definition)
                sanitized = sanitize_ddl_for_storage(definition)
                if sanitized != definition:
                    logger.debug("Sanitized DDL before storage")
                if truncated_before and is_likely_truncated_ddl(sanitized):
                    logger.warning(
                        "DDL still appears truncated after sanitation; storing best-effort definition."
                    )
                item = {**item, "definition": sanitized}

            # Build search text if missing
            if "search_text" in available_fields:
                search_text = item.get("search_text", "")
                if not isinstance(search_text, str) or not search_text.strip():
                    item = {**item, "search_text": self._build_search_text(item)}

            cleaned.append(item)

        super().store_batch(cleaned)

    def search_all_schemas(self, database_name: str = "", catalog_name: str = "") -> Set[str]:
        """
        Get all unique schema names for given database/catalog.

        Args:
            database_name: Database name filter
            catalog_name: Catalog name filter

        Returns:
            Set of schema names
        """
        search_result = self._search_all(
            where=_build_where_clause(database_name=database_name, catalog_name=catalog_name),
            select_fields=["schema_name"],
        )
        return {search_result["schema_name"]}

    def search_top_tables_by_every_schema(
        self,
        query_text: str,
        database_name: str = "",
        catalog_name: str = "",
        all_schemas: Optional[Set[str]] = None,
        top_n: int = 20,
    ) -> pa.Table:
        """
        Search for similar tables within each schema separately.

        Args:
            query_text: Search query
            database_name: Database filter
            catalog_name: Catalog filter
            all_schemas: Pre-computed set of schemas (optional)
            top_n: Number of results per schema

        Returns:
            Concatenated results from all schemas
        """
        if all_schemas is None:
            all_schemas = self.search_all_schemas(catalog_name=catalog_name, database_name=database_name)

        result = []
        for schema in all_schemas:
            result.append(
                self.search_similar(
                    query_text=query_text,
                    database_name=database_name,
                    catalog_name=catalog_name,
                    schema_name=schema,
                    top_n=top_n,
                )
            )
        return pa.concat_tables(result, promote_options="default")

    def get_schema(
        self, table_name: str, catalog_name: str = "", database_name: str = "", schema_name: str = ""
    ) -> pa.Table:
        """
        Get schema for a specific table.

        Args:
            table_name: Table name
            catalog_name: Optional catalog filter
            database_name: Optional database filter
            schema_name: Optional schema filter

        Returns:
            PyArrow table with schema definition
        """
        where = _build_where_clause(
            catalog_name=catalog_name,
            database_name=database_name,
            schema_name=schema_name,
            table_type="full",
        )
        table_condition = eq("table_name", table_name)
        if where:
            where_condition = and_(where, table_condition)
        else:
            where_condition = table_condition

        where_clause = build_where(where_condition)
        return (
            self.table.search()
            .where(where_clause)
            .select(["catalog_name", "database_name", "schema_name", "table_name", "table_type", "definition"])
            .to_arrow()
        )

    def get_table_schemas(
        self,
        table_names: List[str],
        catalog_name: str = "",
        database_name: str = "",
        schema_name: str = "",
    ) -> pa.Table:
        """
        Get schemas for multiple table names.

        Args:
            table_names: List of table names to retrieve
            catalog_name: Optional catalog filter
            database_name: Optional database filter
            schema_name: Optional schema filter

        Returns:
            PyArrow table containing schema definitions
        """
        self._ensure_table_ready()

        # Build OR condition for all tables
        table_conditions = [eq("table_name", tbl) for tbl in table_names]

        # Add additional filters
        base_conditions = []
        if catalog_name:
            base_conditions.append(eq("catalog_name", catalog_name))
        if database_name:
            base_conditions.append(eq("database_name", database_name))
        if schema_name:
            base_conditions.append(eq("schema_name", schema_name))

        # Combine: (table1 OR table2 OR ...) AND catalog AND database AND schema
        if len(table_conditions) == 1:
            table_where = table_conditions[0]
        else:
            table_where = or_(*table_conditions)

        if base_conditions:
            where = and_(table_where, *base_conditions)
        else:
            where = table_where

        where_clause = build_where(where)

        return (
            self.table.search()
            .where(where_clause)
            .select(
                [
                    "identifier",
                    "catalog_name",
                    "database_name",
                    "schema_name",
                    "table_name",
                    "table_type",
                    "definition",
                ]
            )
            .to_arrow()
        )

    def update_table_schema(
        self,
        table_name: str,
        definition: str,
        catalog_name: str = "",
        database_name: str = "",
        schema_name: str = "",
        table_type: TABLE_TYPE = "table",
        table_comment: str = "",
        column_comments: Optional[Dict[str, str]] = None,
        column_enums: Optional[Dict[str, List[Dict[str, str]]]] = None,
        business_tags: Optional[List[str]] = None,
        row_count: int = 0,
        sample_statistics: Optional[Dict[str, Dict]] = None,
        relationship_metadata: Optional[Dict[str, Any]] = None,
        metadata_version: int = 1,
    ) -> bool:
        """
        Update or insert a table's schema definition with enhanced metadata.

        This method is used for metadata repair when DDL is retrieved from the database.

        Args:
            table_name: Name of the table
            definition: DDL or schema definition
            catalog_name: Optional catalog name
            database_name: Optional database name
            schema_name: Optional schema name
            table_type: Type of table (table, view, mv)
            table_comment: Table comment from DDL
            column_comments: Dictionary mapping column names to comments
            column_enums: Dictionary mapping column names to enum values
            business_tags: List of business domain tags
            row_count: Approximate row count
            sample_statistics: Column statistics (min, max, mean, std)
            relationship_metadata: Foreign keys and join paths
            metadata_version: Schema version (0=legacy, 1=enhanced)

        Returns:
            True if update was successful, False otherwise
        """
        try:
            self._ensure_table_ready()

            # Sanitize DDL
            truncated_before = is_likely_truncated_ddl(definition)
            definition = sanitize_ddl_for_storage(definition)
            if truncated_before and is_likely_truncated_ddl(definition):
                logger.warning(
                    f"DDL still appears truncated after sanitation for table {table_name}; "
                    "storing best-effort definition."
                )

            # Build identifier
            identifier_parts = [
                catalog_name or "",
                database_name or "",
                schema_name or "",
                table_name,
                table_type,
            ]
            identifier = ".".join(identifier_parts)

            # Check if record exists
            existing = self.get_schema(
                table_name=table_name,
                catalog_name=catalog_name,
                database_name=database_name,
                schema_name=schema_name,
            )

            # Enhance definition with Chinese comments for better semantic search
            enhanced_definition = self._enhance_definition_with_comments(
                definition=definition,
                table_comment=table_comment or "",
                column_comments=column_comments or {},
            )

            # Prepare data with enhanced fields
            data = {
                "identifier": identifier,
                "catalog_name": catalog_name or "",
                "database_name": database_name or "",
                "schema_name": schema_name or "",
                "table_name": table_name,
                "table_type": table_type,
                "definition": enhanced_definition,
                "table_comment": table_comment or "",
                "column_comments": json.dumps(column_comments or {}, ensure_ascii=False),
                "column_enums": json.dumps(column_enums or {}, ensure_ascii=False),
                "business_tags": business_tags or [],
                "row_count": row_count or 0,
                "sample_statistics": json.dumps(sample_statistics or {}, ensure_ascii=False),
                "relationship_metadata": json.dumps(relationship_metadata or {}, ensure_ascii=False),
                "metadata_version": metadata_version,
                "last_updated": int(time.time()),
            }

            # Generate embedding
            embedded_data = self._embed_and_prepare(data)

            if existing and len(existing) > 0:
                # Update existing record (delete and re-insert)
                where_clause = build_where(
                    _build_where_clause(
                        catalog_name=catalog_name,
                        database_name=database_name,
                        schema_name=schema_name,
                        table_name=table_name,
                        table_type="full",
                    )
                )
                self.table.delete(where_clause)
                self.table.add([embedded_data])
                logger.info(f"Updated enhanced schema for table: {table_name}")
            else:
                # Insert new record
                self.table.add([embedded_data])
                logger.info(f"Inserted enhanced schema for table: {table_name}")

            return True

        except Exception as e:
            logger.error(f"Failed to update table schema for {table_name}: {e}")
            return False


class SchemaValueStorage(BaseMetadataStorage):
    """Storage for sample row values from tables."""

    def __init__(self, db_path: str, embedding_model: EmbeddingModel):
        """
        Initialize the value store.

        Args:
            db_path: Path to the LanceDB database directory
            embedding_model: Embedding model for vector generation
        """
        super().__init__(
            db_path=db_path,
            embedding_model=embedding_model,
            table_name="schema_value",
            vector_source_name="sample_rows",
        )
        self.reranker = None
        # Optional: Configure reranker
        # self.reranker = CrossEncoderReranker(
        #     model_name="BAAI/bge-reranker-large", device=get_device(), column="sample_rows"
        # )


class SchemaWithValueRAG:
    """
    Combined RAG storage for schemas and values.

    This class provides a unified interface for storing and searching
    both schema definitions and sample row values.
    """

    def __init__(
        self,
        agent_config: AgentConfig,
        sub_agent_name: Optional[str] = None,
    ):
        """
        Initialize RAG storage.

        Args:
            agent_config: Agent configuration
            sub_agent_name: Optional sub-agent name for multi-tenancy
        """
        from datus.storage.cache import get_storage_cache_instance

        self.schema_store = get_storage_cache_instance(agent_config).schema_storage(sub_agent_name)
        self.value_store = get_storage_cache_instance(agent_config).schema_value_storage(sub_agent_name)

    def store_batch(self, schemas: List[Dict[str, Any]], values: List[Dict[str, Any]]):
        """
        Store schemas and sample values.

        Args:
            schemas: List of schema dictionaries
            values: List of sample row dictionaries
        """
        if schemas:
            self.schema_store.store_batch(schemas)

        if len(values) == 0:
            return

        # Convert sample rows to CSV format
        final_values = []
        for item in values:
            if "sample_rows" not in item or not item["sample_rows"]:
                continue
            sample_rows = item["sample_rows"]
            if isinstance(sample_rows, list):
                sample_rows = json2csv(sample_rows)
            item["sample_rows"] = sample_rows
            final_values.append(item)

        self.value_store.store_batch(final_values)
        logger.debug(f"Batch stored {len(schemas)} schemas, {len(final_values)} values")

    def after_init(self):
        """Create indices after initialization."""
        self.schema_store.create_indices()
        self.value_store.create_indices()

    def get_schema_size(self) -> int:
        """Get number of stored schemas."""
        return self.schema_store.table_size()

    def get_value_size(self) -> int:
        """Get number of stored value samples."""
        return self.value_store.table_size()

    def search_similar(
        self,
        query_text: str,
        catalog_name: str = "",
        database_name: str = "",
        schema_name: str = "",
        use_rerank: bool = False,
        table_type: TABLE_TYPE = "table",
        top_n: int = 5,
    ) -> Tuple[pa.Table, pa.Table]:
        """
        Search for similar schemas and values.

        Args:
            query_text: Search query
            catalog_name: Optional catalog filter
            database_name: Optional database filter
            schema_name: Optional schema filter
            use_rerank: Whether to use reranker
            table_type: Table type filter
            top_n: Number of results

        Returns:
            Tuple of (schema_results, value_results)
        """
        where = _build_where_clause(
            catalog_name=catalog_name,
            database_name=database_name,
            schema_name=schema_name,
            table_type=table_type,
        )

        schema_results = self.schema_store.do_search_similar(
            query_text,
            top_n=top_n,
            where=where,
            reranker=self.schema_store.reranker if use_rerank else None,
        )

        value_results = self.value_store.do_search_similar(
            query_text,
            top_n=top_n,
            where=where,
            reranker=self.value_store.reranker if use_rerank else None,
        )

        return schema_results, value_results

    def search_all_schemas(
        self,
        catalog_name: str = "",
        database_name: str = "",
        schema_name: str = "",
        table_type: TABLE_TYPE = "full",
        select_fields: Optional[List[str]] = None,
    ) -> pa.Table:
        """
        Search all schemas matching filters.

        Args:
            catalog_name: Optional catalog filter
            database_name: Optional database filter
            schema_name: Optional schema filter
            table_type: Table type filter
            select_fields: Fields to include

        Returns:
            PyArrow table with matching schemas
        """
        return self.schema_store.search_all(
            catalog_name=catalog_name,
            database_name=database_name,
            schema_name=schema_name,
            table_type=table_type,
            select_fields=select_fields,
        )

    def search_all_value(
        self, catalog_name: str = "", database_name: str = "", schema_name: str = "", table_type: TABLE_TYPE = "full"
    ) -> pa.Table:
        """
        Search all sample values matching filters.

        Args:
            catalog_name: Optional catalog filter
            database_name: Optional database filter
            schema_name: Optional schema filter
            table_type: Table type filter

        Returns:
            PyArrow table with matching values
        """
        return self.value_store.search_all(
            catalog_name=catalog_name, database_name=database_name, schema_name=schema_name, table_type=table_type
        )

    def search_tables(
        self,
        tables: list[str],
        catalog_name: str = "",
        database_name: str = "",
        schema_name: str = "",
        dialect: str = DBType.SQLITE,
    ) -> Tuple[List[TableSchema], List[TableValue]]:
        """
        Search schemas and values for given table names.

        Args:
            tables: List of table names (can include catalog/database/schema prefixes)
            catalog_name: Default catalog name
            database_name: Default database name
            schema_name: Default schema name
            dialect: Database dialect for parsing table names

        Returns:
            Tuple of (table_schemas, table_values)
        """
        # Ensure tables are ready
        self.schema_store._ensure_table_ready()
        self.value_store._ensure_table_ready()

        # Parse table names and build where clause
        table_conditions = []
        for full_table in tables:
            parts = full_table.split(".")
            table_name = parts[-1]

            if len(parts) == 4:
                # Format: catalog.database.schema.table
                cat, db, sch = parts[0], parts[1], parts[2]
            elif len(parts) == 3:
                # Format: database.schema.table
                if dialect == DBType.STARROCKS:
                    cat, db, sch = parts[0], parts[1], ""
                else:
                    cat, db, sch = catalog_name, parts[0], parts[1]
            elif len(parts) == 2:
                # Format: database.table
                if dialect in (DBType.SQLITE, DBType.MYSQL, DBType.STARROCKS):
                    cat, db, sch = catalog_name, parts[0], ""
                else:
                    cat, db, sch = catalog_name, database_name, parts[0]
            else:
                # Format: table
                cat, db, sch = catalog_name, database_name, schema_name

            table_conditions.append(
                _build_where_clause(
                    table_name=table_name,
                    catalog_name=cat,
                    database_name=db,
                    schema_name=sch,
                    table_type="full",
                )
            )

        # Build combined condition
        if table_conditions:
            combined_condition = table_conditions[0] if len(table_conditions) == 1 else or_(*table_conditions)
            where_clause = build_where(combined_condition)
            schema_query = self.schema_store.table.search().where(where_clause)
            value_query = self.value_store.table.search().where(where_clause)
        else:
            schema_query = self.schema_store.table.search()
            value_query = self.value_store.table.search()

        # Search schemas
        schema_select_fields = [
            "identifier",
            "catalog_name",
            "database_name",
            "schema_name",
            "table_name",
            "table_type",
            "definition",
            "table_comment",
            "column_comments",
            "column_enums",
            "business_tags",
            "row_count",
            "sample_statistics",
            "relationship_metadata",
            "metadata_version",
            "last_updated",
        ]
        available_schema_fields = set(self.schema_store.table.schema.names)
        schema_select_fields = [field for field in schema_select_fields if field in available_schema_fields]
        schema_results = schema_query.select(schema_select_fields).limit(len(tables)).to_arrow()
        schemas_result = TableSchema.from_arrow(schema_results)

        # Search values
        value_results = (
            value_query.select(
                [
                    "identifier",
                    "catalog_name",
                    "database_name",
                    "schema_name",
                    "table_name",
                    "table_type",
                    "sample_rows",
                ]
            )
            .limit(len(tables))
            .to_arrow()
        )
        values_result = TableValue.from_arrow(value_results)

        return schemas_result, values_result

    def remove_data(
        self,
        catalog_name: str = "",
        database_name: str = "",
        schema_name: str = "",
        table_name: str = "",
        table_type: TABLE_TYPE = "table",
    ):
        """
        Remove schemas and values matching filters.

        Args:
            catalog_name: Optional catalog filter
            database_name: Optional database filter
            schema_name: Optional schema filter
            table_name: Optional table name filter
            table_type: Table type filter
        """
        # Ensure tables are ready
        self.schema_store._ensure_table_ready()
        self.value_store._ensure_table_ready()

        where_condition = _build_where_clause(
            catalog_name=catalog_name,
            database_name=database_name,
            schema_name=schema_name,
            table_name=table_name,
            table_type=table_type,
        )
        where_clause = build_where(where_condition) if where_condition else None

        self.schema_store.table.delete(where_clause)
        self.value_store.table.delete(where_clause)
