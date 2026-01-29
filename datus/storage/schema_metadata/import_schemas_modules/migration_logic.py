# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details

"""
Core migration logic for v0 to v1 schema upgrade.

This module handles the primary migration operations:
- Schema storage migration
- Schema value storage migration
- Metadata enhancement and extraction
- Verification and validation
"""

import json
import os
import sys
import time
from collections import Counter
from typing import Any, Dict, List, Optional

from datus.configuration.agent_config import AgentConfig
from datus.configuration.business_term_config import infer_business_tags
from datus.models.base import LLMBaseModel
from datus.storage.schema_metadata import SchemaStorage, SchemaWithValueRAG
from datus.utils.loggings import get_logger
from datus.utils.sql_utils import (
    extract_enhanced_metadata_from_ddl,
    extract_enum_values_from_comment,
    is_likely_truncated_ddl,
    sanitize_ddl_for_storage,
    validate_comment,
    validate_table_name,
)
from datus.utils.constants import DBType

from .utils import (
    load_backup_json,
    shutdown_requested,
    detect_dialect_from_ddl,
    _normalize_dialect,
    _build_table_name_map,
    _infer_relationships_from_names,
)
from .models import METADATA_VERSION_V1, MigrationResult

logger = get_logger(__name__)


def migrate_schema_storage(
    storage: SchemaStorage,
    backup_path: Optional[str] = None,
    extract_statistics: bool = False,
    extract_relationships: bool = True,
    llm_fallback: bool = False,
    llm_model: Optional[LLMBaseModel] = None,
    update_existing: bool = False,
    db_manager: Optional[Any] = None,
    namespace: Optional[str] = None,
) -> int:
    """
    Migrate schema metadata from v0 to v1 format.

    This function:
    1. Loads data from backup JSON file (if provided) or reads from existing table
    2. Creates a fresh v1 table (assumes cleanup has been run)
    3. Inserts migrated v1 records with enhanced metadata

    IMPORTANT: Before running this migration, ensure the cleanup script has been run:
        python -m datus.storage.schema_metadata.cleanup_v0_table

    Args:
        storage: SchemaStorage instance
        backup_path: Path to JSON backup file from cleanup script (recommended)
        extract_statistics: Whether to extract column statistics (expensive)
        extract_relationships: Whether to extract relationship metadata
        llm_fallback: Use LLM as fallback for DDL parsing
        llm_model: LLM model for fallback parsing
        update_existing: Force update existing records
        db_manager: Optional DBManager for statistics extraction
        namespace: Namespace for DBManager lookups

    Returns:
        Number of records migrated
    """
    def _escape_where_value(value: str) -> str:
        return value.replace("\\", "\\\\").replace('"', '\\"')

    def _llm_fallback_parse_ddl(ddl: str) -> Optional[Dict[str, Any]]:
        if not llm_fallback or not llm_model:
            return None
        if not ddl or not isinstance(ddl, str):
            return None

        prompt = (
            "You are a SQL DDL parser. Extract metadata from the CREATE TABLE statement.\n"
            "Return ONLY a JSON object with this schema:\n"
            "{\n"
            '  "table": {"name": "", "comment": ""},\n'
            '  "columns": [{"name": "", "type": "", "comment": "", "nullable": true}],\n'
            '  "primary_keys": [],\n'
            '  "foreign_keys": [],\n'
            '  "indexes": []\n'
            "}\n"
            "Rules:\n"
            "- Only use information present in the DDL.\n"
            "- If a field is missing, use empty string or empty list.\n"
            "- Keep comments exactly as written (may contain Chinese and punctuation).\n"
            "- If NULL/NOT NULL is not specified, set nullable=true.\n"
            "DDL:\n"
            "```sql\n"
            f"{ddl}\n"
            "```\n"
        )

        try:
            response = llm_model.generate_with_json_output(prompt)
        except Exception as exc:
            logger.warning(f"LLM fallback parse failed: {exc}")
            return None

        if not isinstance(response, dict):
            return None

        table = response.get("table") if isinstance(response.get("table"), dict) else {}
        table_name = table.get("name", "") if isinstance(table.get("name"), str) else ""
        is_valid, _, table_name = validate_table_name(table_name)
        if not is_valid or not table_name:
            table_name = ""

        table_comment = table.get("comment", "") if isinstance(table.get("comment"), str) else ""
        is_valid, _, table_comment = validate_comment(table_comment)
        if not is_valid or table_comment is None:
            table_comment = ""

        columns = []
        raw_columns = response.get("columns", [])
        if isinstance(raw_columns, list):
            for col in raw_columns:
                if not isinstance(col, dict):
                    continue
                col_name = col.get("name", "")
                if not isinstance(col_name, str) or not col_name:
                    continue
                col_type = col.get("type", "")
                if not isinstance(col_type, str):
                    col_type = ""
                col_comment = col.get("comment", "")
                if not isinstance(col_comment, str):
                    col_comment = ""
                is_valid, _, col_comment = validate_comment(col_comment)
                if not is_valid or col_comment is None:
                    col_comment = ""
                nullable = col.get("nullable", True)
                if not isinstance(nullable, bool):
                    nullable = True
                columns.append(
                    {
                        "name": col_name,
                        "type": col_type,
                        "comment": col_comment,
                        "nullable": nullable,
                    }
                )

        if not table_name or not columns:
            return None

        return {
            "table": {"name": table_name, "comment": table_comment},
            "columns": columns,
            "primary_keys": response.get("primary_keys", []) if isinstance(response.get("primary_keys"), list) else [],
            "foreign_keys": response.get("foreign_keys", []) if isinstance(response.get("foreign_keys"), list) else [],
            "indexes": response.get("indexes", []) if isinstance(response.get("indexes"), list) else [],
        }

    from datus.tools.db_tools.metadata_extractor import get_metadata_extractor

    metadata_extractor_cache: Dict[str, Optional[Any]] = {}
    db_configs: Dict[str, Any] = {}
    if db_manager and namespace:
        try:
            db_configs = db_manager.current_db_configs(namespace)
        except Exception as exc:
            logger.warning(f"Failed to load database configs for namespace '{namespace}': {exc}")
            db_configs = {}
    elif extract_statistics:
        logger.warning("Extract statistics requested but no namespace/db connection available; using defaults.")

    has_multiple_configs = len(db_configs) > 1

    def _resolve_connector(database_name: str):
        if not db_manager or not namespace:
            return "__default__", None
        logic_name = ""
        if database_name:
            if database_name in db_configs:
                logic_name = database_name
            else:
                for name, cfg in db_configs.items():
                    if getattr(cfg, "database", "") == database_name:
                        logic_name = name
                        break
        if has_multiple_configs and not logic_name and database_name:
            logger.warning(
                f"Could not match database '{database_name}' to config; using first configured connection."
            )
        try:
            connector = db_manager.get_conn(namespace, logic_name)
        except Exception as exc:
            logger.warning(f"Failed to get database connection for '{database_name or 'default'}': {exc}")
            return logic_name or "__default__", None
        return logic_name or "__default__", connector

    def _get_metadata_extractor(database_name: str):
        if not db_manager or not namespace:
            return None
        cache_key, connector = _resolve_connector(database_name)
        if not connector:
            return None
        if cache_key not in metadata_extractor_cache:
            if not hasattr(connector, "execute_sql"):
                logger.warning(
                    f"Connector for dialect '{connector.dialect}' does not support execute_sql; "
                    "skipping statistics extraction."
                )
                metadata_extractor_cache[cache_key] = None
                return None
            try:
                metadata_extractor_cache[cache_key] = get_metadata_extractor(connector, connector.dialect)
            except TypeError as exc:
                logger.warning(f"No metadata extractor available for dialect '{connector.dialect}': {exc}")
                metadata_extractor_cache[cache_key] = None
            except Exception as exc:
                logger.warning(f"Failed to initialize metadata extractor: {exc}")
                metadata_extractor_cache[cache_key] = None
        return metadata_extractor_cache[cache_key]

    try:
        # Step 1: Load source data
        if backup_path and os.path.exists(backup_path):
            # Load from backup JSON (recommended)
            logger.info(f"Loading data from backup: {backup_path}")
            backup_records = load_backup_json(backup_path)
            if not backup_records:
                logger.info("No records found in backup file")
                return 0
            logger.info(f"Loaded {len(backup_records)} records from backup")
        else:
            # Fallback: Read from existing table (if it exists)
            logger.info("No backup provided, reading from existing table...")
            storage._ensure_table_ready()

            all_data = storage._search_all(
                where=None,
                select_fields=["identifier", "catalog_name", "database_name", "schema_name", "table_name", "table_type", "definition"]
            )

            if not all_data or len(all_data) == 0:
                logger.info("No existing data found in schema storage - creating empty v1 table (successful migration)")
                return 0

            backup_records = all_data.to_pylist()
            logger.info(f"Found {len(backup_records)} records to migrate")

        # Step 2: Create fresh v1 table (assumes cleanup has been run)
        required_fields = [
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
        core_fields = {
            "identifier",
            "catalog_name",
            "database_name",
            "schema_name",
            "table_name",
            "table_type",
            "definition",
        }
        table_exists = storage.table_name in storage.db.table_names(limit=100)
        table_recreated = not table_exists
        recreate_table = update_existing

        if table_exists:
            existing_schema = storage.db.open_table(storage.table_name).schema
            existing_fields = set(existing_schema.names)
            missing_required = [field for field in required_fields if field not in existing_fields]
            if missing_required:
                logger.warning(
                    "Existing schema is missing v1 fields; recreating table: "
                    f"{', '.join(missing_required)}"
                )
                recreate_table = True

        if table_exists and recreate_table:
            if update_existing:
                logger.info("Force mode enabled; recreating table to avoid per-row updates.")
            try:
                storage.db.drop_table(storage.table_name)
                table_recreated = True
            except Exception as exc:
                logger.error(f"Failed to drop existing table: {exc}")
                logger.warning("Migration will proceed with in-place updates; per-record deletes may fail.")

        # Reset initialization to force table creation with v1 schema
        storage.table = None
        storage._table_initialized = False
        storage._ensure_table_ready()
        if table_recreated:
            logger.info(f"Created fresh v1 table: {storage.table_name}")
        else:
            logger.info(f"Using existing v1 table: {storage.table_name}")
        table_fields = set(storage.table.schema.names)
        missing_core = [field for field in core_fields if field not in table_fields]
        if missing_core:
            raise RuntimeError(f"Schema missing required fields: {', '.join(missing_core)}")
        missing_required = [field for field in required_fields if field not in table_fields]
        if missing_required:
            logger.warning(
                "Table schema still missing fields; inserts will omit: "
                f"{', '.join(missing_required)}"
            )

        total_records = len(backup_records)
        migrated_count = 0
        batch_size = 100
        batch_updates = []
        batch_status = []
        existing_identifiers = set()
        table_name_map: Dict[str, List[str]] = {}
        if extract_relationships:
            table_name_map = _build_table_name_map(
                [row.get("table_name", "") for row in backup_records if isinstance(row, dict)]
            )
        update_existing_effective = update_existing and not table_recreated
        if not table_recreated:
            try:
                existing_data = storage._search_all(where=None, select_fields=["identifier"])
                existing_identifiers = {
                    row.get("identifier", "") for row in existing_data.to_pylist() if row.get("identifier")
                }
            except Exception as exc:
                logger.debug(f"Could not load existing identifiers: {exc}")

        for i, row in enumerate(backup_records):
            if shutdown_requested():
                logger.warning("Shutdown requested; stopping migration loop.")
                break
            try:
                identifier = row["identifier"]
                catalog_name = row["catalog_name"]
                database_name = row["database_name"]
                schema_name = row["schema_name"]
                table_name = row["table_name"]
                table_type = row["table_type"]
                definition = row["definition"]
                qualified_table = ".".join(part for part in [database_name, schema_name, table_name] if part)
                logger.info(f"[{i + 1}/{total_records}] Processing {qualified_table} ({table_type})")

                if identifier in existing_identifiers and not update_existing_effective:
                    logger.info("  - STATUS: COMPLETED (SKIPPED - already exists in storage)")
                    continue
                if identifier in existing_identifiers and update_existing_effective:
                    if shutdown_requested():
                        logger.warning("Shutdown requested; skipping in-place update.")
                        break
                    try:
                        escaped_identifier = _escape_where_value(identifier)
                        storage.table.delete(f'identifier = "{escaped_identifier}"')
                    except Exception as exc:
                        logger.warning(f"  - UPDATE: failed to remove existing record: {exc}")
                    else:
                        logger.info("  - UPDATE: replacing existing record")

                # Fix and clean DDL before parsing
                original_definition = definition
                was_truncated = is_likely_truncated_ddl(original_definition)
                definition = sanitize_ddl_for_storage(definition)
                if was_truncated and not is_likely_truncated_ddl(definition):
                    logger.info(
                        f"  - DDL FIXED (truncation): length {len(original_definition)} -> {len(definition)}"
                    )

                detected_dialect = detect_dialect_from_ddl(definition)
                metadata_extractor = None
                if detected_dialect == DBType.STARROCKS or extract_statistics or extract_relationships:
                    metadata_extractor = _get_metadata_extractor(database_name)

                enhanced_metadata = {"table": {"comment": ""}, "columns": [], "foreign_keys": []}
                used_information_schema = False
                table_comment = ""
                column_comments: Dict[str, str] = {}
                column_names: List[str] = []
                skip_ddl_parse = detected_dialect == DBType.STARROCKS

                if (
                    metadata_extractor
                    and _normalize_dialect(getattr(metadata_extractor, "dialect", "")) == DBType.STARROCKS
                    and hasattr(metadata_extractor, "extract_table_metadata")
                ):
                    try:
                        starrocks_metadata = metadata_extractor.extract_table_metadata(
                            table_name,
                            database_name=database_name,
                        )
                    except Exception as exc:
                        logger.debug(f"Failed to read StarRocks metadata for {table_name}: {exc}")
                        starrocks_metadata = {}
                    if starrocks_metadata and starrocks_metadata.get("columns"):
                        used_information_schema = True
                        table_comment = starrocks_metadata.get("table_comment", "") or ""
                        column_comments = starrocks_metadata.get("column_comments", {}) or {}
                        column_names = starrocks_metadata.get("column_names", []) or []

                if not used_information_schema and not skip_ddl_parse:
                    enhanced_metadata = extract_enhanced_metadata_from_ddl(
                        definition,
                        dialect=detected_dialect,
                        warn_on_invalid=False,
                    )
                    if llm_fallback and (not enhanced_metadata["columns"] or not enhanced_metadata["table"].get("name")):
                        llm_metadata = _llm_fallback_parse_ddl(definition)
                        if llm_metadata:
                            enhanced_metadata = llm_metadata
                            logger.info(f"LLM fallback parsed DDL for table: {table_name}")
                    table_comment = enhanced_metadata["table"].get("comment", "")
                    column_comments = {
                        col["name"]: col.get("comment", "")
                        for col in enhanced_metadata["columns"]
                    }
                    column_names = [col["name"] for col in enhanced_metadata["columns"]]

                column_enums: Dict[str, List[Dict[str, str]]] = {}
                for col_name, col_comment in column_comments.items():
                    enum_pairs = extract_enum_values_from_comment(col_comment)
                    if enum_pairs:
                        column_enums[col_name] = [
                            {"value": code, "label": label} for code, label in enum_pairs
                        ]

                # Infer business tags
                business_tags = infer_business_tags(table_name, column_names)

                # Extract statistics (row count + column stats) from live database
                row_count = 0
                sample_statistics: Dict[str, Dict[str, Any]] = {}
                if extract_statistics:
                    if metadata_extractor:
                        stats_table_name = table_name
                        if schema_name:
                            stats_table_name = f"{schema_name}.{table_name}"
                        try:
                            row_count = metadata_extractor.extract_row_count(stats_table_name)
                            logger.debug(f"  Row count: {row_count}")
                        except Exception as exc:
                            logger.debug(f"  Could not extract row count: {exc}")
                        if row_count > 1000:
                            try:
                                sample_statistics = metadata_extractor.extract_column_statistics(stats_table_name)
                                logger.debug(f"  Column statistics: {len(sample_statistics)} columns")
                            except Exception as exc:
                                logger.debug(f"  Could not extract column statistics: {exc}")

                # Build relationship metadata
                relationship_metadata = {}
                if extract_relationships:
                    foreign_keys = enhanced_metadata.get("foreign_keys", [])
                    join_paths: List[str] = []
                    relationship_source = ""
                    if (
                        metadata_extractor
                        and _normalize_dialect(getattr(metadata_extractor, "dialect", "")) == DBType.STARROCKS
                    ):
                        try:
                            relationships = metadata_extractor.detect_relationships(table_name)
                            if relationships:
                                foreign_keys = relationships.get("foreign_keys", []) or foreign_keys
                                join_paths = relationships.get("join_paths", []) or []
                                if foreign_keys or join_paths:
                                    relationship_source = "information_schema"
                        except Exception as exc:
                            logger.debug(f"  Could not detect StarRocks relationships: {exc}")
                    if not foreign_keys and column_names and table_name_map:
                        inferred = _infer_relationships_from_names(column_names, table_name_map)
                        if inferred:
                            foreign_keys = inferred.get("foreign_keys", [])
                            join_paths = inferred.get("join_paths", [])
                            if foreign_keys or join_paths:
                                relationship_source = "heuristic"
                    if foreign_keys:
                        if not join_paths:
                            join_paths = [
                                f"{fk['from_column']} -> {fk['to_table']}.{fk['to_column']}"
                                for fk in foreign_keys
                            ]
                        relationship_metadata = {
                            "foreign_keys": foreign_keys,
                            "join_paths": join_paths,
                        }
                        if relationship_source:
                            relationship_metadata["source"] = relationship_source

                # Prepare update data
                update_data = {
                    "identifier": identifier,
                    "catalog_name": catalog_name,
                    "database_name": database_name,
                    "schema_name": schema_name,
                    "table_name": table_name,
                    "table_type": table_type,
                    "definition": definition,
                    # New v1 fields
                    "table_comment": table_comment,
                    "column_comments": json.dumps(column_comments, ensure_ascii=False),
                    "column_enums": json.dumps(column_enums, ensure_ascii=False),
                    "business_tags": business_tags,
                    "row_count": row_count,
                    "sample_statistics": json.dumps(sample_statistics, ensure_ascii=False),
                    "relationship_metadata": json.dumps(relationship_metadata, ensure_ascii=False),
                    "metadata_version": 1,  # Mark as v1
                    "last_updated": int(time.time())
                }
                if missing_required:
                    update_data = {key: value for key, value in update_data.items() if key in table_fields}

                # LanceDB handles embedding automatically via table's embedding function config
                action = "UPDATE" if identifier in existing_identifiers and update_existing_effective else "INSERT"
                batch_updates.append(update_data)
                batch_status.append((qualified_table, action))
                migrated_count += 1

                # Batch insert (no delete needed since table was recreated)
                if len(batch_updates) >= batch_size:
                    if shutdown_requested():
                        logger.warning("Shutdown requested; skipping batch insert.")
                        break
                    try:
                        storage.table.add(batch_updates)
                    except Exception as exc:
                        for table_name, table_action in batch_status:
                            logger.error(f"  - {table_action} FAILED: {table_name} ({exc})")
                    else:
                        for table_name, table_action in batch_status:
                            logger.info(f"  - {table_action} OK (COMPLETED): {table_name}")
                        logger.info(
                            f"Migrated batch of {len(batch_updates)} records (total: {migrated_count}/{len(backup_records)})"
                        )
                    batch_updates = []
                    batch_status = []

            except Exception as e:
                logger.error(f"Failed to migrate record {i}: {e}")
                continue

        # Flush remaining records
        if batch_updates:
            try:
                storage.table.add(batch_updates)
            except Exception as exc:
                for table_name, table_action in batch_status:
                    logger.error(f"  - {table_action} FAILED: {table_name} ({exc})")
            else:
                for table_name, table_action in batch_status:
                    logger.info(f"  - {table_action} OK (COMPLETED): {table_name}")
                logger.info(f"Migrated final batch of {len(batch_updates)} records")

        if shutdown_requested():
            logger.warning(
                f"Migration interrupted by user: processed {migrated_count}/{total_records} records"
            )
            return migrated_count

        logger.info(f"Migration completed: {migrated_count} records upgraded to v1")
        return migrated_count

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise


def migrate_schema_value_storage(
    storage: SchemaWithValueRAG,
) -> int:
    """
    Migrate schema value storage to v1 format (adds new fields with defaults).

    Args:
        storage: SchemaWithValueRAG instance

    Returns:
        Number of records migrated
    """
    try:
        value_store = storage.value_store
        value_store._ensure_table_ready()

        # Get all existing records
        all_data = value_store._search_all(
            where=None,
            select_fields=["identifier", "catalog_name", "database_name", "schema_name", "table_name", "table_type", "sample_rows"]
        )

        if not all_data or len(all_data) == 0:
            logger.info("No existing data found in schema value storage")
            return 0

        logger.info(f"Schema value storage has {len(all_data)} records (no migration needed, v1 compatible)")
        return len(all_data)

    except Exception as e:
        logger.error(f"Schema value storage check failed: {e}")
        return 0


def verify_migration_detailed(storage: SchemaWithValueRAG) -> bool:
    """
    Verify migration with detailed diagnostic output.

    This function performs comprehensive verification including:
    - Table existence check
    - Record count verification
    - Schema field validation
    - Metadata version distribution analysis
    - Detailed error reporting with actionable suggestions

    Args:
        storage: SchemaWithValueRAG instance

    Returns:
        True if migration appears successful
    """
    try:
        schema_store = storage.schema_store
        schema_store._ensure_table_ready()

        logger.info("=" * 80)
        logger.info("VERIFICATION DETAILS")
        logger.info("=" * 80)

        # Check table state
        table_exists = schema_store.table_name in schema_store.db.table_names()
        logger.info(f"Table exists: {table_exists}")
        logger.info(f"Table name: {schema_store.table_name}")
        logger.info(f"Database path: {schema_store.db_path}")

        if not table_exists:
            logger.error("")
            logger.error("❌ Table was not created - migration FAILED")
            logger.info("")
            logger.info("Possible causes:")
            logger.info("  1. Storage path permission issue")
            logger.info("  2. Disk space exhausted")
            logger.info("  3. Configuration error (wrong namespace/path)")
            logger.info("  4. Database connection failure")
            logger.info("")
            logger.info("Troubleshooting steps:")
            logger.info("  • Check write permissions on storage directory")
            logger.info("  • Verify available disk space")
            logger.info("  • Confirm namespace configuration in agent.yml")
            logger.info("  • Check logs above for specific errors")
            logger.info("=" * 80)
            return False

        # Check table schema
        try:
            table = schema_store.db.open_table(schema_store.table_name)
            schema = table.schema
            field_names = schema.names
            logger.info(f"Schema fields ({len(field_names)}): {field_names}")

            # Verify v1 fields exist
            v1_fields = [
                "table_comment", "column_comments", "column_enums", "business_tags",
                "row_count", "sample_statistics", "relationship_metadata",
                "metadata_version", "last_updated"
            ]
            missing_v1_fields = [f for f in v1_fields if f not in field_names]

            if missing_v1_fields:
                logger.warning("")
                logger.warning(f"⚠️  Missing v1 fields: {missing_v1_fields}")
                logger.warning("  → Table may not have been created with v1 schema")
                logger.warning("  → Re-run cleanup and migration scripts")
                logger.info("=" * 80)
                return False

            logger.info("✅ All v1 fields present in schema")

        except Exception as e:
            logger.error(f"Error checking table schema: {e}")
            logger.info("=" * 80)
            return False

        # Get data
        try:
            all_data = schema_store._search_all(
                where=None,
                select_fields=["metadata_version"]
            )

            logger.info(f"Record count: {len(all_data)}")

            if len(all_data) == 0:
                logger.info("")
                logger.info("✅ Empty v1 table created successfully")
                logger.info("   This is expected for fresh installations")
                logger.info("   The migration completed - no data to migrate")
                logger.info("")
                logger.info("Migration status: SUCCESS (empty v1 table)")
                logger.info("=" * 80)
                return True

            # Has data - check versions
            version_counts = Counter(
                row.get("metadata_version", 0)
                for row in all_data.to_pylist()
            )

            logger.info(f"Version distribution: {dict(version_counts)}")

            v1_count = version_counts.get(1, 0)
            v0_count = version_counts.get(0, 0)
            other_count = sum(v for k, v in version_counts.items() if k not in [0, 1])
            total_count = sum(version_counts.values())

            logger.info(f"  v0 (legacy) records: {v0_count}")
            logger.info(f"  v1 (enhanced) records: {v1_count}")
            if other_count > 0:
                logger.info(f"  other version records: {other_count}")

            # Calculate migration percentage
            if total_count > 0:
                migration_pct = (v1_count / total_count) * 100
                logger.info(f"  Migration completion: {migration_pct:.1f}%")

            # Determine success
            if v1_count > 0 and v0_count == 0:
                logger.info("")
                logger.info("✅ Migration successful - all records upgraded to v1")
                logger.info("=" * 80)
                return True
            elif v1_count > 0:
                logger.warning("")
                logger.warning(f"⚠️  Partial migration: {v1_count}/{total_count} records upgraded")
                logger.warning("  → Some v0 records remain")
                logger.warning("  → Check migration logs for errors")
                logger.warning("  → Re-run migration with --force to retry")
                logger.info("=" * 80)
                return False
            else:
                logger.warning("")
                logger.warning("⚠️  No v1 records created - migration failed")
                logger.warning("  → Check logs above for errors during data processing")
                logger.warning("  → Verify backup file was correctly loaded")
                logger.warning("  → Re-run migration with --force")
                logger.info("=" * 80)
                return False

        except Exception as e:
            logger.error(f"Error reading table data: {e}")
            logger.error("")
            logger.error("Possible causes:")
            logger.error("  1. Table corruption during migration")
            logger.error("  2. Insufficient permissions to read data")
            logger.error("  3. Data format incompatibility")
            logger.info("=" * 80)
            return False

    except Exception as e:
        logger.error(f"Verification failed with exception: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        logger.info("=" * 80)
        return False


def verify_migration(storage: SchemaWithValueRAG) -> bool:
    """
    Verify migration success by checking metadata_version distribution.

    This is a simplified version of verify_migration_detailed() for backward compatibility.

    Args:
        storage: SchemaWithValueRAG instance

    Returns:
        True if migration appears successful
    """
    return verify_migration_detailed(storage)
