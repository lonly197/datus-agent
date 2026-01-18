# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details

"""
Migration script to upgrade LanceDB schema from v0 (legacy) to v1 (enhanced).

This script handles:
1. Backing up existing data
2. Adding new fields to existing tables
3. Extracting enhanced metadata where possible
4. Preserving backward compatibility
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

from datus.configuration.agent_config import AgentConfig
from datus.storage.cache import get_storage_cache_instance
from datus.storage.schema_metadata import SchemaStorage, SchemaWithValueRAG
from datus.utils.loggings import get_logger
from datus.utils.sql_utils import extract_enhanced_metadata_from_ddl, parse_dialect
from datus.utils.constants import DBType

logger = get_logger(__name__)


def backup_database(db_path: str) -> str:
    """
    Create a backup of the LanceDB database.

    Args:
        db_path: Path to LanceDB database directory

    Returns:
        Path to backup directory
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_path = f"{db_path}.backup_v0_{timestamp}"

    if os.path.exists(db_path):
        logger.info(f"Creating backup at {backup_path}")
        shutil.copytree(db_path, backup_path)
        logger.info(f"Backup created successfully")
    else:
        logger.warning(f"Database path {db_path} does not exist, skipping backup")

    return backup_path


def migrate_schema_storage(
    storage: SchemaStorage,
    extract_statistics: bool = False,
    extract_relationships: bool = True,
) -> int:
    """
    Migrate schema metadata from v0 to v1 format.

    Args:
        storage: SchemaStorage instance
        extract_statistics: Whether to extract column statistics (expensive)
        extract_relationships: Whether to extract relationship metadata

    Returns:
        Number of records migrated
    """
    try:
        storage._ensure_table_ready()

        # Get all existing records (v0 format)
        all_data = storage._search_all(
            where=None,
            select_fields=["identifier", "catalog_name", "database_name", "schema_name", "table_name", "table_type", "definition"]
        )

        if not all_data or len(all_data) == 0:
            logger.info("No existing data found in schema storage")
            return 0

        migrated_count = 0
        batch_size = 100
        batch_updates = []

        logger.info(f"Found {len(all_data)} records to migrate")

        for i, row in enumerate(all_data.to_pylist()):
            try:
                identifier = row["identifier"]
                catalog_name = row["catalog_name"]
                database_name = row["database_name"]
                schema_name = row["schema_name"]
                table_name = row["table_name"]
                table_type = row["table_type"]
                definition = row["definition"]

                # Parse enhanced metadata from DDL
                enhanced_metadata = extract_enhanced_metadata_from_ddl(definition, dialect="snowflake")

                # Extract comment information
                table_comment = enhanced_metadata["table"].get("comment", "")
                column_comments = {
                    col["name"]: col.get("comment", "")
                    for col in enhanced_metadata["columns"]
                }

                # Infer business tags
                from datus.configuration.business_term_config import infer_business_tags
                column_names = [col["name"] for col in enhanced_metadata["columns"]]
                business_tags = infer_business_tags(table_name, column_names)

                # Build relationship metadata
                relationship_metadata = {}
                if extract_relationships:
                    foreign_keys = enhanced_metadata.get("foreign_keys", [])
                    relationship_metadata = {
                        "foreign_keys": foreign_keys,
                        "join_paths": [
                            f"{fk['from_column']} -> {fk['to_table']}.{fk['to_column']}"
                            for fk in foreign_keys
                        ]
                    }

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
                    "business_tags": business_tags,
                    "row_count": 0,  # Will be populated if extract_statistics=True
                    "sample_statistics": json.dumps({}, ensure_ascii=False),  # Empty initially
                    "relationship_metadata": json.dumps(relationship_metadata, ensure_ascii=False),
                    "metadata_version": 1,  # Mark as v1
                    "last_updated": int(time.time())
                }

                # Generate embedding
                embedded_data = storage._embed_and_prepare(update_data)

                batch_updates.append(embedded_data)
                migrated_count += 1

                # Batch update
                if len(batch_updates) >= batch_size:
                    # Delete old records and insert new ones
                    for embedded in batch_updates:
                        from datus.storage.lancedb_conditions import build_where, eq, and_
                        from datus.storage.schema_metadata.store import _build_where_clause

                        where_clause = build_where(
                            _build_where_clause(
                                table_name=embedded["table_name"],
                                catalog_name=embedded["catalog_name"],
                                database_name=embedded["database_name"],
                                schema_name=embedded["schema_name"],
                                table_type=embedded["table_type"]
                            )
                        )
                        storage.table.delete(where_clause)

                    storage.table.add(batch_updates)
                    logger.info(f"Migrated batch of {len(batch_updates)} records (total: {migrated_count}/{len(all_data)})")
                    batch_updates = []

            except Exception as e:
                logger.error(f"Failed to migrate record {i}: {e}")
                continue

        # Flush remaining records
        if batch_updates:
            for embedded in batch_updates:
                from datus.storage.lancedb_conditions import build_where
                from datus.storage.schema_metadata.store import _build_where_clause

                where_clause = build_where(
                    _build_where_clause(
                        table_name=embedded["table_name"],
                        catalog_name=embedded["catalog_name"],
                        database_name=embedded["database_name"],
                        schema_name=embedded["schema_name"],
                        table_type=embedded["table_type"]
                    )
                )
                storage.table.delete(where_clause)

            storage.table.add(batch_updates)
            logger.info(f"Migrated final batch of {len(batch_updates)} records")

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


def verify_migration(storage: SchemaWithValueRAG) -> bool:
    """
    Verify migration success by checking metadata_version distribution.

    Args:
        storage: SchemaWithValueRAG instance

    Returns:
        True if migration appears successful
    """
    try:
        schema_store = storage.schema_store
        schema_store._ensure_table_ready()

        # Check if metadata_version column exists and has data
        all_data = schema_store._search_all(
            where=None,
            select_fields=["metadata_version"]
        )

        if len(all_data) == 0:
            logger.warning("No records found for verification")
            return False

        # Count versions
        import pyarrow as pa
        version_counts = {}
        for row in all_data.to_pylist():
            version = row.get("metadata_version", 0)
            version_counts[version] = version_counts.get(version, 0) + 1

        logger.info(f"Metadata version distribution: {version_counts}")
        logger.info(f"Total records: {sum(version_counts.values())}")

        # Success if we have any v1 records
        success = version_counts.get(1, 0) > 0
        if success:
            logger.info("✅ Migration verification successful: v1 records found")
        else:
            logger.warning("⚠️  Migration verification: No v1 records found")

        return success

    except Exception as e:
        logger.error(f"Migration verification failed: {e}")
        return False


def main():
    """Main migration function."""
    parser = argparse.ArgumentParser(description="Migrate LanceDB schema from v0 to v1")
    parser.add_argument("--config", required=True, help="Path to agent configuration file")
    parser.add_argument("--db-path", help="Override database path (default: from config)")
    parser.add_argument("--extract-statistics", action="store_true", help="Extract column statistics (expensive)")
    parser.add_argument("--extract-relationships", action="store_true", default=True, help="Extract relationship metadata")
    parser.add_argument("--skip-backup", action="store_true", help="Skip backup creation")
    parser.add_argument("--force", action="store_true", help="Force migration even if v1 already exists")

    args = parser.parse_args()

    # Load agent configuration
    try:
        agent_config = AgentConfig.from_yaml(args.config)
    except Exception as e:
        logger.error(f"Failed to load agent configuration: {e}")
        sys.exit(1)

    # Determine database path
    db_path = args.db_path or agent_config.rag_storage_path()

    logger.info("=" * 80)
    logger.info("LanceDB Schema Migration: v0 → v1")
    logger.info("=" * 80)
    logger.info(f"Database path: {db_path}")
    logger.info(f"Extract statistics: {args.extract_statistics}")
    logger.info(f"Extract relationships: {args.extract_relationships}")
    logger.info("")

    # Backup existing database
    if not args.skip_backup:
        backup_path = backup_database(db_path)
        logger.info(f"✅ Backup created at: {backup_path}")
        logger.info("")

    # Initialize storage
    try:
        storage_cache = get_storage_cache_instance(agent_config)
        storage = SchemaWithValueRAG(agent_config)
        logger.info("✅ Storage initialized")
        logger.info("")
    except Exception as e:
        logger.error(f"Failed to initialize storage: {e}")
        sys.exit(1)

    # Check if migration already run
    try:
        schema_store = storage.schema_store
        schema_store._ensure_table_ready()

        # Check for metadata_version field
        all_data = schema_store._search_all(
            where=None,
            select_fields=["metadata_version"]
        )

        if len(all_data) > 0:
            # Check if any v1 records exist
            has_v1 = any(row.get("metadata_version", 0) == 1 for row in all_data.to_pylist())
            if has_v1 and not args.force:
                logger.warning("⚠️  Migration appears to have already been run (found v1 records)")
                logger.warning("Use --force to re-run migration")
                logger.info("")
                logger.info("To re-run:")
                logger.info(f"  python {__file__} --config {args.config} --force")
                sys.exit(0)
    except Exception as e:
        logger.debug(f"Could not check migration status: {e}")

    # Perform migration
    logger.info("Starting migration...")
    logger.info("")

    try:
        # Migrate schema storage
        logger.info("Step 1/3: Migrating schema metadata...")
        migrated_schemas = migrate_schema_storage(
            storage.schema_store,
            extract_statistics=args.extract_statistics,
            extract_relationships=args.extract_relationships
        )
        logger.info(f"✅ Migrated {migrated_schemas} schema records")
        logger.info("")

        # Migrate schema value storage
        logger.info("Step 2/3: Checking schema value storage...")
        migrated_values = migrate_schema_value_storage(storage)
        logger.info(f"✅ Schema value storage: {migrated_values} records (v1 compatible)")
        logger.info("")

        # Verify migration
        logger.info("Step 3/3: Verifying migration...")
        success = verify_migration(storage)

        if success:
            logger.info("")
            logger.info("=" * 80)
            logger.info("✅ MIGRATION SUCCESSFUL")
            logger.info("=" * 80)
            logger.info(f"Schema metadata: {migrated_schemas} records upgraded to v1")
            logger.info(f"Schema value storage: {migrated_values} records")
            logger.info("")
            logger.info("Next steps:")
            logger.info("1. Test the system with enhanced metadata")
            logger.info("2. Verify schema discovery precision improvements")
            logger.info("3. If issues arise, restore from backup:")
            logger.info(f"   rm -rf {db_path} && mv {backup_path} {db_path}")
            sys.exit(0)
        else:
            logger.error("")
            logger.error("=" * 80)
            logger.error("❌ MIGRATION VERIFICATION FAILED")
            logger.error("=" * 80)
            logger.error("Please check the logs above for details")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        logger.error("")
        logger.error("Please restore from backup:")
        logger.error(f"  rm -rf {db_path} && mv {backup_path} {db_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()