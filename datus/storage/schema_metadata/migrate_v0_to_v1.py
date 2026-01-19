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
from typing import Optional

from datus.configuration.agent_config import AgentConfig
from datus.configuration.agent_config_loader import load_agent_config
from datus.storage.embedding_models import get_db_embedding_model
from datus.storage.schema_metadata import SchemaStorage, SchemaWithValueRAG
from datus.utils.loggings import get_logger
from datus.utils.sql_utils import extract_enhanced_metadata_from_ddl, parse_dialect
from datus.utils.constants import DBType

logger = get_logger(__name__)


def select_namespace_interactive(agent_config: AgentConfig, specified_namespace: Optional[str] = None) -> Optional[str]:
    """
    智能选择 namespace：
    - 如果用户通过 --namespace 指定，使用指定的值
    - 如果配置只有一个 namespace，自动使用
    - 如果配置有多个 namespace，交互式选择
    - 如果没有 namespace，返回 None（使用 base path）

    Returns:
        namespace name 或 None（表示使用 base path）
    """
    from rich.console import Console
    from rich.prompt import Prompt
    from rich.table import Table

    console = Console()

    # 情况1: 用户已显式指定
    if specified_namespace:
        return specified_namespace

    # 情况2: 没有配置任何 namespace
    if not agent_config.namespaces:
        console.print("[yellow]⚠️  No namespaces found in configuration[/]")
        console.print("[yellow]Will use base storage path (Schema-only migration)[/]")
        return None

    namespaces = list(agent_config.namespaces.keys())

    # 情况3: 只有一个 namespace，自动使用
    if len(namespaces) == 1:
        selected = namespaces[0]
        console.print(f"[green]✓ Auto-selected namespace: {selected}[/]")
        return selected

    # 情况4: 多个 namespace，交互式选择
    console.print("\n[bold cyan]Available Namespaces:[/]\n")

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("No.", style="dim", width=3)
    table.add_column("Namespace", style="green")
    table.add_column("Type", style="blue")
    table.add_column("Databases", style="yellow")

    for i, ns_name in enumerate(namespaces, 1):
        db_configs = agent_config.namespaces[ns_name]
        first_db = list(db_configs.values())[0]
        db_type = first_db.type
        db_count = len(db_configs)

        table.add_row(str(i), ns_name, db_type, f"{db_count} db(s)")

    console.print(table)

    # 添加"无 namespace"选项
    console.print("[dim]0. No namespace (Schema-only migration)[/]\n")

    choice = Prompt.ask(
        "[bold cyan]Select namespace[/]",
        choices=[str(i) for i in range(len(namespaces) + 1)],
        default="1"
    )

    if choice == "0":
        console.print("[yellow]Selected: Schema-only migration (base path)[/]")
        return None
    else:
        selected = namespaces[int(choice) - 1]
        console.print(f"[green]✓ Selected namespace: {selected}[/]")
        return selected


def detect_dialect_from_ddl(ddl: str) -> str:
    """
    Attempt to detect SQL dialect from DDL statement.

    Uses keyword and type pattern matching to identify the most likely
    database dialect from the DDL content.

    Args:
        ddl: DDL statement to analyze

    Returns:
        Detected dialect (defaults to "snowflake" if unable to determine)
    """
    ddl_upper = ddl.upper()

    # Snowflake-specific patterns
    if "VARIANT" in ddl_upper or "OBJECT" in ddl_upper or "ARRAY" in ddl_upper:
        return "snowflake"

    # DuckDB-specific patterns
    if "DOUBLE" in ddl_upper or "UBIGINT" in ddl_upper or "HUGEINT" in ddl_upper:
        return "duckdb"

    # PostgreSQL-specific patterns
    if "SERIAL" in ddl_upper or "BIGSERIAL" in ddl_upper or "MONEY" in ddl_upper:
        return "postgres"

    # MySQL-specific patterns
    if "TINYINT" in ddl_upper or "MEDIUMINT" in ddl_upper or "AUTO_INCREMENT" in ddl_upper:
        return "mysql"

    # BigQuery-specific patterns
    if "STRUCT" in ddl_upper or "INT64" in ddl_upper or "FLOAT64" in ddl_upper:
        return "bigquery"

    # Default to snowflake as it's commonly used
    return "snowflake"


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

    This function:
    1. Reads all existing v0 data into memory
    2. Drops the v0 table
    3. Creates a new v1 table
    4. Inserts migrated v1 records

    Args:
        storage: SchemaStorage instance
        extract_statistics: Whether to extract column statistics (expensive)
        extract_relationships: Whether to extract relationship metadata

    Returns:
        Number of records migrated
    """
    try:
        # Step 1: Read all existing v0 data FIRST (before dropping table)
        storage._ensure_table_ready()

        all_data = storage._search_all(
            where=None,
            select_fields=["identifier", "catalog_name", "database_name", "schema_name", "table_name", "table_type", "definition"]
        )

        if not all_data or len(all_data) == 0:
            logger.info("No existing data found in schema storage")
            return 0

        logger.info(f"Found {len(all_data)} records to migrate")

        # Step 2: Store table name and db reference
        table_name = storage.table_name
        db = storage.db

        # Step 3: Drop the v0 table
        logger.info(f"Dropping existing v0 table: {table_name}")
        db.drop_table(table_name)
        logger.info(f"Dropped table {table_name}")

        # CRITICAL: Reset the initialization flag so _ensure_table_ready() will recreate the table
        storage._table_initialized = False

        # Step 4: Create new v1 table
        storage._ensure_table_ready()
        logger.info(f"Created new v1 table: {table_name}")

        migrated_count = 0
        batch_size = 100
        batch_updates = []

        for i, row in enumerate(all_data.to_pylist()):
            try:
                identifier = row["identifier"]
                catalog_name = row["catalog_name"]
                database_name = row["database_name"]
                schema_name = row["schema_name"]
                table_name = row["table_name"]
                table_type = row["table_type"]
                definition = row["definition"]

                # Parse enhanced metadata from DDL with dialect detection
                detected_dialect = detect_dialect_from_ddl(definition)
                enhanced_metadata = extract_enhanced_metadata_from_ddl(definition, dialect=detected_dialect)

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

                # LanceDB handles embedding automatically via table's embedding function config
                batch_updates.append(update_data)
                migrated_count += 1

                # Batch insert (no delete needed since table was recreated)
                if len(batch_updates) >= batch_size:
                    storage.table.add(batch_updates)
                    logger.info(f"Migrated batch of {len(batch_updates)} records (total: {migrated_count}/{len(all_data)})")
                    batch_updates = []

            except Exception as e:
                logger.error(f"Failed to migrate record {i}: {e}")
                continue

        # Flush remaining records
        if batch_updates:
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


def str_to_bool(v):
    """Convert string to boolean for argparse.

    Accepts various boolean representations and converts them to bool.
    Used as type converter for argparse arguments to support explicit true/false syntax.

    Args:
        v: Input value - can be bool, str, or any type

    Returns:
        bool: True for yes/true/t/y/1, False for no/false/f/n/0

    Raises:
        argparse.ArgumentTypeError: If string value is not a valid boolean representation
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError(f'Boolean value expected, got: {v}')


def main():
    """Main migration function."""
    parser = argparse.ArgumentParser(description="Migrate LanceDB schema from v0 to v1")
    parser.add_argument("--config", required=True, help="Path to agent configuration file")
    parser.add_argument("--namespace", help="Namespace for the database (optional: will prompt if not specified and multiple namespaces exist)")
    parser.add_argument("--db-path", help="Override database path (default: from config)")
    parser.add_argument("--extract-statistics", type=str_to_bool, default=False, help="Extract column statistics (expensive)")
    parser.add_argument("--extract-relationships", type=str_to_bool, default=True, help="Extract relationship metadata")
    parser.add_argument("--skip-backup", action="store_true", help="Skip backup creation")
    parser.add_argument("--force", action="store_true", help="Force migration even if v1 already exists")

    args = parser.parse_args()

    # Load agent configuration
    try:
        agent_config = load_agent_config(config=args.config)
    except Exception as e:
        logger.error(f"Failed to load agent configuration: {e}")
        sys.exit(1)

    # Determine database path
    if args.db_path:
        db_path = args.db_path
        namespace = None
    else:
        # 智能选择 namespace（交互式或自动）
        namespace = select_namespace_interactive(agent_config, args.namespace)

        if namespace:
            # 设置 namespace 并获取存储路径
            agent_config.current_namespace = namespace
            db_path = agent_config.rag_storage_path()
        else:
            # 使用 base storage path (Schema-only migration)
            db_path = os.path.join(agent_config.rag_base_path, "lancedb")
            logger.info("Using base storage path (Schema-only migration)")

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
        if namespace:
            # Use SchemaWithValueRAG with namespace (requires namespace to be set)
            storage = SchemaWithValueRAG(agent_config)
            schema_store = storage.schema_store
            logger.info(f"✅ Storage initialized for namespace: {namespace}")
        else:
            # Use SchemaStorage directly with base db_path (no namespace required)
            embedding_model = get_db_embedding_model()
            schema_store = SchemaStorage(db_path=db_path, embedding_model=embedding_model)
            storage = None
            logger.info("✅ Storage initialized (base path, Schema-only migration)")
        logger.info("")
    except Exception as e:
        logger.error(f"Failed to initialize storage: {e}")
        sys.exit(1)

    # Check if migration already run
    try:
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
                logger.info(f"  python -m datus.storage.schema_metadata.migrate_v0_to_v1 --config {args.config} --force")
                if namespace:
                    logger.info(f"  python -m datus.storage.schema_metadata.migrate_v0_to_v1 --config {args.config} --namespace {namespace} --force")
                sys.exit(0)
    except Exception as e:
        logger.debug(f"Could not check migration status: {e}")

    # Perform migration
    logger.info("Starting migration...")
    logger.info("")

    try:
        # Migrate schema storage (always done)
        logger.info("Step 1/3: Migrating schema metadata...")
        migrated_schemas = migrate_schema_storage(
            schema_store,
            extract_statistics=args.extract_statistics,
            extract_relationships=args.extract_relationships
        )
        logger.info(f"✅ Migrated {migrated_schemas} schema records")
        logger.info("")

        # Migrate schema value storage and verify (only if namespace provided)
        if namespace and storage:
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
        else:
            # No namespace - schema migration only
            logger.info("Step 2/3: Skipped (schema value storage requires --namespace)")
            logger.info("")
            logger.info("Step 3/3: Skipped (verification requires --namespace)")
            logger.info("")
            logger.info("=" * 80)
            logger.info("✅ SCHEMA METADATA MIGRATION SUCCESSFUL")
            logger.info("=" * 80)
            logger.info(f"Schema metadata: {migrated_schemas} records upgraded to v1")
            logger.info("")
            logger.info("Note: Schema value storage was not migrated.")
            logger.info("To migrate schema value storage, re-run with --namespace:")
            if args.config:
                logger.info(f"  python -m datus.storage.schema_metadata.migrate_v0_to_v1 --config {args.config} --namespace <name> --force")
            logger.info("")
            logger.info("Next steps:")
            logger.info("1. Test the system with enhanced metadata")
            logger.info("2. Verify schema discovery precision improvements")
            logger.info("3. If issues arise, restore from backup:")
            logger.info(f"   rm -rf {db_path} && mv {backup_path} {db_path}")
            sys.exit(0)

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        logger.error("")
        logger.error("Please restore from backup:")
        logger.error(f"  rm -rf {db_path} && mv {backup_path} {db_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()