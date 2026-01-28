# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details

"""
Migration script to upgrade LanceDB schema from v0 (legacy) to v1 (enhanced).

This script has been refactored into separate modules for better maintainability.
All public functions are re-exported from the new modules for backward compatibility.

REFACTORED STRUCTURE:
    datus/storage/schema_metadata/migrate_v0_to_v1_modules/
    ├── __init__.py           # Public API exports
    ├── models.py             # Data models and schema definitions
    ├── utils.py              # Helper functions and utilities
    ├── migration_logic.py    # Core migration logic
    └── batch_processor.py    # Batch processing and reporting

MIGRATION GUIDE:
    Old imports (still work):
        from datus.storage.schema_metadata.migrate_v0_to_v1 import migrate_schema_storage
        from datus.storage.schema_metadata.migrate_v0_to_v1 import verify_migration

    New imports (recommended):
        from datus.storage.schema_metadata.migrate_v0_to_v1_modules import migrate_schema_storage
        from datus.storage.schema_metadata.migrate_v0_to_v1_modules import verify_migration

USAGE:
    # CLI usage
    python -m datus.storage.schema_metadata.migrate_v0_to_v1 \\
        --config=agent.yml \\
        --namespace=my_namespace

    # Programmatic usage
    from datus.storage.schema_metadata.migrate_v0_to_v1_modules import migrate_schema_storage

For detailed refactoring information, see the module docstrings in each submodule.
"""

import argparse
import sys

from datus.configuration.agent_config_loader import load_agent_config
from datus.storage.embedding_models import get_db_embedding_model
from datus.storage.schema_metadata import SchemaStorage, SchemaWithValueRAG
from datus.models.base import LLMBaseModel
from datus.utils.loggings import configure_logging, get_logger

# Re-export everything from the new modules for backward compatibility
from .migrate_v0_to_v1_modules import (
    # Core migration functions
    migrate_schema_storage,
    migrate_schema_value_storage,
    verify_migration,
    verify_migration_detailed,
    # Utility functions
    setup_signal_handlers,
    shutdown_requested,
    select_namespace_interactive,
    backup_database,
    report_migration_state,
    str_to_bool,
    # Batch processing and reporting
    import_schema_metadata,
    print_recovery_suggestions,
    print_final_migration_report,
    # Data models
    MigrationRecord,
    MigrationResult,
    METADATA_VERSION_V0,
    METADATA_VERSION_V1,
)

logger = get_logger(__name__)

__all__ = [
    # Core migration functions
    "migrate_schema_storage",
    "migrate_schema_value_storage",
    "verify_migration",
    "verify_migration_detailed",
    # Utility functions
    "setup_signal_handlers",
    "shutdown_requested",
    "select_namespace_interactive",
    "backup_database",
    "report_migration_state",
    "str_to_bool",
    # Batch processing and reporting
    "import_schema_metadata",
    "print_recovery_suggestions",
    "print_final_migration_report",
    # Data models
    "MigrationRecord",
    "MigrationResult",
    "METADATA_VERSION_V0",
    "METADATA_VERSION_V1",
    "main",
]


def main():
    """Main migration function."""
    setup_signal_handlers()
    parser = argparse.ArgumentParser(description="Migrate LanceDB schema from v0 to v1")
    parser.add_argument("--config", required=True, help="Path to agent configuration file")
    parser.add_argument("--namespace", help="Namespace for the database (optional: will prompt if not specified and multiple namespaces exist)")
    parser.add_argument("--db-path", help="Override database path (default: from config)")
    parser.add_argument("--extract-statistics", type=str_to_bool, default=False, help="Extract column statistics (expensive)")
    parser.add_argument("--extract-relationships", type=str_to_bool, default=True, help="Extract relationship metadata")
    parser.add_argument("--backup-path", help="Path to backup JSON file from cleanup script")
    parser.add_argument("--skip-backup", action="store_true", help="Skip backup creation")
    parser.add_argument("--force", action="store_true", help="Force migration even if v1 already exists")
    parser.add_argument("--import-schemas", action="store_true", help="Import schema metadata from database after migration (recommended for fresh installations)")
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing schema_metadata and schema_value before import",
    )
    parser.add_argument(
        "--import-only",
        action="store_true",
        help="Skip migration and only import schemas from database (requires --import-schemas)",
    )
    parser.add_argument("--llm-fallback", action="store_true", help="Use LLM as final fallback for DDL parsing failures")
    parser.add_argument("--llm-model", help="Optional model name for LLM fallback (defaults to active model)")
    parser.add_argument("--llm-enum-extraction", action="store_true", help="Use LLM to enhance enum value extraction from comments")

    args = parser.parse_args()
    configure_logging(debug=False)

    # Load agent configuration
    try:
        agent_config = load_agent_config(config=args.config)
    except Exception as e:
        logger.error(f"Failed to load agent configuration: {e}")
        sys.exit(1)

    # Determine database path
    import os

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
    logger.info(f"LLM fallback: {args.llm_fallback}")
    logger.info(f"LLM enum extraction: {args.llm_enum_extraction}")
    logger.info("")

    # Report current migration state (diagnostics)
    report_migration_state(db_path, "schema_metadata")

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

    if args.clear:
        logger.info("Clearing existing schema_metadata and schema_value before migration...")
        try:
            if schema_store.table_name in schema_store.db.table_names(limit=100):
                schema_store.db.drop_table(schema_store.table_name)
            schema_store.table = None
            schema_store._table_initialized = False
            if storage:
                value_store = storage.value_store
                if value_store.table_name in value_store.db.table_names(limit=100):
                    value_store.db.drop_table(value_store.table_name)
                value_store.table = None
                value_store._table_initialized = False
            schema_store._ensure_table_ready()
            if storage:
                storage.value_store._ensure_table_ready()
        except Exception as exc:
            logger.error(f"Failed to clear schema tables before migration: {exc}")
            sys.exit(1)
        logger.info("✅ Cleared schema tables before migration")
        logger.info("")

    if args.import_only and not args.import_schemas:
        logger.error("--import-only requires --import-schemas")
        sys.exit(1)

    # Check if migration already run (skip when import-only)
    if not args.import_only:
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

    llm_model = None
    if args.llm_fallback:
        try:
            llm_model = LLMBaseModel.create_model(agent_config, model_name=args.llm_model)
        except Exception as e:
            logger.warning(f"Failed to initialize LLM model for fallback: {e}")
            logger.warning("Disabling LLM fallback for this run")
            args.llm_fallback = False

    db_manager = None
    if args.extract_statistics and namespace:
        try:
            from datus.tools.db_tools.db_manager import get_db_manager
            db_manager = get_db_manager(agent_config.namespaces)
        except Exception as e:
            logger.warning(f"Failed to initialize DB manager for statistics extraction: {e}")
            db_manager = None

    # Perform migration
    logger.info("Starting migration...")
    logger.info("")

    # Initialize result tracking
    migration_results = {
        "schemas_migrated": 0,
        "values_migrated": 0,
        "schemas_imported": 0,
        "success": False,
        "verification_passed": False,
        "namespace": namespace,
        "cancelled": False,
    }
    migrated_schemas = 0
    migrated_values = 0

    try:
        if args.import_only:
            logger.info("Step 1/3: Skipped (import-only mode)")
            logger.info("")
            migrated_schemas = 0
        else:
            # Migrate schema storage (always done)
            logger.info("Step 1/3: Migrating schema metadata...")
            migrated_schemas = migrate_schema_storage(
                schema_store,
                backup_path=args.backup_path,
                extract_statistics=args.extract_statistics,
                extract_relationships=args.extract_relationships,
                llm_fallback=args.llm_fallback,
                llm_model=llm_model,
                update_existing=args.force,
                db_manager=db_manager,
                namespace=namespace,
            )
            migration_results["schemas_migrated"] = migrated_schemas
            logger.info(f"✅ Migrated {migrated_schemas} schema records")
            logger.info("")

            if shutdown_requested():
                logger.warning("Migration interrupted by user; skipping remaining steps.")
                migration_results["cancelled"] = True
                sys.exit(130)

        # Migrate schema value storage and verify (only if namespace provided)
        if namespace and storage:
            # Migrate schema value storage
            if args.import_only:
                logger.info("Step 2/3: Skipped (import-only mode)")
                logger.info("")
                logger.info("Step 3/3: Skipped (import-only mode)")
                logger.info("")
                migration_results["success"] = True
                migration_results["verification_passed"] = False
                success = True
            else:
                logger.info("Step 2/3: Checking schema value storage...")
                migrated_values = migrate_schema_value_storage(storage)
                migration_results["values_migrated"] = migrated_values
                logger.info(f"✅ Schema value storage: {migrated_values} records (v1 compatible)")
                logger.info("")

                # Verify migration
                logger.info("Step 3/3: Verifying migration...")
                success = verify_migration(storage)
                migration_results["verification_passed"] = success
                migration_results["success"] = success

            if success:
                logger.info("")
                logger.info("=" * 80)
                logger.info("✅ MIGRATION SUCCESSFUL")
                logger.info("=" * 80)
                logger.info(f"Schema metadata: {migrated_schemas} records upgraded to v1")
                logger.info(f"Schema value storage: {migrated_values} records")
                logger.info("")

                # Import schema metadata if requested
                if args.import_schemas:
                    logger.info("Step 4/4: Importing schema metadata from database...")
                    imported_count = import_schema_metadata(
                        agent_config,
                        namespace,
                        clear_before_import=args.clear,
                        extract_statistics=args.extract_statistics,
                        extract_relationships=args.extract_relationships,
                        llm_enum_extraction=args.llm_enum_extraction,
                    )
                    migration_results["schemas_imported"] = imported_count

                    if imported_count > 0:
                        logger.info("")
                        logger.info("=" * 80)
                        logger.info("✅ MIGRATION + IMPORT COMPLETE")
                        logger.info("=" * 80)
                        logger.info(f"Schema metadata: {migrated_schemas} records upgraded to v1")
                        logger.info(f"Schema import: {imported_count} schemas imported from database")
                        logger.info("")
                        logger.info("Your system is now ready for text2sql queries!")
                        logger.info("")
                        logger.info("To verify the import:")
                        logger.info(f"  python -c \"")
                        logger.info(f"    from datus.storage.schema_metadata import SchemaStorage")
                        logger.info(f"    from datus.configuration.agent_config_loader import load_agent_config")
                        logger.info(f"    config = load_agent_config('{args.config}')")
                        logger.info(f"    config.current_namespace = '{namespace}'")
                        logger.info(f"    storage = SchemaStorage(db_path=config.rag_storage_path())")
                        logger.info(f"    print('Schema count:', len(storage._search_all(where=None)))")
                        logger.info(f"  \"")
                        logger.info("=" * 80)
                        sys.exit(0)
                    else:
                        logger.warning("")
                        logger.warning("Schema import returned 0 schemas. This may indicate:")
                        logger.warning("  1. Database connection issues")
                        logger.warning("  2. Database is empty (no tables)")
                        logger.warning("  3. Namespace/database name mismatch")
                        logger.warning("")
                        logger.warning("The migration was successful, but schema import failed.")
                        logger.warning("You can run schema import separately:")
                        logger.warning(f"  python -m datus.storage.schema_metadata.local_init \\")
                        logger.warning(f"    --config={args.config} --namespace={namespace}")
                        logger.warning("")
                        logger.warning("Or verify database connection:")
                        logger.warning(f"  python -c \"")
                        logger.warning(f"    from datus.tools.db_tools.db_manager import get_db_manager")
                        logger.warning(f"    db = get_db_manager().get_conn('{namespace}', '<database_name>')")
                        logger.warning(f"    print('Tables:', len(db.get_tables_with_ddl()))")
                        logger.warning(f"  \"")
                        logger.info("=" * 80)
                        sys.exit(1)
                else:
                    logger.info("Next steps:")
                    logger.info("1. Test the system with enhanced metadata")
                    logger.info("2. Verify schema discovery precision improvements")
                    logger.info("3. If this is a fresh installation, import schemas:")
                    logger.info(f"   python -m datus.storage.schema_metadata.migrate_v0_to_v1 \\")
                    logger.info(f"     --config={args.config} --namespace={namespace} \\")
                    logger.info(f"     --import-schemas --force")
                    logger.info("")
                    logger.info("4. If issues arise, restore from backup:")
                    logger.info(f"   rm -rf {db_path} && mv {backup_path} {db_path}")
                    sys.exit(0)
            else:
                logger.error("")
                logger.error("=" * 80)
                logger.error("❌ MIGRATION VERIFICATION FAILED")
                logger.error("=" * 80)
                logger.error("Please check the logs above for details")

                # Print recovery suggestions
                print_recovery_suggestions(db_path, "schema_metadata")

                sys.exit(1)
        else:
            # No namespace - schema migration only
            logger.info("Step 2/3: Skipped (schema value storage requires --namespace)")
            logger.info("")
            logger.info("Step 3/3: Skipped (verification requires --namespace)")
            logger.info("")
            migration_results["success"] = True  # Schema migration succeeded
            migration_results["verification_passed"] = False  # But verification was skipped

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
            logger.info("Note: Schema import requires --namespace.")
            logger.info("To import schemas after migration, run:")
            if args.config:
                logger.info(f"  python -m datus.storage.schema_metadata.migrate_v0_to_v1 \\")
                logger.info(f"    --config={args.config} --namespace <name> \\")
                logger.info(f"    --import-schemas --force")
            logger.info("")
            logger.info("Next steps:")
            logger.info("1. Test the system with enhanced metadata")
            logger.info("2. Verify schema discovery precision improvements")
            logger.info("3. If issues arise, restore from backup:")
            logger.info(f"   rm -rf {db_path} && mv {backup_path} {db_path}")
            sys.exit(0)

    except KeyboardInterrupt:
        logger.warning("Migration interrupted by user (KeyboardInterrupt).")
        migration_results["cancelled"] = True
        sys.exit(130)
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        logger.error("")

        # Print recovery suggestions
        print_recovery_suggestions(db_path, "schema_metadata")

        sys.exit(1)

    finally:
        # Always print final migration report
        print_final_migration_report(migration_results, db_path, args)


if __name__ == "__main__":
    main()
