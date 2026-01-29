# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details

"""
Batch processing and reporting for v0 to v1 migration.

This module handles:
- Schema import from database
- Progress reporting and diagnostics
- Recovery suggestions
- Final migration reports
"""

import os
import sys
from typing import Any, Dict, Optional

from datus.configuration.agent_config import AgentConfig
from datus.storage.schema_metadata import SchemaWithValueRAG
from datus.utils.loggings import get_logger

from .migration_logic import verify_migration

logger = get_logger(__name__)


def import_schema_metadata(
    agent_config: AgentConfig,
    namespace: str,
    clear_before_import: bool = False,
    extract_statistics: bool = False,
    extract_relationships: bool = True,
    llm_enum_extraction: bool = False,
) -> int:
    """
    Import schema metadata from database into LanceDB after migration.

    This function populates the empty v1 table with actual schema data
    from the configured database(s). It uses the same import logic as
    the local_init module.

    Args:
        agent_config: Agent configuration with database settings
        namespace: Namespace to import schemas for
        clear_before_import: Clear existing data before import
        extract_statistics: Extract column statistics
        extract_relationships: Extract relationship metadata
        llm_enum_extraction: Use LLM for enum extraction

    Returns:
        Number of schemas imported
    """
    try:
        from datus.storage.schema_metadata.local_init import init_local_schema
        from datus.tools.db_tools.db_manager import get_db_manager

        logger.info("=" * 80)
        logger.info("SCHEMA IMPORT")
        logger.info("=" * 80)
        logger.info(f"Importing schema metadata for namespace: {namespace}")

        # Validate namespace exists in config
        # Note: This check validates agent_config.namespaces, but DBManager also
        # needs to be initialized with the same config (see get_db_manager call below)
        if not agent_config.namespaces or namespace not in agent_config.namespaces:
            logger.error("")
            logger.error("=" * 80)
            logger.error("NAMESPACE NOT FOUND IN CONFIGURATION")
            logger.error("=" * 80)
            logger.error(f"Namespace '{namespace}' does not exist in agent configuration")
            logger.error("")
            if agent_config.namespaces:
                logger.error("Available namespaces:")
                for ns in agent_config.namespaces.keys():
                    logger.error(f"  - {ns}")
            else:
                logger.error("No namespaces configured in agent.yml")
            logger.error("")
            logger.error("To fix this issue:")
            logger.error("  1. Check that the namespace is correctly configured in agent.yml")
            logger.error("  2. Use --namespace with a valid namespace name")
            logger.error("  3. Or add the namespace configuration to agent.yml")
            logger.error("=" * 80)
            logger.error("")
            return 0

        # Initialize storage with namespace
        agent_config.current_namespace = namespace
        storage = SchemaWithValueRAG(agent_config)
        schema_store = storage.schema_store

        if clear_before_import:
            logger.info("Clearing existing schema_metadata and schema_value before import...")
            try:
                if schema_store.table_name in schema_store.db.table_names(limit=100):
                    schema_store.db.drop_table(schema_store.table_name)
                schema_store.table = None
                schema_store._table_initialized = False
                schema_store._ensure_table_ready()
                value_store = storage.value_store
                if value_store.table_name in value_store.db.table_names(limit=100):
                    value_store.db.drop_table(value_store.table_name)
                value_store.table = None
                value_store._table_initialized = False
            except Exception as exc:
                logger.error(f"Failed to clear schema tables before import: {exc}")
                return 0

        # Get database manager - MUST pass namespaces configuration
        db_manager = get_db_manager(agent_config.namespaces)

        # Import schemas from all databases
        # Use 'overwrite' mode to ensure full import for fresh installations
        init_local_schema(
            storage,
            agent_config,
            db_manager,
            build_mode='overwrite',  # Force full import
            table_type='full',  # Import both tables and views
            pool_size=4,
            extract_statistics=extract_statistics,
            extract_relationships=extract_relationships,
            llm_enum_extraction=llm_enum_extraction,
        )

        # Verify import was successful
        schema_store._ensure_table_ready()

        all_data = schema_store._search_all(
            where=None,
            select_fields=["identifier"]
        )

        imported_count = len(all_data)

        logger.info("")
        logger.info(f"✅ Schema import completed: {imported_count} schemas imported")
        logger.info("=" * 80)
        logger.info("")

        return imported_count

    except Exception as e:
        logger.error(f"Schema import failed: {e}")
        logger.info("")
        logger.info("To diagnose the issue, check:")
        logger.info("  1. Database connection parameters in agent.yml")
        logger.info("  2. Database is accessible and contains tables")
        logger.info("  3. Credentials are valid")
        logger.info("  4. Namespace exists in configuration")
        logger.info("")
        logger.info("To run schema import separately:")
        logger.info(f"  python -m datus.storage.schema_metadata.local_init \\")
        logger.info(f"    --config=<config_path> --namespace={namespace}")
        logger.info("=" * 80)
        return 0


def print_recovery_suggestions(db_path: str, table_name: str = "schema_metadata"):
    """
    Print recovery suggestions based on common migration failure scenarios.

    Provides actionable troubleshooting steps for users encountering migration issues.

    Args:
        db_path: Path to LanceDB database directory
        table_name: Name of the table (default: "schema_metadata")
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info("TROUBLESHOOTING SUGGESTIONS")
    logger.info("=" * 80)
    logger.info("")
    logger.info("To diagnose the issue, run the following commands:")
    logger.info("")
    logger.info("1. Check if the v1 table was actually created:")
    logger.info(f"   python -c \"import lancedb; db = lancedb.connect('{db_path}'); print('Tables:', db.table_names())\"")
    logger.info("")
    logger.info("2. Check table contents (if table exists):")
    logger.info(f"   python -c \"import lancedb; t = lancedb.connect('{db_path}').open_table('{table_name}'); print('Count:', len(t.to_arrow()))\"")
    logger.info("")
    logger.info("3. Verify metadata_version field (if table exists):")
    logger.info(f"   python -c \"import lancedb; t = lancedb.connect('{db_path}').open_table('{table_name}'); print('Fields:', t.schema.names)\"")
    logger.info("")
    logger.info("4. Check record count and versions:")
    logger.info(f"   python -c \"")
    logger.info(f"     import lancedb")
    logger.info(f"     from collections import Counter")
    logger.info(f"     t = lancedb.connect('{db_path}').open_table('{table_name}')")
    logger.info(f"     data = t.search().select(['metadata_version']).to_arrow()")
    logger.info(f"     versions = [r.get('metadata_version', 0) for r in data.to_pylist()]")
    logger.info(f"     print('Version distribution:', dict(Counter(versions)))")
    logger.info(f"   \"")
    logger.info("")
    logger.info("5. Check disk space and permissions:")
    logger.info(f"   df -h {os.path.dirname(db_path)}")
    logger.info(f"   ls -la {db_path}")
    logger.info("")
    logger.info("Common scenarios and solutions:")
    logger.info("")
    logger.info("Scenario A: Table exists but migration shows FAILED")
    logger.info("  → The table may have been created but verification logic failed")
    logger.info("  → Check the version distribution using command #4 above")
    logger.info("  → If v1 records exist, migration was successful (ignore the error)")
    logger.info("  → Re-run with --force to see improved verification output")
    logger.info("")
    logger.info("Scenario B: No table exists")
    logger.info("  → Check storage path configuration in agent.yml")
    logger.info("  → Verify write permissions on the directory")
    logger.info("  → Check available disk space")
    logger.info("  → Review logs above for specific errors during table creation")
    logger.info("")
    logger.info("Scenario C: Table exists but empty (0 records)")
    logger.info("  → This is normal for fresh installations")
    logger.info("  → Migration was successful (empty v1 table created)")
    logger.info("  → Ignore the error message")
    logger.info("")
    logger.info("Scenario D: Partial migration (some v0, some v1 records)")
    logger.info("  → Check logs above for errors during data processing")
    logger.info("  → Re-run with --force to complete migration")
    logger.info("  → If issue persists, restore from backup and retry")
    logger.info("")
    logger.info("To restore from backup (if needed):")
    logger.info(f"   # Find backup directory")
    logger.info(f"   ls -la {db_path}.backup_v0_*")
    logger.info(f"   # Restore (replace <backup_path> with actual backup)")
    logger.info(f"   rm -rf {db_path}")
    logger.info(f"   mv <backup_path> {db_path}")
    logger.info("")
    logger.info("=" * 80)


def print_final_migration_report(migration_results: Dict[str, Any], db_path: str, args):
    """
    Print a comprehensive final migration report.

    This function is called in the finally block to ensure that users always receive
    a clear summary of the migration results, even if the script exits unexpectedly.

    Args:
        migration_results: Dictionary containing migration results
        db_path: Path to LanceDB database directory
        args: Command-line arguments
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info("FINAL MIGRATION REPORT")
    logger.info("=" * 80)
    logger.info("")
    logger.info(f"Database Path: {db_path}")
    logger.info(f"Namespace: {migration_results.get('namespace', 'None')}")
    logger.info("")

    # Schema Migration Results
    logger.info("Schema Metadata Migration:")
    schemas_migrated = migration_results.get("schemas_migrated", 0)
    logger.info(f"  - Records migrated: {schemas_migrated}")
    if migration_results.get("cancelled"):
        logger.info("  - Status: ⚠️  CANCELLED")
    elif schemas_migrated > 0:
        logger.info(f"  - Status: ✅ SUCCESS")
    else:
        logger.info(f"  - Status: ⚠️  No records to migrate (empty database or already migrated)")
    logger.info("")

    # Schema Value Storage Results
    if migration_results.get('namespace'):
        logger.info("Schema Value Storage:")
        values_migrated = migration_results.get("values_migrated", 0)
        logger.info(f"  - Records checked: {values_migrated}")
        logger.info(f"  - Status: ✅ V1 compatible")
        logger.info("")
    else:
        logger.info("Schema Value Storage:")
        logger.info(f"  - Status: ⏭️  SKIPPED (requires --namespace)")
        logger.info("")

    # Verification Results
    logger.info("Migration Verification:")
    if migration_results.get('namespace'):
        if migration_results.get("verification_passed"):
            logger.info(f"  - Status: ✅ PASSED")
        else:
            logger.info(f"  - Status: ⚠️  SKIPPED or FAILED")
    else:
        logger.info(f"  - Status: ⏭️  SKIPPED (requires --namespace)")
    logger.info("")

    # Schema Import Results
    if args.import_schemas and migration_results.get('namespace'):
        imported_count = migration_results.get("schemas_imported", 0)
        logger.info("Schema Import from Database:")
        logger.info(f"  - Records imported: {imported_count}")
        if imported_count > 0:
            logger.info(f"  - Status: ✅ SUCCESS")
        else:
            logger.info(f"  - Status: ⚠️  FAILED (check database connection)")
        logger.info("")
    else:
        logger.info("Schema Import from Database:")
        logger.info(f"  - Status: ⏭️  SKIPPED (not requested)")
        logger.info("")

    # Overall Status
    logger.info("=" * 80)
    logger.info("OVERALL STATUS")
    logger.info("=" * 80)

    # Determine overall success
    if migration_results.get("cancelled"):
        logger.info("⚠️ MIGRATION: CANCELLED BY USER")
        logger.info("")
        logger.info("The migration was interrupted before completion.")
    elif migration_results.get("success"):
        if args.import_schemas and migration_results.get('namespace'):
            if migration_results.get("schemas_imported", 0) > 0:
                logger.info("✅ MIGRATION + IMPORT: COMPLETE SUCCESS")
                logger.info("")
                logger.info("Your LanceDB schema has been successfully upgraded to v1!")
                logger.info("The system is now ready for enhanced text2sql queries.")
            else:
                logger.info("✅ MIGRATION: SUCCESSFUL")
                logger.info("⚠️  IMPORT: FAILED (but migration completed)")
                logger.info("")
                logger.info("Your LanceDB schema has been successfully upgraded to v1,")
                logger.info("but schema import from database failed.")
                logger.info("You can re-run the import separately if needed.")
        else:
            logger.info("✅ MIGRATION: SUCCESSFUL")
            logger.info("")
            logger.info("Your LanceDB schema has been successfully upgraded to v1!")
    else:
        logger.info("❌ MIGRATION: FAILED")
        logger.info("")
        logger.info("The migration did not complete successfully.")
        logger.info("Please check the logs above for details.")

    logger.info("")
    logger.info("=" * 80)
    logger.info("NEXT STEPS")
    logger.info("=" * 80)
    logger.info("")

    if migration_results.get("cancelled"):
        logger.info("To resume migration:")
        logger.info(f"  python -m datus.storage.schema_metadata.migrate_v0_to_v1 \\")
        logger.info(f"    --config={args.config} --namespace={migration_results.get('namespace') or '<name>'} \\")
        logger.info("    --force")
        logger.info("")
        logger.info("If you need to restore from backup:")
        logger.info(f"  rm -rf {db_path}")
        logger.info(f"  mv {db_path}.backup_v0_* {db_path}")
        logger.info("")
    elif migration_results.get("success"):
        logger.info("To verify the migration:")
        logger.info(f"  1. Check version distribution:")
        logger.info(f"     python -c \"")
        logger.info(f"       import lancedb")
        logger.info(f"       from collections import Counter")
        logger.info(f"       t = lancedb.connect('{db_path}').open_table('schema_metadata')")
        logger.info(f"       data = t.search().select(['metadata_version']).to_arrow()")
        logger.info(f"       versions = [r.get('metadata_version', 0) for r in data.to_pylist()]")
        logger.info(f"       print('Version distribution:', dict(Counter(versions)))")
        logger.info(f"     \"")
        logger.info("")

        if migration_results.get('namespace') and not args.import_schemas:
            logger.info("To import schema metadata from your database:")
            logger.info(f"  python -m datus.storage.schema_metadata.migrate_v0_to_v1 \\")
            logger.info(f"    --config={args.config} --namespace={migration_results.get('namespace')} \\")
            logger.info(f"    --import-schemas --force")
            logger.info("")

        logger.info("To test the enhanced schema discovery:")
        logger.info("  1. Start the Datus agent")
        logger.info("  2. Try a text2sql query")
        logger.info("  3. Verify improved accuracy with enhanced metadata")
        logger.info("")
    else:
        logger.info("To troubleshoot:")
        logger.info("  1. Check the error messages above")
        logger.info("  2. Verify your configuration file")
        logger.info("  3. Ensure database connections are working")
        logger.info("  4. Re-run with --force if needed")
        logger.info("")
        logger.info("To restore from backup:")
        logger.info(f"  rm -rf {db_path}")
        logger.info(f"  mv {db_path}.backup_v0_* {db_path}")
        logger.info("")

    logger.info("=" * 80)
    logger.info("")
