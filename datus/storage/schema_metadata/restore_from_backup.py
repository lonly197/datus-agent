# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details

"""
Restore script to import v0 schema data from JSON backup.

This script:
1. Loads v0 schema data from JSON backup file
2. Validates backup integrity
3. Imports data into LanceDB v0 table
4. Verifies successful import

Use this script to rollback a migration or recover from cleanup.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

import lancedb
import pyarrow as pa

from datus.configuration.agent_config import AgentConfig
from datus.configuration.agent_config_loader import load_agent_config
from datus.storage.embedding_models import get_db_embedding_model
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


def validate_backup(backup_path: str) -> Dict[str, Any]:
    """
    Validate JSON backup file integrity and structure.

    Args:
        backup_path: Path to JSON backup file

    Returns:
        Validation result with 'valid', 'record_count', 'schema', 'errors'
    """
    result = {
        "valid": False,
        "record_count": 0,
        "schema": None,
        "errors": []
    }

    # Check file exists
    if not os.path.exists(backup_path):
        result["errors"].append(f"Backup file not found: {backup_path}")
        return result

    try:
        with open(backup_path, 'r', encoding='utf-8') as f:
            records = json.load(f)

        if not isinstance(records, list):
            result["errors"].append(f"Invalid backup format: expected list, got {type(records).__name__}")
            return result

        result["record_count"] = len(records)

        if result["record_count"] == 0:
            result["errors"].append("Backup file is empty")
            return result

        # Validate first record structure
        first_record = records[0]
        if "table_name" not in first_record:
            result["errors"].append("Invalid backup schema: missing 'table_name' field")
            return result

        result["schema"] = first_record
        result["valid"] = True

    except json.JSONDecodeError as e:
        result["errors"].append(f"Invalid JSON: {e}")
    except Exception as e:
        result["errors"].append(f"Error reading backup: {e}")

    return result


def import_from_backup(
    db_path: str,
    table_name: str,
    backup_path: str,
    validate: bool = True
) -> int:
    """
    Import v0 schema data from JSON backup into LanceDB table.

    Args:
        db_path: Path to LanceDB database directory
        table_name: Name of the table to import into
        backup_path: Path to JSON backup file
        validate: If True, validate backup before import

    Returns:
        Number of records imported
    """
    logger.info("=" * 80)
    logger.info("LanceDB v0 Table Restore from Backup")
    logger.info("=" * 80)
    logger.info(f"Database path: {db_path}")
    logger.info(f"Table name: {table_name}")
    logger.info(f"Backup path: {backup_path}")
    logger.info("")

    # Step 1: Validate backup
    if validate:
        logger.info("Step 1/4: Validating backup file...")
        validation = validate_backup(backup_path)

        if not validation["valid"]:
            logger.error("Backup validation failed:")
            for error in validation["errors"]:
                logger.error(f"  ✗ {error}")
            return 0

        logger.info(f"✓ Backup is valid ({validation['record_count']} records)")
        logger.info(f"✓ Schema sample: {list(validation['schema'].keys())[:5]}...")
        logger.info("")
    else:
        logger.info("Step 1/4: Skipping validation (--validate=False)")
        logger.info("")

    # Step 2: Load backup data
    logger.info("Step 2/4: Loading backup data...")
    with open(backup_path, 'r', encoding='utf-8') as f:
        records = json.load(f)

    logger.info(f"✓ Loaded {len(records)} records from backup")
    logger.info("")

    # Step 3: Create LanceDB table and import data
    logger.info("Step 3/4: Creating table and importing data...")
    db = lancedb.connect(db_path)

    # Check if table already exists
    if table_name in db.table_names():
        logger.warning(f"⚠️  Table '{table_name}' already exists")
        logger.info("To avoid data loss, please:")
        logger.info("  1. Drop existing table: db.drop_table('{}')".format(table_name))
        logger.info("  2. Or use different table name: --table-name schema_metadata_restored")
        logger.error("Import aborted to prevent data loss")
        return 0

    try:
        # Convert to Arrow table
        logger.info("Converting JSON to Arrow format...")
        arrow_table = pa.Table.from_pylist(records)

        # Create table
        logger.info(f"Creating LanceDB table '{table_name}'...")
        db.create_table(table_name, arrow_table)

        logger.info(f"✓ Created table with {len(records)} records")
        logger.info("")

    except Exception as e:
        logger.error(f"Failed to create table: {e}")
        import traceback
        logger.debug(traceback.format_exc())

        # Cleanup: remove partially created table
        try:
            db.drop_table(table_name)
            logger.info(f"Cleaned up partial table: {table_name}")
        except Exception:
            pass

        return 0

    # Step 4: Verification
    logger.info("Step 4/4: Verifying import...")
    table = db.open_table(table_name)
    imported_count = len(table.to_arrow())

    if imported_count == len(records):
        logger.info(f"✓ Verification successful: {imported_count} records imported")
    else:
        logger.warning(f"⚠️  Record count mismatch: expected {len(records)}, got {imported_count}")

    logger.info("")
    logger.info("=" * 80)
    logger.info("✅ RESTORE SUCCESSFUL")
    logger.info("=" * 80)
    logger.info(f"Records imported: {imported_count}")
    logger.info(f"Table created: {table_name}")
    logger.info(f"Database: {db_path}")
    logger.info("")
    logger.info("Next step: Verify data and re-run migration if needed")
    logger.info("")

    return imported_count


def main():
    """Main restore function."""
    parser = argparse.ArgumentParser(
        description="Restore v0 schema data from JSON backup",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Restore from backup with validation
  python -m datus.storage.schema_metadata.restore_from_backup \\
    --config=config.yml --backup=/tmp/schema_v0_backup.json

  # Skip validation for faster import
  python -m datus.storage.schema_metadata.restore_from_backup \\
    --config=config.yml --backup=/tmp/schema_v0_backup.json --no-validate

  # Restore to different table name
  python -m datus.storage.schema_metadata.restore_from_backup \\
    --config=config.yml --backup=/tmp/schema_v0_backup.json \\
    --table-name=schema_metadata_restored
        """
    )
    parser.add_argument("--config", required=True, help="Path to agent configuration file")
    parser.add_argument("--namespace", help="Namespace for the database (optional)")
    parser.add_argument("--db-path", help="Override database path (default: from config)")
    parser.add_argument("--table-name", default="schema_metadata", help="Table name to create (default: schema_metadata)")
    parser.add_argument("--backup-path", required=True, help="Path to JSON backup file")
    parser.add_argument("--no-validate", action="store_true", help="Skip backup validation")

    args = parser.parse_args()

    # Load agent configuration
    try:
        agent_config = load_agent_config(config=args.config)
    except Exception as e:
        logger.error(f"Failed to load agent configuration: {e}")
        sys.exit(1)

    # Determine database path
    db_path = args.db_path or None
    namespace = None

    if not db_path:
        # Use same logic as cleanup script
        from datus.storage.schema_metadata.cleanup_v0_table import select_namespace_interactive

        namespace = select_namespace_interactive(agent_config, args.namespace)
        if namespace:
            agent_config.current_namespace = namespace
            db_path = agent_config.rag_storage_path()
        else:
            db_path = os.path.join(agent_config.rag_base_path, "lancedb")
            logger.info("Using base storage path (Schema-only restore)")

    # Validate backup exists
    if not os.path.exists(args.backup_path):
        logger.error(f"Backup file not found: {args.backup_path}")
        logger.info("")
        logger.info("Please specify a valid backup file with --backup-path")
        logger.info("Common backup locations:")
        logger.info("  /tmp/schema_v0_backup.json (default cleanup location)")
        logger.info("  ./schema_v0_backup.json")
        sys.exit(1)

    # Run restore
    try:
        imported_count = import_from_backup(
            db_path=db_path,
            table_name=args.table_name,
            backup_path=args.backup_path,
            validate=not args.no_validate
        )

        if imported_count > 0:
            logger.info(f"✅ Successfully restored {imported_count} records")
            sys.exit(0)
        else:
            logger.error("Restore failed")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("\nRestore cancelled by user (Ctrl+C)")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Restore failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
