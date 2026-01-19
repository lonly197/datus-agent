# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details

"""
Cleanup script to export v0 schema data and force-delete the old table.

This script:
1. Connects to the LanceDB database
2. Exports all existing v0 data to a JSON backup file
3. Force-deletes the v0 table using multiple methods:
   - LanceDB API drop_table()
   - File system directory deletion
   - Database connection refresh
4. Verifies the table is completely gone

This should be run BEFORE migrate_v0_to_v1.py to ensure a clean migration.
"""

import argparse
import json
import os
import shutil
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


def report_cleanup_state(db_path: str, table_name: str = "schema_metadata"):
    """
    Report current cleanup state for debugging and diagnostics.

    This function provides comprehensive pre-cleanup state information including:
    - Table existence
    - Record count
    - Schema field information
    - Data summary

    Args:
        db_path: Path to LanceDB database directory
        table_name: Name of the table to check (default: "schema_metadata")
    """
    import lancedb

    logger.info("=" * 80)
    logger.info("CLEANUP STATE CHECK")
    logger.info("=" * 80)
    logger.info(f"Database path: {db_path}")
    logger.info("")

    try:
        # Check if database path exists
        if not os.path.exists(db_path):
            logger.info(f"✗ Database path does not exist: {db_path}")
            logger.info("  → This is normal for fresh installations")
            logger.info("  → No cleanup needed")
            logger.info("=" * 80)
            return

        db = lancedb.connect(db_path)

        # Check if table exists
        table_names = db.table_names()
        logger.info(f"Tables in database: {table_names}")

        if table_name not in table_names:
            logger.info(f"✗ Table '{table_name}' DOES NOT EXIST")
            logger.info("  → This is normal for fresh installations")
            logger.info("  → No cleanup needed (table already removed or never created)")
            logger.info("=" * 80)
            return

        # Table exists - report details
        logger.info(f"✓ Table '{table_name}' EXISTS")
        table = db.open_table(table_name)

        # Get record count
        all_data = table.to_arrow()
        count = len(all_data)
        logger.info(f"  Record count: {count}")

        if count == 0:
            logger.info("  → Table is empty (will be dropped during cleanup)")
        else:
            logger.info(f"  → Table contains {count} records (will be exported before cleanup)")

        # Check schema
        schema = table.schema
        field_names = schema.names
        logger.info(f"  Fields ({len(field_names)}): {field_names}")

        # Check if it has v1 fields
        v1_fields = [
            "table_comment", "column_comments", "business_tags",
            "row_count", "sample_statistics", "relationship_metadata",
            "metadata_version", "last_updated"
        ]
        has_v1_fields = [f for f in v1_fields if f in field_names]

        if has_v1_fields:
            logger.info(f"  Has v1 fields: {len(has_v1_fields)}/{len(v1_fields)}")
            logger.info("  → Table appears to have v1 schema (already migrated?)")
            logger.info("  → Verify if this is the correct table to clean up")

            # Check metadata_version if field exists
            if "metadata_version" in field_names:
                try:
                    version_data = table.search().select(["metadata_version"]).to_arrow()
                    versions = [row.get("metadata_version", 0) for row in version_data.to_pylist()]
                    from collections import Counter
                    version_counts = Counter(versions)
                    logger.info(f"  Version distribution: {dict(version_counts)}")

                    v1_count = version_counts.get(1, 0)
                    if v1_count > 0:
                        logger.warning("  ⚠️  WARNING: Table contains v1 data!")
                        logger.warning("     → This table has already been migrated")
                        logger.warning("     → Cleaning up v1 data will require re-migration")
                        logger.warning("     → Only proceed if you intend to re-run the full migration")
                except Exception as e:
                    logger.debug(f"Could not read version distribution: {e}")
        else:
            logger.info("  Has v1 fields: 0/8")
            logger.info("  → Table has v0 schema structure")

    except Exception as e:
        logger.error(f"Error checking cleanup state: {e}")
        import traceback
        logger.debug(traceback.format_exc())

    logger.info("=" * 80)
    logger.info("")


def select_namespace_interactive(agent_config: AgentConfig, specified_namespace: Optional[str] = None) -> Optional[str]:
    """
    Smart namespace selection:
    - Use specified value if --namespace provided
    - Auto-select if only one namespace exists
    - Interactive prompt if multiple namespaces exist
    - Return None for base path (Schema-only)

    Returns:
        namespace name or None (base path)
    """
    from rich.console import Console
    from rich.prompt import Prompt
    from rich.table import Table

    console = Console()

    # Case 1: User specified namespace
    if specified_namespace:
        return specified_namespace

    # Case 2: No namespaces configured
    if not agent_config.namespaces:
        console.print("[yellow]⚠️  No namespaces found in configuration[/]")
        console.print("[yellow]Will use base storage path (Schema-only cleanup)[/]")
        return None

    namespaces = list(agent_config.namespaces.keys())

    # Case 3: Single namespace - auto-select
    if len(namespaces) == 1:
        selected = namespaces[0]
        console.print(f"[green]✓ Auto-selected namespace: {selected}[/]")
        return selected

    # Case 4: Multiple namespaces - interactive selection
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

    # Add "no namespace" option
    console.print("[dim]0. No namespace (Schema-only cleanup)[/]\n")

    choice = Prompt.ask(
        "[bold cyan]Select namespace[/]",
        choices=[str(i) for i in range(len(namespaces) + 1)],
        default="1"
    )

    if choice == "0":
        console.print("[yellow]Selected: Schema-only cleanup (base path)[/]")
        return None
    else:
        selected = namespaces[int(choice) - 1]
        console.print(f"[green]✓ Selected namespace: {selected}[/]")
        return selected


def export_table_to_json(
    db: lancedb.DBConnection,
    table_name: str,
    backup_path: str
) -> int:
    """
    Export all data from a LanceDB table to a JSON backup file.

    Args:
        db: LanceDB database connection
        table_name: Name of the table to export
        backup_path: Path where JSON backup will be saved

    Returns:
        Number of records exported
    """
    try:
        # Check if table exists
        if table_name not in db.table_names():
            logger.info(f"Table '{table_name}' does not exist, nothing to export")
            return 0

        # Open the table
        table = db.open_table(table_name)

        # Get all data (without vector column to reduce size)
        all_data = table.search().to_arrow()

        # Remove vector column if present
        if "vector" in all_data.column_names:
            all_data = all_data.drop(["vector"])

        # Convert to list of dicts
        records = []
        for row in all_data.to_pylist():
            # Convert Arrow types to Python native types
            record = {}
            for key, value in row.items():
                if value is not None:
                    # Handle binary data
                    if hasattr(value, 'as_py'):
                        record[key] = value.as_py()
                    elif isinstance(value, list):
                        record[key] = [v.as_py() if hasattr(v, 'as_py') else v for v in value]
                    else:
                        record[key] = value
                else:
                    record[key] = None
            records.append(record)

        # Write to JSON file
        os.makedirs(os.path.dirname(backup_path) or ".", exist_ok=True)
        with open(backup_path, 'w', encoding='utf-8') as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

        logger.info(f"Exported {len(records)} records to {backup_path}")
        return len(records)

    except Exception as e:
        logger.error(f"Failed to export table '{table_name}': {e}")
        raise


def force_drop_table(
    db_path: str,
    table_name: str
) -> bool:
    """
    Force-delete a LanceDB table using multiple methods.

    This function tries multiple approaches to ensure the table is completely removed:
    1. LanceDB API: db.drop_table()
    2. File system: Delete table directory
    3. Connection refresh: Close and reopen database

    Args:
        db_path: Path to LanceDB database directory
        table_name: Name of the table to delete

    Returns:
        True if table was successfully removed
    """
    try:
        # Method 1: Connect and drop using LanceDB API
        logger.info(f"Attempting to drop table '{table_name}' using LanceDB API...")
        db = lancedb.connect(db_path)

        if table_name in db.table_names():
            db.drop_table(table_name)
            logger.info(f"Dropped table '{table_name}' using LanceDB API")
        else:
            logger.info(f"Table '{table_name}' not found in database")

        # Method 2: Delete table directory from file system
        # LanceDB stores table data in: <db_path>/<table_name>.lance/
        table_dir = os.path.join(db_path, f"{table_name}.lance")
        if os.path.exists(table_dir):
            logger.info(f"Deleting table directory: {table_dir}")
            shutil.rmtree(table_dir)
            logger.info(f"Deleted table directory")
        else:
            logger.info(f"Table directory not found: {table_dir}")

        # Method 3: Force connection refresh
        # Close the connection by creating a new one
        logger.info("Refreshing database connection...")
        db = lancedb.connect(db_path)
        logger.info("Database connection refreshed")

        return True

    except Exception as e:
        logger.error(f"Error during force drop: {e}")
        return False


def verify_table_removed(
    db_path: str,
    table_name: str
) -> bool:
    """
    Verify that a table has been completely removed from the database.

    Args:
        db_path: Path to LanceDB database directory
        table_name: Name of the table to verify

    Returns:
        True if table is completely gone
    """
    try:
        # Check with fresh connection
        db = lancedb.connect(db_path)

        # Check if table name is in table list
        table_names = db.table_names()
        if table_name in table_names:
            logger.warning(f"Table '{table_name}' still appears in table_names(): {table_names}")
            return False

        # Check if table directory exists on file system
        table_dir = os.path.join(db_path, f"{table_name}.lance")
        if os.path.exists(table_dir):
            logger.warning(f"Table directory still exists: {table_dir}")
            return False

        logger.info(f"Verification: Table '{table_name}' has been completely removed")
        return True

    except Exception as e:
        logger.error(f"Error during verification: {e}")
        return False


def cleanup_v0_table(
    db_path: str,
    table_name: str = "schema_metadata",
    backup_path: str = "/tmp/schema_v0_backup.json",
    dry_run: bool = False,
    skip_verification: bool = False
) -> bool:
    """
    Main cleanup function: export data and force-delete the v0 table.

    Args:
        db_path: Path to LanceDB database directory
        table_name: Name of the table to clean up
        backup_path: Path where JSON backup will be saved
        dry_run: If True, preview changes without executing
        skip_verification: If True, skip post-cleanup verification

    Returns:
        True if cleanup was successful (or dry-run completed)
    """
    logger.info("=" * 80)
    logger.info("LanceDB v0 Table Cleanup" + (" [DRY-RUN]" if dry_run else ""))
    logger.info("=" * 80)
    logger.info(f"Database path: {db_path}")
    logger.info(f"Table name: {table_name}")
    logger.info(f"Backup path: {backup_path}")
    logger.info("")

    # Pre-cleanup checks
    db = lancedb.connect(db_path)
    table_exists = table_name in db.table_names()

    if not table_exists:
        logger.info(f"✓ Table '{table_name}' does not exist - no cleanup needed")
        logger.info("=" * 80)
        return True

    # Get record count
    table = db.open_table(table_name)
    record_count = len(table.to_arrow())
    logger.info(f"✓ Table found with {record_count} records")

    if dry_run:
        logger.info("")
        logger.info("=" * 80)
        logger.info("DRY-RUN SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Would export: {record_count} records")
        logger.info(f"Backup location: {backup_path}")
        logger.info(f"Would delete: {table_name}")
        logger.info("")
        logger.info("To execute cleanup, run without --dry-run flag")
        logger.info("=" * 80)
        return True

    # Step 1: Export data to JSON backup
    logger.info("Step 1/3: Exporting v0 data to JSON backup...")
    exported_count = export_table_to_json(db, table_name, backup_path)

    if exported_count == 0:
        logger.warning(f"Table '{table_name}' exists but is empty")
    else:
        logger.info(f"✅ Exported {exported_count} records to {backup_path}")
    logger.info("")

    # Step 2: Force drop the table
    logger.info("Step 2/3: Force-deleting v0 table...")
    if not force_drop_table(db_path, table_name):
        logger.error("Failed to drop table")
        return False
    logger.info("")

    # Step 3: Verify removal (optional)
    if not skip_verification:
        logger.info("Step 3/3: Verifying table removal...")
        if not verify_table_removed(db_path, table_name):
            logger.error("Table still exists after cleanup")
            return False
    else:
        logger.info("Step 3/3: Skipping verification (--skip-verification)")
    logger.info("")

    # Summary
    logger.info("=" * 80)
    logger.info("✅ CLEANUP SUCCESSFUL")
    logger.info("=" * 80)
    logger.info(f"Records exported: {exported_count}")
    logger.info(f"Backup location: {backup_path}")
    logger.info(f"Table removed: {table_name}")
    logger.info("")
    logger.info("Next step: Run migration script")
    logger.info(f"  python -m datus.storage.schema_metadata.migrate_v0_to_v1 \\")
    logger.info(f"    --config=/path/to/config.yml \\")
    logger.info(f"    --backup-path={backup_path}")
    logger.info("")

    return True


def str_to_bool(v):
    """Convert string to boolean for argparse."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError(f'Boolean value expected, got: {v}')


def main():
    """Main cleanup function with safety checks."""
    parser = argparse.ArgumentParser(
        description="Cleanup v0 LanceDB table and export data to JSON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview cleanup changes (recommended first step)
  python -m datus.storage.schema_metadata.cleanup_v0_table --config=config.yml --dry-run

  # Run cleanup with confirmation prompt
  python -m datus.storage.schema_metadata.cleanup_v0_table --config=config.yml --confirm

  # Skip verification for faster cleanup
  python -m datus.storage.schema_metadata.cleanup_v0_table --config=config.yml --skip-verification
        """
    )
    parser.add_argument("--config", required=True, help="Path to agent configuration file")
    parser.add_argument("--namespace", help="Namespace for the database (optional)")
    parser.add_argument("--db-path", help="Override database path (default: from config)")
    parser.add_argument("--table-name", default="schema_metadata", help="Table name to cleanup (default: schema_metadata)")
    parser.add_argument("--backup-path", default="/tmp/schema_v0_backup.json", help="Path for JSON backup file")
    parser.add_argument("--skip-verification", action="store_true", help="Skip verification step")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without executing")
    parser.add_argument("--confirm", action="store_true", help="Require confirmation before cleanup")
    parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation (auto-confirm)")

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
        namespace = select_namespace_interactive(agent_config, args.namespace)
        if namespace:
            agent_config.current_namespace = namespace
            db_path = agent_config.rag_storage_path()
        else:
            db_path = os.path.join(agent_config.rag_base_path, "lancedb")
            logger.info("Using base storage path (Schema-only cleanup)")

    # Report current cleanup state (diagnostics)
    report_cleanup_state(db_path, args.table_name)

    # Handle confirmation (unless --yes or --dry-run)
    if args.confirm and not args.yes and not args.dry_run:
        from rich.console import Console
        from rich.prompt import Confirm

        console = Console()
        console.print("\n[bold yellow]⚠️  WARNING: This will permanently delete data![/]")
        console.print(f"[yellow]Table: {args.table_name}[/]")
        console.print(f"[yellow]Database: {db_path}[/]")
        console.print("")

        if not Confirm.ask("[bold red]Continue with cleanup?[/]", default=False):
            console.print("[yellow]Cleanup cancelled by user[/]")
            sys.exit(0)

    # Run cleanup
    try:
        success = cleanup_v0_table(
            db_path=db_path,
            table_name=args.table_name,
            backup_path=args.backup_path,
            dry_run=args.dry_run,
            skip_verification=args.skip_verification
        )

        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        logger.info("\nCleanup cancelled by user (Ctrl+C)")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
