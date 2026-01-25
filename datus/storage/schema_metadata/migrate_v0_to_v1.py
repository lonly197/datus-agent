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
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from datus.configuration.agent_config import AgentConfig
from datus.configuration.agent_config_loader import load_agent_config
from datus.storage.embedding_models import get_db_embedding_model
from datus.storage.schema_metadata import SchemaStorage, SchemaWithValueRAG
from datus.models.base import LLMBaseModel
from datus.utils.loggings import configure_logging, get_logger
from datus.utils.sql_utils import (
    extract_enhanced_metadata_from_ddl,
    extract_enum_values_from_comment,
    is_likely_truncated_ddl,
    parse_dialect,
    sanitize_ddl_for_storage,
    validate_comment,
    validate_table_name,
)
from datus.utils.constants import DBType

logger = get_logger(__name__)
_shutdown_event = threading.Event()
_shutdown_signal_count = 0


def shutdown_requested() -> bool:
    return _shutdown_event.is_set()


def setup_signal_handlers() -> None:
    def _handle_signal(sig, frame):
        global _shutdown_signal_count
        _shutdown_signal_count += 1
        signal_name = signal.Signals(sig).name
        logger.warning(f"Received {signal_name}; exiting immediately.")
        _shutdown_event.set()
        raise SystemExit(130)

    try:
        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)
    except Exception as exc:
        logger.debug(f"Failed to register signal handlers: {exc}")


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
        # Validate that the specified namespace exists in configuration
        if not agent_config.namespaces or specified_namespace not in agent_config.namespaces:
            console.print("")
            console.print("[red]" + "=" * 80 + "[/]")
            console.print("[red]NAMESPACE NOT FOUND IN CONFIGURATION[/]")
            console.print("[red]" + "=" * 80 + "[/]")
            console.print("")
            console.print(f"[red]Namespace '{specified_namespace}' does not exist in agent configuration[/]")
            console.print("")
            if agent_config.namespaces:
                console.print("[yellow]Available namespaces:[/]")
                for ns in agent_config.namespaces.keys():
                    console.print(f"  [cyan]{ns}[/]")
            else:
                console.print("[yellow]No namespaces configured in agent.yml[/]")
            console.print("")
            console.print("[yellow]To fix this issue:[/]")
            console.print("  1. Check that the namespace is correctly configured in agent.yml")
            console.print("  2. Use --namespace with a valid namespace name")
            console.print("  3. Or add the namespace configuration to agent.yml")
            console.print("")
            console.print("[red]" + "=" * 80 + "[/]")
            console.print("")
            sys.exit(1)
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


def report_migration_state(db_path: str, table_name: str = "schema_metadata"):
    """
    Report current migration state for debugging and diagnostics.

    This function provides comprehensive pre-migration state information including:
    - Table existence
    - Record count
    - Schema field information
    - Metadata version distribution
    - v1 field presence detection

    Args:
        db_path: Path to LanceDB database directory
        table_name: Name of the table to check (default: "schema_metadata")
    """
    import lancedb
    from collections import Counter

    logger.info("=" * 80)
    logger.info("MIGRATION STATE CHECK")
    logger.info("=" * 80)
    logger.info(f"Database path: {db_path}")
    logger.info("")

    try:
        # Check if database path exists
        if not os.path.exists(db_path):
            logger.info(f"✗ Database path does not exist: {db_path}")
            logger.info("  → This is normal for fresh installations")
            logger.info("  → A new database will be created during migration")
            logger.info("=" * 80)
            return

        db = lancedb.connect(db_path)

        # Check if table exists
        table_names = db.table_names()
        logger.info(f"Tables in database: {table_names}")

        if table_name not in table_names:
            logger.info(f"✗ Table '{table_name}' DOES NOT EXIST")
            logger.info("  → This is normal for fresh installations")
            logger.info("  → A new v1 table will be created during migration")
            logger.info("=" * 80)
            return

        # Table exists - report details
        logger.info(f"✓ Table '{table_name}' EXISTS")
        table = db.open_table(table_name)

        # Get record count (avoid sampling defaults)
        count = table.count_rows()
        logger.info(f"  Record count: {count}")

        # Check schema
        schema = table.schema
        field_names = schema.names
        logger.info(f"  Fields ({len(field_names)}): {field_names}")

        # Check if it has v1 fields
        v1_fields = [
            "table_comment", "column_comments", "column_enums", "business_tags",
            "row_count", "sample_statistics", "relationship_metadata",
            "metadata_version", "last_updated"
        ]
        has_v1_fields = [f for f in v1_fields if f in field_names]
        logger.info(f"  Has v1 fields: {len(has_v1_fields)}/{len(v1_fields)}")

        if has_v1_fields:
            logger.info(f"    Present: {', '.join(has_v1_fields)}")
            missing = [f for f in v1_fields if f not in field_names]
            if missing:
                logger.info(f"    Missing: {', '.join(missing)}")

        # Check metadata_version distribution if field exists
        if "metadata_version" in field_names and count > 0:
            try:
                # Select only metadata_version to reduce payload; explicitly fetch all rows.
                version_data = table.search().select(["metadata_version"]).limit(count).to_arrow()
                versions = [row.get("metadata_version", 0) for row in version_data.to_pylist()]
                version_counts = Counter(versions)
                logger.info(f"  Version distribution: {dict(version_counts)}")

                v0_count = version_counts.get(0, 0)
                v1_count = version_counts.get(1, 0)
                other_count = sum(v for k, v in version_counts.items() if k not in [0, 1])

                if v1_count > 0:
                    logger.info(f"  → Already migrated: {v1_count} v1 records found")
                if v0_count > 0:
                    logger.info(f"  → Legacy data: {v0_count} v0 records need migration")
                if other_count > 0:
                    logger.info(f"  → Other versions: {other_count} records with unexpected versions")

            except Exception as e:
                logger.debug(f"Could not read version distribution: {e}")

        logger.info("")
        logger.info("Summary:")
        if count == 0:
            logger.info("  → Table exists but is empty (fresh installation)")
        elif len(has_v1_fields) == len(v1_fields):
            if "metadata_version" in field_names:
                version_data = table.search().select(["metadata_version"]).to_arrow()
                if any(row.get("metadata_version", 0) == 1 for row in version_data.to_pylist()):
                    logger.info("  → Table already contains v1 data (use --force to re-migrate)")
                else:
                    logger.info("  → Table has v1 schema but v0 data (migration needed)")
            else:
                logger.info("  → Table has v1 schema structure")
        else:
            logger.info("  → Table has v0 schema structure (migration needed)")

    except Exception as e:
        logger.error(f"Error checking migration state: {e}")
        import traceback
        logger.debug(traceback.format_exc())

    logger.info("=" * 80)
    logger.info("")


def detect_dialect_from_ddl(ddl: str) -> str:
    """
    Attempt to detect SQL dialect from DDL statement.

    Uses keyword and type pattern matching to identify the most likely
    database dialect from the DDL content.

    Args:
        ddl: DDL statement to analyze

    Returns:
        Detected dialect (defaults to "starrocks" for StarRocks-like syntax)
    """
    ddl_upper = ddl.upper()

    # StarRocks-specific patterns (MySQL-compatible with backticks + bigint(20) + Chinese comments)
    if "`" in ddl and ("BIGINT(" in ddl_upper or "TINYINT" in ddl_upper or "MEDIUMINT" in ddl_upper):
        return "starrocks"  # Use native starrocks dialect

    # Snowflake-specific patterns
    if "VARIANT" in ddl_upper or ("OBJECT" in ddl_upper and "`" not in ddl) or ("ARRAY" in ddl_upper and "`" not in ddl):
        return "snowflake"

    # DuckDB-specific patterns
    if "DOUBLE" in ddl_upper or "UBIGINT" in ddl_upper or "HUGEINT" in ddl_upper:
        return "duckdb"

    # PostgreSQL-specific patterns
    if "SERIAL" in ddl_upper or "BIGSERIAL" in ddl_upper or "MONEY" in ddl_upper:
        return "postgres"

    # MySQL-specific patterns (includes StarRocks)
    if "TINYINT" in ddl_upper or "MEDIUMINT" in ddl_upper or "AUTO_INCREMENT" in ddl_upper:
        return "mysql"

    # BigQuery-specific patterns
    if "STRUCT" in ddl_upper or "INT64" in ddl_upper or "FLOAT64" in ddl_upper:
        return "bigquery"

    # Default to starrocks for StarRocks-like syntax (backtick identifiers)
    # This is safer than snowflake as it's more permissive
    return "starrocks"


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


def load_backup_json(backup_path: str) -> List[Dict]:
    """
    Load schema data from JSON backup file.

    Args:
        backup_path: Path to JSON backup file

    Returns:
        List of schema records as dictionaries
    """
    if not os.path.exists(backup_path):
        raise FileNotFoundError(f"Backup file not found: {backup_path}")

    with open(backup_path, 'r', encoding='utf-8') as f:
        records = json.load(f)

    logger.info(f"Loaded {len(records)} records from {backup_path}")
    return records


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
                # This handles incomplete DDL statements from SHOW CREATE TABLE
                original_definition = definition
                was_truncated = is_likely_truncated_ddl(original_definition)
                definition = sanitize_ddl_for_storage(definition)
                if was_truncated and not is_likely_truncated_ddl(definition):
                    logger.info(
                        f"  - DDL FIXED (truncation): length {len(original_definition)} -> {len(definition)}"
                    )

                # Parse enhanced metadata from DDL with dialect detection
                detected_dialect = detect_dialect_from_ddl(definition)
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

                # Extract comment information
                table_comment = enhanced_metadata["table"].get("comment", "")
                column_comments = {
                    col["name"]: col.get("comment", "")
                    for col in enhanced_metadata["columns"]
                }
                column_enums: Dict[str, List[Dict[str, str]]] = {}
                for col_name, col_comment in column_comments.items():
                    enum_pairs = extract_enum_values_from_comment(col_comment)
                    if enum_pairs:
                        column_enums[col_name] = [
                            {"value": code, "label": label} for code, label in enum_pairs
                        ]

                # Infer business tags
                from datus.configuration.business_term_config import infer_business_tags
                column_names = [col["name"] for col in enhanced_metadata["columns"]]
                business_tags = infer_business_tags(table_name, column_names)

                # Extract statistics (row count + column stats) from live database
                row_count = 0
                sample_statistics: Dict[str, Dict[str, Any]] = {}
                if extract_statistics:
                    metadata_extractor = _get_metadata_extractor(database_name)
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
    from collections import Counter

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


def import_schema_metadata(agent_config: AgentConfig, namespace: str, clear_before_import: bool = False) -> int:
    """
    Import schema metadata from database into LanceDB after migration.

    This function populates the empty v1 table with actual schema data
    from the configured database(s). It uses the same import logic as
    the local_init module.

    Args:
        agent_config: Agent configuration with database settings
        namespace: Namespace to import schemas for

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
            pool_size=4
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


def print_final_migration_report(migration_results: dict, db_path: str, args):
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
    parser.add_argument("--llm-fallback", action="store_true", help="Use LLM as final fallback for DDL parsing failures")
    parser.add_argument("--llm-model", help="Optional model name for LLM fallback (defaults to active model)")

    args = parser.parse_args()
    configure_logging(debug=False)

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
    logger.info(f"LLM fallback: {args.llm_fallback}")
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

    try:
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
