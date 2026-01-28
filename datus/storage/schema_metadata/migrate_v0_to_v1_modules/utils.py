# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details

"""
Utility functions for v0 to v1 migration.

This module provides helper functions for:
- Signal handling and shutdown management
- Namespace selection and interaction
- Database backup and restore
- DDL parsing and dialect detection
- Relationship inference
- Diagnostic reporting
"""

import argparse
import json
import os
import re
import shutil
import signal
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from datus.configuration.agent_config import AgentConfig
from datus.utils.constants import DBType
from datus.utils.loggings import get_logger
from datus.utils.sql_utils import (
    is_likely_truncated_ddl,
    sanitize_ddl_for_storage,
)

logger = get_logger(__name__)
_shutdown_event = threading.Event()
_shutdown_signal_count = 0


# ============================================================================
# Signal Handling
# ============================================================================

def shutdown_requested() -> bool:
    """Check if shutdown has been requested.

    Returns:
        True if shutdown signal received
    """
    return _shutdown_event.is_set()


def setup_signal_handlers() -> None:
    """Setup signal handlers for graceful shutdown."""
    def _handle_signal(sig, frame):
        global _shutdown_signal_count
        _shutdown_signal_count += 1
        _shutdown_event.set()
        try:
            signal_name = signal.Signals(sig).name
            message = f"Received {signal_name}; exiting immediately.\n"
            os.write(2, message.encode())
        except Exception:
            pass
        os._exit(130)

    try:
        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)
    except Exception as exc:
        logger.debug(f"Failed to register signal handlers: {exc}")


# ============================================================================
# Namespace Selection
# ============================================================================

def select_namespace_interactive(
    agent_config: AgentConfig,
    specified_namespace: Optional[str] = None
) -> Optional[str]:
    """
    智能选择 namespace：
    - 如果用户通过 --namespace 指定，使用指定的值
    - 如果配置只有一个 namespace，自动使用
    - 如果配置有多个 namespace，交互式选择
    - 如果没有 namespace，返回 None（使用 base path）

    Args:
        agent_config: Agent configuration
        specified_namespace: User-specified namespace (optional)

    Returns:
        namespace name 或 None（表示使用 base path）
    """
    from rich.console import Console
    from rich.prompt import Prompt
    from rich.table import Table

    console = Console()

    # 情况1: 用户已显式指定
    if specified_namespace:
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


# ============================================================================
# Database Backup
# ============================================================================

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


# ============================================================================
# DDL Parsing and Dialect Detection
# ============================================================================

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


# ============================================================================
# Relationship Inference
# ============================================================================

_RELATION_PREFIXES = (
    "ods_", "dwd_", "dws_", "dim_", "ads_", "tmp_", "stg_", "stage_", "fact_",
)
_RELATION_SUFFIXES = (
    "_di", "_df", "_tmp", "_temp", "_bak", "_backup",
)
_RELATION_DATE_SUFFIX_RE = re.compile(r"_(?:19|20)\d{6,8}$")


def _normalize_relationship_name(name: str) -> str:
    """Normalize table name for relationship matching."""
    value = (name or "").strip().strip("`").lower()
    value = _RELATION_DATE_SUFFIX_RE.sub("", value)
    for suffix in _RELATION_SUFFIXES:
        if value.endswith(suffix):
            value = value[: -len(suffix)]
            break
    for prefix in _RELATION_PREFIXES:
        if value.startswith(prefix):
            value = value[len(prefix):]
            break
    return value


def _build_table_name_map(table_names: List[str]) -> Dict[str, List[str]]:
    """Build mapping of normalized names to actual table names."""
    mapping: Dict[str, List[str]] = {}
    for name in table_names:
        normalized = _normalize_relationship_name(name)
        if not normalized:
            continue
        mapping.setdefault(normalized, []).append(name)
    return mapping


def _infer_relationships_from_names(
    column_names: List[str],
    table_name_map: Dict[str, List[str]],
) -> Dict[str, Any]:
    """Infer foreign key relationships from column names."""
    foreign_keys = []
    for col in column_names:
        col_lower = (col or "").lower()
        if not col_lower or not col_lower.endswith("_id"):
            continue
        base = col_lower[:-3]
        normalized_base = _normalize_relationship_name(base)
        if len(normalized_base) < 3:
            continue
        candidates = list(table_name_map.get(normalized_base, []))
        if not candidates:
            for normalized, names in table_name_map.items():
                if normalized == normalized_base:
                    continue
                if normalized.endswith(f"_{normalized_base}") or normalized.endswith(normalized_base):
                    candidates.extend(names)
        if not candidates:
            continue
        to_table = sorted(candidates)[0]
        foreign_keys.append({
            "from_column": col,
            "to_table": to_table,
            "to_column": "id",
        })
    if not foreign_keys:
        return {}
    return {
        "foreign_keys": foreign_keys,
        "join_paths": [f"{fk['from_column']} -> {fk['to_table']}.{fk['to_column']}" for fk in foreign_keys],
    }


# ============================================================================
# Diagnostic Reporting
# ============================================================================

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


# ============================================================================
# Argument Parsing Utilities
# ============================================================================

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
