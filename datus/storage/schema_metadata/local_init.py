# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Local schema initialization - Backward compatibility wrapper.

This file has been refactored into separate modules for better maintainability.
All classes and functions are re-exported from the new modules for backward compatibility.

REFACTORED STRUCTURE:
    datus/storage/schema_metadata/local_init_modules/
    ├── __init__.py           # Public API exports
    ├── enum_extraction.py    # Enum extraction from column comments
    ├── utils.py              # Helper functions for relationships and DDL parsing
    ├── table_storage.py      # Table storage and metadata processing
    ├── db_initializers.py    # Database-specific initialization functions
    └── initialization.py     # Main schema initialization orchestrator

MIGRATION GUIDE:
    Old imports (still work):
        from datus.storage.schema_metadata.local_init import init_local_schema
        from datus.storage.schema_metadata.local_init import store_tables

    New imports (recommended):
        from datus.storage.schema_metadata.local_init_modules import init_local_schema
        from datus.storage.schema_metadata.local_init_modules import store_tables
        from datus.storage.schema_metadata.local_init_modules.enum_extraction import _get_enum_extractor
        from datus.storage.schema_metadata.local_init_modules.utils import _infer_relationships_from_names

For detailed refactoring information, see the module docstrings in each submodule.
"""

import argparse
import sys

from datus.configuration.agent_config_loader import load_agent_config
from datus.storage.schema_metadata import SchemaWithValueRAG
from datus.tools.db_tools.db_manager import get_db_manager
from datus.utils.loggings import get_logger

# Re-export everything from the new modules for backward compatibility
from .local_init_modules import (
    # Enum extraction
    _get_enum_extractor,
    _extract_enums,
    # Utility functions
    _normalize_relationship_name,
    _build_table_name_map,
    _infer_relationships_from_names,
    _normalize_dialect,
    _llm_fallback_parse_ddl,
    _fill_sample_rows,
    extract_enhanced_metadata_from_ddl,
    parse_dialect,
    sanitize_ddl_for_storage,
    # Table storage
    store_tables,
    # Database-specific initializers
    init_sqlite_schema,
    init_duckdb_schema,
    init_mysql_schema,
    init_starrocks_schema,
    init_other_three_level_schema,
    # Main initialization
    init_local_schema,
)

logger = get_logger(__name__)

__all__ = [
    # Enum extraction
    "_get_enum_extractor",
    "_extract_enums",
    # Utility functions
    "_normalize_relationship_name",
    "_build_table_name_map",
    "_infer_relationships_from_names",
    "_normalize_dialect",
    "_llm_fallback_parse_ddl",
    "_fill_sample_rows",
    "extract_enhanced_metadata_from_ddl",
    "parse_dialect",
    "sanitize_ddl_for_storage",
    # Table storage
    "store_tables",
    # Database-specific initializers
    "init_sqlite_schema",
    "init_duckdb_schema",
    "init_mysql_schema",
    "init_starrocks_schema",
    "init_other_three_level_schema",
    # Main initialization
    "init_local_schema",
    "main",
]


def main():
    """CLI entry point for schema import.

    Usage:
        python -m datus.storage.schema_metadata.local_init --config=<config_path> --namespace=<namespace>
    """
    parser = argparse.ArgumentParser(description="Import schema metadata from database into LanceDB")
    parser.add_argument("--config", required=True, help="Path to agent configuration file")
    parser.add_argument("--namespace", help="Namespace for the database")
    parser.add_argument("--database", help="Database name (optional, uses all databases if not specified)")
    parser.add_argument("--catalog", default="", help="Catalog name (for StarRocks/Snowflake)")
    parser.add_argument("--table-type", default="full", choices=["table", "view", "mv", "full"], help="Type of tables to import")
    parser.add_argument("--build-mode", default="overwrite", choices=["overwrite", "append"], help="Build mode: overwrite or append")
    parser.add_argument("--prune-missing", action="store_true", help="Delete tables not present in source when build-mode=append")
    parser.add_argument("--llm-enum-extraction", action="store_true", help="Use LLM to enhance enum value extraction from comments")

    args = parser.parse_args()

    # Load agent configuration
    try:
        agent_config = load_agent_config(config=args.config)
    except Exception as e:
        logger.error(f"Failed to load agent configuration: {e}")
        sys.exit(1)

    # Determine namespace
    namespace = args.namespace
    if not namespace:
        namespaces = list(agent_config.namespaces.keys()) if agent_config.namespaces else []
        if len(namespaces) == 1:
            namespace = namespaces[0]
            logger.info(f"Auto-selected namespace: {namespace}")
        elif len(namespaces) > 1:
            logger.error(f"Multiple namespaces found: {namespaces}")
            logger.error("Please specify --namespace")
            sys.exit(1)
        else:
            logger.error("No namespaces found in configuration")
            sys.exit(1)

    # Set current namespace
    agent_config.current_namespace = namespace

    logger.info("=" * 80)
    logger.info("SCHEMA IMPORT")
    logger.info("=" * 80)
    logger.info(f"Namespace: {namespace}")
    logger.info(f"Table type: {args.table_type}")
    logger.info(f"Build mode: {args.build_mode}")
    logger.info(f"Prune missing: {args.prune_missing}")
    logger.info("")

    if args.prune_missing and args.build_mode != "append":
        logger.warning("Prune missing is only effective with --build-mode=append; ignoring for overwrite mode")

    # Initialize storage
    storage = SchemaWithValueRAG(agent_config)

    # Get database manager - MUST pass namespaces configuration
    db_manager = get_db_manager(agent_config.namespaces)

    # Import schemas
    try:
        init_local_schema(
            storage,
            agent_config,
            db_manager,
            build_mode=args.build_mode,
            table_type=args.table_type,
            prune_missing=args.prune_missing,
            init_catalog_name=args.catalog,
            init_database_name=args.database or "",
            llm_enum_extraction=args.llm_enum_extraction,
        )

        # Verify import was successful
        schema_store = storage.schema_store
        schema_store._ensure_table_ready()

        all_data = schema_store._search_all(
            where=None,
            select_fields=["identifier"]
        )

        imported_count = len(all_data)

        logger.info("")
        logger.info("=" * 80)
        logger.info("SCHEMA IMPORT COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total schemas imported: {imported_count}")
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

    except Exception as e:
        logger.error(f"Schema import failed: {e}")
        logger.info("")
        logger.info("To diagnose the issue, check:")
        logger.info("  1. Database connection parameters in agent.yml")
        logger.info("  2. Database is accessible and contains tables")
        logger.info("  3. Credentials are valid")
        logger.info("")
        logger.info("Example connection test:")
        logger.info(f"  python -c \"")
        logger.info(f"    from datus.tools.db_tools.db_manager import get_db_manager")
        logger.info(f"    from datus.configuration.agent_config_loader import load_agent_config")
        logger.info(f"    config = load_agent_config('{args.config}')")
        logger.info(f"    config.current_namespace = '{namespace}'")
        logger.info(f"    db = get_db_manager()")
        logger.info(f"    conn = db.get_conn('{namespace}', '{args.database or '<database_name>'}')")
        logger.info(f"    print('Tables:', len(conn.get_tables_with_ddl()))")
        logger.info(f"  \"")
        logger.info("=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    main()
