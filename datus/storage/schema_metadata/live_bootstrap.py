# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Live database metadata bootstrapping from database connections.

This module provides functionality to bootstrap enhanced schema metadata from live
database connections (not from benchmark CSV files). Supports incremental updates
and parallel processing for large databases.
"""

import asyncio
import json
import time
from typing import Any, Dict, List, Optional

from datus.configuration.agent_config import AgentConfig
from datus.storage.schema_metadata import SchemaWithValueRAG
from datus.tools.db_tools.metadata_extractor import get_metadata_extractor
from datus.utils.loggings import get_logger
from datus.utils.sql_utils import extract_enhanced_metadata_from_ddl, parse_dialect
from datus.configuration.business_term_config import infer_business_tags

logger = get_logger(__name__)


# Comprehensive system table prefixes for different databases
SYSTEM_TABLE_PREFIXES = {
    "information_schema",
    "pg_",
    "pg_catalog",
    "pg_toast",
    "mysql_",
    "performance_schema",
    "sys.",
    "sqlite_",
    "snowflake.",
    "system$",
    "__",
}


def is_system_table(table_name: str) -> bool:
    """
    Check if a table name matches system table patterns.

    System tables are internal database tables that should be excluded
    from metadata bootstrap operations.

    Args:
        table_name: The table name to check

    Returns:
        True if the table appears to be a system table
    """
    table_lower = table_name.lower()
    return any(table_lower.startswith(prefix.lower()) for prefix in SYSTEM_TABLE_PREFIXES)


async def bootstrap_database_metadata(
    storage: SchemaWithValueRAG,
    connector,
    catalog_name: str = "",
    database_name: str = "",
    schema_name: str = "",
    extract_statistics: bool = True,
    extract_relationships: bool = True,
    batch_size: int = 100,
    dialect: str = "duckdb",
) -> Dict[str, Any]:
    """
    Bootstrap schema metadata from a live database connection.

    This function connects to the database and extracts enhanced metadata including:
    - DDL statements
    - Table and column comments
    - Business domain tags
    - Row counts
    - Column statistics (if enabled)
    - Foreign key relationships (if enabled)

    Args:
        storage: SchemaWithValueRAG instance
        connector: Database connector instance
        catalog_name: Optional catalog filter
        database_name: Optional database filter
        schema_name: Optional schema filter
        extract_statistics: Extract column statistics (can be expensive for large tables)
        extract_relationships: Detect foreign key relationships
        batch_size: Number of tables to process per batch
        dialect: Database type identifier

    Returns:
        Dictionary with bootstrap results:
        {
            "total_tables": int,
            "processed_tables": int,
            "skipped_tables": int,
            "failed_tables": int,
            "duration_seconds": float
        }
    """
    start_time = time.time()
    total_tables = 0
    processed_tables = 0
    skipped_tables = 0
    failed_tables = 0

    try:
        logger.info(f"Starting live database metadata bootstrap for {dialect} database")
        logger.info(f"Extract statistics: {extract_statistics}")
        logger.info(f"Extract relationships: {extract_relationships}")
        logger.info("")

        # Get all table names from database
        all_tables = connector.get_all_table_names()

        # Filter by catalog/database/schema if specified
        filtered_tables = []
        for table in all_tables:
            # Skip system tables using comprehensive filter
            if is_system_table(table):
                logger.debug(f"Skipping system table: {table}")
                continue

            # Apply filters if specified
            if catalog_name and hasattr(connector, 'catalog_name') and connector.catalog_name != catalog_name:
                continue
            if database_name and hasattr(connector, 'database_name') and connector.database_name != database_name:
                continue
            if schema_name and hasattr(connector, 'schema_name') and connector.schema_name != schema_name:
                continue

            filtered_tables.append(table)

        total_tables = len(filtered_tables)
        logger.info(f"Found {total_tables} tables after filtering")

        if total_tables == 0:
            logger.warning("No tables found to bootstrap")
            return {
                "total_tables": 0,
                "processed_tables": 0,
                "skipped_tables": 0,
                "failed_tables": 0,
                "duration_seconds": 0.0
            }

        # Get metadata extractor for this database type
        metadata_extractor = get_metadata_extractor(connector, dialect)

        # Process tables in batches
        batch_records = []
        batch_value_records = []

        for i, table_name in enumerate(filtered_tables):
            try:
                logger.info(f"[{i+1}/{total_tables}] Processing table: {table_name}")

                # Get DDL for the table
                ddl = None
                if hasattr(connector, "get_table_ddl"):
                    ddl = connector.get_table_ddl(table_name)
                elif hasattr(connector, "get_ddl"):
                    ddl = connector.get_ddl(table_name)

                if not ddl:
                    logger.warning(f"  ⚠️  No DDL found for {table_name}, skipping")
                    skipped_tables += 1
                    continue

                # Parse enhanced metadata from DDL
                parsed_metadata = extract_enhanced_metadata_from_ddl(ddl, dialect=dialect)

                # Extract table comment
                table_comment = parsed_metadata["table"].get("comment", "")

                # Extract column comments
                column_comments = {
                    col["name"]: col.get("comment", "")
                    for col in parsed_metadata["columns"]
                }

                # Infer business tags
                column_names = [col["name"] for col in parsed_metadata["columns"]]
                business_tags = infer_business_tags(table_name, column_names)

                # Extract row count
                row_count = 0
                if extract_statistics and metadata_extractor:
                    try:
                        row_count = metadata_extractor.extract_row_count(table_name)
                        logger.debug(f"  Row count: {row_count}")
                    except Exception as e:
                        logger.debug(f"  Could not extract row count: {e}")

                # Extract column statistics
                sample_statistics = {}
                if extract_statistics and metadata_extractor and row_count > 1000:  # Only for tables with sufficient data
                    try:
                        sample_statistics = metadata_extractor.extract_column_statistics(table_name)
                        logger.debug(f"  Column statistics: {len(sample_statistics)} columns")
                    except Exception as e:
                        logger.debug(f"  Could not extract column statistics: {e}")

                # Extract relationships
                relationship_metadata = {}
                if extract_relationships and metadata_extractor:
                    try:
                        relationships = metadata_extractor.detect_relationships(table_name)
                        relationship_metadata = relationships
                        logger.debug(f"  Relationships: {len(relationships.get('foreign_keys', []))} FKs found")
                    except Exception as e:
                        logger.debug(f"  Could not detect relationships: {e}")

                # Build identifier
                identifier_parts = [
                    catalog_name or "",
                    database_name or "",
                    schema_name or "",
                    table_name,
                    "table"
                ]
                identifier = ".".join(identifier_parts)

                # Prepare schema record
                schema_record = {
                    "identifier": identifier,
                    "catalog_name": catalog_name or "",
                    "database_name": database_name or "",
                    "schema_name": schema_name or "",
                    "table_name": table_name,
                    "table_type": "table",  # Could be enhanced to detect view/mv
                    "definition": ddl,
                    # Enhanced metadata fields
                    "table_comment": table_comment,
                    "column_comments": json.dumps(column_comments, ensure_ascii=False),
                    "business_tags": business_tags,
                    "row_count": row_count,
                    "sample_statistics": json.dumps(sample_statistics, ensure_ascii=False),
                    "relationship_metadata": json.dumps(relationship_metadata, ensure_ascii=False),
                    "metadata_version": 1,
                    "last_updated": int(time.time())
                }

                batch_records.append(schema_record)

                processed_tables += 1

                # Batch store
                if len(batch_records) >= batch_size:
                    logger.info(f"  Storing batch of {len(batch_records)} tables...")
                    storage.store_batch(batch_records, [])
                    batch_records = []
                    logger.info("  ✅ Batch stored")

            except Exception as e:
                logger.error(f"  ❌ Failed to process table {table_name}: {e}")
                failed_tables += 1
                continue

        # Store remaining records
        if batch_records:
            logger.info(f"Storing final batch of {len(batch_records)} tables...")
            storage.store_batch(batch_records, [])
            logger.info("✅ Final batch stored")

        # Create indices for optimized search
        logger.info("")
        logger.info("Creating indices...")
        storage.after_init()
        logger.info("✅ Indices created")

    except Exception as e:
        logger.error(f"Bootstrap failed: {e}")
        raise

    duration = time.time() - start_time

    return {
        "total_tables": total_tables,
        "processed_tables": processed_tables,
        "skipped_tables": skipped_tables,
        "failed_tables": failed_tables,
        "duration_seconds": duration
    }


async def bootstrap_incremental(
    storage: SchemaWithValueRAG,
    connector,
    catalog_name: str = "",
    database_name: str = "",
    schema_name: str = "",
    batch_size: int = 100,
    dialect: str = "duckdb",
) -> Dict[str, Any]:
    """
    Incrementally bootstrap metadata - only update tables that have changed.

    This compares existing metadata with current database state and updates only
    tables where DDL or statistics have changed.

    Args:
        storage: SchemaWithValueRAG instance
        connector: Database connector instance
        catalog_name: Optional catalog filter
        database_name: Optional database filter
        schema_name: Optional schema filter
        batch_size: Number of tables to process per batch
        dialect: Database type identifier

    Returns:
        Dictionary with bootstrap results
    """
    start_time = time.time()
    updated_tables = 0
    unchanged_tables = 0
    failed_tables = 0

    try:
        logger.info(f"Starting incremental metadata bootstrap for {dialect} database")

        # Get all table names
        all_tables = connector.get_all_table_names()
        logger.info(f"Checking {len(all_tables)} tables for updates...")

        metadata_extractor = get_metadata_extractor(connector, dialect)

        for table_name in all_tables:
            try:
                # Check if table exists in storage
                existing = storage.schema_store.get_schema(
                    table_name=table_name,
                    catalog_name=catalog_name,
                    database_name=database_name,
                    schema_name=schema_name
                )

                # Get current DDL
                ddl = None
                if hasattr(connector, "get_table_ddl"):
                    ddl = connector.get_table_ddl(table_name)
                elif hasattr(connector, "get_ddl"):
                    ddl = connector.get_ddl(table_name)

                if not ddl:
                    logger.debug(f"  No DDL for {table_name}, skipping")
                    continue

                # Check if DDL has changed
                needs_update = True
                if existing and len(existing) > 0:
                    existing_ddl = existing["definition"][0] if len(existing) > 0 else ""
                    if ddl.strip() == existing_ddl.strip():
                        unchanged_tables += 1
                        logger.debug(f"  {table_name}: unchanged (DDL matches)")
                        needs_update = False
                    else:
                        logger.info(f"  {table_name}: DDL changed, updating...")

                if needs_update:
                    # Parse enhanced metadata
                    parsed_metadata = extract_enhanced_metadata_from_ddl(ddl, dialect=dialect)

                    # Extract all enhanced metadata
                    table_comment = parsed_metadata["table"].get("comment", "")
                    column_comments = {col["name"]: col.get("comment", "") for col in parsed_metadata["columns"]}
                    column_names = [col["name"] for col in parsed_metadata["columns"]]
                    business_tags = infer_business_tags(table_name, column_names)

                    # Extract statistics
                    row_count = 0
                    sample_statistics = {}
                    relationship_metadata = {}

                    if metadata_extractor:
                        try:
                            row_count = metadata_extractor.extract_row_count(table_name)
                            sample_statistics = metadata_extractor.extract_column_statistics(table_name)
                            relationship_metadata = metadata_extractor.detect_relationships(table_name)
                        except Exception as e:
                            logger.debug(f"  Could not extract enhanced metadata: {e}")

                    # Update schema storage
                    success = storage.schema_store.update_table_schema(
                        table_name=table_name,
                        definition=ddl,
                        catalog_name=catalog_name,
                        database_name=database_name,
                        schema_name=schema_name,
                        table_type="table",
                        table_comment=table_comment,
                        column_comments=column_comments,
                        business_tags=business_tags,
                        row_count=row_count,
                        sample_statistics=sample_statistics,
                        relationship_metadata=relationship_metadata,
                        metadata_version=1
                    )

                    if success:
                        updated_tables += 1
                        logger.info(f"  ✅ {table_name}: updated")
                    else:
                        failed_tables += 1
                        logger.error(f"  ❌ {table_name}: update failed")

            except Exception as e:
                logger.error(f"  ❌ Failed to process {table_name}: {e}")
                failed_tables += 1
                continue

        duration = time.time() - start_time

        return {
            "total_tables": len(all_tables),
            "updated_tables": updated_tables,
            "unchanged_tables": unchanged_tables,
            "failed_tables": failed_tables,
            "duration_seconds": duration
        }

    except Exception as e:
        logger.error(f"Incremental bootstrap failed: {e}")
        raise


def print_bootstrap_results(results: Dict[str, Any]):
    """Print bootstrap results in a formatted way."""
    logger.info("")
    logger.info("=" * 80)
    logger.info("BOOTSTRAP RESULTS")
    logger.info("=" * 80)
    logger.info(f"Total tables: {results.get('total_tables', 0)}")
    logger.info(f"Processed: {results.get('processed_tables', 0)}")
    logger.info(f"Updated: {results.get('updated_tables', 0)}")
    logger.info(f"Unchanged: {results.get('unchanged_tables', 0)}")
    logger.info(f"Skipped: {results.get('skipped_tables', 0)}")
    logger.info(f"Failed: {results.get('failed_tables', 0)}")
    logger.info(f"Duration: {results.get('duration_seconds', 0):.2f} seconds")
    logger.info("=" * 80)


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


async def main():
    """Main function for CLI usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Bootstrap metadata from live database")
    parser.add_argument("--config", required=True, help="Path to agent configuration file")
    parser.add_argument("--catalog", default="", help="Catalog name filter")
    parser.add_argument("--database", default="", help="Database name filter")
    parser.add_argument("--schema", default="", help="Schema name filter")
    parser.add_argument("--extract-statistics", type=str_to_bool, default=False, help="Extract column statistics (default: false)")
    parser.add_argument("--extract-relationships", type=str_to_bool, default=True, help="Extract relationship metadata (default: true)")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for processing")
    parser.add_argument("--dialect", default="duckdb", help="Database type")
    parser.add_argument("--incremental", action="store_true", help="Incremental update mode")

    args = parser.parse_args()

    # Load agent configuration
    try:
        agent_config = AgentConfig.from_yaml(args.config)
    except Exception as e:
        logger.error(f"Failed to load agent configuration: {e}")
        return

    # Get database connector
    try:
        from datus.tools.db_tools.db_manager import get_db_manager
        db_manager = get_db_manager()
        connector = db_manager.get_conn(agent_config.current_namespace, args.database or agent_config.database_name)

        if not connector:
            logger.error(f"Failed to get database connector for {args.database}")
            return

    except Exception as e:
        logger.error(f"Failed to initialize database connector: {e}")
        return

    # Initialize storage
    try:
        storage = SchemaWithValueRAG(agent_config)
        logger.info("✅ Storage initialized")
    except Exception as e:
        logger.error(f"Failed to initialize storage: {e}")
        return

    # Run bootstrap
    try:
        if args.incremental:
            logger.info("Running incremental bootstrap...")
            results = await bootstrap_incremental(
                storage=storage,
                connector=connector,
                catalog_name=args.catalog,
                database_name=args.database,
                schema_name=args.schema,
                batch_size=args.batch_size,
                dialect=args.dialect
            )
        else:
            logger.info("Running full bootstrap...")
            results = await bootstrap_database_metadata(
                storage=storage,
                connector=connector,
                catalog_name=args.catalog,
                database_name=args.database,
                schema_name=args.schema,
                extract_statistics=args.extract_statistics,
                extract_relationships=args.extract_relationships,
                batch_size=args.batch_size,
                dialect=args.dialect
            )

        print_bootstrap_results(results)

    except Exception as e:
        logger.error(f"Bootstrap failed: {e}")
        return


if __name__ == "__main__":
    asyncio.run(main())
