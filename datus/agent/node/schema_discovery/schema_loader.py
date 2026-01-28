# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details

"""
Schema loading and DDL fallback for schema discovery.

This module provides functionality to load table schemas from storage,
repair missing metadata, and handle DDL fallbacks when storage is empty.
"""

from typing import Any, Dict, List

from datus.schemas.node_models import TableSchema
from datus.storage.schema_metadata import SchemaWithValueRAG
from datus.tools.db_tools.db_manager import get_db_manager
from datus.utils.context_lock import safe_context_update
from datus.utils.loggings import get_logger
from datus.utils.sql_utils import (
    ddl_has_missing_commas,
    is_likely_truncated_ddl,
)

logger = get_logger(__name__)


async def get_all_database_tables(
    agent_config,
    workflow_task,
) -> List[str]:
    """
    Fetch all table names from the database(s).
    Supports multi-database configuration by iterating through all configured databases.

    Args:
        agent_config: Agent configuration
        workflow_task: Workflow task with database information

    Returns:
        List of all table names in the database(s)
    """
    try:
        if not workflow_task:
            return []

        task = workflow_task
        db_manager = get_db_manager()

        # Check if multiple databases are configured
        db_configs = agent_config.current_db_configs()
        is_multi_db = len(db_configs) > 1

        all_tables = []

        if is_multi_db:
            logger.info(f"Multi-database mode detected: scanning {len(db_configs)} databases...")
            for logic_name, db_config in db_configs.items():
                try:
                    connector = db_manager.get_conn(agent_config.current_namespace, logic_name)
                    if not connector:
                        continue

                    tables = connector.get_tables(
                        catalog_name=db_config.catalog or "",
                        database_name=db_config.database or "",
                        schema_name=db_config.schema or "",
                    )
                    # Prefix with logic_name to distinguish tables from different DBs
                    prefixed_tables = [f"{logic_name}.{t}" for t in tables]
                    all_tables.extend(prefixed_tables)
                except Exception as e:
                    logger.warning(f"Failed to get tables for database {logic_name}: {e}")
        else:
            # Single database mode (legacy behavior)
            connector = db_manager.get_conn(agent_config.current_namespace, task.database_name)

            if not connector:
                logger.warning(f"Could not get database connector for {task.database_name}")
                return []

            all_tables = connector.get_tables(
                catalog_name=task.catalog_name or "",
                database_name=task.database_name or "",
                schema_name=task.schema_name or "",
            )

        logger.debug(f"Retrieved {len(all_tables)} tables from database(s)")
        return all_tables

    except Exception as e:
        logger.warning(f"Failed to retrieve database tables: {e}")
        return []


async def fallback_get_all_tables(
    agent_config,
    task,
) -> List[str]:
    """
    Fallback: Get all tables from the database if no candidates found.

    Args:
        agent_config: Agent configuration
        task: The SQL task

    Returns:
        List of all table names (limited to top N)
    """
    try:
        db_manager = get_db_manager()
        connector = db_manager.get_conn(agent_config.current_namespace, task.database_name)

        if not connector:
            return []

        all_tables = connector.get_tables(
            catalog_name=task.catalog_name or "",
            database_name=task.database_name or "",
            schema_name=task.schema_name or "",
        )

        # Limit to reasonable number to avoid context overflow
        max_tables = 50
        if agent_config and hasattr(agent_config, "schema_discovery_config"):
            max_tables = agent_config.schema_discovery_config.fallback_table_limit
        if len(all_tables) > max_tables:
            logger.warning(f"Too many tables ({len(all_tables)}), limiting to first {max_tables} for fallback")
            return all_tables[:max_tables]

        return all_tables

    except Exception as e:
        logger.warning(f"Fallback table discovery failed: {e}")
        return []


async def load_table_schemas(
    task,
    candidate_tables: List[str],
    agent_config,
    workflow,
    repair_metadata_func,
    ddl_fallback_func,
) -> None:
    """
    Load schemas for candidate tables and update workflow context.
    Includes automatic metadata repair if definitions are missing.

    Fixed: Uses thread-safe context update to prevent race conditions.

    Args:
        task: The SQL task
        candidate_tables: List of table names to load schemas for
        agent_config: Agent configuration
        workflow: Workflow instance
        repair_metadata_func: Function to repair metadata
        ddl_fallback_func: Function for DDL fallback
    """
    if not candidate_tables or not workflow:
        return

    try:
        # Check if multiple databases are configured
        if agent_config is None:
            db_configs = {}
        else:
            db_configs = agent_config.current_db_configs()
        is_multi_db = len(db_configs) > 1

        # Repair Logic
        if agent_config:
            try:
                from datus.storage.cache import get_storage_cache_instance

                storage_cache = get_storage_cache_instance(agent_config)
                schema_storage = storage_cache.schema_storage()

                # Check for missing definitions
                current_schemas = schema_storage.get_table_schemas(candidate_tables)
                missing_tables = []
                invalid_tables = []

                schema_dicts = current_schemas.to_pylist()
                definition_by_table = {
                    str(schema.get("table_name", "")): schema.get("definition") for schema in schema_dicts
                }
                for table_name in candidate_tables:
                    # Handle potential prefix
                    lookup_name = table_name.split(".")[-1] if "." in table_name else table_name
                    definition = definition_by_table.get(lookup_name)

                    if not definition or not str(definition).strip():
                        missing_tables.append(table_name)
                        continue

                    definition_text = str(definition)
                    if ddl_has_missing_commas(definition_text) or is_likely_truncated_ddl(definition_text):
                        invalid_tables.append(table_name)

                if missing_tables or invalid_tables:
                    logger.info(
                        "Found %d tables with missing metadata and %d with invalid DDL. Attempting repair...",
                        len(missing_tables),
                        len(invalid_tables),
                    )
                    tables_to_repair = sorted(set(missing_tables + invalid_tables))
                    await repair_metadata_func(tables_to_repair, schema_storage, task)
                    logger.info("Retrying DDL fallback for tables with missing or invalid metadata...")
                    await ddl_fallback_func(tables_to_repair, task)

            except Exception as e:
                logger.warning(f"Metadata repair pre-check failed: {e}")

        # Use SchemaWithValueRAG to load table schemas
        if agent_config:
            rag = SchemaWithValueRAG(agent_config=agent_config)

            # Determine target database for search
            target_database = "" if is_multi_db else (task.database_name or "")

            schemas, values = rag.search_tables(
                tables=candidate_tables,
                catalog_name=task.catalog_name or "",
                database_name=target_database,
                schema_name=task.schema_name or "",
                dialect=task.database_type if hasattr(task, "database_type") else None,
            )

            if schemas:
                # Use thread-safe context update
                def update_schemas():
                    workflow.context.update_schema_and_values(schemas, values)
                    return {"loaded_count": len(schemas)}

                result = safe_context_update(
                    workflow.context,
                    update_schemas,
                    operation_name="load_table_schemas",
                )

                if result["success"]:
                    logger.debug(f"Loaded schemas for {result['result']['loaded_count']} tables")
                else:
                    logger.warning(f"Context update failed: {result.get('error')}")
            else:
                logger.warning(f"No schemas found for candidate tables: {candidate_tables}")

                # DDL Fallback: If storage is empty, retrieve DDL from database
                if candidate_tables:
                    logger.info("Schema storage empty, attempting DDL retrieval fallback...")
                    await ddl_fallback_func(candidate_tables, task)

    except Exception as e:
        logger.warning(f"Failed to load table schemas: {e}")


async def repair_metadata(
    table_names: List[str],
    schema_storage,
    task,
    agent_config,
) -> int:
    """
    Repair metadata for tables with missing or invalid definitions.

    Args:
        table_names: List of table names to repair
        schema_storage: Schema storage instance
        task: SQL task
        agent_config: Agent configuration

    Returns:
        Number of tables successfully repaired
    """
    repaired_count = 0
    if not table_names:
        return repaired_count

    try:
        db_manager = get_db_manager()
        connector = db_manager.get_conn(agent_config.current_namespace, task.database_name)

        if not connector:
            logger.warning("Cannot repair metadata: no database connector available")
            return repaired_count

        for table_name in table_names:
            try:
                # Get DDL from database
                ddl = await get_ddl_with_fallbacks(
                    table_name=table_name,
                    connector=connector,
                    task=task,
                )

                if ddl:
                    # Update storage
                    schema_storage.update_table_schema(
                        table_name=table_name,
                        updates={"definition": ddl}
                    )
                    repaired_count += 1
                    logger.info(f"Repaired metadata for table: {table_name}")

            except Exception as e:
                logger.warning(f"Failed to repair metadata for {table_name}: {e}")

        return repaired_count

    except Exception as e:
        logger.warning(f"Metadata repair failed: {e}")
        return repaired_count


async def ddl_fallback_and_retry(
    candidate_tables: List[str],
    task,
    agent_config,
    get_ddl_func,
) -> None:
    """
    DDL fallback: Retrieve DDL from database and populate storage.

    Args:
        candidate_tables: List of table names
        task: SQL task
        agent_config: Agent configuration
        get_ddl_func: Function to get DDL with fallbacks
    """
    if not candidate_tables:
        return

    try:
        db_manager = get_db_manager()
        connector = db_manager.get_conn(agent_config.current_namespace, task.database_name)

        if not connector:
            logger.warning("Cannot perform DDL fallback: no database connector available")
            return

        from datus.storage.cache import get_storage_cache_instance

        storage_cache = get_storage_cache_instance(agent_config)
        schema_storage = storage_cache.schema_storage()

        for table_name in candidate_tables:
            try:
                ddl = await get_ddl_func(table_name, connector, task)

                if ddl:
                    schema_storage.update_table_schema(
                        table_name=table_name,
                        updates={"definition": ddl}
                    )
                    logger.info(f"Populated DDL for table: {table_name}")

            except Exception as e:
                logger.warning(f"Failed to populate DDL for {table_name}: {e}")

    except Exception as e:
        logger.warning(f"DDL fallback failed: {e}")


async def get_ddl_with_fallbacks(
    table_name: str,
    connector,
    task,
) -> str:
    """
    Get DDL for a table with multiple fallback strategies.

    Args:
        table_name: Name of the table
        connector: Database connector
        task: SQL task

    Returns:
        DDL statement as string
    """
    # Implementation would try multiple methods to get DDL
    # This is a placeholder for the actual implementation
    try:
        # Try primary method
        if hasattr(connector, "get_ddl"):
            ddl = connector.get_ddl(
                table_name=table_name,
                catalog_name=task.catalog_name or "",
                database_name=task.database_name or "",
                schema_name=task.schema_name or "",
            )
            if ddl:
                return ddl
    except Exception as e:
        logger.debug(f"Primary DDL method failed for {table_name}: {e}")

    # Fallback methods would be implemented here
    return ""


def build_ddl_from_schema(
    table_name: str,
    schema_info: List[Dict[str, Any]],
) -> str:
    """
    Build a DDL statement from schema information.

    Args:
        table_name: Name of the table
        schema_info: List of column information dictionaries

    Returns:
        DDL CREATE TABLE statement
    """
    if not schema_info:
        return ""

    columns = []
    for col in schema_info:
        col_name = col.get("name", "")
        col_type = col.get("type", "TEXT")
        nullable = "NOT NULL" if not col.get("nullable", True) else ""
        columns.append(f"    {col_name} {col_type} {nullable}".strip())

    ddl = f"CREATE TABLE {table_name} (\n"
    ddl += ",\n".join(columns)
    ddl += "\n);"

    return ddl


def filter_candidate_tables_for_db(
    candidate_tables: List[str],
    all_db_tables: List[str],
) -> List[str]:
    """
    Filter candidate tables to only include tables that exist in the database.

    Args:
        candidate_tables: List of candidate table names
        all_db_tables: List of all tables in the database

    Returns:
        Filtered list of table names
    """
    if not all_db_tables:
        return candidate_tables

    db_table_set = set(all_db_tables)
    filtered = []

    for table in candidate_tables:
        # Handle prefixed tables (multi-db mode)
        lookup_name = table.split(".")[-1] if "." in table else table
        if lookup_name in db_table_set or table in db_table_set:
            filtered.append(table)

    return filtered
