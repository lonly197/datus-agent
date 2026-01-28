# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Main schema initialization orchestrator.

This module provides the main entry point for initializing local schemas
from different database types.
"""

from typing import Optional

from datus.configuration.agent_config import AgentConfig, DbConfig
from datus.models.base import LLMBaseModel
from datus.schemas.base import TABLE_TYPE
from datus.storage.schema_metadata.store import SchemaWithValueRAG
from datus.tools.db_tools.db_manager import DBManager
from datus.utils.constants import DBType
from datus.utils.loggings import get_logger

from .db_initializers import (
    init_sqlite_schema,
    init_duckdb_schema,
    init_mysql_schema,
    init_starrocks_schema,
    init_other_three_level_schema,
)

logger = get_logger(__name__)


def init_local_schema(
    table_lineage_store: SchemaWithValueRAG,
    agent_config: AgentConfig,
    db_manager: DBManager,
    build_mode: str = "overwrite",
    table_type: TABLE_TYPE = "full",
    init_catalog_name: str = "",
    init_database_name: str = "",
    pool_size: int = 4,  # TODO: support multi-threading
    prune_missing: bool = False,
    llm_fallback: bool = False,
    llm_model: Optional[LLMBaseModel] = None,
    extract_statistics: bool = False,
    extract_relationships: bool = True,
    llm_enum_extraction: bool = False,
):
    """Initialize local schema from the configured database."""
    logger.info(f"Initializing local schema for namespace: {agent_config.current_namespace}")
    db_configs = agent_config.namespaces[agent_config.current_namespace]
    if len(db_configs) == 1:
        db_configs = list(db_configs.values())[0]

    if isinstance(db_configs, DbConfig):
        # Single database configuration (like StarRocks, MySQL, PostgreSQL, etc.)
        logger.info(f"Processing single database configuration: {db_configs.type}")
        if db_configs.type == DBType.SQLITE:
            init_sqlite_schema(
                table_lineage_store,
                agent_config,
                db_configs,
                db_manager,
                table_type=table_type,
                build_mode=build_mode,
                prune_missing=prune_missing,
                llm_fallback=llm_fallback,
                llm_model=llm_model,
                extract_statistics=extract_statistics,
                extract_relationships=extract_relationships,
                llm_enum_extraction=llm_enum_extraction,
            )
        elif db_configs.type == DBType.DUCKDB:
            init_duckdb_schema(
                table_lineage_store,
                agent_config,
                db_configs,
                db_manager,
                schema_name=init_database_name,
                table_type=table_type,
                build_mode=build_mode,
                prune_missing=prune_missing,
                llm_fallback=llm_fallback,
                llm_model=llm_model,
                extract_statistics=extract_statistics,
                extract_relationships=extract_relationships,
            )
        elif db_configs.type == DBType.MYSQL:
            init_mysql_schema(
                table_lineage_store,
                agent_config,
                db_configs,
                db_manager,
                database_name=init_database_name,
                table_type=table_type,
                build_mode=build_mode,
                prune_missing=prune_missing,
                llm_fallback=llm_fallback,
                llm_model=llm_model,
                extract_statistics=extract_statistics,
                extract_relationships=extract_relationships,
                llm_enum_extraction=llm_enum_extraction,
            )
        elif db_configs.type == DBType.STARROCKS:
            init_starrocks_schema(
                table_lineage_store,
                agent_config,
                db_configs,
                db_manager,
                catalog_name=init_catalog_name,
                database_name=init_database_name,
                table_type=table_type,
                build_mode=build_mode,
                prune_missing=prune_missing,
                llm_fallback=llm_fallback,
                llm_model=llm_model,
                extract_statistics=extract_statistics,
                extract_relationships=extract_relationships,
            )
        else:
            init_other_three_level_schema(
                table_lineage_store,
                agent_config,
                db_configs,
                db_manager,
                database_name=init_database_name,
                table_type=table_type,
                build_mode=build_mode,
                prune_missing=prune_missing,
                llm_fallback=llm_fallback,
                llm_model=llm_model,
                extract_statistics=extract_statistics,
                extract_relationships=extract_relationships,
                llm_enum_extraction=llm_enum_extraction,
            )

    else:
        # Multiple database configuration (like multiple SQLite files)
        logger.info("Processing multiple database configuration")
        if not db_configs:
            logger.warning("No database configuration found")
            return

        for database_name, db_config in db_configs.items():
            logger.info(f"Processing database: {database_name}")
            if init_database_name and init_database_name != database_name:
                logger.info(f"Skip database: {database_name} because it is not the same as {init_database_name}")
                continue
            # only sqlite and duckdb support multiple databases
            if db_config.type == DBType.SQLITE:
                init_sqlite_schema(
                    table_lineage_store,
                    agent_config,
                    db_config,
                    db_manager,
                    table_type=table_type,
                    build_mode=build_mode,
                    prune_missing=prune_missing,
                    llm_fallback=llm_fallback,
                    llm_model=llm_model,
                    extract_statistics=extract_statistics,
                    extract_relationships=extract_relationships,
                    llm_enum_extraction=llm_enum_extraction,
                )
            elif db_config.type == DBType.DUCKDB:
                init_duckdb_schema(
                    table_lineage_store,
                    agent_config,
                    db_config,
                    db_manager,
                    database_name=database_name,
                    schema_name=init_database_name,
                    table_type=table_type,
                    build_mode=build_mode,
                    prune_missing=prune_missing,
                    llm_fallback=llm_fallback,
                    llm_model=llm_model,
                    extract_statistics=extract_statistics,
                    extract_relationships=extract_relationships,
                    llm_enum_extraction=llm_enum_extraction,
                )
            else:
                logger.warning(f"Unsupported database type {db_config.type} for multi-database configuration")
    # Create indices after initialization
    table_lineage_store.after_init()
    logger.info("Local schema initialization completed")


__all__ = [
    "init_local_schema",
]
