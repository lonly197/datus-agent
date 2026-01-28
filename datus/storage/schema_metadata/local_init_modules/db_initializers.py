# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Database-specific schema initialization functions.

This module provides initialization functions for different database types:
- SQLite
- DuckDB
- MySQL
- StarRocks
- Other three-level databases (Snowflake, etc.)
"""

from typing import Optional

from datus.configuration.agent_config import AgentConfig, DbConfig
from datus.models.base import LLMBaseModel
from datus.schemas.base import TABLE_TYPE
from datus.storage.schema_metadata.store import SchemaWithValueRAG
from datus.tools.db_tools.db_manager import DBManager
from datus.utils.constants import DBType
from datus.utils.loggings import get_logger

from ..init_utils import exists_table_value
from .table_storage import store_tables

logger = get_logger(__name__)


def init_sqlite_schema(
    table_lineage_store: SchemaWithValueRAG,
    agent_config: AgentConfig,
    db_config: DbConfig,
    db_manager: DBManager,
    table_type: TABLE_TYPE = "table",
    build_mode: str = "overwrite",
    prune_missing: bool = False,
    llm_fallback: bool = False,
    llm_model: Optional[LLMBaseModel] = None,
    extract_statistics: bool = False,
    extract_relationships: bool = True,
    llm_enum_extraction: bool = False,
):
    """Initialize schema from SQLite database."""
    database_name = getattr(db_config, "database", "")
    sql_connector = db_manager.get_conn(agent_config.current_namespace, database_name)
    all_schema_tables, all_value_tables, all_schema_metadata = exists_table_value(
        table_lineage_store,
        database_name=database_name,
        table_type=table_type,
        build_mode=build_mode,
        return_metadata=True,
    )
    logger.info(
        f"Exists data from LanceDB {database_name}, tables={len(all_schema_tables)}, values={len(all_value_tables)}"
    )
    if table_type == "table" or table_type == "full":
        tables = sql_connector.get_tables_with_ddl(schema_name=database_name)
        for table in tables:
            table["database_name"] = database_name
        store_tables(
            table_lineage_store,
            database_name,
            all_schema_tables,
            all_value_tables,
            all_schema_metadata if prune_missing else None,
            tables,
            "table",
            sql_connector,
            build_mode=build_mode,
            prune_missing=prune_missing,
            llm_fallback=llm_fallback,
            llm_model=llm_model,
            extract_statistics=extract_statistics,
            extract_relationships=extract_relationships,
            llm_enum_extraction=llm_enum_extraction,
        )

    if (table_type == "view" or table_type == "full") and hasattr(sql_connector, "get_views_with_ddl"):
        views = sql_connector.get_views_with_ddl()
        store_tables(
            table_lineage_store,
            database_name,
            all_schema_tables,
            all_value_tables,
            all_schema_metadata if prune_missing else None,
            views,
            "view",
            sql_connector,
            build_mode=build_mode,
            prune_missing=prune_missing,
            llm_fallback=llm_fallback,
            llm_model=llm_model,
            extract_statistics=extract_statistics,
            extract_relationships=extract_relationships,
            llm_enum_extraction=llm_enum_extraction,
        )


def init_duckdb_schema(
    table_lineage_store: SchemaWithValueRAG,
    agent_config: AgentConfig,
    db_config: DbConfig,
    db_manager: DBManager,
    database_name: str = "",  # init database_name
    schema_name: str = "",
    table_type: TABLE_TYPE = "table",
    build_mode: str = "overwrite",
    prune_missing: bool = False,
    llm_fallback: bool = False,
    llm_model: Optional[LLMBaseModel] = None,
    extract_statistics: bool = False,
    extract_relationships: bool = True,
    llm_enum_extraction: bool = False,
):
    """Initialize schema from DuckDB database."""
    # means schema_name here
    database_name = database_name or getattr(db_config, "database", "")
    schema_name = schema_name or getattr(db_config, "schema", "")

    all_schema_tables, all_value_tables, all_schema_metadata = exists_table_value(
        table_lineage_store,
        database_name=database_name,
        table_type=table_type,
        build_mode=build_mode,
        return_metadata=True,
    )

    logger.info(
        f"Exists data from LanceDB {database_name}, tables={len(all_schema_tables)}," f"values={len(all_value_tables)}"
    )
    sql_connector = db_manager.get_conn(agent_config.current_namespace, database_name)
    if table_type == "table" or table_type == "full":
        # Get all tables with DDL
        tables = sql_connector.get_tables_with_ddl(schema_name=schema_name)
        for table in tables:
            if not table.get("database_name"):
                table["database_name"] = database_name

        logger.info(f"Found {len(tables)} tables")
        store_tables(
            table_lineage_store,
            database_name,
            all_schema_tables,
            all_value_tables,
            all_schema_metadata if prune_missing else None,
            tables,
            "table",
            sql_connector,
            build_mode=build_mode,
            prune_missing=prune_missing,
            llm_fallback=llm_fallback,
            llm_model=llm_model,
            extract_statistics=extract_statistics,
            extract_relationships=extract_relationships,
            llm_enum_extraction=llm_enum_extraction,
        )

    if (table_type == "view" or table_type == "full") and hasattr(sql_connector, "get_views_with_ddl"):
        views = sql_connector.get_views_with_ddl(schema_name=database_name)
        store_tables(
            table_lineage_store,
            database_name,
            all_schema_tables,
            all_value_tables,
            all_schema_metadata if prune_missing else None,
            views,
            "view",
            sql_connector,
            build_mode=build_mode,
            prune_missing=prune_missing,
            llm_fallback=llm_fallback,
            llm_model=llm_model,
            extract_statistics=extract_statistics,
            extract_relationships=extract_relationships,
            llm_enum_extraction=llm_enum_extraction,
        )


def init_mysql_schema(
    table_lineage_store: SchemaWithValueRAG,
    agent_config: AgentConfig,
    db_config: DbConfig,
    db_manager: DBManager,
    database_name: str = "",
    table_type: TABLE_TYPE = "table",
    build_mode: str = "overwrite",
    prune_missing: bool = False,
    llm_fallback: bool = False,
    llm_model: Optional[LLMBaseModel] = None,
    extract_statistics: bool = False,
    extract_relationships: bool = True,
    llm_enum_extraction: bool = False,
):
    """Initialize schema from MySQL database."""
    database_name = database_name or getattr(db_config, "database", "")

    sql_connector = db_manager.get_conn(agent_config.current_namespace)

    all_schema_tables, all_value_tables, all_schema_metadata = exists_table_value(
        storage=table_lineage_store,
        database_name=database_name,
        catalog_name="",
        schema_name="",
        table_type=table_type,
        build_mode=build_mode,
        return_metadata=True,
    )

    logger.info(
        f"Exists data from LanceDB database={database_name}, tables={len(all_schema_tables)}, "
        f"values={len(all_value_tables)}"
    )
    if table_type in ("full", "table"):
        # Get all tables with DDL
        tables = sql_connector.get_tables_with_ddl(database_name=database_name)
        logger.info(f"Found {len(tables)} tables from database {database_name}")
        store_tables(
            table_lineage_store,
            database_name,
            all_schema_tables,
            all_value_tables,
            all_schema_metadata if prune_missing else None,
            tables,
            "table",
            sql_connector,
            build_mode=build_mode,
            prune_missing=prune_missing,
            llm_fallback=llm_fallback,
            llm_model=llm_model,
            extract_statistics=extract_statistics,
            extract_relationships=extract_relationships,
            llm_enum_extraction=llm_enum_extraction,
        )

    if table_type in ("full", "view"):
        views = sql_connector.get_views_with_ddl(database_name=database_name)

        store_tables(
            table_lineage_store,
            database_name,
            all_schema_tables,
            all_value_tables,
            all_schema_metadata if prune_missing else None,
            views,
            "view",
            sql_connector,
            build_mode=build_mode,
            prune_missing=prune_missing,
            llm_fallback=llm_fallback,
            llm_model=llm_model,
            extract_statistics=extract_statistics,
            extract_relationships=extract_relationships,
            llm_enum_extraction=llm_enum_extraction,
        )


def init_starrocks_schema(
    table_lineage_store: SchemaWithValueRAG,
    agent_config: AgentConfig,
    db_config: DbConfig,
    db_manager: DBManager,
    catalog_name: str = "",
    database_name: str = "",
    table_type: TABLE_TYPE = "table",
    build_mode: str = "overwrite",
    prune_missing: bool = False,
    llm_fallback: bool = False,
    llm_model: Optional[LLMBaseModel] = None,
    extract_statistics: bool = False,
    extract_relationships: bool = True,
    llm_enum_extraction: bool = False,
):
    """Initialize schema from StarRocks database."""
    sql_connector = db_manager.get_conn(agent_config.current_namespace)
    catalog_name = catalog_name or getattr(db_config, "catalog", "") or sql_connector.catalog_name
    database_name = database_name or getattr(db_config, "database", "") or sql_connector.database_name

    all_schema_tables, all_value_tables, all_schema_metadata = exists_table_value(
        table_lineage_store,
        database_name=database_name,
        catalog_name=catalog_name,
        schema_name="",
        table_type=table_type,
        build_mode=build_mode,
        return_metadata=True,
    )

    logger.info(
        f"Exists data from LanceDB {catalog_name}.{database_name}, tables={len(all_schema_tables)}, "
        f"values={len(all_value_tables)}"
    )
    if table_type in ("full", "table"):
        # Get all tables with DDL
        tables = sql_connector.get_tables_with_ddl(catalog_name=catalog_name, database_name=database_name)
        logger.info(f"Found {len(tables)} tables from {database_name}")
        store_tables(
            table_lineage_store,
            database_name,
            all_schema_tables,
            all_value_tables,
            all_schema_metadata if prune_missing else None,
            tables,
            "table",
            sql_connector,
            build_mode=build_mode,
            prune_missing=prune_missing,
            llm_fallback=llm_fallback,
            llm_model=llm_model,
            extract_statistics=extract_statistics,
            extract_relationships=extract_relationships,
            llm_enum_extraction=llm_enum_extraction,
        )

    if table_type in ("full", "view"):
        views = sql_connector.get_views_with_ddl(catalog_name=catalog_name, database_name=database_name)
        store_tables(
            table_lineage_store,
            database_name,
            all_schema_tables,
            all_value_tables,
            all_schema_metadata if prune_missing else None,
            views,
            "view",
            sql_connector,
            build_mode=build_mode,
            prune_missing=prune_missing,
            llm_fallback=llm_fallback,
            llm_model=llm_model,
            extract_statistics=extract_statistics,
            extract_relationships=extract_relationships,
            llm_enum_extraction=llm_enum_extraction,
        )
    if table_type in ("full", "view"):
        materialized_views = sql_connector.get_materialized_views_with_ddl(
            catalog_name=catalog_name, database_name=database_name
        )
        store_tables(
            table_lineage_store,
            database_name,
            all_schema_tables,
            all_value_tables,
            all_schema_metadata if prune_missing else None,
            materialized_views,
            "mv",
            sql_connector,
            build_mode=build_mode,
            prune_missing=prune_missing,
            llm_fallback=llm_fallback,
            llm_model=llm_model,
            extract_statistics=extract_statistics,
            extract_relationships=extract_relationships,
            llm_enum_extraction=llm_enum_extraction,
        )


def init_other_three_level_schema(
    table_lineage_store: SchemaWithValueRAG,
    agent_config: AgentConfig,
    db_config: DbConfig,
    db_manager: DBManager,
    database_name: str = "",
    table_type: TABLE_TYPE = "table",
    build_mode: str = "overwrite",
    prune_missing: bool = False,
    llm_fallback: bool = False,
    llm_model: Optional[LLMBaseModel] = None,
    extract_statistics: bool = False,
    extract_relationships: bool = True,
    llm_enum_extraction: bool = False,
):
    """Initialize schema from other three-level databases (Snowflake, etc.)."""
    db_type = db_config.type
    database_name = database_name or getattr(db_config, "database", "")
    schema_name = getattr(db_config, "schema", "")
    catalog_name = getattr(db_config, "catalog", "")

    sql_connector = db_manager.get_conn(agent_config.current_namespace)

    if not database_name and hasattr(sql_connector, "database_name"):
        database_name = getattr(sql_connector, "database_name", "")

    if db_type == DBType.STARROCKS:
        if hasattr(sql_connector, "default_catalog"):
            catalog_name = catalog_name or sql_connector.default_catalog()
        elif hasattr(sql_connector, "catalog_name"):
            catalog_name = catalog_name or getattr(sql_connector, "catalog_name", "")
        schema_name = ""
    elif db_type == DBType.SNOWFLAKE:
        catalog_name = ""
        if not schema_name and hasattr(sql_connector, "schema_name"):
            schema_name = getattr(sql_connector, "schema_name", "")
    else:
        if hasattr(sql_connector, "default_catalog"):
            catalog_name = catalog_name or sql_connector.default_catalog()
        elif hasattr(sql_connector, "catalog_name"):
            catalog_name = catalog_name or getattr(sql_connector, "catalog_name", "")
        if not schema_name and hasattr(sql_connector, "schema_name"):
            schema_name = getattr(sql_connector, "schema_name", "")

    all_schema_tables, all_value_tables, all_schema_metadata = exists_table_value(
        table_lineage_store,
        database_name=database_name,
        catalog_name=catalog_name,
        schema_name=schema_name,
        table_type=table_type,
        build_mode=build_mode,
        return_metadata=True,
    )

    logger.info(
        f"Exists data from LanceDB {catalog_name or '[no catalog]'}.{database_name}, tables={len(all_schema_tables)}, "
        f"values={len(all_value_tables)}"
    )
    if table_type == "table" or table_type == "full":
        # Get all tables with DDL
        if hasattr(sql_connector, "get_tables_with_ddl"):
            tables = sql_connector.get_tables_with_ddl(
                catalog_name=catalog_name, database_name=database_name, schema_name=schema_name
            )
        else:
            # Fallback: get table names and generate basic schema
            table_names = sql_connector.get_tables(database_name=database_name, schema_name=schema_name)
            tables = []
            for table_name in table_names:
                tables.append(
                    {
                        "identifier": sql_connector.identifier(
                            catalog_name=catalog_name,
                            database_name=database_name,
                            schema_name=schema_name,
                            table_name=table_name,
                        ),
                        "catalog_name": catalog_name,
                        "database_name": database_name,
                        "schema_name": schema_name,
                        "table_name": table_name,
                        "definition": f"-- Table: {table_name} (DDL not available)",
                    }
                )
        for table in tables:
            if not table.get("catalog_name"):
                table["catalog_name"] = catalog_name
            if not table.get("database_name"):
                table["database_name"] = database_name
            if db_type == DBType.STARROCKS:
                table["schema_name"] = ""
            elif not table.get("schema_name"):
                table["schema_name"] = schema_name
        logger.info(f"Found {len(tables)} tables from {database_name}")
        store_tables(
            table_lineage_store,
            database_name,
            all_schema_tables,
            all_value_tables,
            all_schema_metadata if prune_missing else None,
            tables,
            "table",
            sql_connector,
            build_mode=build_mode,
            prune_missing=prune_missing,
            llm_fallback=llm_fallback,
            llm_model=llm_model,
            extract_statistics=extract_statistics,
            extract_relationships=extract_relationships,
            llm_enum_extraction=llm_enum_extraction,
        )

    if (table_type == "view" or table_type == "full") and hasattr(sql_connector, "get_views_with_ddl"):
        views = sql_connector.get_views_with_ddl(
            catalog_name=catalog_name, database_name=database_name, schema_name=schema_name
        )
        for view in views:
            if not view.get("catalog_name"):
                view["catalog_name"] = catalog_name
            if not view.get("database_name"):
                view["database_name"] = database_name
            if db_type == DBType.STARROCKS:
                view["schema_name"] = ""
            elif not view.get("schema_name"):
                view["schema_name"] = schema_name
        store_tables(
            table_lineage_store,
            database_name,
            all_schema_tables,
            all_value_tables,
            all_schema_metadata if prune_missing else None,
            views,
            "view",
            sql_connector,
            build_mode=build_mode,
            prune_missing=prune_missing,
            llm_fallback=llm_fallback,
            llm_model=llm_model,
            extract_statistics=extract_statistics,
            extract_relationships=extract_relationships,
            llm_enum_extraction=llm_enum_extraction,
        )
    if (table_type == "mv" or table_type == "full") and hasattr(sql_connector, "get_materialized_views_with_ddl"):
        materialized_views = sql_connector.get_materialized_views_with_ddl(
            catalog_name=catalog_name, database_name=database_name, schema_name=schema_name
        )
        for mv in materialized_views:
            if not mv.get("catalog_name"):
                mv["catalog_name"] = catalog_name
            if not mv.get("database_name"):
                mv["database_name"] = database_name
            if db_type == DBType.STARROCKS:
                mv["schema_name"] = ""
            elif not mv.get("schema_name"):
                mv["schema_name"] = schema_name
        store_tables(
            table_lineage_store,
            database_name,
            all_schema_tables,
            all_value_tables,
            all_schema_metadata if prune_missing else None,
            materialized_views,
            "mv",
            sql_connector,
            build_mode=build_mode,
            prune_missing=prune_missing,
            llm_fallback=llm_fallback,
            llm_model=llm_model,
            extract_statistics=extract_statistics,
            extract_relationships=extract_relationships,
            llm_enum_extraction=llm_enum_extraction,
        )


__all__ = [
    "init_sqlite_schema",
    "init_duckdb_schema",
    "init_mysql_schema",
    "init_starrocks_schema",
    "init_other_three_level_schema",
]
