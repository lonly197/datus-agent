# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import json
from typing import Any, Dict, List, Optional, Set

from datus.configuration.agent_config import AgentConfig, DbConfig
from datus.models.base import LLMBaseModel
from datus.schemas.base import TABLE_TYPE
from datus.storage.schema_metadata.store import SchemaWithValueRAG
from datus.tools.db_tools.base import BaseSqlConnector
from datus.tools.db_tools.db_manager import DBManager
from datus.utils.constants import DBType
from datus.utils.loggings import get_logger
from datus.utils.sql_utils import (
    extract_enhanced_metadata_from_ddl,
    extract_enum_values_from_comment,
    parse_dialect,
    validate_comment,
    validate_table_name,
)

from .init_utils import exists_table_value

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
                )
            else:
                logger.warning(f"Unsupported database type {db_config.type} for multi-database configuration")
    # Create indices after initialization
    table_lineage_store.after_init()
    logger.info("Local schema initialization completed")


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
):
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
):
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
):
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
):
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
):
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
        )


def store_tables(
    table_lineage_store: SchemaWithValueRAG,
    database_name: str,
    exists_tables: Dict[str, str],
    exists_values: Set[str],
    exists_metadata: Optional[Dict[str, Dict[str, str]]],
    tables: List[Dict[str, str]],
    table_type: TABLE_TYPE,
    connector: BaseSqlConnector,
    build_mode: str = "overwrite",
    prune_missing: bool = False,
    llm_fallback: bool = False,
    llm_model: Optional[LLMBaseModel] = None,
):
    """
    Store tables to the table_lineage_store.
    params:
        exists_tables: {full_name: schema_text}
        return the new tables.
    """
    if not tables and not (prune_missing and build_mode == "append"):
        logger.info(f"No schemas of {table_type} to store for {database_name}")
        return
    new_tables: List[Dict[str, Any]] = []
    new_values: List[Dict[str, Any]] = []
    incoming_identifiers: Set[str] = set()
    for table in tables:
        if not table.get("database_name"):
            table["database_name"] = database_name
        if not table.get("identifier"):
            table["identifier"] = connector.identifier(
                catalog_name=table["catalog_name"],
                database_name=table["database_name"],
                schema_name=table["schema_name"],
                table_name=table["table_name"],
            )

        # Fix truncated DDL before storing
        table["definition"] = _fix_truncated_ddl(table["definition"])
        definition = table.get("definition", "")
        if definition:
            try:
                parsed_metadata = extract_enhanced_metadata_from_ddl(
                    definition, dialect=parse_dialect(getattr(connector, "dialect", ""))
                )
                if (
                    llm_fallback
                    and llm_model
                    and "CREATE TABLE" in definition.upper()
                    and (not parsed_metadata.get("columns") or not parsed_metadata.get("table", {}).get("name"))
                ):
                    llm_metadata = _llm_fallback_parse_ddl(definition, llm_model)
                    if llm_metadata:
                        parsed_metadata = llm_metadata
                        logger.info(f"LLM fallback parsed DDL for table: {table.get('table_name', '')}")
                table_comment = parsed_metadata["table"].get("comment", "") or ""
                column_comments = {
                    col["name"]: col.get("comment", "")
                    for col in parsed_metadata.get("columns", [])
                    if col.get("comment")
                }
                column_enums: Dict[str, List[Dict[str, str]]] = {}
                for col_name, col_comment in column_comments.items():
                    enum_pairs = extract_enum_values_from_comment(col_comment)
                    if enum_pairs:
                        column_enums[col_name] = [
                            {"value": code, "label": label} for code, label in enum_pairs
                        ]
                if table_comment:
                    table["table_comment"] = table_comment
                if column_comments:
                    table["column_comments"] = json.dumps(column_comments, ensure_ascii=False)
                if column_enums:
                    table["column_enums"] = json.dumps(column_enums, ensure_ascii=False)
                if table_comment or column_comments:
                    table["definition"] = table_lineage_store.schema_store._enhance_definition_with_comments(
                        definition=definition,
                        table_comment=table_comment,
                        column_comments=column_comments,
                    )
            except Exception as e:
                logger.debug(f"Failed to parse DDL comments for {table.get('table_name', '')}: {e}")

        identifier = table["identifier"]
        incoming_identifiers.add(identifier)
        if identifier not in exists_tables:
            logger.debug(f"Add {table_type} {identifier}")
            new_tables.append(table)
            if identifier in exists_values:
                continue
            _fill_sample_rows(new_values=new_values, identifier=identifier, table_data=table, connector=connector)

        elif exists_tables[identifier] != table["definition"]:
            # update table and value
            logger.debug(f"Update {table_type} {identifier}")
            table_lineage_store.remove_data(
                catalog_name=table["catalog_name"],
                database_name=table["database_name"],
                schema_name=table["schema_name"],
                table_name=table["table_name"],
                table_type=table_type,
            )
            new_tables.append(table)

            _fill_sample_rows(new_values=new_values, identifier=identifier, table_data=table, connector=connector)

        elif identifier not in exists_values:
            logger.debug(f"Just add sample rows for {identifier}")

            _fill_sample_rows(new_values=new_values, identifier=identifier, table_data=table, connector=connector)

    if prune_missing and build_mode == "append" and exists_metadata is not None:
        filtered_existing = {
            identifier: meta
            for identifier, meta in exists_metadata.items()
            if meta.get("table_type") == table_type
        }
        missing_identifiers = set(filtered_existing.keys()) - incoming_identifiers
        if missing_identifiers:
            for missing_identifier in missing_identifiers:
                missing_meta = filtered_existing.get(missing_identifier, {})
                table_lineage_store.remove_data(
                    catalog_name=missing_meta.get("catalog_name", ""),
                    database_name=missing_meta.get("database_name", ""),
                    schema_name=missing_meta.get("schema_name", ""),
                    table_name=missing_meta.get("table_name", ""),
                    table_type=missing_meta.get("table_type", table_type),
                )
            logger.info(
                f"Pruned {len(missing_identifiers)} {table_type}(s) missing from source for {database_name}"
            )

    if new_tables or new_values:
        for item in new_values:
            item["table_type"] = table_type
        table_lineage_store.store_batch(new_tables, new_values)
        logger.info(f"Stored {len(new_tables)} {table_type}s and {len(new_values)} values for {database_name}")
    else:
        logger.info(f"No new {table_type}s or values to store for {database_name}")


def _fix_truncated_ddl(ddl: str) -> str:
    """
    Fix truncated DDL statements by detecting common truncation patterns and attempting completion.

    This function handles DDL that may have been truncated during retrieval from the database,
    such as when SHOW CREATE TABLE returns incomplete results due to length limits.

    Args:
        ddl: Potentially truncated DDL statement

    Returns:
        Fixed DDL statement or original if not recognized as truncated
    """
    if not ddl or not isinstance(ddl, str):
        return ddl

    # Check if DDL appears to be truncated
    ddl_upper = ddl.upper().strip()

    stripped_ddl = ddl.rstrip()
    # Patterns that indicate truncation
    truncation_indicators = [
        stripped_ddl.endswith(','),  # Ends with comma (incomplete column list)
        not stripped_ddl.endswith(')'),  # Missing closing paren
    ]

    # Check if missing closing paren for CREATE TABLE
    open_parens = ddl.count('(')
    close_parens = ddl.count(')')
    missing_closing_paren = open_parens > close_parens

    # If any truncation indicators or missing closing paren, try to fix
    if sum(truncation_indicators) >= 1 or missing_closing_paren:
        logger.debug(f"Detected potentially truncated DDL (indicators: {sum(truncation_indicators)}, missing paren: {missing_closing_paren})")

        # Try to complete the basic structure
        fixed_ddl = ddl

        # Remove trailing comma if present
        if fixed_ddl.rstrip().endswith(','):
            # Remove the trailing comma from the end of the DDL
            fixed_ddl = fixed_ddl.rstrip()[:-1].rstrip()  # Remove comma and any trailing whitespace

        stripped_fixed = fixed_ddl.rstrip()
        if missing_closing_paren or not stripped_fixed.endswith(')'):
            if stripped_fixed.endswith(';'):
                fixed_ddl = stripped_fixed.rstrip(';')
            fixed_ddl += '\n)'

        fixed_upper = fixed_ddl.upper()
        starrocks_indicators = [
            'DUPLICATE KEY' in fixed_upper,
            'AGGREGATE KEY' in fixed_upper,
            'UNIQUE KEY' in fixed_upper,
            'PRIMARY KEY' in fixed_upper,
            'DISTRIBUTED BY' in fixed_upper,
            'PARTITION BY' in fixed_upper,
            'PROPERTIES' in fixed_upper,
        ]

        # Add basic StarRocks table structure if missing
        if 'ENGINE=' not in fixed_upper and any(starrocks_indicators):
            if not fixed_ddl.rstrip().endswith(';'):
                fixed_ddl += ' ENGINE=OLAP;'
            else:
                fixed_ddl = fixed_ddl.rstrip(';') + ' ENGINE=OLAP;'
        elif not fixed_ddl.rstrip().endswith(';'):
            fixed_ddl += ';'

        # Clean up
        fixed_ddl = fixed_ddl.strip()

        if fixed_ddl != ddl:
            logger.info(f"Fixed truncated DDL for table (length: {len(ddl)} -> {len(fixed_ddl)})")
            return fixed_ddl

    return ddl


def _llm_fallback_parse_ddl(ddl: str, llm_model: Optional[LLMBaseModel]) -> Optional[Dict[str, Any]]:
    if not llm_model or not ddl or not isinstance(ddl, str):
        return None

    prompt = (
        "You are a SQL DDL parser. Extract metadata from the CREATE TABLE statement.\n"
        "Return ONLY a JSON object with this schema:\n"
        "{\n"
        '  \"table\": {\"name\": \"\", \"comment\": \"\"},\n'
        '  \"columns\": [{\"name\": \"\", \"type\": \"\", \"comment\": \"\", \"nullable\": true}],\n'
        '  \"primary_keys\": [],\n'
        '  \"foreign_keys\": [],\n'
        '  \"indexes\": []\n'
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

    if not columns:
        return None

    return {
        "table": {"name": table_name, "comment": table_comment},
        "columns": columns,
        "primary_keys": response.get("primary_keys", []) if isinstance(response.get("primary_keys"), list) else [],
        "foreign_keys": response.get("foreign_keys", []) if isinstance(response.get("foreign_keys"), list) else [],
        "indexes": response.get("indexes", []) if isinstance(response.get("indexes"), list) else [],
    }


def _fill_sample_rows(
    new_values: List[Dict[str, Any]], identifier: str, table_data: Dict[str, Any], connector: BaseSqlConnector
):
    sample_rows = connector.get_sample_rows(
        tables=[table_data["table_name"]],
        top_n=5,
        catalog_name=table_data["catalog_name"],
        database_name=table_data["database_name"],
        schema_name=table_data["schema_name"],
    )
    if sample_rows:
        for row in sample_rows:
            if not row.get("identifier"):
                row["identifier"] = identifier
        new_values.extend(sample_rows)


def main():
    """CLI entry point for schema import.

    Usage:
        python -m datus.storage.schema_metadata.local_init --config=<config_path> --namespace=<namespace>
    """
    import argparse
    import sys

    from datus.configuration.agent_config_loader import load_agent_config
    from datus.tools.db_tools.db_manager import get_db_manager

    parser = argparse.ArgumentParser(description="Import schema metadata from database into LanceDB")
    parser.add_argument("--config", required=True, help="Path to agent configuration file")
    parser.add_argument("--namespace", help="Namespace for the database")
    parser.add_argument("--database", help="Database name (optional, uses all databases if not specified)")
    parser.add_argument("--catalog", default="", help="Catalog name (for StarRocks/Snowflake)")
    parser.add_argument("--table-type", default="full", choices=["table", "view", "mv", "full"], help="Type of tables to import")
    parser.add_argument("--build-mode", default="overwrite", choices=["overwrite", "append"], help="Build mode: overwrite or append")
    parser.add_argument("--prune-missing", action="store_true", help="Delete tables not present in source when build-mode=append")

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
    from datus.storage.schema_metadata import SchemaWithValueRAG

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
        logger.info("✅ SCHEMA IMPORT COMPLETE")
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
