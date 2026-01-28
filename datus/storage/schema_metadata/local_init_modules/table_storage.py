# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Table storage functionality for schema initialization.

This module handles the core logic for storing table metadata,
including DDL parsing, relationship inference, and sample data collection.
"""

import json
import time
from typing import Any, Dict, List, Optional, Set

from datus.models.base import LLMBaseModel
from datus.schemas.base import TABLE_TYPE
from datus.storage.schema_metadata.store import SchemaWithValueRAG
from datus.configuration.business_term_config import infer_business_tags
from datus.tools.db_tools.base import BaseSqlConnector
from datus.tools.db_tools.metadata_extractor import get_metadata_extractor
from datus.utils.constants import DBType
from datus.utils.loggings import get_logger
from datus.utils.sql_utils import extract_metadata_from_ddl_regex_only

from .enum_extraction import _get_enum_extractor, _extract_enums
from .utils import (
    _normalize_dialect,
    _infer_relationships_from_names,
    _build_table_name_map,
    _llm_fallback_parse_ddl,
    _fill_sample_rows,
    extract_enhanced_metadata_from_ddl,
    parse_dialect,
    sanitize_ddl_for_storage,
)

logger = get_logger(__name__)


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
    extract_statistics: bool = False,
    extract_relationships: bool = True,
    llm_enum_extraction: bool = False,  # New: LLM-enhanced enum extraction
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
    metadata_extractor = None
    is_starrocks = _normalize_dialect(getattr(connector, "dialect", "")) == DBType.STARROCKS
    if is_starrocks:
        try:
            metadata_extractor = get_metadata_extractor(connector, connector.dialect)
        except Exception as exc:
            logger.debug(f"Failed to initialize StarRocks metadata extractor: {exc}")
    table_name_map: Dict[str, List[str]] = {}
    if extract_relationships:
        table_name_map = _build_table_name_map([t.get("table_name", "") for t in tables])
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
        if "metadata_version" not in table:
            table["metadata_version"] = 1
        if "last_updated" not in table:
            table["last_updated"] = int(time.time())

        # Fix and clean DDL before storing
        table["definition"] = sanitize_ddl_for_storage(table["definition"])
        definition = table.get("definition", "")
        used_information_schema = False
        skip_ddl_parse = is_starrocks
        row_count = 0
        sample_statistics: Dict[str, Dict[str, Any]] = {}
        if metadata_extractor and hasattr(metadata_extractor, "extract_table_metadata"):
            try:
                starrocks_metadata = metadata_extractor.extract_table_metadata(
                    table.get("table_name", ""),
                    database_name=table.get("database_name", ""),
                )
            except Exception as exc:
                logger.debug(f"Failed to read StarRocks metadata for {table.get('table_name', '')}: {exc}")
                starrocks_metadata = {}
            if starrocks_metadata and starrocks_metadata.get("columns"):
                used_information_schema = True
                table_comment = starrocks_metadata.get("table_comment", "") or ""
                column_comments = starrocks_metadata.get("column_comments", {}) or {}
                column_names = starrocks_metadata.get("column_names", []) or []
                if definition and (not table_comment or not column_comments):
                    regex_metadata = extract_metadata_from_ddl_regex_only(
                        definition,
                        dialect=DBType.STARROCKS,
                        warn_on_invalid=False,
                    )
                    regex_table_comment = regex_metadata.get("table", {}).get("comment", "") or ""
                    if not table_comment and regex_table_comment:
                        table_comment = regex_table_comment
                    regex_column_comments = {
                        col.get("name", ""): col.get("comment", "")
                        for col in regex_metadata.get("columns", [])
                        if col.get("name") and col.get("comment")
                    }
                    for col_name, col_comment in regex_column_comments.items():
                        if col_name not in column_comments:
                            column_comments[col_name] = col_comment
                if not column_names and column_comments:
                    column_names = list(column_comments.keys())
                column_enums: Dict[str, List[Dict[str, str]]] = {}
                llm_extractor = _get_enum_extractor(llm_model, llm_enum_extraction)
                for col_name, col_comment in column_comments.items():
                    enum_values = _extract_enums(col_comment, col_name, llm_extractor)
                    if enum_values:
                        column_enums[col_name] = enum_values
                business_tags = infer_business_tags(table.get("table_name", ""), column_names)
                relationship_metadata: Dict[str, Any] = {}
                if extract_relationships:
                    relationship_source = ""
                    try:
                        relationships = metadata_extractor.detect_relationships(table.get("table_name", ""))
                        if relationships and (relationships.get("foreign_keys") or relationships.get("join_paths")):
                            relationship_metadata = relationships
                            relationship_source = "information_schema"
                    except Exception as exc:
                        logger.debug(f"Failed to detect StarRocks relationships: {exc}")
                    if not relationship_metadata and column_names and table_name_map:
                        inferred = _infer_relationships_from_names(
                            table.get("table_name", ""),
                            column_names,
                            table_name_map,
                        )
                        if inferred:
                            relationship_metadata = inferred
                            relationship_source = "heuristic"
                    if relationship_metadata and relationship_source and "source" not in relationship_metadata:
                        relationship_metadata["source"] = relationship_source
                if extract_statistics:
                    try:
                        row_count = metadata_extractor.extract_row_count(table.get("table_name", ""))
                    except Exception as exc:
                        logger.debug(f"Failed to extract StarRocks row count: {exc}")
                    if row_count > 1000:
                        try:
                            sample_statistics = metadata_extractor.extract_column_statistics(
                                table.get("table_name", "")
                            )
                        except Exception as exc:
                            logger.debug(f"Failed to extract StarRocks column statistics: {exc}")
                if table_comment:
                    table["table_comment"] = table_comment
                if column_comments:
                    table["column_comments"] = json.dumps(column_comments, ensure_ascii=False)
                if column_enums:
                    table["column_enums"] = json.dumps(column_enums, ensure_ascii=False)
                if business_tags:
                    table["business_tags"] = business_tags
                if relationship_metadata:
                    table["relationship_metadata"] = json.dumps(relationship_metadata, ensure_ascii=False)
                if row_count:
                    table["row_count"] = row_count
                if sample_statistics:
                    table["sample_statistics"] = json.dumps(sample_statistics, ensure_ascii=False)
                if table_comment or column_comments:
                    table["definition"] = table_lineage_store.schema_store._enhance_definition_with_comments(
                        definition=definition,
                        table_comment=table_comment,
                        column_comments=column_comments,
                    )
        if is_starrocks and definition and not used_information_schema:
            regex_metadata = extract_metadata_from_ddl_regex_only(
                definition,
                dialect=DBType.STARROCKS,
                warn_on_invalid=False,
            )
            table_comment = regex_metadata.get("table", {}).get("comment", "") or ""
            column_comments = {
                col.get("name", ""): col.get("comment", "")
                for col in regex_metadata.get("columns", [])
                if col.get("name") and col.get("comment")
            }
            column_names = [col.get("name", "") for col in regex_metadata.get("columns", []) if col.get("name")]
            column_enums: Dict[str, List[Dict[str, str]]] = {}
            llm_extractor = _get_enum_extractor(llm_model, llm_enum_extraction)
            for col_name, col_comment in column_comments.items():
                enum_values = _extract_enums(col_comment, col_name, llm_extractor)
                if enum_values:
                    column_enums[col_name] = enum_values
            business_tags = infer_business_tags(table.get("table_name", ""), column_names)
            relationship_metadata: Dict[str, Any] = {}
            if extract_relationships and column_names and table_name_map:
                inferred = _infer_relationships_from_names(
                    table.get("table_name", ""),
                    column_names,
                    table_name_map,
                )
                if inferred:
                    relationship_metadata = inferred
                    if "source" not in relationship_metadata:
                        relationship_metadata["source"] = "heuristic"
            if table_comment:
                table["table_comment"] = table_comment
            if column_comments:
                table["column_comments"] = json.dumps(column_comments, ensure_ascii=False)
            if column_enums:
                table["column_enums"] = json.dumps(column_enums, ensure_ascii=False)
            if business_tags:
                table["business_tags"] = business_tags
            if relationship_metadata:
                table["relationship_metadata"] = json.dumps(relationship_metadata, ensure_ascii=False)
            if table_comment or column_comments:
                table["definition"] = table_lineage_store.schema_store._enhance_definition_with_comments(
                    definition=definition,
                    table_comment=table_comment,
                    column_comments=column_comments,
                )
        if extract_statistics and metadata_extractor and row_count == 0:
            try:
                row_count = metadata_extractor.extract_row_count(table.get("table_name", ""))
            except Exception as exc:
                logger.debug(f"Failed to extract StarRocks row count: {exc}")
            if row_count > 1000 and not sample_statistics:
                try:
                    sample_statistics = metadata_extractor.extract_column_statistics(table.get("table_name", ""))
                except Exception as exc:
                    logger.debug(f"Failed to extract StarRocks column statistics: {exc}")
            if row_count:
                table["row_count"] = row_count
            if sample_statistics:
                table["sample_statistics"] = json.dumps(sample_statistics, ensure_ascii=False)

        if definition and not used_information_schema and not skip_ddl_parse:
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
                llm_extractor = _get_enum_extractor(llm_model, llm_enum_extraction)
                for col_name, col_comment in column_comments.items():
                    enum_values = _extract_enums(col_comment, col_name, llm_extractor)
                    if enum_values:
                        column_enums[col_name] = enum_values
                column_names = [col["name"] for col in parsed_metadata.get("columns", [])]
                business_tags = infer_business_tags(table.get("table_name", ""), column_names)
                foreign_keys = parsed_metadata.get("foreign_keys", [])
                relationship_metadata: Dict[str, Any] = {}
                if foreign_keys:
                    relationship_metadata = {
                        "foreign_keys": foreign_keys,
                        "join_paths": [
                            f"{fk.get('from_column', '')} -> {fk.get('to_table', '')}.{fk.get('to_column', '')}"
                            for fk in foreign_keys
                            if fk.get("from_column") and fk.get("to_table") and fk.get("to_column")
                        ],
                    }
                if table_comment:
                    table["table_comment"] = table_comment
                if column_comments:
                    table["column_comments"] = json.dumps(column_comments, ensure_ascii=False)
                if column_enums:
                    table["column_enums"] = json.dumps(column_enums, ensure_ascii=False)
                if business_tags:
                    table["business_tags"] = business_tags
                if relationship_metadata:
                    table["relationship_metadata"] = json.dumps(relationship_metadata, ensure_ascii=False)
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


__all__ = [
    "store_tables",
]
