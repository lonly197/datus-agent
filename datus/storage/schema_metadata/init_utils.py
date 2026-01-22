# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from typing import Dict, Set

from datus.schemas.base import TABLE_TYPE
from datus.storage.schema_metadata.store import SchemaWithValueRAG
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


def exists_table_value(
    storage: SchemaWithValueRAG,
    database_name: str = "",
    catalog_name: str = "",
    schema_name: str = "",
    table_type: TABLE_TYPE = "table",
    build_mode: str = "overwrite",
    return_metadata: bool = False,
) -> tuple[Dict[str, str], Set[str]] | tuple[Dict[str, str], Set[str], Dict[str, Dict[str, str]]]:
    """
    Get the existing tables and values from the storage.
    Return:
        all_schema_tables: Dict[str,  str]] identifier -> definition
        all_value_tables: Set[str]
    """
    all_schema_tables: Dict[str, str] = {}
    all_value_tables: Set[str] = set()
    all_schema_metadata: Dict[str, Dict[str, str]] = {}
    if build_mode == "overwrite":
        if return_metadata:
            return all_schema_tables, all_value_tables, all_schema_metadata
        return all_schema_tables, all_value_tables

    try:
        select_fields = [
            "identifier",
            "catalog_name",
            "database_name",
            "schema_name",
            "table_name",
            "table_type",
            "definition",
        ]
        schemas = storage.search_all_schemas(
            catalog_name=catalog_name,
            database_name=database_name,
            schema_name=schema_name,
            table_type=table_type,
            select_fields=select_fields,
        )
        if schemas.num_rows > 0:
            identifier_idx = schemas.schema.get_field_index("identifier")
            definition_idx = schemas.schema.get_field_index("definition")
            catalog_idx = schemas.schema.get_field_index("catalog_name")
            database_idx = schemas.schema.get_field_index("database_name")
            schema_idx = schemas.schema.get_field_index("schema_name")
            table_idx = schemas.schema.get_field_index("table_name")
            type_idx = schemas.schema.get_field_index("table_type")

            batch_size = 500
            for i in range(0, schemas.num_rows, batch_size):
                batch = schemas.slice(i, min(batch_size, schemas.num_rows - i))
                identifiers = batch.column(identifier_idx).to_pylist()
                definitions = batch.column(definition_idx).to_pylist()

                all_schema_tables.update(zip(identifiers, definitions))
                if return_metadata:
                    catalogs = batch.column(catalog_idx).to_pylist()
                    databases = batch.column(database_idx).to_pylist()
                    schemas_names = batch.column(schema_idx).to_pylist()
                    table_names = batch.column(table_idx).to_pylist()
                    table_types = batch.column(type_idx).to_pylist()
                    for idx, ident in enumerate(identifiers):
                        all_schema_metadata[ident] = {
                            "catalog_name": catalogs[idx],
                            "database_name": databases[idx],
                            "schema_name": schemas_names[idx],
                            "table_name": table_names[idx],
                            "table_type": table_types[idx],
                        }

        values = storage.search_all_value(
            catalog_name=catalog_name, database_name=database_name, schema_name=schema_name, table_type=table_type
        )
        if values.num_rows > 0:
            identifier_idx = values.schema.get_field_index("identifier")

            batch_size = 500
            for i in range(0, values.num_rows, batch_size):
                batch = values.slice(i, min(batch_size, values.num_rows - i))
                identifiers = batch.column(identifier_idx).to_pylist()

                all_value_tables.update(identifiers)

    except Exception as e:
        raise DatusException(
            ErrorCode.COMMON_UNKNOWN, message=f"Failed to load already existing metadata, reason: {str(e)}"
        ) from e

    if return_metadata:
        return all_schema_tables, all_value_tables, all_schema_metadata
    return all_schema_tables, all_value_tables
