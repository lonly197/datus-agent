#!/usr/bin/env python3
import argparse
import json
import random
from typing import Any, Dict, List

from datus.configuration.agent_config_loader import load_agent_config
from datus.storage.schema_metadata import SchemaWithValueRAG
from datus.tools.db_tools.db_manager import get_db_manager
from datus.utils.sql_utils import (
    ddl_has_missing_commas,
    is_likely_truncated_ddl,
    sanitize_ddl_for_storage,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose stored DDL quality in schema metadata.")
    parser.add_argument("--config", default="conf/agent.yml", help="Path to agent.yml")
    parser.add_argument("--namespace", default="", help="Namespace name (optional)")
    parser.add_argument("--catalog", default="", help="Catalog filter (optional)")
    parser.add_argument("--database", default="", help="Database filter (optional)")
    parser.add_argument("--schema", default="", help="Schema filter (optional)")
    parser.add_argument("--limit", type=int, default=5, help="Number of tables to sample")
    parser.add_argument("--random", action="store_true", help="Randomly sample tables")
    parser.add_argument("--compare-db", action="store_true", help="Fetch DDL from database for comparison")
    parser.add_argument("--output", default="", help="Write JSON output to file")
    return parser.parse_args()


def _fetch_db_ddl(
    namespace: str,
    table_name: str,
    catalog_name: str,
    database_name: str,
    schema_name: str,
    agent_config: Any,
) -> Dict[str, Any]:
    db_manager = get_db_manager(agent_config.namespaces)
    connector = db_manager.get_conn(namespace, database_name)
    if not connector:
        return {"error": f"No connector for database {database_name}"}
    try:
        results = connector.get_tables_with_ddl(
            catalog_name=catalog_name,
            database_name=database_name,
            schema_name=schema_name,
            tables=[table_name],
        )
        if not results:
            return {"error": "DDL not found"}
        return {"definition": results[0].get("definition", "")}
    except Exception as exc:
        return {"error": str(exc)}


def main() -> None:
    args = _parse_args()
    agent_config = load_agent_config(config=args.config)
    if args.namespace:
        agent_config.current_namespace = args.namespace
    elif not agent_config.current_namespace and agent_config.namespaces:
        agent_config.current_namespace = list(agent_config.namespaces.keys())[0]

    rag = SchemaWithValueRAG(agent_config=agent_config)
    schema_store = rag.schema_store

    schemas = schema_store.search_all(
        catalog_name=args.catalog,
        database_name=args.database,
        schema_name=args.schema,
        table_type="full",
        select_fields=["catalog_name", "database_name", "schema_name", "table_name", "definition"],
    )
    rows: List[Dict[str, Any]] = schemas.to_pylist() if schemas else []
    if not rows:
        print("No schemas found for the given filters.")
        return

    if args.random:
        random.shuffle(rows)
    sample = rows[: max(args.limit, 0)]
    output: List[Dict[str, Any]] = []

    for row in sample:
        table_name = row.get("table_name", "")
        definition = row.get("definition", "") or ""
        catalog_name = row.get("catalog_name", "") or ""
        database_name = row.get("database_name", "") or ""
        schema_name = row.get("schema_name", "") or ""

        missing_commas = ddl_has_missing_commas(definition)
        truncated = is_likely_truncated_ddl(definition)
        sanitized = sanitize_ddl_for_storage(definition)
        sanitized_changed = sanitized != definition

        record: Dict[str, Any] = {
            "catalog_name": catalog_name,
            "database_name": database_name,
            "schema_name": schema_name,
            "table_name": table_name,
            "missing_commas": missing_commas,
            "truncated": truncated,
            "sanitized_changed": sanitized_changed,
            "definition_preview": definition[:500],
        }

        if args.compare_db:
            record["database_ddl"] = _fetch_db_ddl(
                agent_config.current_namespace,
                table_name,
                catalog_name,
                database_name,
                schema_name,
                agent_config,
            )

        output.append(record)

    print(json.dumps(output, indent=2, ensure_ascii=True))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(output, handle, indent=2, ensure_ascii=True)


if __name__ == "__main__":
    main()
