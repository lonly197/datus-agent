#!/usr/bin/env python3
# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import argparse
import json
import sys
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datus.configuration.agent_config_loader import load_agent_config
from datus.storage.embedding_models import get_db_embedding_model
from datus.storage.schema_metadata.store import SchemaStorage


DEFAULT_QUERIES = [
    "线索统计 铂智3X 渠道",
    "转化漏斗 试驾 订单实绩",
    "有效线索到店",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check search_text coverage and FTS recall for schema metadata.")
    parser.add_argument("--config", required=True, help="Path to agent configuration file")
    parser.add_argument("--namespace", help="Namespace to check (uses base path if omitted)")
    parser.add_argument("--db-path", help="Override LanceDB path (optional)")
    parser.add_argument("--database", default="", help="Database filter for FTS search")
    parser.add_argument("--schema", default="", help="Schema filter for FTS search")
    parser.add_argument("--table-type", default="table", choices=["table", "view", "mv", "full"])
    parser.add_argument("--top-n", type=int, default=5, help="Top N FTS results to show")
    parser.add_argument("--queries", default="", help="Comma-separated queries for FTS test")
    parser.add_argument("--queries-file", default="", help="Path to a JSON or txt file with queries")
    parser.add_argument("--show-examples", action="store_true", help="Print example queries and exit")
    return parser.parse_args()


def _load_queries(args: argparse.Namespace) -> List[str]:
    if args.show_examples:
        return []
    if args.queries:
        return [q.strip() for q in args.queries.split(",") if q.strip()]
    if args.queries_file:
        path = Path(args.queries_file)
        if not path.exists():
            raise FileNotFoundError(f"queries file not found: {path}")
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return []
        if path.suffix.lower() == ".json":
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(q).strip() for q in parsed if str(q).strip()]
            raise ValueError("queries JSON must be a list of strings")
        return [line.strip() for line in text.splitlines() if line.strip()]
    return DEFAULT_QUERIES


def _build_db_path(args: argparse.Namespace) -> str:
    agent_config = load_agent_config(config=args.config)
    if args.db_path:
        return args.db_path
    if args.namespace:
        agent_config.current_namespace = args.namespace
        return agent_config.rag_storage_path()
    return f"{agent_config.rag_base_path}/lancedb"


def main() -> int:
    args = _parse_args()
    if args.show_examples:
        print("Example queries:")
        for query in DEFAULT_QUERIES:
            print(f"  - {query}")
        return 0

    db_path = _build_db_path(args)
    storage = SchemaStorage(db_path=db_path, embedding_model=get_db_embedding_model())
    storage._ensure_table_ready()

    fields = storage.table.schema.names
    has_search_text = "search_text" in fields
    print("")
    print("SEARCH_TEXT + FTS CHECK")
    print("=" * 80)
    print(f"DB Path: {db_path}")
    print(f"search_text field: {'yes' if has_search_text else 'no'}")

    if has_search_text:
        rows = storage._search_all(where=None, select_fields=["search_text"])
        data = rows.to_pylist() if rows is not None else []
        total = len(data)
        non_empty = sum(1 for row in data if isinstance(row.get("search_text"), str) and row["search_text"].strip())
        coverage = (non_empty / total) * 100 if total else 0.0
        print(f"search_text coverage: {non_empty}/{total} ({coverage:.1f}%)")
    else:
        print("search_text coverage: N/A (field missing)")

    queries = _load_queries(args)
    if not queries:
        print("")
        print("No queries provided; skipping FTS tests.")
        print("")
        return 0

    print("")
    print("FTS TEST QUERIES")
    print("=" * 80)
    for query in queries:
        print(f"\nQUERY: {query}")
        try:
            results = storage.search_fts(
                query_text=query,
                database_name=args.database,
                schema_name=args.schema,
                table_type=args.table_type,
                top_n=args.top_n,
                select_fields=["table_name", "table_comment", "_score"],
            )
        except Exception as exc:
            print(f"  FTS search failed: {exc}")
            continue

        rows = results.to_pylist() if results is not None else []
        if not rows:
            print("  No results.")
            continue

        for row in rows:
            table_name = row.get("table_name", "")
            table_comment = row.get("table_comment", "")
            score = row.get("_score", "")
            score_repr = f"{score:.4f}" if isinstance(score, (int, float)) else str(score)
            print(f"  - {table_name} | {table_comment} | score={score_repr}")

    print("")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
