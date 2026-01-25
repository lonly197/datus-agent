# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datus.configuration.agent_config_loader import load_agent_config
from datus.storage.embedding_models import get_db_embedding_model
from datus.storage.schema_metadata.store import SchemaStorage


def _load_table_rows(storage: SchemaStorage) -> List[Dict[str, Any]]:
    fields = [
        "identifier",
        "database_name",
        "schema_name",
        "table_name",
        "table_comment",
        "column_comments",
        "column_enums",
        "business_tags",
        "row_count",
        "sample_statistics",
        "relationship_metadata",
        "metadata_version",
    ]
    data = storage._search_all(where=None, select_fields=fields)
    return data.to_pylist() if data is not None else []


def _safe_json(value: Any) -> Dict[str, Any]:
    if not value or not isinstance(value, str):
        return {}
    try:
        parsed = json.loads(value)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _non_empty_json(value: Any) -> bool:
    parsed = _safe_json(value)
    return bool(parsed)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check migration coverage for schema metadata")
    parser.add_argument("--config", required=True, help="Path to agent configuration file")
    parser.add_argument("--namespace", help="Namespace to check (uses base path if omitted)")
    parser.add_argument("--db-path", help="Override LanceDB path (optional)")
    parser.add_argument("--top-tags", type=int, default=10, help="Top N business tags to display")
    args = parser.parse_args()

    agent_config = load_agent_config(config=args.config)

    if args.db_path:
        db_path = args.db_path
    elif args.namespace:
        agent_config.current_namespace = args.namespace
        db_path = agent_config.rag_storage_path()
    else:
        db_path = f"{agent_config.rag_base_path}/lancedb"

    storage = SchemaStorage(db_path=db_path, embedding_model=get_db_embedding_model())
    rows = _load_table_rows(storage)

    total = len(rows)
    if total == 0:
        print("No records found in schema_metadata.")
        return 1

    def ratio(count: int) -> str:
        return f"{count}/{total} ({(count / total) * 100:.1f}%)"

    table_comment_count = sum(1 for row in rows if (row.get("table_comment") or "").strip())
    column_comments_count = sum(1 for row in rows if _non_empty_json(row.get("column_comments")))
    column_enums_count = sum(1 for row in rows if _non_empty_json(row.get("column_enums")))
    business_tags_count = sum(1 for row in rows if row.get("business_tags"))
    relationship_count = sum(1 for row in rows if _non_empty_json(row.get("relationship_metadata")))
    row_count_count = sum(1 for row in rows if (row.get("row_count") or 0) > 0)
    stats_count = sum(1 for row in rows if _non_empty_json(row.get("sample_statistics")))
    v1_count = sum(1 for row in rows if row.get("metadata_version") == 1)

    tag_counter = Counter()
    for row in rows:
        tags = row.get("business_tags") or []
        tag_counter.update(tags)

    print("")
    print("MIGRATION COVERAGE REPORT")
    print("=" * 80)
    print(f"DB Path: {db_path}")
    print(f"Total records: {total}")
    print(f"v1 records: {ratio(v1_count)}")
    print("")
    print("Field coverage:")
    print(f"  table_comment:        {ratio(table_comment_count)}")
    print(f"  column_comments:      {ratio(column_comments_count)}")
    print(f"  column_enums:         {ratio(column_enums_count)}")
    print(f"  business_tags:        {ratio(business_tags_count)}")
    print(f"  relationship_metadata:{ratio(relationship_count)}")
    print(f"  row_count:            {ratio(row_count_count)}")
    print(f"  sample_statistics:    {ratio(stats_count)}")
    print("")
    print(f"Top {args.top_tags} business tags:")
    for tag, count in tag_counter.most_common(args.top_tags):
        print(f"  {tag}: {count}")
    print("")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
