#!/usr/bin/env python3
# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datus.configuration.agent_config_loader import load_agent_config
from datus.storage.embedding_models import get_db_embedding_model
from datus.storage.schema_metadata.store import SchemaStorage


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild FTS index for schema metadata.")
    parser.add_argument("--config", required=True, help="Path to agent configuration file")
    parser.add_argument("--namespace", help="Namespace to target (uses base path if omitted)")
    parser.add_argument("--db-path", help="Override LanceDB path (optional)")
    return parser.parse_args()


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
    db_path = _build_db_path(args)
    storage = SchemaStorage(db_path=db_path, embedding_model=get_db_embedding_model())
    storage.create_indices()
    print(f"FTS index rebuilt for schema_metadata at {db_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
