# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details

"""
Schema discovery node modules.

This package contains the refactored components of the original schema_discovery_node.py,
split into focused modules for better maintainability.

Modules:
    - core: SchemaDiscoveryNode class with main execution logic
    - utils: Helper functions for system resources, Chinese text processing
    - discovery_engine: Table discovery orchestrator and candidate finalization
    - search_strategies: Semantic, keyword, and context-based search strategies
    - llm_discovery: LLM-driven table discovery and schema matching
    - schema_loader: Schema loading, metadata repair, and DDL fallbacks
    - knowledge_enhancement: External knowledge enhancement
"""

# Public API - re-export the main class
from datus.agent.node.schema_discovery.core import SchemaDiscoveryNode

# Optional: export main functions for testing
from datus.agent.node.schema_discovery.discovery_engine import (
    discover_candidate_tables,
    finalize_candidates,
)
from datus.agent.node.schema_discovery.llm_discovery import llm_based_table_discovery
from datus.agent.node.schema_discovery.schema_loader import (
    ddl_fallback_and_retry,
    get_all_database_tables,
    load_table_schemas,
)
from datus.agent.node.schema_discovery.search_strategies import (
    context_based_discovery,
    keyword_table_discovery,
    semantic_table_discovery,
)

__all__ = [
    "SchemaDiscoveryNode",
    # Optional exports for testing
    "discover_candidate_tables",
    "finalize_candidates",
    "llm_based_table_discovery",
    "get_all_database_tables",
    "load_table_schemas",
    "context_based_discovery",
    "keyword_table_discovery",
    "semantic_table_discovery",
]
