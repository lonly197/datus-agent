# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details

"""
SchemaDiscoveryNode implementation for discovering relevant schema and tables.

This file is a backward-compatible wrapper for the refactored schema_discovery package.
All functionality has been split into focused modules for better maintainability.

For new code, import from the module package directly:
    from datus.agent.node.schema_discovery import SchemaDiscoveryNode

Legacy imports (still supported):
    from datus.agent.node.schema_discovery_node import SchemaDiscoveryNode
"""

# Re-export all public classes from the new module package
from datus.agent.node.schema_discovery import (
    SchemaDiscoveryNode,
)

__all__ = [
    "SchemaDiscoveryNode",
]
