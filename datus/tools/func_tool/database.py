# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Database function tools for Datus.

This module has been split into multiple submodules for better organization:
- patterns.py: Table coordinate and pattern matching utilities
- validation.py: Validation functions for SQL, table conflicts, and partitioning
- db_functions.py: Core database operation methods
- utils.py: Utility functions for query plan analysis and coordinate building
- core.py: Main DBFuncTool class initialization

This file maintains backward compatibility by re-exporting the public API.
"""

# Re-export all public APIs from the database subpackage
from datus.tools.func_tool.database.core import (
    DBFuncTool,
    db_function_tool_instance,
    db_function_tools,
)
from datus.tools.func_tool.database.patterns import (
    ScopedTablePattern,
    TableCoordinate,
    _pattern_matches,
)

__all__ = [
    # Core classes and functions
    "DBFuncTool",
    "db_function_tool_instance",
    "db_function_tools",
    # Pattern matching
    "TableCoordinate",
    "ScopedTablePattern",
    "_pattern_matches",
]
