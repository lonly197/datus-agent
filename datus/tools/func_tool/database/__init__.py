# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Database function tools for Datus.

This package provides database operations including table listing,
schema retrieval, DDL extraction, and query execution with support
for scoped table access patterns.
"""

from datus.tools.func_tool.database.core import DBFuncTool, db_function_tool_instance, db_function_tools
from datus.tools.func_tool.database.patterns import ScopedTablePattern, TableCoordinate, _pattern_matches

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
