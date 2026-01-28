# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Database table coordinate and pattern matching utilities.

This module provides classes for representing table coordinates and scoped
table patterns with wildcard matching capabilities.
"""

from dataclasses import dataclass
from fnmatch import fnmatchcase
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class TableCoordinate:
    """
    Represents a fully qualified table coordinate.

    Attributes:
        catalog: Catalog name (empty if not applicable)
        database: Database name
        schema: Schema name
        table: Table name
    """

    catalog: str = ""
    database: str = ""
    schema: str = ""
    table: str = ""


@dataclass(frozen=True)
class ScopedTablePattern:
    """
    Immutable pattern for matching table coordinates with wildcards.

    Supports wildcard matching using SQL-like patterns (% or * for any characters).
    Used for scoping database operations to specific tables.

    Attributes:
        raw: Original pattern string provided by user
        catalog: Catalog pattern (empty = wildcard)
        database: Database pattern (empty = wildcard)
        schema: Schema pattern (empty = wildcard)
        table: Table pattern (empty = wildcard)

    Example:
        >>> pattern = ScopedTablePattern("mydb.public.*", catalog="", database="mydb", schema="public", table="*")
        >>> coord = TableCoordinate(catalog="", database="mydb", schema="public", table="users")
        >>> pattern.matches(coord)
        True
    """

    raw: str
    catalog: str = ""
    database: str = ""
    schema: str = ""
    table: str = ""

    def matches(self, coordinate: TableCoordinate) -> bool:
        """
        Check if this pattern matches the given table coordinate.

        Args:
            coordinate: TableCoordinate to check against

        Returns:
            True if all non-wildcard fields match
        """
        return all(
            _pattern_matches(getattr(self, field), getattr(coordinate, field))
            for field in ("catalog", "database", "schema", "table")
        )


def _pattern_matches(pattern: str, value: str) -> bool:
    """
    Match a value against a pattern with wildcard support.

    Empty pattern or "*" / "%" matches any value.
    Uses case-sensitive fnmatch for wildcard matching.

    Args:
        pattern: Pattern string (may contain % or * wildcards)
        value: Value to match against

    Returns:
        True if value matches the pattern

    Examples:
        >>> _pattern_matches("", "anything")
        True
        >>> _pattern_matches("*", "anything")
        True
        >>> _pattern_matches("test*", "test123")
        True
        >>> _pattern_matches("test%", "test123")
        True
        >>> _pattern_matches("user_", "user_1")
        True
    """
    if not pattern or pattern in ("*", "%"):
        return True
    normalized_pattern = pattern.replace("%", "*")
    return fnmatchcase(value or "", normalized_pattern)
