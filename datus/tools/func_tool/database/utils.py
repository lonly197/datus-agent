# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Database utility functions.

This module provides utility functions for query plan analysis,
coordinate building, scope matching, and other helper operations.
"""

from typing import Any, Dict, List, Optional, Sequence

from datus.tools.func_tool.database.patterns import ScopedTablePattern, TableCoordinate


def normalize_identifier_part(value: Optional[str]) -> str:
    """
    Normalize a database identifier part by stripping quotes and whitespace.

    Args:
        value: Identifier string to normalize

    Returns:
        Normalized identifier without quotes or whitespace

    Examples:
        >>> normalize_identifier_part('"my_table"')
        'my_table'
        >>> normalize_identifier_part('[my_schema]')
        'my_schema'
        >>> normalize_identifier_part('  my_column  ')
        'my_column'
    """
    if value is None:
        return ""
    normalized = str(value).strip()
    if not normalized:
        return ""
    # Strip common quoting characters
    return normalized.strip("`\"'[]")


def build_table_coordinate(
    raw_name: str,
    field_order: Sequence[str],
    default_field_value_func: callable,
    catalog: Optional[str] = "",
    database: Optional[str] = "",
    schema: Optional[str] = "",
) -> TableCoordinate:
    """
    Build a TableCoordinate from a raw table name and optional overrides.

    Args:
        raw_name: Raw table name (may be qualified with catalog.database.schema)
        field_order: Ordered list of field names for the dialect
        default_field_value_func: Function to get default values for fields
        catalog: Optional catalog override
        database: Optional database override
        schema: Optional schema override

    Returns:
        TableCoordinate with all fields populated
    """
    coordinate = TableCoordinate(
        catalog=default_field_value_func("catalog", catalog),
        database=default_field_value_func("database", database),
        schema=default_field_value_func("schema", schema),
        table=normalize_identifier_part(raw_name),
    )
    parts = [normalize_identifier_part(part) for part in raw_name.split(".") if part.strip()]
    if parts:
        coordinate.table = parts[-1]
        idx = len(parts) - 2
        for field in reversed(field_order[:-1]):
            if idx < 0:
                break
            setattr(coordinate, field, parts[idx])
            idx -= 1
    return coordinate


def parse_scope_token(token: str, field_order: Sequence[str]) -> Optional[ScopedTablePattern]:
    """
    Parse a scope token into a ScopedTablePattern.

    Args:
        token: Token string (e.g., "catalog.database.schema.table")
        field_order: Ordered list of field names for the dialect

    Returns:
        ScopedTablePattern if token is valid, None otherwise
    """
    token = (token or "").strip()
    if not token:
        return None
    parts = [normalize_identifier_part(part) for part in token.split(".") if part.strip()]
    if not parts:
        return None
    values: Dict[str, str] = {field: "" for field in field_order}
    for idx, part in enumerate(parts[: len(field_order)]):
        field = field_order[idx]
        values[field] = part
    return ScopedTablePattern(raw=token, **values)


def matches_catalog_database(pattern: ScopedTablePattern, catalog: str, database: str) -> bool:
    """
    Check if a pattern matches given catalog and database.

    Args:
        pattern: ScopedTablePattern to check
        catalog: Catalog value
        database: Database value

    Returns:
        True if pattern matches catalog and database
    """
    from datus.tools.func_tool.database.patterns import _pattern_matches

    if pattern.catalog and not _pattern_matches(pattern.catalog, catalog):
        return False
    if pattern.database and not _pattern_matches(pattern.database, database):
        return False
    return True


def parse_execution_plan(plan_data: Any, dialect: str) -> Dict[str, Any]:
    """
    Parse execution plan data and extract performance insights.

    Args:
        plan_data: Raw execution plan data from database
        dialect: Database dialect

    Returns:
        Structured analysis of the execution plan
    """
    try:
        # Initialize analysis structure
        analysis = {
            "success": True,
            "plan_text": "",
            "estimated_rows": 0,
            "estimated_cost": 0.0,
            "hotspots": [],
            "join_analysis": {"join_count": 0, "join_types": [], "join_order_issues": []},
            "index_usage": {"indexes_used": [], "missing_indexes": [], "index_effectiveness": "unknown"},
            "warnings": [],
        }

        # Convert plan data to string for analysis
        if isinstance(plan_data, list) and plan_data:
            plan_text = "\n".join([str(row) for row in plan_data if row])
        else:
            plan_text = str(plan_data)

        analysis["plan_text"] = plan_text

        # Parse based on dialect
        if dialect in ["starrocks", "mysql", "mariadb"]:
            _parse_mysql_like_plan(plan_text, analysis)
        elif dialect in ["postgresql", "postgres"]:
            _parse_postgres_plan(plan_text, analysis)
        elif dialect in ["duckdb", "sqlite"]:
            _parse_sqlite_plan(plan_text, analysis)
        else:
            # Generic parsing for unknown dialects
            _parse_generic_plan(plan_text, analysis)

        # Generate overall assessment
        _generate_plan_assessment(analysis)

        return analysis

    except Exception as e:
        return {
            "success": False,
            "error": f"Plan parsing failed: {str(e)}",
            "plan_text": str(plan_data) if plan_data else "",
            "hotspots": [],
            "warnings": ["Failed to parse execution plan"],
        }


def _parse_mysql_like_plan(plan_text: str, analysis: Dict[str, Any]) -> None:
    """Parse MySQL-like execution plans (MySQL, StarRocks, MariaDB)."""
    lines = plan_text.strip().split("\n")

    for line in lines:
        line = line.strip().lower()

        # Check for table scans
        if "table scan" in line or "all" in line:
            analysis["hotspots"].append(
                {
                    "reason": "full_table_scan",
                    "node": line,
                    "severity": "high",
                    "recommendation": "Consider adding appropriate indexes for WHERE conditions",
                }
            )

        # Check for expensive joins
        if "join" in line:
            analysis["join_analysis"]["join_count"] += 1
            if "block nested loop" in line:
                analysis["hotspots"].append(
                    {
                        "reason": "expensive_join",
                        "node": line,
                        "severity": "high",
                        "recommendation": "Consider adding indexes on join columns",
                    }
                )
                analysis["join_analysis"]["join_types"].append("block_nested_loop")

        # Check for filesort
        if "filesort" in line:
            analysis["hotspots"].append(
                {
                    "reason": "filesort_operation",
                    "node": line,
                    "severity": "medium",
                    "recommendation": "Consider adding indexes to avoid sorting",
                }
            )


def _parse_postgres_plan(plan_text: str, analysis: Dict[str, Any]) -> None:
    """Parse PostgreSQL execution plans."""
    lines = plan_text.strip().split("\n")

    for line in lines:
        line = line.strip().lower()

        # Check for sequential scans
        if "seq scan" in line:
            analysis["hotspots"].append(
                {
                    "reason": "sequential_scan",
                    "node": line,
                    "severity": "medium",
                    "recommendation": "Consider adding indexes for better performance",
                }
            )

        # Check for hash joins
        if "hash join" in line:
            analysis["join_analysis"]["join_count"] += 1
            analysis["join_analysis"]["join_types"].append("hash_join")

        # Check for nested loop joins
        if "nested loop" in line:
            analysis["join_analysis"]["join_count"] += 1
            analysis["join_analysis"]["join_types"].append("nested_loop")

        # Extract cost estimates
        if "cost=" in line:
            cost_match = line.split("cost=")[1].split()[0]
            if ".." in cost_match:
                total_cost = float(cost_match.split("..")[1])
                analysis["estimated_cost"] = total_cost

        # Extract row estimates
        if "rows=" in line:
            rows_match = line.split("rows=")[1].split()[0]
            try:
                row_count = int(float(rows_match))
                analysis["estimated_rows"] = max(analysis["estimated_rows"], row_count)
            except ValueError:
                pass


def _parse_sqlite_plan(plan_text: str, analysis: Dict[str, Any]) -> None:
    """Parse SQLite/DuckDB execution plans."""
    lines = plan_text.strip().split("\n")

    for line in lines:
        line = line.strip().lower()

        # Check for table scans
        if "scan" in line and "table" in line:
            analysis["hotspots"].append(
                {
                    "reason": "table_scan",
                    "node": line,
                    "severity": "low",
                    "recommendation": "Consider query optimization if performance is an issue",
                }
            )


def _parse_generic_plan(plan_text: str, analysis: Dict[str, Any]) -> None:
    """Generic parsing for unknown dialects."""
    lines = plan_text.strip().split("\n")

    # Basic heuristics for any execution plan
    for line in lines:
        line = line.strip().lower()

        # Look for common performance indicators
        if any(keyword in line for keyword in ["scan", "table scan", "full scan"]):
            analysis["hotspots"].append(
                {
                    "reason": "potential_scan_operation",
                    "node": line,
                    "severity": "medium",
                    "recommendation": "Review query for potential optimization opportunities",
                }
            )

        if "join" in line:
            analysis["join_analysis"]["join_count"] += 1

    analysis["warnings"].append("Using generic plan analysis - results may be limited")


def _generate_plan_assessment(analysis: Dict[str, Any]) -> None:
    """Generate overall assessment and recommendations."""
    hotspots = analysis.get("hotspots", [])

    # Assess index effectiveness
    if hotspots:
        high_severity = len([h for h in hotspots if h.get("severity") == "high"])
        if high_severity > 0:
            analysis["index_usage"]["index_effectiveness"] = "poor"
            analysis["index_usage"]["missing_indexes"].append(
                "Consider adding indexes for high-severity operations"
            )
        else:
            analysis["index_usage"]["index_effectiveness"] = "fair"
    else:
        analysis["index_usage"]["index_effectiveness"] = "good"

    # Generate warnings based on analysis
    if len(hotspots) > 5:
        analysis["warnings"].append("Multiple performance hotspots detected - comprehensive optimization needed")

    if analysis["join_analysis"]["join_count"] > 3:
        analysis["warnings"].append("Complex join operations detected - consider query restructuring")

    if analysis.get("estimated_cost", 0) > 10000:
        analysis["warnings"].append("High estimated query cost - optimization recommended")
