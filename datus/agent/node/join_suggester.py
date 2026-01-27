# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Join Path Suggester for intelligent multi-table query recommendations.

This module provides functionality to suggest optimal join paths between tables
based on relationship metadata extracted during schema discovery.
"""

import json
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

from datus.configuration.node_config import DEFAULT_JOIN_DEPTH
from datus.storage.schema_metadata import SchemaStorage
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


async def suggest_join_paths(
    storage: SchemaStorage,
    source_tables: List[str],
    target_tables: List[str],
    max_depth: int = DEFAULT_JOIN_DEPTH,
    catalog_name: str = "",
    database_name: str = "",
    schema_name: str = "",
) -> List[Dict[str, Any]]:
    """
    Suggest join paths between source and target tables using relationship metadata.

    This function analyzes foreign key relationships to find optimal join paths
    for multi-table queries. Uses BFS (Breadth-First Search) to find shortest paths.

    Args:
        storage: SchemaStorage instance
        source_tables: List of source table names
        target_tables: List of target table names
        max_depth: Maximum path length (number of tables in path)
        catalog_name: Optional catalog filter
        database_name: Optional database filter
        schema_name: Optional schema filter

    Returns:
        List of suggested join paths:
        [
            {
                "path": ["table1", "table2", "table3"],
                "join_conditions": ["table1.id = table2.fk_id", "table2.id = table3.fk_id"],
                "confidence": 0.95,
                "reason": "Direct foreign key relationship",
                "path_length": 3
            }
        ]
    """
    try:
        # Get relationship metadata for all involved tables
        all_tables = list(set(source_tables + target_tables))
        schemas = storage.get_table_schemas(
            table_names=all_tables, catalog_name=catalog_name, database_name=database_name, schema_name=schema_name
        )

        if not schemas or len(schemas) == 0:
            logger.warning(f"No schemas found for tables: {all_tables}")
            return []

        # Build relationship graph
        graph = _build_relationship_graph(schemas)

        # Find shortest paths using BFS
        paths = []
        for src in source_tables:
            for tgt in target_tables:
                if src == tgt:
                    continue  # Skip same table

                path = _bfs_shortest_path(graph, src, tgt, max_depth)
                if path:
                    join_conditions = _generate_join_conditions(graph, path)
                    confidence = _calculate_confidence(graph, path)
                    reason = _infer_reason(graph, path)

                    paths.append(
                        {
                            "path": path,
                            "join_conditions": join_conditions,
                            "confidence": confidence,
                            "reason": reason,
                            "path_length": len(path),
                        }
                    )

        # Sort by confidence (highest first) and path length (shortest first)
        paths.sort(key=lambda p: (-p["confidence"], p["path_length"]))

        return paths

    except Exception as e:
        logger.error(f"Failed to suggest join paths: {e}")
        return []


def _build_relationship_graph(schemas) -> Dict[str, Dict[str, str]]:
    """
    Build a relationship graph from schema metadata.

    Args:
        schemas: List of TableSchema objects

    Returns:
        Graph represented as adjacency list:
        {
            "table1": {
                "table2": "table1.id = table2.fk_id",
                "table3": "table1.id = table3.user_id"
            }
        }
    """
    graph = {}

    for schema in schemas:
        table_name = schema.table_name
        graph[table_name] = {}

        # Parse relationship_metadata if exists
        if hasattr(schema, "relationship_metadata") and schema.relationship_metadata:
            try:
                relationships = json.loads(schema.relationship_metadata)
                foreign_keys = relationships.get("foreign_keys", [])

                for fk in foreign_keys:
                    to_table = fk.get("to_table", "")
                    from_column = fk.get("from_column", "")
                    to_column = fk.get("to_column", "")

                    if to_table and from_column:
                        # Add bidirectional edge
                        join_condition = f"{table_name}.{from_column} = {to_table}.{to_column}"
                        graph[table_name][to_table] = join_condition

                        # Add reverse edge (undirected graph)
                        if to_table not in graph:
                            graph[to_table] = {}
                        graph[to_table][table_name] = f"{to_table}.{to_column} = {table_name}.{from_column}"

            except (json.JSONDecodeError, TypeError) as e:
                logger.debug(f"Failed to parse relationship_metadata for {table_name}: {e}")

    return graph


def _bfs_shortest_path(
    graph: Dict[str, Dict[str, str]], start: str, end: str, max_depth: int = 3
) -> Optional[List[str]]:
    """
    Find shortest path between two tables using BFS algorithm.

    Args:
        graph: Relationship graph as adjacency list
        start: Starting table name
        end: Target table name
        max_depth: Maximum path length

    Returns:
        List of table names forming the path, or None if no path exists
    """
    if start not in graph:
        return None

    queue = deque([(start, [start])])
    visited = {start}

    while queue:
        current_node, path = queue.popleft()

        if len(path) > max_depth:
            continue

        if current_node == end:
            return path

        # Explore neighbors
        for neighbor, join_condition in graph[current_node].items():
            if neighbor not in visited:
                visited.add(neighbor)
                new_path = path + [neighbor]
                queue.append((neighbor, new_path))

    return None


def _generate_join_conditions(graph: Dict[str, Dict[str, str]], path: List[str]) -> List[str]:
    """
    Generate SQL join conditions from a path.

    Args:
        graph: Relationship graph
        path: List of table names in the path

    Returns:
        List of join condition strings
    """
    join_conditions = []

    for i in range(len(path) - 1):
        table1 = path[i]
        table2 = path[i + 1]

        if table1 in graph and table2 in graph[table1]:
            join_conditions.append(graph[table1][table2])
        elif table2 in graph and table1 in graph[table2]:
            join_conditions.append(graph[table2][table1])

    return join_conditions


def _calculate_confidence(graph: Dict[str, Dict[str, str]], path: List[str]) -> float:
    """
    Calculate confidence score for a join path.

    Shorter paths with direct foreign key relationships have higher confidence.

    Args:
        graph: Relationship graph
        path: List of table names in the path

    Returns:
        Confidence score between 0.0 and 1.0
    """
    # Base confidence: shorter paths are better
    path_length = len(path)
    base_confidence = 1.0 / path_length

    # Check if all edges are foreign key relationships (strong signal)
    fk_count = 0
    for i in range(len(path) - 1):
        table1 = path[i]
        table2 = path[i + 1]

        # Check if edge exists
        if table1 in graph and table2 in graph[table1]:
            fk_count += 1

    # Boost confidence if all edges are FK relationships
    if fk_count == path_length - 1:
        base_confidence *= 1.2

    # Cap at 1.0
    return min(base_confidence, 1.0)


def _infer_reason(graph: Dict[str, Dict[str, str]], path: List[str]) -> str:
    """
    Infer human-readable reason for the suggested path.

    Args:
        graph: Relationship graph
        path: List of table names in the path

    Returns:
        Human-readable reason string
    """
    if len(path) == 2:
        return "Direct foreign key relationship"
    elif len(path) == 3:
        return "Single hop through intermediate table"
    elif len(path) <= 5:
        return f"Multi-hop path through {len(path) - 1} tables"
    else:
        return f"Complex path through {len(path) - 1} tables"


async def suggest_drill_down_paths(
    storage: SchemaStorage,
    fact_table: str,
    catalog_name: str = "",
    database_name: str = "",
    schema_name: str = "",
) -> List[Dict[str, Any]]:
    """
    Suggest drill-down paths from fact table to dimension hierarchies.

    This is specifically designed for drill-down analysis scenarios, identifying
    dimension tables and hierarchy levels from relationship metadata and column comments.

    Args:
        storage: SchemaStorage instance
        fact_table: Fact table name
        catalog_name: Optional catalog filter
        database_name: Optional database filter
        schema_name: Optional schema filter

    Returns:
        List of drill-down path suggestions:
        [
            {
                "dimension_table": "date_dim",
                "levels": ["year", "quarter", "month", "day"],
                "join_path": "fact_table.date_id = date_dim.date_id",
                "level_comments": {
                    "year": "Calendar year",
                    "quarter": "Fiscal quarter",
                    "month": "Calendar month",
                    "day": "Calendar day"
                }
            }
        ]
    """
    try:
        # Get fact table schema
        fact_schemas = storage.get_table_schemas(
            table_names=[fact_table], catalog_name=catalog_name, database_name=database_name, schema_name=schema_name
        )

        if not fact_schemas or len(fact_schemas) == 0:
            logger.warning(f"No schema found for fact table: {fact_table}")
            return []

        fact_schema = fact_schemas[0]

        # Parse relationship metadata
        if not fact_schema.relationship_metadata:
            logger.info(f"No relationship metadata found for {fact_table}")
            return []

        try:
            relationships = json.loads(fact_schema.relationship_metadata)
            foreign_keys = relationships.get("foreign_keys", [])

            if not foreign_keys:
                return []

            drill_downs = []
            for fk in foreign_keys:
                dim_table = fk.get("to_table", "")
                from_column = fk.get("from_column", "")

                if not dim_table:
                    continue

                # Get dimension table schema
                dim_schemas = storage.get_table_schemas(
                    table_names=[dim_table],
                    catalog_name=catalog_name,
                    database_name=database_name,
                    schema_name=schema_name,
                )

                if not dim_schemas or len(dim_schemas) == 0:
                    continue

                dim_schema = dim_schemas[0]

                # Parse column comments
                column_comments = {}
                if dim_schema.column_comments:
                    try:
                        column_comments = json.loads(dim_schema.column_comments)
                    except (json.JSONDecodeError, TypeError):
                        pass

                # Detect hierarchy from column names and comments
                levels = _detect_hierarchy_levels(dim_table, column_comments)

                if levels:
                    drill_downs.append(
                        {
                            "dimension_table": dim_table,
                            "levels": levels,
                            "join_path": f"{fact_table}.{from_column} = {dim_table}.{fk.get('to_column', '')}",
                            "level_comments": {col: column_comments.get(col, "") for col in levels},
                        }
                    )

            return drill_downs

        except (json.JSONDecodeError, TypeError) as e:
            logger.error(f"Failed to parse relationship metadata for {fact_table}: {e}")
            return []

    except Exception as e:
        logger.error(f"Failed to suggest drill-down paths for {fact_table}: {e}")
        return []


def _detect_hierarchy_levels(table_name: str, column_comments: Dict[str, str]) -> List[str]:
    """
    Detect hierarchy levels from dimension table structure.

    Args:
        table_name: Dimension table name
        column_comments: Dictionary mapping column names to comments

    Returns:
        Ordered list of column names representing hierarchy levels
    """
    from datus.configuration.business_term_config import \
        detect_temporal_granularity

    # Temporal hierarchy patterns
    temporal_order = ["year", "quarter", "month", "day", "hour", "minute"]

    # Check if this looks like a temporal dimension
    if any(temporal_kw in table_name.lower() for temporal_kw in ["date", "time", "calendar", "fiscal"]):
        # Filter columns that match temporal patterns
        temporal_cols = []
        for col in column_comments.keys():
            col_lower = col.lower()
            if any(temp_kw in col_lower for temporal_kw in temporal_order):
                temporal_cols.append(col)

        # Sort by predefined order
        ordered_cols = []
        for level in temporal_order:
            for col in temporal_cols:
                if level in col.lower():
                    ordered_cols.append(col)

        return ordered_cols if ordered_cols else temporal_cols

    # Geographic hierarchy patterns
    geo_order = ["country", "state", "province", "region", "city", "district", "zip"]

    if any(geo_kw in table_name.lower() for geo_kw in ["geo", "location", "address"]):
        geo_cols = []
        for col in column_comments.keys():
            col_lower = col.lower()
            if any(geo_kw in col_lower for geo_kw in geo_order):
                geo_cols.append(col)

        ordered_cols = []
        for level in geo_order:
            for col in geo_cols:
                if level in col.lower():
                    ordered_cols.append(col)

        return ordered_cols if ordered_cols else geo_cols

    # Fallback: return columns in order (no hierarchy detected)
    return list(column_comments.keys())[:5]  # Limit to top 5 columns
