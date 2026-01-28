# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Database validation utilities.

This module provides validation functions for SQL syntax, table existence,
partitioning strategies, and table conflicts.
"""

from typing import Any, Dict, List, Optional

from datus.tools.func_tool.base import FuncToolResult
from datus.tools.func_tool.database.patterns import TableCoordinate


def validate_sql_syntax(
    sql: str, dialect: Optional[str] = None, agent_config: Optional[Any] = None
) -> FuncToolResult:
    """
    Validate SQL syntax without executing the query.

    Args:
        sql: SQL statement to validate
        dialect: Database dialect (optional, will be detected from agent_config if not provided)
        agent_config: Agent configuration with db_type attribute

    Returns:
        FuncToolResult with validation status and extracted tables
    """
    try:
        import sqlglot
        from sqlglot import exp

        # Get dialect from agent config or parameter
        if not dialect and agent_config and hasattr(agent_config, "db_type"):
            dialect = agent_config.db_type

        # Parse SQL syntax tree
        parsed = sqlglot.parse_one(sql, read=dialect)

        # Basic syntax checks
        issues = []

        # Check for basic SELECT/FROM/INSERT/UPDATE/DELETE structure
        has_main_operation = any(
            isinstance(node, (exp.Select, exp.Insert, exp.Update, exp.Delete)) for node in parsed.walk()
        )

        if not has_main_operation:
            issues.append("SQL语句缺少基本的查询/修改操作关键词(SELECT, INSERT, UPDATE, DELETE等)")

        # Extract table references
        tables = [table.name for table in parsed.find_all(exp.Table)]
        if not tables and has_main_operation:
            # Some valid operations might not have explicit table references (like SELECT 1)
            # Only flag as issue if it's clearly intended to have tables
            pass

        if issues:
            return FuncToolResult(success=0, error="; ".join(issues))

        return FuncToolResult(
            result={
                "syntax_valid": True,
                "tables_referenced": tables,
                "sql_type": type(parsed).__name__,
                "dialect": dialect or "default",
            }
        )

    except Exception as e:
        return FuncToolResult(success=0, error=f"SQL语法错误: {str(e)}")


def check_table_conflicts(
    table_name: str,
    catalog: Optional[str] = "",
    database: Optional[str] = "",
    schema_name: Optional[str] = "",
    connector: Optional[Any] = None,
    schema_rag: Optional[Any] = None,
    has_schema: bool = False,
    coordinate_builder: Optional[Any] = None,
) -> FuncToolResult:
    """
    Check for potential table conflicts and duplicate data structures.

    Args:
        table_name: Table to check for conflicts
        catalog: Optional catalog override
        database: Optional database override
        schema_name: Optional schema override
        connector: Database connector instance
        schema_rag: Schema RAG storage instance
        has_schema: Whether schema storage is available
        coordinate_builder: Function to build table coordinates

    Returns:
        FuncToolResult with conflict analysis
    """
    try:
        if not coordinate_builder:
            return FuncToolResult(success=0, error="Coordinate builder not provided")

        # Build target table coordinate
        target_coordinate = coordinate_builder(
            raw_name=table_name,
            catalog=catalog,
            database=database,
            schema=schema_name,
        )

        # Get target table metadata
        target_metadata = _get_table_metadata_impl(target_coordinate, schema_rag, has_schema, connector)
        if not target_metadata:
            return FuncToolResult(success=0, error=f"Target table '{table_name}' not found in metadata store")

        # Search for similar tables
        similar_tables = _find_similar_tables_impl(
            target_coordinate, target_metadata, schema_rag, has_schema, connector
        )

        # Analyze conflicts
        conflict_analysis = _analyze_table_conflicts_impl(target_metadata, similar_tables)

        return FuncToolResult(result=conflict_analysis)

    except Exception as e:
        return FuncToolResult(success=0, error=f"Table conflict check failed: {str(e)}")


def validate_partitioning(
    table_name: str,
    ddl_text: str,
    catalog: Optional[str] = "",
    database: Optional[str] = "",
    schema_name: Optional[str] = "",
    coordinate_builder: Optional[Any] = None,
) -> FuncToolResult:
    """
    Validate table partitioning strategy and provide optimization recommendations.

    Args:
        table_name: Table to validate partitioning
        ddl_text: DDL text of the table
        catalog: Optional catalog override
        database: Optional database override
        schema_name: Optional schema override
        coordinate_builder: Function to build table coordinates

    Returns:
        FuncToolResult with partitioning analysis
    """
    try:
        if not coordinate_builder:
            return FuncToolResult(success=0, error="Coordinate builder not provided")

        # Build table coordinate
        coordinate = coordinate_builder(
            raw_name=table_name,
            catalog=catalog,
            database=database,
            schema=schema_name,
        )

        # Parse partitioning information
        partitioning_info = _parse_partitioning_info_impl(ddl_text, coordinate)

        # Validate partitioning strategy
        validation_results = _validate_partitioning_strategy_impl(partitioning_info, coordinate)

        # Generate recommendations
        recommendations = _generate_partitioning_recommendations_impl(partitioning_info, validation_results)

        result = {
            "success": True,
            "partitioned": partitioning_info["is_partitioned"],
            "partition_info": partitioning_info,
            "validation_results": validation_results,
            "issues": validation_results.get("issues", []),
            "recommended_partition": recommendations.get("recommended_partition", {}),
            "performance_impact": recommendations.get("performance_impact", {}),
            "error": "",
        }

        return FuncToolResult(result=result)

    except Exception as e:
        return FuncToolResult(success=0, error=f"Partitioning validation failed: {str(e)}")


def _get_table_metadata_impl(
    coordinate: TableCoordinate,
    schema_rag: Optional[Any],
    has_schema: bool,
    connector: Optional[Any],
) -> Optional[Dict[str, Any]]:
    """Get metadata for a specific table."""
    if not has_schema or not schema_rag:
        return None

    try:
        from datus.utils.loggings import get_logger

        logger = get_logger(__name__)

        # Search for the specific table
        schemas, _ = schema_rag.search_tables(
            tables=[coordinate.table],
            catalog_name=coordinate.catalog,
            database_name=coordinate.database,
            schema_name=coordinate.schema,
            dialect=connector.dialect if connector else "",
        )

        if schemas:
            schema = schemas[0]
            return {
                "table_name": schema.table_name,
                "catalog": schema.catalog_name,
                "database": schema.database_name,
                "schema": schema.schema_name,
                "columns": schema.columns or [],
                "table_type": getattr(schema, "table_type", "table"),
                "ddl_hash": _calculate_ddl_hash_impl(schema),
            }

        return None
    except Exception as e:
        from datus.utils.loggings import get_logger

        logger = get_logger(__name__)
        logger.warning(f"Failed to get table metadata: {e}")
        return None


def _calculate_ddl_hash_impl(schema) -> str:
    """Calculate a hash of table DDL for comparison."""
    import hashlib

    # Create a normalized representation of table structure
    ddl_components = [schema.table_name or "", str(sorted(schema.columns or [], key=lambda x: x.get("name", "")))]

    ddl_string = "|".join(ddl_components)
    return hashlib.md5(ddl_string.encode()).hexdigest()[:16]


def _find_similar_tables_impl(
    target_coordinate: TableCoordinate,
    target_metadata: Dict[str, Any],
    schema_rag: Optional[Any],
    has_schema: bool,
    connector: Optional[Any],
) -> List[Dict[str, Any]]:
    """Find tables similar to the target table."""
    similar_tables = []

    if not has_schema or not schema_rag:
        return similar_tables

    try:
        from datus.utils.loggings import get_logger

        logger = get_logger(__name__)

        # Search by table name similarity
        candidates = _search_similar_by_name_impl(target_coordinate.table, schema_rag)

        for candidate in candidates:
            if _is_same_table_impl(candidate, target_coordinate):
                continue  # Skip the target table itself

            similarity_score = _calculate_table_similarity_impl(target_metadata, candidate)
            if similarity_score > 0.3:  # Only include tables with meaningful similarity
                candidate["similarity_score"] = similarity_score
                similar_tables.append(candidate)

        # Sort by similarity score
        similar_tables.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)

        return similar_tables[:10]  # Return top 10 most similar

    except Exception as e:
        from datus.utils.loggings import get_logger

        logger = get_logger(__name__)
        logger.warning(f"Failed to find similar tables: {e}")
        return similar_tables


def _search_similar_by_name_impl(table_name: str, schema_rag: Any) -> List[Dict[str, Any]]:
    """Search for tables with similar names."""
    candidates = []

    try:
        from datus.utils.loggings import get_logger

        logger = get_logger(__name__)

        # Use schema search to find tables
        search_query = f"table named {table_name}"

        metadata, _ = schema_rag.search_similar(
            query_text=search_query,
            catalog_name="",
            database_name="",
            schema_name="",
            table_type="full",
            top_n=50,  # Get more candidates for name matching
        )

        if metadata:
            for row in metadata.select(
                ["catalog_name", "database_name", "schema_name", "table_name", "table_type", "identifier"]
            ).to_pylist():
                candidates.append(
                    {
                        "table_name": row.get("table_name", ""),
                        "catalog": row.get("catalog_name", ""),
                        "database": row.get("database_name", ""),
                        "schema": row.get("schema_name", ""),
                        "table_type": row.get("table_type", ""),
                        "identifier": row.get("identifier", ""),
                    }
                )

    except Exception as e:
        from datus.utils.loggings import get_logger

        logger = get_logger(__name__)
        logger.warning(f"Failed to search similar by name: {e}")

    return candidates


def _is_same_table_impl(candidate: Dict[str, Any], target: TableCoordinate) -> bool:
    """Check if candidate table is the same as target table."""
    return (
        candidate.get("table_name") == target.table
        and candidate.get("catalog") == target.catalog
        and candidate.get("database") == target.database
        and candidate.get("schema") == target.schema
    )


def _calculate_table_similarity_impl(target: Dict[str, Any], candidate: Dict[str, Any]) -> float:
    """Calculate similarity score between two tables."""
    score = 0.0

    # Name similarity (basic string similarity)
    target_name = target.get("table_name", "").lower()
    candidate_name = candidate.get("table_name", "").lower()

    if target_name == candidate_name:
        score += 0.5  # Exact name match
    elif target_name in candidate_name or candidate_name in target_name:
        score += 0.3  # Partial name match

    # Column similarity (if we can get column info)
    target_columns = target.get("columns", [])
    candidate_columns = candidate.get("columns", [])

    if target_columns and candidate_columns:
        # Compare column names and types
        target_col_names = {col.get("name", "").lower() for col in target_columns}
        candidate_col_names = {col.get("name", "").lower() for col in candidate_columns}

        intersection = target_col_names & candidate_col_names
        union = target_col_names | candidate_col_names

        if union:
            jaccard_similarity = len(intersection) / len(union)
            score += jaccard_similarity * 0.4  # Column overlap contributes to score

    # DDL hash similarity
    target_hash = target.get("ddl_hash", "")
    candidate_hash = candidate.get("ddl_hash", "")

    if target_hash and candidate_hash and target_hash == candidate_hash:
        score += 0.1  # Exact DDL match bonus

    return min(score, 1.0)  # Cap at 1.0


def _analyze_table_conflicts_impl(target_metadata: Dict[str, Any], similar_tables: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze conflicts between target table and similar tables."""
    analysis = {
        "success": True,
        "exists_similar": len(similar_tables) > 0,
        "target_table": {
            "name": target_metadata.get("table_name", ""),
            "columns": [col.get("name", "") for col in target_metadata.get("columns", [])],
            "ddl_hash": target_metadata.get("ddl_hash", ""),
            "estimated_rows": 0,  # Would need to get from actual table stats
        },
        "matches": [],
        "duplicate_build_risk": "low",
        "layering_violations": [],
        "error": "",
    }

    # Analyze each similar table
    for similar in similar_tables:
        similarity_score = similar.get("similarity_score", 0)
        conflict_type = _classify_conflict_type_impl(similarity_score, similar)

        match_info = {
            "table_name": similar.get("table_name", ""),
            "similarity_score": similarity_score,
            "conflict_type": conflict_type,
            "matching_columns": [],  # Would need more detailed column comparison
            "column_similarity": similarity_score,  # Simplified
            "business_conflict": _assess_business_conflict_impl(target_metadata, similar),
            "recommendation": _generate_conflict_recommendation_impl(conflict_type, similarity_score),
        }

        analysis["matches"].append(match_info)

    # Assess overall risk
    if analysis["matches"]:
        high_similarity = [m for m in analysis["matches"] if m["similarity_score"] > 0.8]
        if high_similarity:
            analysis["duplicate_build_risk"] = "high"
        elif len([m for m in analysis["matches"] if m["similarity_score"] > 0.6]) > 2:
            analysis["duplicate_build_risk"] = "medium"

    # Check for layering violations
    layering_issues = _check_layering_violations_impl(target_metadata, similar_tables)
    analysis["layering_violations"] = layering_issues

    return analysis


def _classify_conflict_type_impl(similarity_score: float, table_info: Dict[str, Any]) -> str:
    """Classify the type of conflict."""
    if similarity_score > 0.9:
        return "duplicate"  # Nearly identical tables
    elif similarity_score > 0.7:
        return "similar_business"  # Similar business purpose
    elif similarity_score > 0.5:
        return "structural_overlap"  # Structural similarities
    else:
        return "minor_overlap"  # Minor similarities only


def _assess_business_conflict_impl(target: Dict[str, Any], candidate: Dict[str, Any]) -> str:
    """Assess business logic conflicts (simplified version)."""
    target_name = target.get("table_name", "").lower()
    candidate_name = candidate.get("table_name", "").lower()

    # Simple heuristic: similar names often indicate similar business purpose
    if "fact" in target_name and "fact" in candidate_name:
        return "可能存在事实表重复建设"
    elif "dim" in target_name and "dim" in candidate_name:
        return "可能存在维度表重复建设"
    elif any(keyword in target_name for keyword in ["user", "customer", "client"]) and any(
        keyword in candidate_name for keyword in ["user", "customer", "client"]
    ):
        return "可能存在用户相关数据重复"

    return "表结构相似，建议进一步评估业务需求"


def _generate_conflict_recommendation_impl(conflict_type: str, similarity_score: float) -> str:
    """Generate recommendations based on conflict type."""
    if conflict_type == "duplicate":
        return "建议删除重复表或合并数据，优先保留数据更完整的一份"
    elif conflict_type == "similar_business":
        return "建议评估是否可以复用现有表，或明确业务边界"
    elif conflict_type == "structural_overlap":
        return "建议检查是否可以标准化表结构设计"
    else:
        return "建议定期review表设计，避免逐渐偏离规范"


def _check_layering_violations_impl(target: Dict[str, Any], similar_tables: List[Dict[str, Any]]) -> List[str]:
    """Check for data warehouse layering violations."""
    violations = []
    target_name = target.get("table_name", "").lower()

    # Simplified layering checks
    if "ads_" in target_name:
        # ADS layer should not have direct duplicates in other layers
        for similar in similar_tables:
            similar_name = similar.get("table_name", "").lower()
            if any(layer in similar_name for layer in ["ods_", "dwd_", "dws_"]):
                violations.append(f"ADS层表不应与{similar_name}直接对应，可能违反分层规范")

    elif "dws_" in target_name:
        # DWS layer aggregations should be unique
        duplicate_aggregations = [t for t in similar_tables if "dws_" in t.get("table_name", "").lower()]
        if duplicate_aggregations:
            violations.append("DWS层存在相似汇总逻辑，可能存在重复计算")

    return violations


def _parse_partitioning_info_impl(ddl_text: str, coordinate: TableCoordinate) -> Dict[str, Any]:
    """Parse partitioning information from DDL text."""
    info = {
        "is_partitioned": False,
        "partition_key": "",
        "partition_type": "",
        "partition_count": 0,
        "partition_expression": "",
        "partition_values": [],
        "subpartition_info": {},
    }

    if not ddl_text:
        return info

    ddl_lower = ddl_text.lower()

    # Check if table is partitioned
    if "partitioned by" in ddl_lower or "partition by" in ddl_lower:
        info["is_partitioned"] = True

        # Extract partition key and type
        _extract_partition_details_impl(ddl_text, info)

    return info


def _extract_partition_details_impl(ddl_text: str, info: Dict[str, Any]) -> None:
    """Extract detailed partitioning information from DDL."""
    import re

    ddl_lower = ddl_text.lower()

    # StarRocks partitioning patterns
    starrocks_patterns = [
        r"partitioned\s+by\s+\(([^)]+)\)",  # PARTITIONED BY (column)
        r"partition\s+by\s+\(([^)]+)\)",  # PARTITION BY (column)
        r"partitioned\s+by\s+date_trunc\([^)]+\)",  # PARTITIONED BY date_trunc
    ]

    for pattern in starrocks_patterns:
        match = re.search(pattern, ddl_lower, re.IGNORECASE)
        if match:
            partition_expr = match.group(1).strip()
            info["partition_expression"] = partition_expr

            # Try to extract partition key
            if "(" in partition_expr and ")" in partition_expr:
                # Extract column name from function calls like date_trunc('day', column)
                inner_match = re.search(r'[\'"]([^\'"]+)[\'"],?\s*([^)]+)', partition_expr)
                if inner_match:
                    info["partition_key"] = inner_match.group(2).strip()
                    info["partition_type"] = "time_based"
                else:
                    # Simple column partitioning
                    info["partition_key"] = partition_expr.strip("()")
                    info["partition_type"] = "range"
            else:
                info["partition_key"] = partition_expr
                info["partition_type"] = "range"

            break

    # Try to estimate partition count (simplified)
    if info["partition_type"] == "time_based":
        info["partition_count"] = 30  # Assume monthly partitions for time-based
    else:
        info["partition_count"] = 10  # Default estimate


def _validate_partitioning_strategy_impl(partition_info: Dict[str, Any], coordinate: TableCoordinate) -> Dict[str, Any]:
    """Validate partitioning strategy against best practices."""
    results = {
        "partition_key_valid": True,
        "granularity_appropriate": True,
        "data_distribution_even": True,
        "pruning_opportunities": True,
        "issues": [],
    }

    if not partition_info["is_partitioned"]:
        results["issues"].append(
            {
                "severity": "medium",
                "issue_type": "no_partitioning",
                "description": "表未进行分区，可能影响查询性能和大表维护",
                "recommendation": "建议根据数据特点和查询模式添加分区",
            }
        )
        return results

    partition_key = partition_info.get("partition_key", "").lower()
    partition_type = partition_info.get("partition_type", "")

    # Validate partition key choice
    if partition_type == "time_based":
        # Time-based partitioning validation
        time_keywords = ["create_time", "update_time", "event_time", "date", "time"]
        if not any(keyword in partition_key for keyword in time_keywords):
            results["partition_key_valid"] = False
            results["issues"].append(
                {
                    "severity": "high",
                    "issue_type": "poor_key_choice",
                    "description": f"时间分区键'{partition_key}'不是标准时间字段",
                    "recommendation": "建议使用create_time、update_time等标准时间字段作为分区键",
                }
            )

    elif partition_type == "range":
        # Range partitioning validation
        if not partition_key:
            results["partition_key_valid"] = False
            results["issues"].append(
                {
                    "severity": "high",
                    "issue_type": "missing_partition_key",
                    "description": "分区表缺少明确的partition_key定义",
                    "recommendation": "明确指定分区键字段",
                }
            )

    # Check partition granularity
    partition_count = partition_info.get("partition_count", 0)
    if partition_count > 1000:
        results["granularity_appropriate"] = False
        results["issues"].append(
            {
                "severity": "medium",
                "issue_type": "too_many_partitions",
                "description": f"分区数量({partition_count})过多，可能影响查询性能",
                "recommendation": "考虑增大分区粒度或使用动态分区",
            }
        )
    elif partition_count < 3:
        results["granularity_appropriate"] = False
        results["issues"].append(
            {
                "severity": "low",
                "issue_type": "too_few_partitions",
                "description": f"分区数量({partition_count})过少，限制了分区裁剪效果",
                "recommendation": "考虑减小分区粒度以提高查询性能",
            }
        )

    # Assess pruning opportunities (simplified)
    partition_key = partition_info.get("partition_key", "")
    if not partition_key:
        results["pruning_opportunities"] = False

    return results


def _generate_partitioning_recommendations_impl(
    partition_info: Dict[str, Any], validation_results: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate partitioning recommendations and performance impact assessment."""
    recommendations = {
        "recommended_partition": {},
        "performance_impact": {
            "query_speed_improvement": "unknown",
            "storage_efficiency": "neutral",
            "maintenance_overhead": "unknown",
        },
    }

    if not partition_info["is_partitioned"]:
        # Recommend partitioning for large tables
        recommendations["recommended_partition"] = {
            "suggested_key": "create_time",
            "suggested_type": "time_based",
            "estimated_partitions": 30,
            "rationale": "基于创建时间按月分区，适合大多数业务场景，支持时间范围查询优化",
        }

        recommendations["performance_impact"] = {
            "query_speed_improvement": "significant",
            "storage_efficiency": "improved",
            "maintenance_overhead": "acceptable",
        }

    else:
        issues = validation_results.get("issues", [])
        partition_key = partition_info.get("partition_key", "")

        if any(issue["issue_type"] == "poor_key_choice" for issue in issues):
            # Suggest better partition key
            table_name_lower = partition_info.get("table_name", "").lower()

            if any(keyword in table_name_lower for keyword in ["log", "event", "action"]):
                suggested_key = "event_time"
            elif any(keyword in table_name_lower for keyword in ["order", "transaction"]):
                suggested_key = "order_date"
            else:
                suggested_key = "create_time"

            recommendations["recommended_partition"] = {
                "suggested_key": suggested_key,
                "suggested_type": "time_based",
                "estimated_partitions": 30,
                "rationale": f"建议使用{suggested_key}作为分区键，更符合业务查询模式",
            }

        elif any(issue["issue_type"] == "too_many_partitions" for issue in issues):
            # Suggest coarser granularity
            recommendations["recommended_partition"] = {
                "suggested_key": partition_key,
                "suggested_type": "time_based",
                "estimated_partitions": 12,
                "rationale": "建议改用季度或年度分区，减少分区数量",
            }

    return recommendations
