# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Enhanced Preflight Tools for SQL Review (v2.4)

This module implements the three enhanced preflight tools introduced in v2.4:
- analyze_query_plan: Query execution plan analysis
- check_table_conflicts: Table structure conflict detection
- validate_partitioning: Partition strategy validation
"""

import time
from typing import Any, Dict, List, Optional

from datus.configuration.agent_config import AgentConfig
from datus.storage.metric.store import SemanticMetricsRAG
from datus.storage.schema_metadata.store import SchemaWithValueRAG
from datus.tools.db_tools import BaseSqlConnector
from datus.tools.db_tools.db_manager import get_db_manager
from datus.tools.func_tool.base import FuncToolResult
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class EnhancedPreflightTools:
    """
    v2.4 Enhanced Preflight Tools Collection

    Provides advanced SQL analysis capabilities for comprehensive SQL review.
    """

    def __init__(
        self,
        agent_config: AgentConfig,
        *,
        sub_agent_name: Optional[str] = None,
    ):
        """Initialize enhanced preflight tools."""
        self.agent_config = agent_config
        self.sub_agent_name = sub_agent_name

        # Initialize storage modules
        self.schema_rag = SchemaWithValueRAG(agent_config, sub_agent_name) if agent_config else None
        self.metrics_rag = SemanticMetricsRAG(agent_config, sub_agent_name) if agent_config else None

        # Initialize database manager
        self.db_manager = get_db_manager(agent_config.namespaces) if agent_config else None

        # Check availability
        self.has_schema = self.schema_rag and self.schema_rag.schema_store.table_size() > 0
        self.has_metrics = self.metrics_rag and self.metrics_rag.get_semantic_model_size() > 0

    def _get_connector(self, catalog: str = "", database: str = "", schema: str = "") -> Optional[BaseSqlConnector]:
        """Get database connector for the specified context."""
        if not self.db_manager:
            return None

        try:
            # Use the current namespace or find appropriate one
            namespace = self.db_manager.get_current_namespace()
            if not namespace:
                # Try to find namespace by database type
                for ns_name, ns_config in self.db_manager.namespaces.items():
                    if ns_config.get(self.agent_config.db_type):
                        namespace = ns_name
                        break

            if not namespace:
                logger.warning("No suitable namespace found for enhanced tools")
                return None

            connector = self.db_manager.get_connector(namespace)
            return connector
        except Exception as e:
            logger.warning(f"Failed to get database connector: {e}")
            return None

    async def analyze_query_plan(
        self, sql: str, catalog: str = "default_catalog", database: str = "", schema: str = "", **kwargs
    ) -> FuncToolResult:
        """
        Analyze query execution plan for performance insights.

        Args:
            sql: SQL query to analyze
            catalog: Catalog name
            database: Database name
            schema: Schema name

        Returns:
            FuncToolResult with plan analysis
        """
        try:
            connector = self._get_connector(catalog, database, schema)
            if not connector:
                return FuncToolResult(success=0, error="No database connector available for query plan analysis")

            # Determine EXPLAIN syntax based on database type
            db_type = connector.dialect.lower()

            if db_type in ["mysql", "starrocks"]:
                explain_sql = f"EXPLAIN {sql}"
            elif db_type == "postgresql":
                explain_sql = f"EXPLAIN ANALYZE {sql}"
            elif db_type == "sqlite":
                explain_sql = f"EXPLAIN QUERY PLAN {sql}"
            elif db_type == "duckdb":
                explain_sql = f"EXPLAIN ANALYZE {sql}"
            else:
                # Generic fallback
                explain_sql = f"EXPLAIN {sql}"

            # Execute EXPLAIN query
            start_time = time.time()
            result = connector.execute_arrow(explain_sql)
            execution_time = time.time() - start_time

            if not result or not result.success:
                return FuncToolResult(
                    success=0, error=f"Failed to execute EXPLAIN query: {result.error if result else 'Unknown error'}"
                )

            plan_text = ""
            if result.sql_return:
                # Convert result to readable format
                if hasattr(result.sql_return, "to_pylist"):
                    plan_rows = result.sql_return.to_pylist()
                    plan_text = "\n".join(
                        [
                            " | ".join([str(row.get(col, "")) for col in result.sql_return.column_names])
                            for row in plan_rows
                        ]
                    )
                else:
                    plan_text = str(result.sql_return)

            # Basic plan analysis
            analysis = self._analyze_plan_text(plan_text, db_type)

            return FuncToolResult(
                success=1,
                result={
                    "plan_text": plan_text,
                    "estimated_cost": analysis.get("estimated_cost", 0),
                    "estimated_rows": analysis.get("estimated_rows", 0),
                    "hotspots": analysis.get("hotspots", []),
                    "join_analysis": analysis.get("join_analysis", {}),
                    "index_usage": analysis.get("index_usage", {}),
                    "execution_time": execution_time,
                    "recommendations": analysis.get("recommendations", []),
                },
            )

        except Exception as e:
            logger.warning(f"Query plan analysis failed: {e}, using fallback rule-based analysis")
            # Fallback: Use rule-based analysis when EXPLAIN fails
            return self._fallback_query_analysis(sql, str(e))

    def _analyze_plan_text(self, plan_text: str, db_type: str) -> Dict[str, Any]:
        """Analyze execution plan text and extract insights."""
        analysis = {
            "estimated_cost": 0,
            "estimated_rows": 0,
            "hotspots": [],
            "join_analysis": {},
            "index_usage": {},
            "recommendations": [],
        }

        if not plan_text:
            return analysis

        plan_lower = plan_text.lower()

        # Extract cost estimates
        if "cost=" in plan_lower:
            import re

            cost_match = re.search(r"cost=([\d.]+)", plan_lower)
            if cost_match:
                analysis["estimated_cost"] = float(cost_match.group(1))

        # Extract row estimates
        if "rows=" in plan_lower:
            import re

            rows_match = re.search(r"rows=([\d.]+)", plan_lower)
            if rows_match:
                analysis["estimated_rows"] = float(rows_match.group(1))

        # Identify potential hotspots
        hotspots = []
        if "table scan" in plan_lower or "seq scan" in plan_lower:
            hotspots.append("Full table scan detected - consider adding indexes")
        if "nested loop" in plan_lower and "no index" in plan_lower:
            hotspots.append("Inefficient join detected - check join conditions")
        if "filesort" in plan_lower or "sort" in plan_lower:
            hotspots.append("Sorting operation detected - may impact performance")

        analysis["hotspots"] = hotspots

        # Basic join analysis
        join_analysis = {}
        if "join" in plan_lower:
            join_analysis["join_detected"] = True
            if "hash join" in plan_lower:
                join_analysis["join_type"] = "hash"
            elif "merge join" in plan_lower:
                join_analysis["join_type"] = "merge"
            elif "nested loop" in plan_lower:
                join_analysis["join_type"] = "nested_loop"
        else:
            join_analysis["join_detected"] = False

        analysis["join_analysis"] = join_analysis

        # Index usage analysis
        index_usage = {}
        if "index" in plan_lower:
            index_usage["index_used"] = True
            if "using index" in plan_lower:
                index_usage["index_scan"] = True
        else:
            index_usage["index_used"] = False
            analysis["recommendations"].append("Consider adding appropriate indexes for better performance")

        analysis["index_usage"] = index_usage

        return analysis

    def _fallback_query_analysis(self, sql: str, error: str) -> FuncToolResult:
        """
        Fallback analysis when EXPLAIN execution fails.

        Uses static analysis rules to identify potential issues without executing EXPLAIN.
        This provides useful insights even when the database connection fails or SQL has syntax errors.

        Args:
            sql: SQL query to analyze
            error: Original error message from EXPLAIN attempt

        Returns:
            FuncToolResult with rule-based analysis
        """
        import re

        hotspots = []
        warnings = [f"EXPLAIN execution failed: {error}"]
        index_usage = {"indexes_used": [], "missing_indexes": [], "index_effectiveness": "unknown"}

        sql_upper = sql.upper()

        # Rule 1: Detect SELECT * (full table scan risk)
        if re.search(r'SELECT\s+\*\s+FROM', sql_upper):
            hotspots.append({
                "reason": "select_star",
                "node": "SELECT *",
                "severity": "medium",
                "recommendation": "Specify only needed columns instead of SELECT * to reduce data transfer"
            })
            index_usage["index_effectiveness"] = "poor"

        # Rule 2: Detect LIKE '%...%' (leading wildcard prevents index usage)
        like_patterns = re.findall(r'LIKE\s+[\'"]?%\w+%[\'"]?', sql_upper)
        if like_patterns:
            hotspots.append({
                "reason": "leading_wildcard_like",
                "node": f"LIKE '%...%' pattern",
                "severity": "medium",
                "recommendation": "Leading wildcards in LIKE prevent index usage. Consider full-text search or removing leading %"
            })
            index_usage["missing_indexes"].append("fulltext_index")
            index_usage["index_effectiveness"] = "poor"

        # Rule 3: Detect functions on indexed columns in WHERE
        # Pattern: WHERE UPPER(col) = ... or WHERE DATE(col) = ...
        func_patterns = re.findall(r'WHERE\s+\w+\((\w+)\)', sql_upper)
        if func_patterns:
            hotspots.append({
                "reason": "function_on_indexed_column",
                "node": f"Function on {func_patterns[0]}",
                "severity": "low",
                "recommendation": f"Functions on columns prevent index usage. Consider functional indexes or restructuring query"
            })
            if index_usage["index_effectiveness"] == "unknown":
                index_usage["index_effectiveness"] = "reduced"

        # Rule 4: Check for JOIN without join conditions
        join_count = len(re.findall(r'\bJOIN\b', sql_upper))
        if join_count > 0:
            join_analysis = {"join_count": join_count, "join_types": [], "join_order_issues": []}

            # Check for missing ON clause
            if not re.search(r'\bJOIN\b.+\bON\b', sql_upper, re.DOTALL):
                hotspots.append({
                    "reason": "cross_join",
                    "node": "JOIN without ON",
                    "severity": "high",
                    "recommendation": "Missing join condition. This creates a Cartesian product which can be very expensive"
                })
                join_analysis["join_order_issues"].append("Missing join condition - potential cross join")
        else:
            join_analysis = {"join_count": 0, "join_detected": False}

        # Rule 5: Detect ORDER BY without LIMIT (could return many rows)
        if re.search(r'\bORDER\s+BY\b', sql_upper) and not re.search(r'\bLIMIT\b', sql_upper):
            warnings.append("ORDER BY without LIMIT may return many rows and consume memory")

        # Rule 6: Detect DISTINCT (can be expensive)
        if re.search(r'\bDISTINCT\b', sql_upper):
            hotspots.append({
                "reason": "distinct_operation",
                "node": "DISTINCT",
                "severity": "low",
                "recommendation": "DISTINCT operations can be expensive. Consider GROUP BY or redesign if possible"
            })

        # Rule 7: Detect subqueries
        if re.search(r'\bSELECT\b.*\bFROM\b.*\bWHERE\b.*\bSELECT\b', sql_upper, re.DOTALL):
            hotspots.append({
                "reason": "subquery",
                "node": "Subquery detected",
                "severity": "low",
                "recommendation": "Subqueries can impact performance. Consider JOINs or CTEs for better optimization"
            })

        return FuncToolResult(
            success=1,  # Mark as success because we provided fallback analysis
            result={
                "plan_text": f"[Fallback Analysis - EXPLAIN failed: {error}]",
                "estimated_cost": 0,
                "estimated_rows": 0,
                "hotspots": hotspots,
                "join_analysis": join_analysis,
                "index_usage": index_usage,
                "warnings": warnings,
                "fallback_used": True,  # Flag to indicate this is fallback analysis
                "fallback_reason": error,
                "recommendations": [h["recommendation"] for h in hotspots],
            },
        )

    async def check_table_conflicts(
        self, table_name: str, catalog: str = "default_catalog", database: str = "", schema: str = "", **kwargs
    ) -> FuncToolResult:
        """
        Check for table structure conflicts and duplicate builds.

        Args:
            table_name: Name of table to check
            catalog: Catalog name
            database: Database name
            schema: Schema name

        Returns:
            FuncToolResult with conflict analysis
        """
        try:
            if not self.has_schema:
                return FuncToolResult(success=0, error="Schema metadata not available for conflict checking")

            # Get table schema information
            table_info = self.schema_rag.get_table_schema(
                table_name=table_name, catalog_name=catalog, database_name=database, schema_name=schema
            )

            if not table_info:
                return FuncToolResult(
                    success=1,
                    result={
                        "exists_similar": False,
                        "matches": [],
                        "duplicate_build_risk": "unknown",
                        "layering_violations": [],
                        "analysis": "Table not found in schema metadata",
                    },
                )

            # Search for similar table structures
            similar_tables = self._find_similar_tables(table_info, catalog, database, schema)

            # Analyze layering violations
            layering_violations = self._analyze_layering_violations(table_name, similar_tables)

            # Assess duplicate build risk
            duplicate_risk = self._assess_duplicate_risk(similar_tables, layering_violations)

            return FuncToolResult(
                success=1,
                result={
                    "exists_similar": len(similar_tables) > 0,
                    "matches": similar_tables,
                    "duplicate_build_risk": duplicate_risk,
                    "layering_violations": layering_violations,
                    "recommendations": self._generate_conflict_recommendations(
                        similar_tables, layering_violations, duplicate_risk
                    ),
                },
            )

        except Exception as e:
            logger.error(f"Table conflict checking failed: {e}")
            return FuncToolResult(success=0, error=str(e))

    def _find_similar_tables(self, table_info: Dict, catalog: str, database: str, schema: str) -> List[Dict]:
        """Find tables with similar structure."""
        similar_tables = []

        try:
            # Get all tables in the same database/schema
            all_tables = self.schema_rag.search_tables(
                catalog_name=catalog, database_name=database, schema_name=schema, limit=100
            )

            if not all_tables:
                return similar_tables

            table_columns = set(table_info.get("columns", {}).keys())
            table_column_count = len(table_columns)

            for candidate_table in all_tables:
                if candidate_table.get("table_name") == table_info.get("table_name"):
                    continue  # Skip self

                candidate_columns = set(candidate_table.get("columns", {}).keys())
                candidate_count = len(candidate_columns)

                # Check for structural similarity
                if abs(table_column_count - candidate_count) <= 2:  # Allow small differences
                    intersection = table_columns.intersection(candidate_columns)
                    union = table_columns.union(candidate_columns)

                    if union and (len(intersection) / len(union)) > 0.6:  # 60% similarity threshold
                        similar_tables.append(
                            {
                                "table_name": candidate_table.get("table_name"),
                                "similarity_score": len(intersection) / len(union),
                                "shared_columns": list(intersection),
                                "column_count": candidate_count,
                            }
                        )

        except Exception as e:
            logger.warning(f"Error finding similar tables: {e}")

        return similar_tables

    def _analyze_layering_violations(self, table_name: str, similar_tables: List[Dict]) -> List[str]:
        """Analyze data warehouse layering violations."""
        violations = []

        # Check for naming patterns that indicate layering issues
        ods_patterns = ["ods_", "origin_", "raw_"]
        dwd_patterns = ["dwd_", "dim_", "fact_"]
        dws_patterns = ["dws_", "summary_", "agg_"]
        ads_patterns = ["ads_", "report_", "dashboard_"]

        table_lower = table_name.lower()

        layer_patterns = [("ODS", ods_patterns), ("DWD", dwd_patterns), ("DWS", dws_patterns), ("ADS", ads_patterns)]

        detected_layers = []
        for layer_name, patterns in layer_patterns:
            if any(pattern in table_lower for pattern in patterns):
                detected_layers.append(layer_name)

        # Check for potential violations
        if len(detected_layers) > 1:
            violations.append(f"Table name suggests multiple layers: {detected_layers}")

        # Check similar tables for layering conflicts
        for similar in similar_tables:
            similar_name = similar.get("table_name", "").lower()
            similar_layers = []
            for layer_name, patterns in layer_patterns:
                if any(pattern in similar_name for pattern in patterns):
                    similar_layers.append(layer_name)

            # If similar table is in different layer, it might be a violation
            for detected_layer in detected_layers:
                if detected_layer not in similar_layers:
                    violations.append(
                        f"Similar structure table '{similar.get('table_name')}' appears to be in different layer"
                    )
                    break

        return violations

    def _assess_duplicate_risk(self, similar_tables: List[Dict], violations: List[str]) -> str:
        """Assess the risk level of duplicate table builds."""
        if not similar_tables:
            return "low"

        high_similarity_count = sum(1 for t in similar_tables if t.get("similarity_score", 0) > 0.8)
        if high_similarity_count > 0:
            return "high"

        if violations:
            return "medium"

        return "low"

    def _generate_conflict_recommendations(
        self, similar_tables: List[Dict], violations: List[str], risk_level: str
    ) -> List[str]:
        """Generate recommendations based on conflict analysis."""
        recommendations = []

        if risk_level == "high":
            recommendations.append(
                "HIGH RISK: Significant structural duplication detected - review business requirements"
            )
            recommendations.append("Consider consolidating similar tables or clarifying business logic differences")

        elif risk_level == "medium":
            recommendations.append("MEDIUM RISK: Potential layering violations or moderate duplication")
            recommendations.append("Verify table purposes and consider refactoring if appropriate")

        if violations:
            recommendations.append("Layering violations detected - ensure compliance with data warehouse standards")
            for violation in violations:
                recommendations.append(f"• {violation}")

        for similar in similar_tables[:3]:  # Limit to top 3 recommendations
            table_name = similar.get("table_name")
            score = similar.get("similarity_score", 0)
            recommendations.append(f"Review relationship with similar table '{table_name}' (similarity: {score:.1%})")

        return recommendations

    async def validate_partitioning(
        self, table_name: str, catalog: str = "default_catalog", database: str = "", schema: str = "", **kwargs
    ) -> FuncToolResult:
        """
        Validate table partitioning strategy and provide recommendations.

        Args:
            table_name: Name of table to validate
            catalog: Catalog name
            database: Database name
            schema: Schema name

        Returns:
            FuncToolResult with partitioning analysis
        """
        try:
            connector = self._get_connector(catalog, database, schema)
            if not connector:
                return FuncToolResult(success=0, error="No database connector available for partitioning validation")

            # Get table DDL
            ddl_result = connector.get_table_ddl(table_name, catalog, database, schema)
            if not ddl_result or not ddl_result.success:
                return FuncToolResult(
                    success=1,  # Not an error, just not partitioned
                    result={
                        "partitioned": False,
                        "partition_info": {},
                        "validation_results": {},
                        "issues": ["Unable to retrieve table DDL"],
                        "recommended_partition": {},
                        "performance_impact": {},
                    },
                )

            ddl_text = ddl_result.result or ""
            partition_info = self._parse_partition_info(ddl_text)

            # Validate partitioning strategy
            validation_results = self._validate_partition_strategy(partition_info, table_name)

            # Generate recommendations
            recommendations = self._generate_partition_recommendations(partition_info, validation_results, table_name)

            # Assess performance impact
            performance_impact = self._assess_partition_performance_impact(partition_info, validation_results)

            return FuncToolResult(
                success=1,
                result={
                    "partitioned": partition_info.get("is_partitioned", False),
                    "partition_info": partition_info,
                    "validation_results": validation_results,
                    "issues": validation_results.get("issues", []),
                    "recommended_partition": recommendations,
                    "performance_impact": performance_impact,
                },
            )

        except Exception as e:
            logger.error(f"Partitioning validation failed: {e}")
            return FuncToolResult(success=0, error=str(e))

    def _parse_partition_info(self, ddl_text: str) -> Dict[str, Any]:
        """Parse partitioning information from DDL."""
        partition_info = {
            "is_partitioned": False,
            "partition_type": None,
            "partition_key": None,
            "partition_expression": None,
            "subpartition_type": None,
            "subpartition_key": None,
        }

        ddl_lower = ddl_text.lower()

        # Check for partitioning keywords
        if "partition" in ddl_lower:
            partition_info["is_partitioned"] = True

            # Try to extract partition details
            import re

            # Range partitioning
            if "partition by range" in ddl_lower:
                partition_info["partition_type"] = "RANGE"
                range_match = re.search(r"partition by range\s*\(([^)]+)\)", ddl_lower)
                if range_match:
                    partition_info["partition_key"] = range_match.group(1).strip()

            # List partitioning
            elif "partition by list" in ddl_lower:
                partition_info["partition_type"] = "LIST"
                list_match = re.search(r"partition by list\s*\(([^)]+)\)", ddl_lower)
                if list_match:
                    partition_info["partition_key"] = list_match.group(1).strip()

            # Hash partitioning
            elif "partition by hash" in ddl_lower:
                partition_info["partition_type"] = "HASH"
                hash_match = re.search(r"partition by hash\s*\(([^)]+)\)", ddl_lower)
                if hash_match:
                    partition_info["partition_key"] = hash_match.group(1).strip()

        return partition_info

    def _validate_partition_strategy(self, partition_info: Dict, table_name: str) -> Dict[str, Any]:
        """Validate partitioning strategy."""
        validation_results = {"is_valid": True, "issues": [], "warnings": [], "score": 100}

        if not partition_info.get("is_partitioned"):
            # Not partitioned - check if it should be
            table_lower = table_name.lower()
            if any(keyword in table_lower for keyword in ["fact", "log", "event", "metric"]):
                validation_results["warnings"].append(
                    "Large fact/log table should consider partitioning for performance"
                )
                validation_results["score"] -= 20
            return validation_results

        partition_type = partition_info.get("partition_type")
        partition_key = partition_info.get("partition_key")

        # Validate partition key
        if not partition_key:
            validation_results["issues"].append("Partition key not clearly defined")
            validation_results["is_valid"] = False
            validation_results["score"] -= 50

        # Check partition key quality
        elif partition_key:
            key_lower = partition_key.lower()

            # Time-based partitioning is preferred
            time_indicators = ["date", "time", "timestamp", "created_at", "updated_at", "year", "month", "day"]
            has_time_key = any(indicator in key_lower for indicator in time_indicators)

            if not has_time_key:
                validation_results["warnings"].append(
                    "Consider using time-based partition key for better query performance"
                )
                validation_results["score"] -= 15

            # Avoid high-cardinality keys
            if any(indicator in key_lower for indicator in ["id", "uuid", "hash"]):
                validation_results["warnings"].append("High-cardinality partition key may cause performance issues")
                validation_results["score"] -= 10

        # Validate partition type
        if partition_type == "HASH" and not any(indicator in partition_key.lower() for indicator in ["id", "key"]):
            validation_results["warnings"].append("HASH partitioning typically works best with ID/key columns")
            validation_results["score"] -= 10

        return validation_results

    def _generate_partition_recommendations(
        self, partition_info: Dict, validation_results: Dict, table_name: str
    ) -> Dict[str, Any]:
        """Generate partitioning recommendations."""
        recommendations = {}

        if not partition_info.get("is_partitioned"):
            # Suggest partitioning
            table_lower = table_name.lower()

            if any(keyword in table_lower for keyword in ["fact", "log", "event"]):
                recommendations["suggested_type"] = "RANGE"
                recommendations["suggested_key"] = "date_column"  # Generic suggestion
                recommendations["rationale"] = "Time-series data benefits from range partitioning"
            elif any(keyword in table_lower for keyword in ["dim", "master"]):
                recommendations["suggested_type"] = "HASH"
                recommendations["suggested_key"] = "primary_key"
                recommendations["rationale"] = "Dimension tables benefit from hash partitioning for joins"
            else:
                recommendations["suggested_type"] = "RANGE"
                recommendations["suggested_key"] = "created_at"
                recommendations["rationale"] = "General recommendation for temporal partitioning"

        else:
            # Optimize existing partitioning
            issues = validation_results.get("issues", [])
            warnings = validation_results.get("warnings", [])

            if warnings:
                recommendations["optimizations"] = warnings
                recommendations["rationale"] = "Address partitioning warnings for better performance"

            if issues:
                recommendations["critical_fixes"] = issues
                recommendations["rationale"] = "Fix critical partitioning issues"

        return recommendations

    def _assess_partition_performance_impact(self, partition_info: Dict, validation_results: Dict) -> Dict[str, Any]:
        """Assess performance impact of partitioning strategy."""
        impact = {
            "query_performance": "neutral",
            "storage_efficiency": "neutral",
            "maintenance_overhead": "low",
            "score": validation_results.get("score", 100),
        }

        if not partition_info.get("is_partitioned"):
            impact["query_performance"] = "degraded"
            impact["rationale"] = "Lack of partitioning may impact query performance on large tables"
            return impact

        score = impact["score"]

        if score >= 80:
            impact["query_performance"] = "good"
            impact["storage_efficiency"] = "good"
        elif score >= 60:
            impact["query_performance"] = "fair"
            impact["storage_efficiency"] = "fair"
        else:
            impact["query_performance"] = "poor"
            impact["storage_efficiency"] = "poor"
            impact["maintenance_overhead"] = "high"

        return impact
