# Preflight Tools Design Document

## Overview

This document defines the design specifications for three new preflight tools to enhance SQL review capabilities: `analyze_query_plan`, `check_table_conflicts`, and `validate_partitioning`.

## 1. analyze_query_plan Tool

### Purpose
Analyze SQL query execution plan to identify performance bottlenecks and optimization opportunities.

### Input Parameters
- `sql_query`: The SQL query to analyze
- `catalog`: Database catalog (optional)
- `database`: Database name (optional)
- `schema`: Schema name (optional)

### Implementation Strategy
1. Execute `EXPLAIN` or `EXPLAIN ANALYZE` depending on database type
2. Parse the execution plan output
3. Extract key performance metrics
4. Identify potential hotspots and optimization opportunities

### Output Schema
```python
{
    "success": bool,
    "plan_text": str,  # Raw execution plan text
    "estimated_rows": int,  # Estimated result rows
    "estimated_cost": float,  # Estimated execution cost
    "hotspots": [
        {
            "reason": str,  # e.g., "full_table_scan", "expensive_join", "missing_index"
            "node": str,  # Plan node description
            "severity": str,  # "low", "medium", "high", "critical"
            "recommendation": str  # Specific optimization suggestion
        }
    ],
    "join_analysis": {
        "join_count": int,
        "join_types": [str],  # e.g., ["hash_join", "merge_join", "nested_loop"]
        "join_order_issues": [str]  # Join order optimization suggestions
    },
    "index_usage": {
        "indexes_used": [str],  # Index names being used
        "missing_indexes": [str],  # Suggested indexes
        "index_effectiveness": str  # "good", "fair", "poor"
    },
    "warnings": [str],  # General warnings about the plan
    "error": str  # Error message if execution failed
}
```

### Error Handling
- `permission_error`: No permission to execute EXPLAIN
- `timeout_error`: Query plan analysis timed out
- `syntax_error`: SQL syntax invalid for EXPLAIN
- `unsupported_error`: Database doesn't support EXPLAIN

### Caching Strategy
- Cache key: `tool_name + hashed_sql_query + database + schema`
- Cache TTL: 30 minutes (plans can change with data updates)
- Cache invalidation: Manual invalidation when table schema changes

## 2. check_table_conflicts Tool

### Purpose
Detect potential table conflicts and duplicate data structures within the same namespace/database.

### Input Parameters
- `table_name`: Table to check for conflicts
- `catalog`: Database catalog (optional)
- `database`: Database name (optional)
- `schema`: Schema name (optional)

### Implementation Strategy
1. Query metadata store for tables with similar names or structures
2. Compare column definitions and data types
3. Check for semantic similarity in table purposes
4. Identify potential duplicate data or conflicting business logic

### Output Schema
```python
{
    "success": bool,
    "exists_similar": bool,
    "target_table": {
        "name": str,
        "columns": [str],  # Column names
        "ddl_hash": str,   # Hash of DDL for comparison
        "estimated_rows": int
    },
    "matches": [
        {
            "table_name": str,
            "similarity_score": float,  # 0.0 to 1.0
            "conflict_type": str,  # "duplicate", "similar_business", "structural_overlap"
            "matching_columns": [str],  # Columns that match
            "column_similarity": float,  # Column structure similarity
            "business_conflict": str,   # Description of business logic conflict
            "recommendation": str  # Action recommendation
        }
    ],
    "duplicate_build_risk": str,  # "low", "medium", "high"
    "layering_violations": [str],  # Num warehouse layering issues
    "error": str
}
```

### Conflict Detection Heuristics
1. **Name Similarity**: Levenshtein distance < 3 for table names
2. **Structure Similarity**: Column count and type matches > 80%
3. **Business Logic**: Check for same column patterns (user_id, create_time, etc.)
4. **DDL Hash Comparison**: Exact DDL duplicates

### Error Handling
- `metadata_access_error`: Cannot access metadata store
- `table_not_found`: Target table doesn't exist
- `parsing_error`: Cannot parse table DDL

### Caching Strategy
- Cache key: `tool_name + table_name + database + schema`
- Cache TTL: 1 hour (table structures don't change frequently)
- Cache invalidation: When tables are created/dropped/altered

## 3. validate_partitioning Tool

### Purpose
Validate table partitioning strategy and provide optimization recommendations.

### Input Parameters
- `table_name`: Table to validate partitioning
- `catalog`: Database catalog (optional)
- `database`: Database name (optional)
- `schema`: Schema name (optional)

### Implementation Strategy
1. Extract partitioning information from table DDL or metadata
2. Analyze partition key selection
3. Validate partition granularity
4. Check for partition pruning opportunities
5. Assess data distribution across partitions

### Output Schema
```python
{
    "success": bool,
    "partitioned": bool,  # Whether table is partitioned
    "partition_info": {
        "partition_key": str,      # Partition column name
        "partition_type": str,     # "range", "list", "hash", "time_based"
        "partition_count": int,    # Number of partitions
        "partition_expression": str # Raw partition expression
    },
    "validation_results": {
        "partition_key_valid": bool,    # Is partition key appropriate?
        "granularity_appropriate": bool, # Is partition size reasonable?
        "data_distribution_even": bool,  # Are partitions evenly distributed?
        "pruning_opportunities": bool    # Can queries benefit from partition pruning?
    },
    "issues": [
        {
            "severity": str,  # "low", "medium", "high"
            "issue_type": str, # "no_partitioning", "poor_key_choice", "uneven_distribution"
            "description": str,
            "recommendation": str
        }
    ],
    "recommended_partition": {
        "suggested_key": str,      # Recommended partition column
        "suggested_type": str,     # Recommended partition type
        "estimated_partitions": int, # Recommended partition count
        "rationale": str           # Why this partitioning is better
    },
    "performance_impact": {
        "query_speed_improvement": str,  # "significant", "moderate", "minimal"
        "storage_efficiency": str,       # "improved", "neutral", "worse"
        "maintenance_overhead": str      # "acceptable", "high", "excessive"
    },
    "error": str
}
```

### Partitioning Rules
1. **Time-series Data**: Should partition by date/timestamp columns
2. **Granularity**: Daily/monthly partitions for most use cases
3. **Key Selection**: Choose columns frequently used in WHERE clauses
4. **Distribution**: Avoid data skew across partitions

### Error Handling
- `ddl_access_error`: Cannot retrieve table DDL
- `parsing_error`: Cannot parse partitioning information
- `unsupported_partitioning`: Database doesn't support partitioning

### Caching Strategy
- Cache key: `tool_name + table_name + database + schema`
- Cache TTL: 2 hours (partitioning changes infrequently)
- Cache invalidation: When table is altered or re-partitioned

## Common Design Patterns

### Cache Key Normalization
All tools follow the same cache key pattern:
```python
cache_key = f"{tool_name}:{namespace}:{database}:{schema}:{table_name}:{hash(params)}"
```

### Error Classification
Standard error types across all tools:
- `permission_error`: Access denied
- `timeout_error`: Operation timed out
- `parsing_error`: Cannot parse results
- `connection_error`: Database connection issues
- `unsupported_error`: Feature not supported

### Structured Output Format
All tools return consistent structure:
```python
{
    "success": bool,
    "error": str,  # Only present if success=False
    # Tool-specific fields...
}
```

### Integration with Existing Architecture
- Tools integrate with existing `QueryCache` and `ExecutionMonitor`
- Follow same event emission pattern for SSE streaming
- Support batch processing where applicable
- Maintain backwards compatibility with existing preflight sequence
