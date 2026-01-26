# Reference SQL Intelligence

## Overview

Bootstrap-KB Reference SQL processes `.sql` files, analyzes comment-SQL pairs, and indexes them into a searchable knowledge base. It turns raw SQL files into structured, searchable reference entries backed by semantic search.

## Core Value

### What Problem Does It Solve?

- **SQL Knowledge Silos**: SQL queries scattered across files without organization
- **SQL Reusability**: Difficulty finding existing queries for similar needs
- **Query Discovery**: No efficient way to search SQL by business intent
- **Knowledge Management**: SQL expertise locked in individual developers' minds

### What Value Does It Provide?

- **Intelligent Organization**: Automatically categorizes and classifies SQL queries
- **Semantic Search**: Find SQL queries using natural language descriptions
- **Knowledge Preservation**: Captures SQL expertise in a searchable format
- **Query Reusability**: Easily discover and reuse existing SQL patterns

## Usage

### Basic Command

```bash
# Initialize reference SQL component
datus-agent bootstrap-kb \
    --namespace your_namespace \
    --components reference_sql \
    --sql_dir /path/to/sql/directory \
    --kb_update_strategy overwrite
```

### Key Parameters

| Parameter | Required | Description | Example |
|-----------|----------|-------------|---------|
| `--namespace` | ✅ | Database namespace | `analytics_db` |
| `--components` | ✅ | Components to initialize | `reference_sql` |
| `--sql_dir` | ⚠️ | Directory containing SQL files (required to ingest) | `/sql/queries` |
| `--kb_update_strategy` | ✅ | Update strategy | `overwrite`/`incremental` |
| `--validate-only` | ❌ | Only validate, don't store | `true`/`false` |
| `--pool_size` | ❌ | Concurrent processing threads | `8` |
| `--subject_tree` | ❌ | Predefined subject taxonomy (comma-separated) | `Sales/Reporting/Daily,Sales/Analytics/Trends` |

## SQL File Format

### Expected Format

SQL files should use `--` comment blocks immediately before each query:

```sql
-- Daily active users count
-- Count unique users who logged in each day
SELECT
    DATE(created_at) as activity_date,
    COUNT(DISTINCT user_id) as daily_active_users
FROM user_activity
WHERE created_at >= '2025-01-01'
GROUP BY DATE(created_at)
ORDER BY activity_date;

-- Monthly revenue summary
-- Total revenue grouped by month and category
SELECT
    DATE_TRUNC('month', order_date) as month,
    category,
    SUM(amount) as total_revenue,
    COUNT(*) as order_count
FROM orders
WHERE order_date >= '2025-01-01'
GROUP BY DATE_TRUNC('month', order_date), category
ORDER BY month, total_revenue DESC;
```

### Processing Rules

- Only `SELECT` queries are indexed (DDL/DML are skipped).
- Only `*.sql` files in the given directory are scanned (non-recursive).
- Comment/SQL pairs are separated by semicolons that appear outside `--` comments.
- Parameter placeholders like `#param#`, `:param`, `@param`, and `${param}` are normalized for validation.
- Empty comments are allowed; they are stored as empty strings.

## Advanced Features

### 1. Multi-Dialect SQL Validation

Validation uses `sqlglot` with multiple dialects:

- **MySQL**: Standard MySQL syntax
- **Hive**: Hadoop Hive SQL dialect
- **Spark**: Apache Spark SQL syntax

### 2. Intelligent Classification

SqlSummaryAgenticNode generates `name`, `summary`, `subject_path`, and `tags` for each SQL entry:

```json
{
    "name": "daily_active_users",
    "subject_path": ["Analytics", "User Analytics", "Activity Metrics"],
    "tags": "daily,users,engagement"
}
```

### 3. Vector Search Capabilities

- **Semantic Search**: Vector search over `search_text` (derived from name + summary)
- **Full-text Index**: FTS index over SQL/name/comment/summary/tags for fast filtering
- **Subject Filtering**: Limit results to a subject path in the taxonomy (see [Subject Tree](subject_tree.md))

### 4. Incremental Updates

- **Incremental Mode**: Add new queries to existing index
- **Overwrite Mode**: Complete rebuild of the index

## Stored Fields

Reference SQL entries are stored with the following fields:

- `id`: Hash of SQL + comment (used for incremental de-duplication)
- `name`: LLM-generated title
- `sql`: Cleaned SQL statement
- `comment`: Combined `--` comment text (may be empty)
- `summary`: LLM-generated summary
- `search_text`: Embedding source text
- `subject_path`: Subject taxonomy path (resolved into `subject_node_id` in LanceDB; tree persisted in `subject_tree.db`)
- `tags`: LLM-generated tags
- `filepath`: Source `.sql` file path

## Best Practices

### 1. File Organization

```
/sql_queries/
├── user_analytics.sql
├── financial_reports.sql
├── product_metrics.sql
└── system_monitoring.sql
```

### 2. Comment Standards

```sql
-- Clear, descriptive title
-- Detailed business context and purpose
-- Important assumptions or business rules
SELECT
    column1,
    column2
FROM table_name
WHERE conditions;
```

### 3. Performance Optimization

```bash
# High-performance processing
datus-agent bootstrap-kb \
    --namespace your_db \
    --components reference_sql \
    --sql_dir /large_sql_directory \
    --kb_update_strategy incremental \
    --pool_size 16

# Validation only (fast check)
datus-agent bootstrap-kb \
    --namespace your_db \
    --components reference_sql \
    --sql_dir /new_sql_files \
    --validate-only
```

### 4. Maintenance Strategy

- **Regular Updates**: Add new SQL files incrementally
- **Quality Checks**: Use validate-only mode for new files
- **Index Optimization**: Periodic full rebuild for large updates

## Usage Examples

### Initial Setup

```bash
# First time setup with complete SQL directory
datus-agent bootstrap-kb \
    --namespace production_db \
    --components reference_sql \
    --sql_dir /company/sql_repository \
    --kb_update_strategy overwrite \
    --pool_size 8
```

### Adding New Queries

```bash
# Add new SQL files incrementally
datus-agent bootstrap-kb \
    --namespace production_db \
    --components reference_sql \
    --sql_dir /new_sql_queries \
    --kb_update_strategy incremental
```

### Validation

```bash
# Validate SQL files before processing
datus-agent bootstrap-kb \
    --namespace production_db \
    --components reference_sql \
    --sql_dir /untested_queries \
    --validate-only
```

## Summary

The Bootstrap-KB Reference SQL component transforms scattered SQL files into an intelligent, searchable knowledge base. It combines LLM-based summarization with robust SQL processing to create a powerful tool for SQL discovery and reuse.

By implementing Reference SQL, teams can break down knowledge silos and build a collective SQL intelligence asset that grows over time.
