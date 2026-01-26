# Enhanced SQL Review cURL Examples

This document provides practical cURL examples for using the enhanced SQL review functionality.

## Prerequisites

1. Datus API server running on `http://localhost:8000`
2. Proper namespace and database configuration
3. Tables imported into the knowledge base

## Basic SQL Review

### Simple SELECT Query Review

```bash
curl -X POST "http://localhost:8000/workflows/chat_research" \
  -H "Accept: text/event-stream" \
  -H "Content-Type: application/json" \
  -d '{
    "namespace": "prod",
    "task": "请审查这个查询SQL：SELECT * FROM users WHERE status = '\''active'\''",
    "database_name": "user_db",
    "plan_mode": true
  }'
```

**Expected Response**: SSE stream with enhanced preflight analysis including query plan, table conflicts, and partitioning validation.

### JOIN Query Performance Analysis

```bash
curl -X POST "http://localhost:8000/workflows/chat_research" \
  -H "Accept: text/event-stream" \
  -H "Content-Type: application/json" \
  -d '{
    "namespace": "prod",
    "catalog_name": "default_catalog",
    "database_name": "ecommerce",
    "schema_name": "public",
    "task": "性能审查：SELECT u.user_name, COUNT(o.order_id) as order_count FROM users u LEFT JOIN orders o ON u.user_id = o.user_id WHERE u.created_at >= '\''2024-01-01'\'' GROUP BY u.user_id, u.user_name HAVING COUNT(o.order_id) > 5",
    "ext_knowledge": "这是一个用户活跃度分析查询，需要确保JOIN性能和分组效率",
    "plan_mode": true
  }'
```

## Advanced Scenarios

### Data Warehouse Layering Compliance Check

```bash
curl -X POST "http://localhost:8000/workflows/chat_research" \
  -H "Accept: text/event-stream" \
  -H "Content-Type: application/json" \
  -d '{
    "namespace": "dwh",
    "database_name": "warehouse",
    "task": "审查DWS层汇总表创建SQL是否符合数仓规范：CREATE TABLE dws_user_behavior_agg (user_id BIGINT, behavior_date DATE, page_views BIGINT, session_duration BIGINT) PARTITION BY (behavior_date) DISTRIBUTED BY HASH(user_id) BUCKETS 32 PROPERTIES('\''replication_num'\''='\''1'\'') AS SELECT user_id, behavior_date, COUNT(*) as page_views, SUM(duration) as session_duration FROM dwd_user_behavior WHERE behavior_date >= '\''2024-01-01'\'' GROUP BY user_id, behavior_date",
    "domain": "data_warehouse",
    "layer1": "dws",
    "layer2": "user_analysis",
    "plan_mode": true,
    "auto_execute_plan": true
  }'
```

### Partitioning Strategy Validation

```bash
curl -X POST "http://localhost:8000/workflows/chat_research" \
  -H "Accept: text/event-stream" \
  -H "Content-Type: application/json" \
  -d '{
    "namespace": "prod",
    "database_name": "analytics",
    "task": "分区策略审查：SELECT event_type, COUNT(*) as event_count FROM user_events WHERE event_time >= '\''2024-01-15 00:00:00'\'' AND event_time < '\''2024-01-16 00:00:00'\'' GROUP BY event_type ORDER BY event_count DESC",
    "current_date": "2024-01-15",
    "ext_knowledge": "user_events表按event_time每日分区，需要验证分区裁剪是否生效",
    "plan_mode": true
  }'
```

### Table Conflict Detection

```bash
curl -X POST "http://localhost:8000/workflows/chat_research" \
  -H "Accept: text/event-stream" \
  -H "Content-Type: application/json" \
  -d '{
    "namespace": "prod",
    "database_name": "crm",
    "task": "重复建设检查：创建新的客户信息汇总表，检查是否与现有表重复：CREATE TABLE customer_summary_new (customer_id BIGINT, customer_name STRING, total_orders BIGINT, last_order_date DATE) DISTRIBUTED BY HASH(customer_id) BUCKETS 16 AS SELECT c.customer_id, c.customer_name, COUNT(o.order_id) as total_orders, MAX(o.order_date) as last_order_date FROM customers c LEFT JOIN orders o ON c.customer_id = o.customer_id GROUP BY c.customer_id, c.customer_name",
    "domain": "customer_analytics",
    "layer1": "summary",
    "layer2": "customer_behavior",
    "plan_mode": true
  }'
```

## Error Handling Examples

### Handling Table Not Found

```bash
curl -X POST "http://localhost:8000/workflows/chat_research" \
  -H "Accept: text/event-stream" \
  -H "Content-Type: application/json" \
  -d '{
    "namespace": "prod",
    "task": "尝试审查不存在的表：SELECT * FROM nonexistent_table",
    "database_name": "test_db",
    "plan_mode": true
  }'
```

**Expected**: Graceful error handling with suggestions for table name correction.

### Timeout Handling

```bash
curl -X POST "http://localhost:8000/workflows/chat_research" \
  -H "Accept: text/event-stream" \
  -H "Content-Type: application/json" \
  -d '{
    "namespace": "prod",
    "task": "复杂查询性能分析：WITH RECURSIVE complex_cte AS (SELECT * FROM large_table UNION ALL SELECT * FROM complex_cte WHERE id > 1000000) SELECT * FROM complex_cte",
    "database_name": "analytics",
    "plan_mode": true,
    "tool_timeout_seconds": 30
  }'
```

## Real-time Event Monitoring

### Monitor Preflight Progress

```bash
# Save response to file for analysis
curl -X POST "http://localhost:8000/workflows/chat_research" \
  -H "Accept: text/event-stream" \
  -H "Content-Type: application/json" \
  -d '{
    "namespace": "prod",
    "task": "SELECT COUNT(*) FROM sales WHERE sale_date >= '\''2024-01-01'\''",
    "database_name": "ecommerce",
    "plan_mode": true
  }' > review_events.txt

# Parse events
grep "event:" review_events.txt
```

### Extract Tool Results

```bash
# Extract analyze_query_plan results
curl -X POST "http://localhost:8000/workflows/chat_research" \
  -H "Accept: text/event-stream" \
  -H "Content-Type: application/json" \
  -d '{
    "namespace": "prod",
    "task": "性能分析：SELECT * FROM products p JOIN categories c ON p.category_id = c.id WHERE p.price > 100",
    "database_name": "catalog",
    "plan_mode": true
  }' | grep "tool_call_result" | jq '.data'
```

## Batch Processing Examples

### Multiple SQL Reviews

```bash
#!/bin/bash
# Batch review multiple SQL files

SQL_FILES=("query1.sql" "query2.sql" "query3.sql")

for sql_file in "${SQL_FILES[@]}"; do
  echo "Reviewing $sql_file..."
  sql_content=$(cat "$sql_file")

  curl -X POST "http://localhost:8000/workflows/chat_research" \
    -H "Accept: text/event-stream" \
    -H "Content-Type: application/json" \
    -d "{
      \"namespace\": \"prod\",
      \"task\": \"审查SQL文件 $sql_file：$sql_content\",
      \"database_name\": \"production\",
      \"plan_mode\": true
    }" > "review_$sql_file.txt" &

  # Limit concurrent requests
  sleep 2
done

wait
echo "All reviews completed"
```

## Integration Examples

### CI/CD Pipeline Integration

```yaml
# .github/workflows/sql-review.yml
name: SQL Review
on:
  pull_request:
    paths:
      - 'sql/**'
      - 'queries/**'

jobs:
  sql-review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Review SQL Changes
        run: |
          for sql_file in $(git diff --name-only HEAD~1 | grep '\.sql$'); do
            echo "Reviewing $sql_file..."
            sql_content=$(cat "$sql_file")

            curl -X POST "${{ secrets.DATUS_API_URL }}/workflows/chat_research" \
              -H "Accept: text/event-stream" \
              -H "Authorization: Bearer ${{ secrets.DATUS_API_TOKEN }}" \
              -H "Content-Type: application/json" \
              -d "{
                \"namespace\": \"prod\",
                \"task\": \"PR SQL审查：$sql_content\",
                \"database_name\": \"production\",
                \"plan_mode\": true
              }" | tee review_output.txt

            # Check for critical issues
            if grep -q '"severity": "high"' review_output.txt; then
              echo "High severity issues found in $sql_file"
              exit 1
            fi
          done
```

### Database Migration Review

```bash
#!/bin/bash
# Review database migration scripts

MIGRATION_FILE="migrations/001_add_user_preferences.sql"

echo "Reviewing migration: $MIGRATION_FILE"

# Extract DDL statements
ddl_statements=$(grep -E "(CREATE|ALTER|DROP)" "$MIGRATION_FILE")

curl -X POST "http://localhost:8000/workflows/chat_research" \
  -H "Accept: text/event-stream" \
  -H "Content-Type: application/json" \
  -d "{
    \"namespace\": \"prod\",
    \"task\": \"数据库迁移审查：$ddl_statements\",
    \"database_name\": \"production\",
    \"domain\": \"database_migration\",
    \"layer1\": \"schema_changes\",
    \"layer2\": \"user_preferences\",
    \"plan_mode\": true,
    \"auto_execute_plan\": true
  }" > migration_review.txt

# Check review results
if grep -q '"duplicate_build_risk": "high"' migration_review.txt; then
  echo "⚠️  High risk of duplicate data structures detected"
  exit 1
fi

if grep -q '"layering_violations"' migration_review.txt; then
  echo "⚠️  Data warehouse layering violations detected"
  exit 1
fi

echo "✅ Migration review passed"
```

## Performance Testing

### Benchmark Enhanced Tools

```bash
#!/bin/bash
# Performance test script

echo "Benchmarking enhanced SQL review tools..."

# Test queries of different complexities
TEST_QUERIES=(
  "SELECT COUNT(*) FROM users"
  "SELECT u.name, COUNT(o.id) FROM users u JOIN orders o ON u.id = o.user_id GROUP BY u.id, u.name"
  "SELECT * FROM events WHERE created_at >= '2024-01-01' AND event_type IN ('click', 'view', 'purchase')"
)

for query in "${TEST_QUERIES[@]}"; do
  echo "Testing query: $query"

  start_time=$(date +%s.%3N)

  curl -X POST "http://localhost:8000/workflows/chat_research" \
    -H "Accept: text/event-stream" \
    -H "Content-Type: application/json" \
    -w "%{time_total}" \
    -o /dev/null \
    -d "{
      \"namespace\": \"benchmark\",
      \"task\": \"性能测试：$query\",
      \"database_name\": \"test_db\",
      \"plan_mode\": true
    }"

  end_time=$(date +%s.%3N)
  duration=$(echo "$end_time - $start_time" | bc)

  echo "Query review completed in ${duration}s"
done

echo "Benchmark complete"
```

## Monitoring and Metrics

### Check Tool Performance Metrics

```bash
# Query execution monitor metrics
curl -X GET "http://localhost:8000/metrics" | jq '.enhanced_tools'

# Expected output:
# {
#   "analyze_query_plan": {
#     "calls": 150,
#     "successes": 145,
#     "avg_time": 0.85,
#     "cache_hits": 45
#   },
#   "check_table_conflicts": {
#     "calls": 89,
#     "successes": 87,
#     "avg_time": 1.2,
#     "cache_hits": 23
#   },
#   "validate_partitioning": {
#     "calls": 67,
#     "successes": 65,
#     "avg_time": 0.95,
#     "cache_hits": 18
#   }
# }
```

### Cache Performance Analysis

```bash
# Analyze cache effectiveness
curl -X POST "http://localhost:8000/workflows/chat_research" \
  -H "Accept: text/event-stream" \
  -H "Content-Type: application/json" \
  -d '{
    "namespace": "prod",
    "task": "缓存性能测试：SELECT * FROM users WHERE created_at >= '\''2024-01-01'\''",
    "database_name": "production",
    "plan_mode": true
  }' 2>&1 | grep -E "(cache_hit|cache_miss|execution_time)"
```

These examples demonstrate the full range of enhanced SQL review capabilities, from basic query analysis to complex CI/CD integration and performance monitoring.
