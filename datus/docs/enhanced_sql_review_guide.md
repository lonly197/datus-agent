# Enhanced SQL Review Guide

## Overview

The enhanced SQL review functionality provides comprehensive SQL quality assessment through advanced preflight tools that analyze query performance, detect table conflicts, and validate partitioning strategies. This guide explains how to use the enhanced SQL review capabilities.

## New Preflight Tools

### 1. Query Plan Analysis (`analyze_query_plan`)

**Purpose**: Analyzes SQL execution plans to identify performance bottlenecks and optimization opportunities.

**What it does**:
- Executes `EXPLAIN` or `EXPLAIN ANALYZE` queries
- Parses execution plans for different database types (MySQL, PostgreSQL, StarRocks, etc.)
- Identifies performance hotspots and optimization opportunities
- Provides structured recommendations

**Analysis Output**:
```json
{
  "success": true,
  "plan_text": "EXPLAIN output...",
  "estimated_rows": 1000,
  "estimated_cost": 150.5,
  "hotspots": [
    {
      "reason": "full_table_scan",
      "node": "TableScan(test_table)",
      "severity": "high",
      "recommendation": "Add index on id column"
    }
  ],
  "join_analysis": {
    "join_count": 2,
    "join_types": ["hash_join", "nested_loop"],
    "join_order_issues": []
  },
  "index_usage": {
    "indexes_used": ["idx_user_id"],
    "missing_indexes": ["idx_created_at"],
    "index_effectiveness": "fair"
  }
}
```

### 2. Table Conflict Detection (`check_table_conflicts`)

**Purpose**: Detects potential table conflicts and duplicate data structures within the namespace.

**What it does**:
- Searches metadata store for similar table names and structures
- Compares column definitions and data types
- Assesses business logic conflicts
- Evaluates duplicate build risks

**Analysis Output**:
```json
{
  "success": true,
  "exists_similar": true,
  "target_table": {
    "name": "user_orders",
    "columns": ["id", "user_id", "order_date", "amount"],
    "ddl_hash": "abc123def",
    "estimated_rows": 1000000
  },
  "matches": [
    {
      "table_name": "user_orders_backup",
      "similarity_score": 0.95,
      "conflict_type": "duplicate",
      "matching_columns": ["id", "user_id", "order_date", "amount"],
      "business_conflict": "疑似数据备份表，可能存在重复建设",
      "recommendation": "建议删除备份表或明确数据生命周期管理"
    }
  ],
  "duplicate_build_risk": "high",
  "layering_violations": [
    "ODS层表不应与DWS层表直接对应，可能违反分层规范"
  ]
}
```

### 3. Partition Validation (`validate_partitioning`)

**Purpose**: Validates table partitioning strategy and provides optimization recommendations.

**What it does**:
- Analyzes partitioning configuration from table DDL
- Validates partition key selection and granularity
- Assesses data distribution across partitions
- Evaluates query pruning effectiveness

**Analysis Output**:
```json
{
  "success": true,
  "partitioned": true,
  "partition_info": {
    "partition_key": "created_at",
    "partition_type": "time_based",
    "partition_count": 30,
    "partition_expression": "date_trunc('day', created_at)"
  },
  "validation_results": {
    "partition_key_valid": true,
    "granularity_appropriate": true,
    "data_distribution_even": false,
    "pruning_opportunities": true
  },
  "issues": [
    {
      "severity": "medium",
      "issue_type": "uneven_distribution",
      "description": "某些分区数据量偏大，可能影响查询性能",
      "recommendation": "考虑调整分区策略，按月 instead of 按日分区"
    }
  ],
  "recommended_partition": {
    "suggested_key": "created_at",
    "suggested_type": "time_based",
    "estimated_partitions": 12,
    "rationale": "建议按月分区以获得更好的数据分布和查询性能"
  },
  "performance_impact": {
    "query_speed_improvement": "significant",
    "storage_efficiency": "improved",
    "maintenance_overhead": "acceptable"
  }
}
```

## API Usage

### Basic SQL Review Request

```bash
curl -X POST "http://localhost:8000/workflows/chat_research" \
  -H "Accept: text/event-stream" \
  -H "Content-Type: application/json" \
  -d '{
    "namespace": "your_namespace",
    "task": "审查以下SQL：SELECT * FROM user_orders WHERE created_at >= '\''2024-01-01'\''",
    "database_name": "ecommerce",
    "plan_mode": true
  }'
```

### Advanced SQL Review with Context

```bash
curl -X POST "http://localhost:8000/workflows/chat_research" \
  -H "Accept: text/event-stream" \
  -H "Content-Type: application/json" \
  -d '{
    "namespace": "prod",
    "catalog_name": "default_catalog",
    "database_name": "ecommerce",
    "schema_name": "public",
    "task": "请详细审查这个用户订单查询SQL的性能和规范性：SELECT u.user_name, o.order_id, o.amount FROM users u JOIN orders o ON u.user_id = o.user_id WHERE o.created_at >= '\''2024-01-01'\'' ORDER BY o.amount DESC LIMIT 100",
    "ext_knowledge": "这是一个用户订单分析查询，需要确保查询性能和数据准确性",
    "plan_mode": true,
    "auto_execute_plan": true
  }'
```

## Real-time Event Streaming

The enhanced SQL review provides real-time progress updates through Server-Sent Events (SSE). Here's what to expect:

### Event Flow Example

```
1. Plan Update Event
event: plan_update
data: {"id":"plan_001","event":"plan_update","todos":[...]}

2. Tool Call Events
event: tool_call
data: {"id":"tool_001","event":"tool_call","toolCallId":"call_analyze","toolName":"analyze_query_plan"}

3. Tool Result Events
event: tool_call_result
data: {"id":"result_001","event":"tool_call_result","toolCallId":"call_analyze","data":{...}}

4. Chat Response (with enhanced context)
event: chat
data: {"id":"chat_001","event":"chat","content":"基于查询计划分析，发现以下问题：..."}

5. Completion Event
event: complete
data: {"id":"complete_001","event":"complete","content":"SQL审查完成"}
```

### Error Handling Events

```
event: tool_call_result
data: {"toolCallId":"call_analyze","data":{"success":false,"error":"Table not found"}}

event: error
data: {"event":"error","error":"Database table not found","suggestions":[...]}
```

## Enhanced Review Report Structure

The enhanced SQL review generates comprehensive markdown reports with the following structure:

### 📋 审查概览
- 总体评估结果
- 严重问题数量统计
- 基于预检工具的综合评分

### 🔍 审查规则
- 使用的StarRocks规范版本
- 数据仓库分层要求
- 性能优化标准

### 📊 执行计划分析
- 查询性能指标（预估行数、成本、执行时间）
- 性能热点识别
- 索引使用情况评估
- JOIN操作优化建议

### 🏗️ 表结构与分区评估
- 表冲突检测结果
- 分区策略验证
- 数据模型合规性检查

### ⚠️ 发现问题
- 按严重程度排序的问题列表
- 每个问题的详细描述和影响分析
- 具体的修复建议

### 💡 优化建议
- SQL结构优化
- 索引添加建议
- 分区策略调整
- 查询重写建议

### 🛠️ 优化后的SQL
- 提供多个优化版本
- 性能对比分析
- 实施复杂度评估

### 📈 预期效果
- 性能提升量化指标
- 资源使用优化效果
- 业务价值评估

## Configuration

### Required Tool Sequence

The enhanced SQL review automatically includes these tools in the preflight sequence:

```yaml
required_tool_sequence:
  - describe_table          # 表结构分析
  - search_external_knowledge  # StarRocks规则检索
  - read_query              # SQL语法验证
  - get_table_ddl           # DDL定义获取
  - analyze_query_plan      # 查询计划分析 (新增)
  - check_table_conflicts   # 表冲突检测 (新增)
  - validate_partitioning   # 分区验证 (新增)
```

### Cache Configuration

```yaml
plan_hooks:
  enable_query_caching: true
  cache_ttl_seconds:
    analyze_query_plan: 1800      # 30分钟
    check_table_conflicts: 3600   # 1小时
    validate_partitioning: 7200   # 2小时
```

### Monitoring

Enhanced metrics are automatically collected:

```python
# Access via ExecutionMonitor
monitor.metrics["enhanced_tools"]["analyze_query_plan"]
# {
#   "calls": 150,
#   "successes": 145,
#   "avg_time": 0.85,
#   "cache_hits": 45
# }
```

## Best Practices

### 1. When to Use Enhanced Review
- 生产环境SQL上线前
- 性能问题排查
- 数据模型重构
- 新功能开发

### 2. Interpreting Results
- **查询计划分析**: 关注高严重度热点，优先修复
- **表冲突检测**: 中高风险冲突需要业务评审
- **分区验证**: 重点检查数据分布均匀性和查询裁剪效果

### 3. Performance Considerations
- 预检工具会增加响应时间（通常1-3秒）
- 缓存机制减少重复分析开销
- 可以配置超时控制避免长时间等待

### 4. Error Handling
- 单个工具失败不影响整体审查
- 错误信息会通过事件流实时反馈
- 系统会尝试基于可用信息继续分析

## Troubleshooting

### Common Issues

**1. Tool Execution Timeout**
```
Error: Query plan analysis timed out
Solution: 检查数据库连接和查询复杂度，考虑简化SQL或增加超时时间
```

**2. Table Not Found**
```
Error: Target table not found in metadata store
Solution: 确保表已正确导入知识库，或运行metadata bootstrap
```

**3. DDL Parsing Failed**
```
Error: Cannot parse table DDL
Solution: 检查表权限和DDL格式，可能需要手动提供表结构信息
```

### Performance Optimization

**1. Enable Caching**
```yaml
plan_hooks:
  enable_query_caching: true
```

**2. Batch Processing**
系统自动对相同类型的工具调用进行批处理，减少数据库连接开销。

**3. Monitoring**
定期检查监控指标，优化缓存命中率和平均执行时间。

## Examples

### Example 1: Simple Query Review

**Input**:
```sql
SELECT * FROM users WHERE created_at >= '2024-01-01'
```

**Enhanced Analysis Results**:
- ⚠️ Full table scan detected
- 💡 Suggest adding index on `created_at`
- 🏗️ Partitioning validation passed

### Example 2: Complex Join Query

**Input**:
```sql
SELECT u.name, o.amount, p.product_name
FROM users u
JOIN orders o ON u.id = o.user_id
JOIN products p ON o.product_id = p.id
WHERE o.created_at >= '2024-01-01'
```

**Enhanced Analysis Results**:
- 📊 JOIN analysis: 2 hash joins detected
- 🏗️ Table conflict: Potential duplicate user data detected
- ⚠️ Missing composite index recommendation

### Example 3: Partitioned Table Validation

**Input**:
```sql
SELECT * FROM sales_data PARTITION (p202401)
WHERE sale_date >= '2024-01-01'
```

**Enhanced Analysis Results**:
- ✅ Partition pruning effective
- 📊 Query plan shows partition scan only
- 🏗️ Partition distribution analysis passed

This enhanced SQL review functionality provides comprehensive, data-driven SQL quality assessment that goes beyond traditional syntax checking to include performance analysis, architectural validation, and business logic verification.
