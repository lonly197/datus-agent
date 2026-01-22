# Schema Retrieval Test Script

## 概述

`tests/integration/test_schema_retrieval_script.py` 是一个CLI测试脚本，用于验证迁移后的schema检索功能是否正常工作。它模拟了text2sql工作流中的schema discovery阶段，检查RAG和Keyword检索是否能正确找到所需的表和字段信息。

## 测试目标

1. ✅ 验证迁移是否成功将DDL元数据导入向量库
2. ✅ 检查RAG（语义）检索功能
3. ✅ 检查Keyword（关键词）检索功能
4. ✅ 验证表和字段的comment信息是否正确提取
5. ✅ 测试"销售线索"等业务场景的检索效果

## 使用场景

- **生产环境StarRocks数仓**：连接生产环境数据库
- **建表语句来源**：`ods_ddl.sql`, `dws_ddl.sql` 等
- **测试场景**：销售线索分析（text2sql任务）
- **测试范围**：仅schema检索阶段，无需完整text2sql流程

## 使用方法

### 基本用法

```bash
# 测试带命名空间的配置
python tests/integration/test_schema_retrieval_script.py --config=conf/agent.yml --namespace=production

# 测试不带命名空间的配置
python tests/integration/test_schema_retrieval_script.py --config=conf/agent.yml

# 自定义测试查询
python tests/integration/test_schema_retrieval_script.py --config=conf/agent.yml --namespace=production --query="客户转化"
```

### 参数说明

| 参数 | 必需 | 说明 | 默认值 |
|------|------|------|--------|
| `--config` | ✅ | agent.yml配置文件路径 | - |
| `--namespace` | ❌ | 数据库命名空间 | None |
| `--query` | ❌ | 测试查询关键词 | "销售线索" |
| `--top-n` | ❌ | 返回结果数量 | 20 |

### 示例命令

```bash
# 1. 测试生产环境（使用命名空间）
python tests/integration/test_schema_retrieval_script.py \
    --config=/Users/lonlyhuang/workspace/myway/Examples/datus-docs/conf/agent.yml \
    --namespace=starrocks_prod

# 2. 测试开发环境
python tests/integration/test_schema_retrieval_script.py \
    --config=conf/agent.dev.yml \
    --namespace=dev_db

# 3. 测试特定业务场景
python tests/integration/test_schema_retrieval_script.py \
    --config=conf/agent.yml \
    --namespace=analytics \
    --query="销售转化率" \
    --top-n=30
```

## 测试内容

### Test 1: Database Connection
检查向量库连接是否正常
- ✅ 向量库可访问
- ✅ schema_metadata表存在

### Test 2: Migration Success
验证迁移是否成功
- ✅ 总记录数统计
- ✅ v0（legacy）记录数
- ✅ v1（enhanced）记录数
- ✅ 迁移进度百分比

### Test 3: Comment Extraction
检查comment信息提取
- ✅ 表注释（table_comment）
- ✅ 列注释（column_comments）
- ✅ 业务标签（business_tags）
- ✅ 提取率统计

### Test 4: Keyword Search
测试关键词检索
- ✅ 基于表名匹配
- ✅ 基于注释匹配
- ✅ 基于标签匹配
- ✅ 相关性评分

### Test 5: RAG (Semantic) Search
测试语义检索
- ✅ 向量相似度搜索
- ✅ 语义相关性排序
- ✅ Top-N结果返回

### Test 6: Sales Lead Scenario
测试销售线索业务场景
- ✅ "销售线索" 查询
- ✅ "线索" 查询
- ✅ "销售" 查询
- ✅ "客户" 查询
- ✅ "潜在客户" 查询
- ✅ "商机" 查询

## 输出示例

### 成功场景

```
================================================================================
Schema Retrieval Test Suite
================================================================================

Test 1: Database Connection
--------------------------------------------------------------------------------
✅ Vector database accessible
   Tables: ['schema_metadata', 'schema_value_storage']

Test 2: Migration Success
--------------------------------------------------------------------------------
┌─────────────────────┬───────────┐
│ Metric              │ Value     │
├─────────────────────┼───────────┤
│ Total Records       │ 1250      │
│ v0 (Legacy) Records│ 0         │
│ v1 (Enhanced) Records│ 1250     │
│ Migration Progress   │ 100.0%    │
└─────────────────────┴───────────┘

✅ Migration successful - v1 records found

Test 3: Comment Information Extraction
--------------------------------------------------------------------------------
┌──────────────────────────────────────────────────────────────────────────┐
│ Comment Extraction Results (10 samples)                                │
├──────────────────────────────────────────────────────────────────────────┤
│ Table Name        │ Table Comment        │ Has Column Comments │ Business Tags │
├──────────────────┼─────────────────────┼─────────────────────┼────────────────┤
│ ods_leads        │ 销售线索原始数据表   │ ✅                   │ sales, ods     │
│ dws_customer     │ 客户维度表         │ ✅                   │ customer, dws  │
│ ...              │ ...                │ ...                 │ ...            │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│ Comment Extraction Summary                                              │
│ Tables with Table Comments: 1180/1250 (94.4%)                          │
│ Tables with Column Comments: 1150/1250 (92.0%)                         │
│ Tables with Business Tags: 1250/1250 (100.0%)                          │
└──────────────────────────────────────────────────────────────────────────┘

✅ Comment extraction successful

Test 4: Keyword Search - '销售线索'
--------------------------------------------------------------------------------
┌──────────────────────────────────────────────────────────────────────────┐
│ Keyword Search Results (15 matches)                                     │
├──────────────────────────────────────────────────────────────────────────┤
│ Table Name        │ Relevance │ Comment              │ Tags             │
├──────────────────┼───────────┼──────────────────────┼──────────────────┤
│ ods_leads        │ 15.00     │ 销售线索原始数据表   │ sales, ods      │
│ dws_leads        │ 12.00     │ 销售线索维度表      │ sales, dws      │
│ ...              │ ...       │ ...                  │ ...             │
└──────────────────────────────────────────────────────────────────────────┘

✅ Keyword search found 15 tables

Test 5: RAG (Semantic) Search - '销售线索'
--------------------------------------------------------------------------------
┌──────────────────────────────────────────────────────────────────────────┐
│ RAG Search Results (20 results)                                        │
├──────────────────────────────────────────────────────────────────────────┤
│ Table Name        │ Similarity │ Table Comment        │ Business Tags    │
├──────────────────┼────────────┼──────────────────────┼──────────────────┤
│ ods_leads        │ 95.2%      │ 销售线索原始数据表   │ sales, ods       │
│ dws_leads        │ 92.8%      │ 销售线索维度表      │ sales, dws       │
│ fact_sales       │ 88.5%      │ 销售事实表          │ sales, fact      │
│ ...              │ ...        │ ...                 │ ...              │
└──────────────────────────────────────────────────────────────────────────┘

✅ RAG search returned 20 results

Test 6: Sales Lead Analysis Scenario
--------------------------------------------------------------------------------

Testing query: 销售线索
  ✅ Found results for '销售线索'

Testing query: 线索
  ✅ Found results for '线索'

Testing query: 销售
  ✅ Found results for '销售'

Testing query: 客户
  ✅ Found results for '客户'

Testing query: 潜在客户
  ✅ Found results for '潜在客户'

Testing query: 商机
  ✅ Found results for '商机'

================================================================================
Test Summary
================================================================================
┌──────────────────────┬───────────┐
│ Test                │ Status    │
├──────────────────────┼───────────┤
│ Database            │ ✅ PASS   │
│ Migration           │ ✅ PASS   │
│ Comments            │ ✅ PASS   │
│ Keyword             │ ✅ PASS   │
│ Rag                 │ ✅ PASS   │
│ Scenario            │ ✅ PASS   │
└──────────────────────┴───────────┘

✅ ALL TESTS PASSED
Schema retrieval is working correctly!
```

### 失败场景

```
Test 3: Comment Information Extraction
--------------------------------------------------------------------------------
┌──────────────────────────────────────────────────────────────────────────┐
│ Comment Extraction Results (10 samples)                                 │
├──────────────────────────────────────────────────────────────────────────┤
│ Table Name        │ Table Comment        │ Has Column Comments │ Business Tags │
├──────────────────┼─────────────────────┼─────────────────────┼────────────────┤
│ table_1          │ [dim]None[/dim]     │ ❌                   │ [dim]None[/dim]│
│ table_2          │ [dim]None[/dim]     │ ❌                   │ [dim]None[/dim]│
└──────────────────────────────────────────────────────────────────────────┘

⚠️ Low comment extraction rate

❌ Comment extraction test failed
```

## 退出码

- `0`: 所有测试通过
- `1`: 有一个或多个测试失败
- `130`: 用户中断测试

## 常见问题

### Q1: 测试失败，提示"Database connection failed"

**A**: 检查以下几点：
1. 确认agent.yml配置文件正确
2. 确认向量库路径可访问
3. 确认已运行迁移脚本：`migrate_v0_to_v1.py`

### Q2: 迁移测试失败，提示"No v1 records"

**A**: 需要先运行迁移脚本：
```bash
python -m datus.storage.schema_metadata.migrate_v0_to_v1 \
    --config=conf/agent.yml \
    --namespace=your_namespace \
    --import-schemas \
    --force
```

### Q3: RAG搜索返回结果很少

**A**: 可能的原因：
1. 向量嵌入未生成（重新导入schema）
2. 测试查询与实际数据不匹配
3. 相似度阈值过高

### Q4: 注释提取率很低

**A**: 检查以下几点：
1. 源DDL文件是否包含COMMENT
2. 迁移脚本是否正确解析COMMENT
3. 数据库类型是否支持COMMENT提取

## 与生产环境的集成

### 步骤1：确认配置文件

确认agent.yml包含正确的StarRocks连接信息：

```yaml
agent:
  storage:
    base_path: /root/.datus/data

  namespace:
    starrocks_prod:
      name: analytics
      type: starrocks
      host: prod-starrocks.company.com
      port: 9030
      username: ${STARROCKS_USER}
      password: ${STARROCKS_PASSWORD}
      database: analytics_db
      catalog: ""
```

### 步骤2：运行迁移

```bash
python -m datus.storage.schema_metadata.migrate_v0_to_v1 \
    --config=conf/agent.prod.yml \
    --namespace=starrocks_prod \
    --extract-relationships=true \
    --import-schemas \
    --force
```

### 步骤3：测试检索功能

```bash
python tests/integration/test_schema_retrieval_script.py \
    --config=conf/agent.prod.yml \
    --namespace=starrocks_prod
```

### 步骤4：验证业务场景

```bash
# 测试销售线索相关查询
python tests/integration/test_schema_retrieval_script.py \
    --config=conf/agent.prod.yml \
    --namespace=starrocks_prod \
    --query="销售线索转化"
```

## 验证指标

### 关键指标

| 指标 | 期望值 | 说明 |
|------|--------|------|
| Migration Progress | 100% | 所有记录都应升级到v1 |
| Table Comment Rate | >80% | 80%以上的表应有注释 |
| Column Comment Rate | >70% | 70%以上的表应有列注释 |
| Business Tags Rate | >90% | 90%以上的表应有业务标签 |
| Keyword Search Results | >5 | 关键词搜索应返回足够结果 |
| RAG Search Results | >10 | RAG搜索应返回足够结果 |

### 业务验证

对于"销售线索"业务场景，应能检索到：
- ✅ ODS层：销售线索原始数据表
- ✅ DWS层：销售线索汇总表
- ✅ 客户相关表
- ✅ 转化相关表
- ✅ 商机相关表

## 故障排除

如果测试失败，参考以下步骤：

### 1. 检查向量库状态

```bash
python -c "
from datus.configuration.agent_config_loader import load_agent_config
from datus.storage.schema_metadata import SchemaStorage

config = load_agent_config('conf/agent.yml')
storage = SchemaStorage(db_path=config.rag_storage_path())
print(f'Table count: {len(storage._search_all(where=None))}')
"
```

### 2. 检查迁移状态

```bash
python -c "
from datus.storage.schema_metadata.migrate_v0_to_v1 import report_migration_state
report_migration_state('/root/.datus/data/lancedb')
"
```

### 3. 手动验证RAG搜索

```bash
python -c "
from datus.configuration.agent_config_loader import load_agent_config
from datus.storage.schema_metadata import SchemaWithValueRAG

config = load_agent_config('conf/agent.yml')
config.current_namespace = 'your_namespace'
storage = SchemaWithValueRAG(config)

results = storage.search_similar('销售线索', top_n=10)
print(f'Found {len(results)} results')
for r in results:
    print(f'  - {r.get(\"table_name\")}: {r.get(\"table_comment\")}')
"
```

## 总结

该测试脚本提供了全面的schema检索功能验证，包括：

1. ✅ **连接验证**：确保向量库可访问
2. ✅ **迁移验证**：确认DDL元数据已正确导入
3. ✅ **提取验证**：检查comment和标签信息
4. ✅ **检索验证**：测试RAG和Keyword检索
5. ✅ **场景验证**：测试业务相关的查询场景

通过这些测试，可以确保迁移后的schema检索功能正常工作，为text2sql任务提供准确的基础数据。
