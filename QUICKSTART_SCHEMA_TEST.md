# Schema Retrieval Test - Quick Start Guide

## 快速开始

### 1. 确认环境

确保已运行迁移脚本：
```bash
python -m datus.storage.schema_metadata.migrate_v0_to_v1 \
    --config=conf/agent.yml \
    --namespace=your_namespace \
    --import-schemas \
    --force
```

### 2. 运行测试

```bash
# 基本测试（销售线索场景）
python test_schema_retrieval.py --config=conf/agent.yml --namespace=your_namespace

# 测试自定义查询
python test_schema_retrieval.py --config=conf/agent.yml --namespace=your_namespace --query="客户转化"

# 查看帮助
python test_schema_retrieval.py --help
```

### 3. 解读结果

**✅ 通过的测试**：
- 数据库连接
- 迁移成功（v1记录）
- 注释提取率 > 50%
- 关键词搜索有结果
- RAG搜索有结果

**❌ 失败的测试**：
- 数据库连接失败 → 检查配置文件和路径
- 迁移失败 → 重新运行迁移脚本
- 注释提取率低 → 检查源DDL是否包含COMMENT
- 搜索无结果 → 确认数据已导入

### 4. 典型输出

```
Test 1: Database Connection
✅ Vector database accessible

Test 2: Migration Success
Migration Progress: 100.0%
✅ Migration successful - v1 records found

Test 3: Comment Extraction
Tables with Table Comments: 1180/1250 (94.4%)
✅ Comment extraction successful

Test 4: Keyword Search - '销售线索'
✅ Keyword search found 15 tables

Test 5: RAG (Semantic) Search - '销售线索'
✅ RAG search returned 20 results

Test 6: Sales Lead Analysis Scenario
✅ Found results for '销售线索'
✅ Found results for '线索'
✅ ALL TESTS PASSED
```

## 故障排除

### 错误：Database connection failed
```bash
# 检查配置
cat conf/agent.yml | grep -A 10 "namespace:"

# 检查路径权限
ls -la /root/.datus/data/
```

### 错误：Migration failed - no v1 records
```bash
# 重新运行迁移
python -m datus.storage.schema_metadata.migrate_v0_to_v1 \
    --config=conf/agent.yml \
    --namespace=your_namespace \
    --force
```

### 错误：Low comment extraction rate
```bash
# 检查源DDL是否包含COMMENT
grep -i "COMMENT" /path/to/ods_ddl.sql | head -5
```

## 生产环境测试

```bash
# 测试生产环境StarRocks
python test_schema_retrieval.py \
    --config=/Users/lonlyhuang/workspace/myway/Examples/datus-docs/conf/agent.yml \
    --namespace=starrocks_prod \
    --query="销售线索" \
    --top-n=30
```

## 验证要点

1. **迁移成功**：v1记录数 > 0
2. **注释完整**：表注释率 > 80%
3. **检索有效**：RAG和Keyword都有结果
4. **业务相关**：能检索到销售线索相关表

## 下一步

测试通过后，可以进行完整的text2sql测试：
```bash
# 使用Datus-Agent进行实际查询测试
datus-agent chat --config=conf/agent.yml --namespace=starrocks_prod
```

## 参考

- 详细文档：`TEST_SCHEMA_RETRIEVAL.md`
- 迁移脚本：`datus/storage/schema_metadata/migrate_v0_to_v1.py`
