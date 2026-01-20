# Schema Retrieval Test - Complete Guide

## 概述

本项目包含一个完整的CLI测试脚本，用于验证DDL迁移后的schema检索功能。测试脚本模拟text2sql工作流中的schema discovery阶段，检查RAG和Keyword检索是否能正确找到表和字段信息。

## 🎯 测试目标

1. ✅ 验证迁移是否成功将DDL元数据导入向量库
2. ✅ 检查RAG（语义）检索功能
3. ✅ 检查Keyword（关键词）检索功能
4. ✅ 验证表和字段的comment信息是否正确提取
5. ✅ 测试"销售线索"等业务场景的检索效果

## 📁 文件清单

### 核心文件

| 文件 | 说明 | 大小 |
|------|------|------|
| `test_schema_retrieval.py` | 主测试脚本（可执行） | 26KB |
| `verify_implementation.sh` | 实现验证脚本（可执行） | - |

### 文档文件

| 文件 | 说明 |
|------|------|
| `TEST_SCHEMA_RETRIEVAL.md` | 完整使用指南（16KB） |
| `QUICKSTART_SCHEMA_TEST.md` | 快速开始指南（2.8KB） |
| `SCHEMA_TEST_SUMMARY.md` | 实现总结（7.7KB） |
| `README_SCHEMA_TEST.md` | 本文件 |
| `test_example_usage.sh` | 使用示例（可执行） |
| `MIGRATION_OUTPUT_IMPROVEMENTS.md` | 迁移脚本改进文档 |

## 🚀 快速开始

### 1. 验证环境

```bash
# 运行验证脚本
./verify_implementation.sh
```

### 2. 运行迁移

```bash
# 迁移DDL元数据到向量库
python -m datus.storage.schema_metadata.migrate_v0_to_v1 \
    --config=conf/agent.yml \
    --namespace=your_namespace \
    --import-schemas \
    --force
```

### 3. 执行测试

```bash
# 基本测试（销售线索场景）
python test_schema_retrieval.py \
    --config=conf/agent.yml \
    --namespace=your_namespace

# 自定义查询
python test_schema_retrieval.py \
    --config=conf/agent.yml \
    --namespace=your_namespace \
    --query="客户转化" \
    --top-n=30
```

### 4. 查看帮助

```bash
python test_schema_retrieval.py --help
```

## 📊 测试内容

### Test 1: Database Connection
- ✅ 向量库连接检查
- ✅ schema_metadata表存在性验证

### Test 2: Migration Success
- ✅ 总记录数统计
- ✅ v0（legacy）记录数
- ✅ v1（enhanced）记录数
- ✅ 迁移进度百分比

### Test 3: Comment Extraction
- ✅ 表注释（table_comment）
- ✅ 列注释（column_comments）
- ✅ 业务标签（business_tags）
- ✅ 提取率统计

### Test 4: Keyword Search
- ✅ 表名匹配
- ✅ 注释匹配
- ✅ 标签匹配
- ✅ 相关性评分

### Test 5: RAG (Semantic) Search
- ✅ 向量相似度搜索
- ✅ 语义相关性排序
- ✅ Top-N结果返回

### Test 6: Sales Lead Scenario
测试6个业务查询：
- "销售线索"
- "线索"
- "销售"
- "客户"
- "潜在客户"
- "商机"

## 📋 使用示例

### 生产环境测试

```bash
# StarRocks生产环境
python test_schema_retrieval.py \
    --config=/Users/lonlyhuang/workspace/myway/Examples/datus-docs/conf/agent.yml \
    --namespace=starrocks_prod \
    --query="销售线索" \
    --top-n=50
```

### 自动化测试

```bash
# 使用示例脚本
./test_example_usage.sh
```

### 故障排除

```bash
# 检查迁移状态
python -c "
from datus.storage.schema_metadata.migrate_v0_to_v1 import report_migration_state
report_migration_state('/root/.datus/data/lancedb')
"

# 验证数据导入
python -c "
from datus.configuration.agent_config_loader import load_agent_config
from datus.storage.schema_metadata import SchemaStorage

config = load_agent_config('conf/agent.yml')
storage = SchemaStorage(db_path=config.rag_storage_path())
print(f'Total tables: {len(storage._search_all(where=None))}')
"
```

## 📊 输出示例

### 成功场景

```
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
Test 3: Comment Extraction
--------------------------------------------------------------------------------
⚠️ Low comment extraction rate

❌ Comment extraction test failed

# 解决方案：
# 1. 检查DDL文件是否包含COMMENT
grep -i "COMMENT" /path/to/ods_ddl.sql | head -10
# 2. 重新运行迁移
python -m datus.storage.schema_metadata.migrate_v0_to_v1 \
    --config=conf/agent.yml \
    --namespace=your_namespace \
    --force
```

## 🔍 验证指标

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

## 🛠️ 故障排除

### 常见错误

| 错误 | 原因 | 解决方案 |
|------|------|----------|
| Database connection failed | 配置文件或路径错误 | 检查agent.yml和路径权限 |
| Migration failed - no v1 records | 未运行迁移脚本 | 运行迁移脚本 |
| Low comment extraction rate | 源DDL缺少COMMENT | 检查DDL文件 |
| RAG search returns no results | 向量未生成 | 重新导入schema |

### 调试命令

```bash
# 1. 检查数据库连接
python test_schema_retrieval.py --config=conf/agent.yml --namespace=ns

# 2. 检查迁移状态
python -m datus.storage.schema_metadata.migrate_v0_to_v1 \
    --config=conf/agent.yml \
    --namespace=ns \
    --force

# 3. 验证注释提取
grep -i "COMMENT" /path/to/ods_ddl.sql | head -10

# 4. 手动测试RAG
python -c "
from datus.storage.schema_metadata import SchemaWithValueRAG
storage = SchemaWithValueRAG(config)
results = storage.search_similar('销售线索', top_n=10)
print(f'Found {len(results)} results')
"
```

## 📚 文档索引

### 详细文档
- **[TEST_SCHEMA_RETRIEVAL.md](TEST_SCHEMA_RETRIEVAL.md)** - 完整使用指南
  - 所有测试详细说明
  - 参数配置
  - 高级用法
  - 生产环境集成

### 快速参考
- **[QUICKSTART_SCHEMA_TEST.md](QUICKSTART_SCHEMA_TEST.md)** - 快速开始
  - 基本命令
  - 常见问题
  - 故障排除

### 实现细节
- **[SCHEMA_TEST_SUMMARY.md](SCHEMA_TEST_SUMMARY.md)** - 实现总结
  - 架构设计
  - 代码结构
  - 性能指标

### 示例
- **[test_example_usage.sh](test_example_usage.sh)** - 使用示例
  - 8个实际用例
  - 自动化脚本
  - 故障排除命令

## 🎓 最佳实践

### 1. 迁移前准备
- ✅ 备份现有数据
- ✅ 确认DDL文件包含COMMENT
- ✅ 验证配置文件正确

### 2. 测试执行
- ✅ 先运行迁移
- ✅ 再运行测试
- ✅ 检查所有测试通过

### 3. 生产部署
- ✅ 在测试环境验证
- ✅ 检查所有指标
- ✅ 监控生产环境

### 4. 持续维护
- ✅ 定期重新测试
- ✅ 监控检索质量
- ✅ 更新测试场景

## 📞 支持

### 获取帮助

```bash
# 查看帮助
python test_schema_retrieval.py --help

# 查看示例
./test_example_usage.sh

# 验证实现
./verify_implementation.sh
```

### 常见问题

**Q: 如何测试特定的业务场景？**
A: 使用`--query`参数指定查询：
```bash
python test_schema_retrieval.py \
    --config=conf/agent.yml \
    --namespace=ns \
    --query="客户转化率"
```

**Q: 测试失败怎么办？**
A: 检查测试输出中的错误信息，通常是：
1. 数据库连接问题
2. 迁移未运行
3. DDL缺少COMMENT

**Q: 如何验证RAG检索质量？**
A: 运行测试后检查：
- RAG Search Results数量
- Similarity分数
- 是否返回相关表

## 🎯 总结

### 实现成果

✅ **完整的测试脚本**：650+行代码，6大测试套件
✅ **丰富的输出格式**：彩色表格，清晰指示
✅ **灵活的参数配置**：支持自定义查询和结果数
✅ **生产环境就绪**：兼容真实StarRocks环境
✅ **完善的文档**：4个文档文件，详细的指南
✅ **易于使用**：简单CLI界面，智能默认值

### 测试覆盖

✅ 数据库连接验证
✅ 迁移成功验证
✅ 注释提取验证
✅ 关键词检索验证
✅ RAG语义检索验证
✅ 业务场景验证

### 使用场景

- **迁移验证**：检查DDL元数据是否成功导入
- **功能测试**：验证RAG和Keyword检索
- **质量保证**：确保comment信息正确提取
- **回归测试**：检测schema变更影响
- **生产检查**：部署前验证环境

## 📈 后续步骤

测试通过后：

1. **进行Text2SQL测试**
   ```bash
   datus-agent chat --config=conf/agent.yml
   ```

2. **监控生产指标**
   - Schema discovery precision
   - Query response times
   - User feedback

3. **定期重新测试**
   - Schema变更后
   - 迁移更新后
   - 月度健康检查

---

**版本**: 1.0
**创建日期**: 2026-01-20
**兼容性**: Python 3.7+, Datus-Agent v1.5+
**测试环境**: StarRocks, DuckDB, SQLite
