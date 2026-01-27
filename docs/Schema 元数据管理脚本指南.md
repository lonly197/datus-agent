# Schema 元数据管理脚本指南

> **文档版本**: v1.2
> **更新日期**: 2026-01-27

> 本指南介绍用于 Schema 元数据管理的脚本。如需查看所有脚本（包括模型下载、开发测试等），请参考 [脚本清单](../scripts/脚本清单.md)。

## 快速索引

| 脚本 | 功能 | 推荐场景 |
|------|------|----------|
| [migrate_v0_to_v1.py](#migrate_v0_to_v1py---schema-迁移) | v0→v1 Schema 迁移 | 升级元数据结构 |
| [diagnose_schema_ddl.py](#diagnose_schema_ddlpy---ddl-诊断) | 抽样诊断 DDL | 排查解析问题 |
| [check_migration_report.py](#check_migration_reportpy---迁移报告) | 验证迁移质量 | 检查填充率 |
| [rebuild_schema.sh](#rebuild_schemash---重建-schema) | 清空+重新导入 | 完全刷新 |
| [backup_storage.sh](#backup_storagesh---备份) | 备份存储目录 | 迁移前保护 |
| [live_bootstrap.py](#live_bootstrappy---实时引导) | 从数据库引导 | 增量更新 |
| [local_init.py](#local_initpy---schema-导入) | 直接导入 Schema | 快速导入 |
| [check_search_text_fts.py](#check_search_text_ftspy---fts-检查) | search_text 覆盖率 + FTS 召回 | 验证中文检索优化 |
| [rebuild_schema_fts_index.py](#rebuild_schema_fts_indexpy---fts-索引重建) | 重建 Schema FTS 索引 | FTS 故障排查 |

---

## migrate_v0_to_v1.py - Schema 迁移

**功能**: 将 Schema 元数据从 v0 版本升级到 v1 增强版本。

**核心参数**:

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--config` | 配置文件路径 | 必填 |
| `--namespace` | 命名空间 | - |
| `--extract-statistics` | 收集统计信息 | false |
| `--extract-relationships` | 提取关系 | true |
| `--import-schemas` | 迁移后导入 Schema | false |
| `--llm-enum-extraction` | LLM 增强枚举提取 | false |
| `--clear` | 导入前清除数据 | false |
| `--force` | 强制重新迁移 | false |

**推荐命令**:
```bash
python scripts/migrate_v0_to_v1.py \
  --config=conf/agent.yml \
  --namespace=test \
  --extract-statistics=true \
  --extract-relationships=true \
  --import-schemas \
  --llm-enum-extraction \
  --clear \
  --force
```

---

## diagnose_schema_ddl.py - DDL 诊断

**功能**: 检查存储的 DDL 是否存在缺逗号/截断等问题。

**参数**:

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--config` | 配置文件路径 | conf/agent.yml |
| `--namespace` | 命名空间 | 自动选择 |
| `--limit` | 抽样数量 | 5 |
| `--random` | 随机抽样 | false |
| `--compare-db` | 与数据库 DDL 对比 | false |

**示例**:
```bash
python scripts/diagnose_schema_ddl.py \
  --config=conf/agent.yml \
  --namespace=test \
  --limit=10 \
  --random \
  --compare-db
```

---

## check_migration_report.py - 迁移报告

**功能**: 检查迁移覆盖率、字段填充率。

**输出指标**:
- v1 记录比例
- `table_comment` / `column_comments` / `column_enums` / `business_tags` 填充率
- `relationship_metadata` 覆盖率
- `row_count` / `sample_statistics` 覆盖率

**示例**:
```bash
python scripts/check_migration_report.py \
  --config=conf/agent.yml \
  --namespace=test
```

---

## rebuild_schema.sh - 重建 Schema

**功能**: 清空现有数据后重新导入。

**示例**:
```bash
# 标准重建
bash scripts/rebuild_schema.sh --namespace=my_database

# 清除后重建
bash scripts/rebuild_schema.sh --namespace=my_database --clear
```

---

## backup_storage.sh - 备份

**功能**: 创建带时间戳的备份。

```bash
bash scripts/backup_storage.sh
```

---

## live_bootstrap.py - 实时引导

**功能**: 从数据库直接拉取 Schema 元数据。

**支持方言**: `duckdb`, `sqlite`, `mysql`, `starrocks`, `snowflake`

**示例**:
```bash
python scripts/live_bootstrap.py \
  --config=conf/agent.yml \
  --database=my_db \
  --dialect=starrocks \
  --extract-statistics=true \
  --incremental
```

---

## local_init.py - Schema 导入（模块）

**功能**: 直接从数据库导入 Schema。

**使用方式**:
```bash
python -m datus.storage.schema_metadata.local_init \
  --config=conf/agent.yml \
  --namespace=test \
  --llm-enum-extraction
```

---

## check_search_text_fts.py - FTS 检查

**功能**: 检查 `search_text` 覆盖率并执行 FTS 查询，验证中文检索是否起效。

**示例**:
```bash
python scripts/check_search_text_fts.py \
  --config=conf/agent.yml \
  --namespace=test
```

**自定义查询**:
```bash
python scripts/check_search_text_fts.py \
  --config=conf/agent.yml \
  --namespace=test \
  --queries="线索统计 铂智3X 渠道,转化漏斗 试驾 订单实绩,有效线索到店"
```

**调试中文短语拆分**:
```bash
python scripts/check_search_text_fts.py \
  --config=conf/agent.yml \
  --namespace=test \
  --queries="有效线索到店,铂智3X线索转化漏斗（按月份）" \
  --debug-simplify
```

---

## rebuild_schema_fts_index.py - FTS 索引重建

**功能**: 重建 `schema_metadata` 的 FTS 索引，用于修复 FTS 查询失败或索引字段变更的场景。

**示例**:
```bash
python scripts/rebuild_schema_fts_index.py \
  --config=conf/agent.yml \
  --namespace=test
```

---

## 完整操作流程

### 流程 1: 完整迁移

```bash
# 1. 备份
bash scripts/backup_storage.sh

# 2. 执行迁移
python scripts/migrate_v0_to_v1.py \
  --config=conf/agent.yml \
  --namespace=test \
  --extract-statistics=true \
  --extract-relationships=true \
  --import-schemas \
  --llm-enum-extraction \
  --clear \
  --force

# 3. 验证报告
python scripts/check_migration_report.py \
  --config=conf/agent.yml \
  --namespace=test

# 4. 抽样诊断
python scripts/diagnose_schema_ddl.py \
  --config=conf/agent.yml \
  --namespace=test \
  --limit=10 \
  --random \
  --compare-db

# 5. FTS 检查
python scripts/check_search_text_fts.py \
  --config=conf/agent.yml \
  --namespace=test

# 6. FTS 索引重建（必要时）
python scripts/rebuild_schema_fts_index.py \
  --config=conf/agent.yml \
  --namespace=test
```

### 流程 2: 快速重建

```bash
bash scripts/backup_storage.sh
bash scripts/rebuild_schema.sh --namespace=my_database --clear
python scripts/check_migration_report.py \
  --config=conf/agent.yml \
  --namespace=my_database
```

---

## 故障排除

| 问题 | 解决方案 |
|------|----------|
| LLM 回退失败 | 禁用 `--llm-fallback` |
| 权限错误 | 检查数据库用户权限 |
| 回滚恢复 | `bash scripts/backup_storage.sh` 后手动恢复 |

---

## 相关文档

- [脚本清单](../scripts/脚本清单.md) - 所有脚本完整说明
- [Schema 元数据管理 - 程序化使用](Schema%20元数据管理%20-%20程序化使用.md) - API 使用方式

---

## 版本记录

### v1.2 (2026-01-27)
- 精简文档，聚焦 Schema 元数据管理
- 新增指向完整脚本清单的引用

### v1.1 (2026-01-27)
- 添加 `--llm-enum-extraction` 参数说明

### v1.0 (2026-01-26)
- 初始版本
