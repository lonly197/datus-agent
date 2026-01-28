# Schema 元数据管理脚本指南

> **文档版本**: v1.2
> **更新日期**: 2026-01-27

> 本指南介绍用于 Schema 元数据管理的脚本。如需查看所有脚本（包括模型下载、开发测试等），请参考 [脚本清单](../scripts/脚本清单.md)。

## 快速索引

| 脚本 | 功能 | 推荐场景 |
|------|------|----------|
| [migrate_v0_to_v1.py](#migrate_v0_to_v1py---schema-迁移) | v0→v1 Schema 迁移 | 升级元数据结构 |
| [enrich_schema_from_design.py](#enrich_schema_from_designpy---schema-增强) | 从设计文档增强元数据 | 补齐业务注释/标签 |
| [verify_schema_enrichment.py](#verify_schema_enrichmentpy---增强验证) | 验证增强效果 | 对比增强前后质量 |
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

**LLM 重写查询（可选）**:
```bash
python scripts/check_search_text_fts.py \
  --config=conf/agent.yml \
  --namespace=test \
  --queries="线索统计 铂智3X 渠道,转化漏斗 试驾 订单实绩" \
  --llm-rewrite
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

## enrich_schema_from_design.py - Schema 增强

**功能**: 从数据架构设计文档（Excel）中提取业务元数据，增强 LanceDB 中的 Schema 信息。支持按数据层过滤和 LLM 智能匹配。

**核心特性**:
- **数据层过滤**: 支持按 DWD/DWS/DIM 等数据层选择性处理
- **智能表名匹配**: 精确匹配 + 模糊匹配 + LLM 增强匹配
- **元数据增强**: 表注释、列注释、业务标签、枚举值推断
- **匹配追踪**: 记录匹配来源（exact/fuzzy/none）便于分析

**核心增强内容**:
| 元数据字段 | 增强来源 | 说明 |
|------------|----------|------|
| `table_comment` | 逻辑实体业务含义 | 表的业务描述 |
| `column_comments` | 属性业务定义 + LLM生成 | 字段的详细业务含义 |
| `business_tags` | 主题域分组、分析对象 | 用于检索和分类 |
| `column_enums` | 字段名模式+安全分类 | 推断枚举值 |

**核心参数**:

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--config` | 配置文件路径 | 必填 |
| `--namespace` | 命名空间 | 必填 |
| `--arch-xlsx` | 数据架构设计 Excel 路径 | 必填 |
| `--arch-sheet-name` | 工作表名称/索引 | 自动检测 |
| `--layers` | 目标数据层（逗号分隔） | `dwd,dws,dim` |
| `--dry-run` | 预览变更，不实际修改 | - |
| `--apply` | 应用变更到 LanceDB | - |
| `--use-llm` | 启用 LLM 智能匹配和注释生成 | false |
| `--output` | 保存报告路径 | - |
| `-v, --verbose` | 详细输出 | false |

**使用示例**:

```bash
# 1. 预览变更 - DWD/DWS/DIM 层（默认）
python scripts/enrich_schema_from_design.py \
  --config=/root/.datus/conf/agent.yml \
  --namespace=test \
  --arch-xlsx=/path/to/数据架构详细设计v2.3.xlsx \
  --layers=dwd,dws,dim \
  --dry-run \
  --output=/tmp/enrichment_preview.txt

# 2. 应用增强 - 只处理 DWS 层
python scripts/enrich_schema_from_design.py \
  --config=/root/.datus/conf/agent.yml \
  --namespace=test \
  --arch-xlsx=/path/to/数据架构详细设计v2.3.xlsx \
  --layers=dws \
  --apply

# 3. LLM 增强模式（智能匹配模糊表名）
python scripts/enrich_schema_from_design.py \
  --config=/root/.datus/conf/agent.yml \
  --namespace=test \
  --arch-xlsx=/path/to/数据架构详细设计v2.3.xlsx \
  --layers=dwd,dws,dim \
  --apply \
  --use-llm

# 4. 验证增强效果
python scripts/check_migration_report.py \
  --config=/root/.datus/conf/agent.yml \
  --namespace=test
```

**设计文档 Excel 格式要求**:
- 第1行: 分组标题
- 第2行: 实际列名（**逻辑实体（英文）列实际存储物理表名**）⭐
- 第3行: 列说明/注释
- 关键列: `逻辑实体（英文）`（物理表名）、`字段名`、`属性业务定义`、`逻辑实体业务含义`

**匹配策略说明**:
1. **精确匹配**: Excel `逻辑实体（英文）` = LanceDB `table_name`
2. **模糊匹配**: 基于表名关键词相似度匹配
3. **LLM 匹配**: 使用 LLM 分析业务语义选择最佳匹配（需 `--use-llm`）

---

## verify_schema_enrichment.py - 增强验证

**功能**: 验证 Schema 增强效果，对比增强前后的元数据质量变化。

**核心参数**:

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--config` | 配置文件路径 | 必填 |
| `--namespace` | 命名空间 | 必填 |
| `--before-backup` | 增强前的备份路径 | - |
| `--output-json` | 保存详细结果到 JSON | - |

**使用示例**:

```bash
# 对比增强前后（需要备份）
python scripts/verify_schema_enrichment.py \
  --config=/root/.datus/conf/agent.yml \
  --namespace=test \
  --before-backup=/root/.datus/data/datus_db_test.backup_v0_20260128_120000 \
  --output-json=/tmp/enrichment_comparison.json

# 仅查看当前状态
python scripts/verify_schema_enrichment.py \
  --config=/root/.datus/conf/agent.yml \
  --namespace=test
```

---

## 完整操作流程

### 流程 1: 完整迁移（含设计文档增强）

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

# 3. 【新增】从设计文档增强 Schema 元数据（DWD/DWS/DIM 层）
python scripts/enrich_schema_from_design.py \
  --config=/root/.datus/conf/agent.yml \
  --namespace=test \
  --arch-xlsx=/path/to/数据架构详细设计v2.3.xlsx \
  --layers=dwd,dws,dim \
  --apply

# 4. 【可选】使用 LLM 增强模式（智能匹配和注释生成）
python scripts/enrich_schema_from_design.py \
  --config=/root/.datus/conf/agent.yml \
  --namespace=test \
  --arch-xlsx=/path/to/数据架构详细设计v2.3.xlsx \
  --layers=dwd,dws,dim \
  --apply \
  --use-llm

# 5. 【新增】验证增强效果
python scripts/verify_schema_enrichment.py \
  --config=/root/.datus/conf/agent.yml \
  --namespace=test

# 5. 验证报告
python scripts/check_migration_report.py \
  --config=conf/agent.yml \
  --namespace=test

# 6. 抽样诊断
python scripts/diagnose_schema_ddl.py \
  --config=conf/agent.yml \
  --namespace=test \
  --limit=10 \
  --random \
  --compare-db

# 7. FTS 检查
python scripts/check_search_text_fts.py \
  --config=conf/agent.yml \
  --namespace=test

# 8. FTS 索引重建（必要时）
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
| LLM 回退失败 | 禁用 `--use-llm` 参数 |
| 匹配率为 0% | 检查 `--layers` 参数是否与设计文档层一致 |
| 只加载了少量表 | 确认 Excel 中 `逻辑实体（英文）` 列包含物理表名 |
| 权限错误 | 检查数据库用户权限 |
| 回滚恢复 | `bash scripts/backup_storage.sh` 后手动恢复 |

---

## 相关文档

- [脚本清单](../scripts/脚本清单.md) - 所有脚本完整说明
- [Schema 元数据管理 - 程序化使用](Schema%20元数据管理%20-%20程序化使用.md) - API 使用方式

---

## 版本记录

### v1.4 (2026-01-28)
- 更新 `enrich_schema_from_design.py` 脚本说明
  - 新增 `--layers` 参数支持数据层过滤
  - 新增 `--use-llm` 参数支持 LLM 智能匹配
  - 新增匹配来源追踪（exact/fuzzy/none）
  - 修复 LanceDB 加载数量问题

### v1.3 (2026-01-28)
- 新增 `enrich_schema_from_design.py` 脚本说明
- 新增 `verify_schema_enrichment.py` 脚本说明
- 更新完整操作流程，包含设计文档增强步骤

### v1.2 (2026-01-27)
- 精简文档，聚焦 Schema 元数据管理
- 新增指向完整脚本清单的引用

### v1.1 (2026-01-27)
- 添加 `--llm-enum-extraction` 参数说明

### v1.0 (2026-01-26)
- 初始版本
