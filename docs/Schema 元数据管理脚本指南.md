# Schema 元数据管理脚本指南

> **文档版本**: v1.0
> **更新日期**: 2026-01-26

> 本指南介绍 `scripts/` 目录下所有 Schema 管理脚本的用法。如需程序化使用，请参考 `docs/Schema 元数据管理 - 程序化使用.md`。

## 脚本概览

| 脚本 | 类型 | 说明 | 应用场景 |
|------|------|------|----------|
| `check_storage_path.py` | Python | 检查存储路径配置 | 验证配置正确性 |
| `backup_storage.sh` | Bash | 备份存储目录 | 迁移前数据保护 |
| `migrate_v0_to_v1.py` | Python | v0 到 v1 Schema 迁移 | 升级元数据结构 |
| `rebuild_schema.sh` | Bash | 重建 Schema（清空+导入） | 完全刷新元数据 |
| `live_bootstrap.py` | Python | 实时数据库引导 | 从生产库拉取元数据 |
| `check_migration_report.py` | Python | 验证迁移报告 | 检查迁移覆盖率 |

---

## 1. check_storage_path.py - 检查存储路径

### 脚本说明
验证 `agent.yml` 配置的存储路径是否正确。

### 使用方式
```bash
python scripts/check_storage_path.py
```

### 输出示例
```
Storage path: /path/to/lancedb/storage
```

---

## 2. backup_storage.sh - 备份存储目录

### 脚本说明
创建存储目录的带时间戳备份。

### 使用方式
```bash
bash scripts/backup_storage.sh
```

### 输出示例
```
检测到存储路径: /path/to/lancedb/storage
备份完成: /path/to/lancedb/storage.backup_v0_20260126_104800
```

---

## 3. migrate_v0_to_v1.py - v0 到 v1 Schema 迁移

### 脚本说明
将 LanceDB 中的 Schema 元数据从 v0 版本升级到 v1 增强版本。

### 使用方式
```bash
python scripts/migrate_v0_to_v1.py --config=conf/agent.yml [选项]
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--config` | 配置文件路径 | 必填 |
| `--namespace` | 命名空间（可选，多命名空间时会提示） | - |
| `--extract-statistics` | 收集行/列统计信息 | false |
| `--extract-relationships` | 提取 FK/JOIN 关系 | true |
| `--import-schemas` | 迁移后从 DB 导入 Schema | false |
| `--import-only` | 仅导入，跳过迁移 | false |
| `--clear` | 导入前清除现有数据 | false |
| `--force` | 强制重新迁移 | false |
| `--backup-path` | 备份 JSON 路径（来自 cleanup 脚本） | - |
| `--skip-backup` | 跳过备份创建 | false |
| `--llm-fallback` | 启用 LLM 作为 DDL 解析回退 | false |
| `--llm-model` | LLM 模型名称 | - |
| `--db-path` | 覆盖存储路径 | - |

带值布尔参数（如 `--extract-statistics`、`--extract-relationships`）接受：`true/false`、`yes/no`、`1/0`。

### 应用场景

**完整迁移（统计信息 + 关系）**
```bash
python scripts/migrate_v0_to_v1.py \
  --config=conf/agent.yml \
  --extract-statistics=true \
  --extract-relationships=true \
  --import-schemas \
  --force
```

**快速迁移（跳过统计信息）**
```bash
python scripts/migrate_v0_to_v1.py \
  --config=conf/agent.yml \
  --extract-statistics=false \
  --extract-relationships=true
```

**仅提取关系（最快）**
```bash
python scripts/migrate_v0_to_v1.py \
  --config=conf/agent.yml \
  --extract-statistics=false
```

**LLM 回退解析 DDL**
```bash
python scripts/migrate_v0_to_v1.py \
  --config=conf/agent.yml \
  --extract-relationships=true \
  --llm-fallback \
  --llm-model=deepseek-chat
```

---

## 4. rebuild_schema.sh - 重建 Schema

### 脚本说明
清空现有 Schema 数据后重新从数据库导入。元数据完全刷新。

### 使用方式
```bash
bash scripts/rebuild_schema.sh --namespace=<name> [选项]
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--namespace` | 命名空间名称 | 必填 |
| `--config` | 配置文件路径 | conf/agent.yml |
| `--clear` | 清除现有数据 | false |

### 应用场景

**标准重建**
```bash
bash scripts/rebuild_schema.sh --namespace=my_database
```

**带清除数据重建**
```bash
bash scripts/rebuild_schema.sh --namespace=my_database --clear
```

**指定配置文件**
```bash
bash scripts/rebuild_schema.sh --namespace=my_database --config=custom/agent.yml
```

---

## 5. live_bootstrap.py - 实时数据库引导

### 脚本说明
从实时数据库直接拉取 Schema 元数据，支持增量更新。

### 使用方式
```bash
python scripts/live_bootstrap.py --config=conf/agent.yml [选项]
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--config` | 配置文件路径 | 必填 |
| `--namespace` | 命名空间（可选，多命名空间时会提示） | - |
| `--catalog` | Catalog 名称过滤 | 空（不过滤） |
| `--database` | 数据库名称过滤 | 空（不过滤） |
| `--schema` | Schema 名称过滤 | 空（不过滤） |
| `--dialect` | SQL 方言 | duckdb |
| `--extract-statistics` | 收集统计信息 | false |
| `--extract-relationships` | 提取关系 | true |
| `--batch-size` | 批处理大小 | 100 |
| `--incremental` | 增量更新模式 | false |

### 支持的方言
`duckdb`、`sqlite`、`mysql`、`starrocks`、`snowflake` 等

### 应用场景

**标准引导**
```bash
python scripts/live_bootstrap.py \
  --config=conf/agent.yml \
  --database=my_db \
  --schema=public \
  --dialect=duckdb \
  --extract-statistics=true \
  --extract-relationships=true
```

**快速引导（无统计）**
```bash
python scripts/live_bootstrap.py \
  --config=conf/agent.yml \
  --database=my_db \
  --schema=public \
  --dialect=duckdb
```

---

## 6. check_migration_report.py - 验证迁移报告

### 脚本说明
检查迁移覆盖率、字段填充率，生成质量报告。

### 使用方式
```bash
python scripts/check_migration_report.py --config=conf/agent.yml [选项]
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--config` | 配置文件路径 | 必填 |
| `--namespace` | 命名空间名称（可选，省略时使用 base path） | - |
| `--db-path` | 覆盖存储路径 | - |
| `--top-tags` | 显示的标签数量 | 10 |

### 报告内容
- v1 记录比例
- `table_comment` 填充率
- `column_comments` 填充率
- `column_enums` 填充率
- `business_tags` 填充率
- `relationship_metadata` 覆盖率及来源分析
- `row_count` 覆盖率
- `sample_statistics` 覆盖率

### 应用场景

**标准报告**
```bash
python scripts/check_migration_report.py \
  --config=conf/agent.yml \
  --namespace=my_database
```

**自定义存储路径**
```bash
python scripts/check_migration_report.py \
  --config=conf/agent.yml \
  --namespace=my_database \
  --db-path=/custom/path/to/lancedb \
  --top-tags=20
```

---

## 常见操作流程

### 完整迁移流程
```bash
# 1. 备份
bash scripts/backup_storage.sh

# 2. 执行迁移
python scripts/migrate_v0_to_v1.py \
  --config=conf/agent.yml \
  --extract-statistics=true \
  --extract-relationships=true \
  --import-schemas \
  --force

# 3. 验证报告
python scripts/check_migration_report.py \
  --config=conf/agent.yml \
  --namespace=my_database
```

### 完全重建流程
```bash
# 1. 备份
bash scripts/backup_storage.sh

# 2. 重建（清空+导入）
bash scripts/rebuild_schema.sh --namespace=my_database --clear

# 3. 验证报告
python scripts/check_migration_report.py \
  --config=conf/agent.yml \
  --namespace=my_database
```

---

## 故障排除

### 缺少依赖
```bash
pip install sqlglot
```

### LLM 回退失败
禁用 LLM 回退，使用纯规则解析：
```bash
python scripts/migrate_v0_to_v1.py \
  --config=conf/agent.yml \
  --extract-relationships=true
```

### 回滚（从备份恢复）
```bash
bash scripts/backup_storage.sh  # 先创建新备份
rm -rf /path/to/lancedb/storage
mv /path/to/lancedb/storage.backup_v0_* /path/to/lancedb/storage
```
