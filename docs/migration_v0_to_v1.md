# LanceDB Schema 迁移指南：v0 -> v1（脚本方式）

> 本指南专注于运行迁移/验证脚本。如需程序化使用，请参考
> `docs/migration_v0_to_v1_code_usage.md`。

## 0. 开始之前

### 0.1 确认配置和存储路径

```bash
python scripts/check_storage_path.py
```

### 0.2 备份存储

```bash
bash scripts/backup_storage.sh
```

## 1. 运行迁移脚本

脚本入口：

- `python scripts/migrate_v0_to_v1.py`

### 1.1 推荐流程

完整迁移（统计信息 + 关系）：

```bash
python scripts/migrate_v0_to_v1.py \
  --config=conf/agent.yml \
  --extract-statistics=true \
  --extract-relationships=true \
  --import-schemas \
  --force
```

快速迁移（跳过统计信息）：

```bash
python scripts/migrate_v0_to_v1.py \
  --config=conf/agent.yml \
  --extract-statistics=false \
  --extract-relationships=true
```

仅关系（无统计，最快）：

```bash
python scripts/migrate_v0_to_v1.py \
  --config=conf/agent.yml \
  --extract-statistics=false
```

仅导入 Schema（跳过迁移，从 DB 刷新）：

```bash
python scripts/migrate_v0_to_v1.py \
  --config=conf/agent.yml \
  --namespace=<name> \
  --import-schemas \
  --import-only \
  --clear \
  --force
```

可选的 LLM 回退用于 DDL 解析：

```bash
python scripts/migrate_v0_to_v1.py \
  --config=conf/agent.yml \
  --extract-relationships=true \
  --llm-fallback \
  --llm-model=deepseek-chat
```

### 1.2 重要参数

| 参数 | 说明 | 备注 |
| --- | --- | --- |
| `--config` | `agent.yml` 路径 | 必填 |
| `--extract-statistics` | 收集行/列统计信息 | 较慢；需要数据库访问权限 |
| `--extract-relationships` | 提取 FK/JOIN 关系 | 默认值: true |
| `--import-schemas` | 迁移后从 DB 导入 Schema | 使用 namespace 配置 |
| `--import-only` | 跳过迁移，仅导入 Schema | 与 `--import-schemas` 一起使用 |
| `--clear` | 导入前清除现有数据 | 谨慎使用 |
| `--force` | 强制重新迁移 | 覆盖 v1 数据 |
| `--llm-fallback` | LLM 作为最后的 DDL 解析器 | 需要模型配置 |
| `--llm-model` | LLM 模型名称 | 可选 |
| `--db-path` | 覆盖存储路径 | 建议使用配置文件 |

布尔标志接受：`true/false`、`yes/no`、`1/0`。

## 2. 验证迁移

使用报告脚本检查覆盖率和字段填充率：

```bash
python scripts/check_migration_report.py \
  --config=conf/agent.yml \
  --namespace=<name>
```

可选覆盖：

```bash
python scripts/check_migration_report.py \
  --config=conf/agent.yml \
  --db-path=/custom/path/to/lancedb \
  --top-tags=20
```

报告包含：
- v1 记录比例
- `table_comment`、`column_comments`、`column_enums`、`business_tags` 的填充率
- `relationship_metadata` 覆盖率和来源分析
- `row_count` / `sample_statistics` 覆盖率

## 3. 使用附加脚本（实时引导）

如果要从实时数据库直接拉取元数据，请使用：

```bash
python scripts/live_bootstrap.py \
  --config=conf/agent.yml \
  --database=my_db \
  --schema=public \
  --dialect=duckdb \
  --extract-statistics=true \
  --extract-relationships=true
```

注意：
- 脚本从 `agent.yml` 读取存储路径
- 连接信息取自配置的 `namespace`
- 使用 `--extract-statistics=false` 可加快运行速度

## 4. 回滚（从备份恢复）

```bash
DB_PATH=$(python3 -c "
import yaml
with open('conf/agent.yml', 'r') as f:
    config = yaml.safe_load(f)
    print(config['agent']['storage']['base_path'])
")

echo "正在恢复 $DB_PATH"
rm -rf "$DB_PATH"
mv "$DB_PATH.backup_v0_"* "$DB_PATH"
```

## 5. 故障排除

缺少 `sqlglot`：

```bash
pip install sqlglot
```

LLM 回退失败：

```bash
# 禁用 LLM 回退
python scripts/migrate_v0_to_v1.py \
  --config=conf/agent.yml \
  --extract-relationships=true
```
