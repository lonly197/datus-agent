# LanceDB Schema Migration: v0 -> v1 (Script Guide)

> This guide focuses on running migration/validation scripts. For programmatic usage, see
> `docs/migration_v0_to_v1_code_usage.md`.

## 0. Before You Start

### 0.1 Confirm config and storage path

The scripts read `storage.base_path` from `agent.yml` automatically.

```bash
python3 -c "
import yaml
with open('conf/agent.yml', 'r') as f:
    config = yaml.safe_load(f)
    print('Storage path:', config['agent']['storage']['base_path'])
"
```

### 0.2 Backup storage

```bash
DB_PATH=$(python3 -c "
import yaml
with open('conf/agent.yml', 'r') as f:
    config = yaml.safe_load(f)
    print(config['agent']['storage']['base_path'])
")

echo "Detected storage path: $DB_PATH"
cp -r "$DB_PATH" "$DB_PATH.backup_v0_$(date +%Y%m%d_%H%M%S)"
```

## 1. Run the migration script

Script entry:

- `python -m datus.storage.schema_metadata.migrate_v0_to_v1`

### 1.1 Recommended flows

Full migration (statistics + relationships):

```bash
python -m datus.storage.schema_metadata.migrate_v0_to_v1 \
  --config=conf/agent.yml \
  --extract-statistics=true \
  --extract-relationships=true \
  --import-schemas \
  --force
```

Fast migration (skip statistics):

```bash
python -m datus.storage.schema_metadata.migrate_v0_to_v1 \
  --config=conf/agent.yml \
  --extract-statistics=false \
  --extract-relationships=true
```

Relationships only (no stats, fastest):

```bash
python -m datus.storage.schema_metadata.migrate_v0_to_v1 \
  --config=conf/agent.yml \
  --extract-statistics=false
```

Schema import only (skip migration, refresh from DB):

```bash
python -m datus.storage.schema_metadata.migrate_v0_to_v1 \
  --config=conf/agent.yml \
  --namespace=<name> \
  --import-schemas \
  --import-only \
  --clear \
  --force
```

Optional LLM fallback for DDL parsing:

```bash
python -m datus.storage.schema_metadata.migrate_v0_to_v1 \
  --config=conf/agent.yml \
  --extract-relationships=true \
  --llm-fallback \
  --llm-model=deepseek-chat
```

### 1.2 Important parameters

| Parameter | Description | Notes |
| --- | --- | --- |
| `--config` | Path to `agent.yml` | Required |
| `--extract-statistics` | Collect row/column stats | Slower; needs DB access |
| `--extract-relationships` | Extract FK/JOIN relationships | Default: true |
| `--import-schemas` | Import schemas from DB after migration | Uses namespace config |
| `--import-only` | Skip migration, only import schemas | Use with `--import-schemas` |
| `--clear` | Clear existing data before import | Use with care |
| `--force` | Force re-migration | Overwrites v1 data |
| `--llm-fallback` | LLM as last-resort DDL parser | Requires model config |
| `--llm-model` | LLM model name | Optional |
| `--db-path` | Override storage path | Prefer config file |

Boolean flags accept: `true/false`, `yes/no`, `1/0`.

## 2. Validate the migration

Use the report script to check coverage and field fill rates:

```bash
python scripts/check_migration_report.py \
  --config=conf/agent.yml \
  --namespace=<name>
```

Optional overrides:

```bash
python scripts/check_migration_report.py \
  --config=conf/agent.yml \
  --db-path=/custom/path/to/lancedb \
  --top-tags=20
```

Report includes:
- v1 record ratio
- Fill rates for `table_comment`, `column_comments`, `column_enums`, `business_tags`
- `relationship_metadata` coverage and source breakdown
- `row_count` / `sample_statistics` coverage

## 3. Use additional scripts (live bootstrap)

If you want to pull metadata directly from a live database, use:

```bash
python -m datus.storage.schema_metadata.live_bootstrap \
  --config=conf/agent.yml \
  --database=my_db \
  --schema=public \
  --dialect=duckdb \
  --extract-statistics=true \
  --extract-relationships=true
```

Notes:
- The script reads storage path from `agent.yml`.
- Connection info is taken from the configured `namespace`.
- Use `--extract-statistics=false` for faster runs.

## 4. Rollback (restore from backup)

```bash
DB_PATH=$(python3 -c "
import yaml
with open('conf/agent.yml', 'r') as f:
    config = yaml.safe_load(f)
    print(config['agent']['storage']['base_path'])
")

echo "Restoring $DB_PATH"
rm -rf "$DB_PATH"
mv "$DB_PATH.backup_v0_"* "$DB_PATH"
```

## 5. Troubleshooting

Missing `sqlglot`:

```bash
pip install sqlglot
```

LLM fallback failed:

```bash
# Disable LLM fallback
python -m datus.storage.schema_metadata.migrate_v0_to_v1 \
  --config=conf/agent.yml \
  --extract-relationships=true
```
