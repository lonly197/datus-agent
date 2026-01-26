# Schema 元数据管理 - 程序化使用

> **文档版本**: v1.0
> **更新日期**: 2026-01-26

> 本文档介绍在 Python 代码中调用迁移/引导 API 的方式。如需使用命令行脚本，请参考 `docs/migration_v0_to_v1.md`。

> **注意**：脚本文件已迁移至 `scripts/` 目录，API 模块仍在 `datus.storage.schema_metadata` 中。

---

## 1. 在代码中运行迁移

```python
from datus.storage.schema_metadata import SchemaWithValueRAG
from datus.storage.schema_metadata.migrate_v0_to_v1 import migrate_schema_storage, verify_migration

# agent_config = AgentConfig.from_yaml("conf/agent.yml")
storage = SchemaWithValueRAG(agent_config)

migrated = migrate_schema_storage(
    storage=storage.schema_store,
    extract_statistics=False,
    extract_relationships=True,
)

ok = verify_migration(storage)
print(f"migrated={migrated}, ok={ok}")
```

---

## 2. 检查版本分布

```python
from collections import Counter
from datus.configuration.agent_config import AgentConfig
from datus.storage.schema_metadata import SchemaStorage

config = AgentConfig.from_yaml("conf/agent.yml")
db_path = config.rag_storage_path()

store = SchemaStorage(db_path=db_path)
store._ensure_table_ready()
rows = store._search_all(where=None, select_fields=["metadata_version"]).to_pylist()

print(Counter(r.get("metadata_version", 0) for r in rows))
```

---

## 3. 从实时数据库增量引导

```python
from datus.storage.schema_metadata import SchemaWithValueRAG
from datus.storage.schema_metadata.live_bootstrap import bootstrap_incremental

storage = SchemaWithValueRAG(agent_config)

results = await bootstrap_incremental(
    storage=storage,
    connector=database_connector,
    catalog_name="",
    database_name="my_db",
    schema_name="public",
)

print(results)
```

---

## 4. 直接调用脚本模块（与命令行等价）

```python
import subprocess
import sys

# 等价于: python scripts/migrate_v0_to_v1.py --config=conf/agent.yml
result = subprocess.run(
    [sys.executable, "scripts/migrate_v0_to_v1.py", "--config=conf/agent.yml"],
    capture_output=True,
    text=True
)
print(result.stdout)
if result.returncode != 0:
    print(result.stderr)
```

---

## 5. 相关文件

| 文件路径 | 说明 |
|----------|------|
| `scripts/migrate_v0_to_v1.py` | 迁移脚本入口 |
| `scripts/live_bootstrap.py` | 实时引导脚本入口 |
| `scripts/rebuild_schema.sh` | 重建脚本（Bash 封装） |
| `datus/storage/schema_metadata/migrate_v0_to_v1.py` | 迁移模块源码 |
| `datus/storage/schema_metadata/live_bootstrap.py` | 引导模块源码 |
| `datus/storage/schema_metadata/__init__.py` | 存储模块导出 |
