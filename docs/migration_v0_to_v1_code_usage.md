# LanceDB Schema 迁移指南：v0 -> v1（代码方式）

> 本文档涵盖程序化使用方式。如需基于脚本的迁移和验证，请参考
> `docs/migration_v0_to_v1.md`。

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
