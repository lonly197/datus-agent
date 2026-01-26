# Datus Storage 模块介绍

> **文档版本**: v2.1
> **更新日期**: 2026-01-23
> **相关模块**: `datus/storage/`
> **代码仓库**: [Datus Agent](https://github.com/Datus-ai/Datus-agent)

---

## 模块概述

### 核心功能

Datus Storage 模块是一个基于 **LanceDB 向量数据库** 的多层知识存储系统，为 Text2SQL 和数据工程应用提供智能化的元数据管理和语义检索能力。

### 设计架构

```
┌─────────────────────────────────────────────────────────────┐
│                    StorageCache (缓存层)                      │
│  ┌─────────────┬──────────────┬──────────────┬────────────┐│
│  │SchemaStorage│MetricStorage │RefSqlStorage │DocumentStore││
│  └─────────────┴──────────────┴──────────────┴────────────┘│
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                   Base Classes (基类层)                       │
│  ┌─────────────────┬──────────────────┬───────────────────┐│
│  │StorageBase      │BaseEmbeddingStore│BaseSubjectEmbedding││
│  │(LanceDB连接)    │(向量嵌入存储)     │Store(主题树集成)   ││
│  └─────────────────┴──────────────────┴───────────────────┘│
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                   Storage Engines (存储引擎)                  │
│  ┌─────────────┬──────────────┬──────────────┬────────────┐│
│  │  LanceDB    │   SQLite     │  FastEmbed   │ SubjectTree││
│  │(向量数据库)  │(主题树存储)   │(嵌入模型)    │(层次结构)  ││
│  └─────────────┴──────────────┴──────────────┴────────────┘│
└─────────────────────────────────────────────────────────────┘
```

---

## 存储类型详解

### 1. SchemaStorage - 表结构存储

**用途**: 存储数据库表、视图、物化视图的 DDL 定义和增强元数据

**核心字段**:
| 字段名 | 类型 | 说明 |
|--------|------|------|
| `identifier` | string | 唯一标识符: `catalog.database.schema.table.type` |
| `catalog_name` | string | 目录名 (StarRocks/Snowflake) |
| `database_name` | string | 数据库名 |
| `schema_name` | string | 模式名/命名空间 |
| `table_name` | string | 表名 |
| `table_type` | string | 类型: `table`/`view`/`mv` |
| `definition` | string | DDL 定义 (含中文注释增强) |
| `table_comment` | string | 表注释 |
| `column_comments` | string | 列注释 JSON |
| `column_enums` | string | 列枚举值 JSON |
| `business_tags` | list[string] | 业务标签 |
| `row_count` | int64 | 行数统计 |
| `sample_statistics` | string | 列统计 JSON |
| `relationship_metadata` | string | 外键和关联路径 JSON |
| `metadata_version` | int32 | 元数据版本 (0=旧版, 1=增强) |
| `last_updated` | int64 | 更新时间戳 |
| `vector` | list[float32] | 向量嵌入 |

**核心方法**:
```python
# 语义搜索表结构
search_similar(query_text, catalog_name, database_name, schema_name, top_n, table_type, reranker)

# 内部搜索（支持 reranker）
do_search_similar(query_text, top_n, where, reranker)

# 获取所有表
search_all(catalog_name, database_name, schema_name, table_type, select_fields)

# 创建索引
create_indices()

# 批量存储
store_batch(data)

# 更新记录
update(where, update_values, unique_filter)
```

### 2. SchemaValueStorage - 样本数据存储

**用途**: 存储表的样本数据 (通常为前5行)，用于数据理解

**核心字段**:
| 字段名 | 类型 | 说明 |
|--------|------|------|
| `identifier` | string | 唯一标识符 |
| `sample_rows` | string | 样本数据 (CSV 格式) |
| `vector` | list[float32] | 向量嵌入 |

**使用场景**: 配合 SchemaStorage 提供完整的表结构理解

### 3. MetricStorage - 业务指标存储

**用途**: 存储业务指标和 KPI，支持主题树层次化组织

**核心字段**:
| 字段名 | 类型 | 说明 |
|--------|------|------|
| `subject_node_id` | int64 | 主题树节点 ID |
| `name` | string | 指标名称 |
| `semantic_model_name` | string | 所属语义模型 |
| `llm_text` | string | LLM 描述文本 (用于嵌入) |
| `vector` | list[float32] | 向量嵌入 |

**核心方法**:
```python
# 语义搜索指标
search_metrics(query_text, subject_path, semantic_model_names, top_n)

# 获取所有指标
search_all_metrics(subject_path, semantic_model_names)

# 批量存储指标
batch_store_metrics(metrics)
```

**主题树组织示例**:
```
Finance
├── Revenue
│   ├── Q1
│   │   ├── total_revenue
│   │   └── growth_rate
│   └── Q2
└── Costs
    └── Operations
```

### 4. SemanticModelStorage - 语义模型存储

**用途**: 存储语义模型定义 (维度、度量、标识符)

**核心字段**:
| 字段名 | 类型 | 说明 |
|--------|------|------|
| `id` | string | 模型 ID |
| `semantic_model_name` | string | 语义模型名称 |
| `semantic_model_desc` | string | 模型描述 |
| `identifiers` | string | 标识符 JSON |
| `dimensions` | string | 维度定义 |
| `measures` | string | 度量定义 |
| `vector` | list[float32] | 向量嵌入 |

### 5. ReferenceSqlStorage - 参考SQL存储

**用途**: 存储历史 SQL 查询和最佳实践

**核心字段**:
| 字段名 | 类型 | 说明 |
|--------|------|------|
| `subject_node_id` | int64 | 主题树节点 ID |
| `name` | string | SQL 名称 |
| `sql` | string | SQL 语句 |
| `comment` | string | 注释 |
| `summary` | string | 摘要 (用于嵌入) |
| `search_text` | string | 搜索文本 |
| `filepath` | string | 文件路径 |
| `tags` | string | 标签 |

**核心方法**:
```python
# 语义搜索 SQL
search_reference_sql(query_text, subject_path, top_n)

# 获取所有 SQL
search_all_reference_sql(subject_path)

# 批量存储 SQL
batch_store_sql(sql_items)
```

### 6. DocumentStore - 文档存储

**用途**: 存储文档块，支持长文档的语义检索

**核心字段**:
| 字段名 | 类型 | 说明 |
|--------|------|------|
| `title` | string | 文档标题 |
| `hierarchy` | string | 层次结构路径 |
| `keywords` | list[string] | 关键词 |
| `language` | string | 语言 |
| `chunk_text` | string | 文档块文本 |
| `vector` | list[float64] | 向量嵌入 |

### 7. SubjectTreeStore - 主题树存储

**用途**: SQLite 实现的层次化主题分类系统

**数据结构** (邻接表模型):
```sql
CREATE TABLE subject_nodes (
    node_id INTEGER PRIMARY KEY AUTOINCREMENT,
    parent_id INTEGER,              -- 父节点 ID (NULL 表示根节点)
    name TEXT NOT NULL,             -- 节点名称
    description TEXT DEFAULT '',    -- 描述
    created_at TEXT NOT NULL,       -- 创建时间
    updated_at TEXT NOT NULL,       -- 更新时间
    UNIQUE(parent_id, name)         -- 同一父节点下名称唯一
)

-- 索引
CREATE INDEX idx_subject_parent_id ON subject_nodes(parent_id)
CREATE UNIQUE INDEX idx_subject_parent_name ON subject_nodes(parent_id, name)
```

**核心方法**:
```python
# CRUD 操作
create_node(parent_id, name, description)
get_node(node_id)
update_node(node_id, name, description, parent_id)
delete_node(node_id, cascade)

# 树遍历
get_children(parent_id)              # 获取直接子节点
get_descendants(node_id)             # 获取所有后代
get_ancestors(node_id)               # 获取所有祖先
get_full_path(node_id)               # 获取完整路径

# 路径操作
find_or_create_path(path_components) # 查找或创建路径
get_node_by_path(path)               # 通过路径获取节点
rename(old_path, new_path)           # 重命名/移动节点

# 树结构
get_tree_structure()                 # 获取嵌套树结构
get_simple_tree_structure()          # 获取简单树结构

# 模式匹配
get_matched_children_id(path, descendant=True)  # 支持通配符的节点匹配
```

**路径示例**: `['Finance', 'Revenue', 'Q1']`

---

## 索引策略

### 向量索引

根据数据集大小自动选择索引类型:

```python
# 代码位置: datus/storage/base.py:191-256
if row_count >= 5000:
    index_type = "IVF_PQ"  # 大数据集: 产品量化压缩
else:
    index_type = "IVF_FLAT"  # 小数据集: 精确搜索
```

**IVF 参数自动计算**:
- `num_partitions`: √n (最大1024)
- `num_sub_vectors`: 根据向量维度和数据量调整 (8-96)

### 标量索引

为常用字段创建标量索引，加速精确查找:
```python
table.create_scalar_index("database_name", replace=True)
table.create_scalar_index("table_name", replace=True)
table.create_scalar_index("subject_node_id", replace=True)
```

### 全文搜索索引 (FTS)

为文本字段创建全文搜索索引:
```python
create_fts_index(["definition", "table_name", "schema_name"])
```

---

## 子代理作用域存储

### 设计理念

支持为每个子代理 (Sub-Agent) 创建独立的知识库，实现上下文隔离。

### 存储路径管理

```python
# 全局存储路径
agent_config.rag_storage_path()  # ~/.datus/data/kb

# 子代理存储路径
agent_config.sub_agent_storage_path(sub_agent_name)
# ~/.datus/data/sub_agents/{sub_agent_name}/data
```

### 作用域配置

在 `agent.yml` 中配置子代理的作用域上下文:

```yaml
sub_agents:
  sales_agent:
    scoped_context:
      tables: true    # 使用独立的表元数据
      metrics: true   # 使用独立的指标
      sqls: false     # 共享全局 SQL
```

### 缓存机制

```python
# 代码位置: datus/storage/cache.py

# LRU 缓存装饰器
@lru_cache(maxsize=12)
def _cached_storage(factory, path, model_name):
    return factory(path, get_embedding_model(model_name))

# StorageCacheHolder - 子代理存储持有者
class StorageCacheHolder:
    def storage_instance(sub_agent_name: Optional[str] = None) -> T:
        # 检查子代理是否有作用域上下文
        # 如果有，使用子代理独立存储路径
        # 否则，使用全局存储并通过 LRU 缓存返回

# StorageCache - 主缓存管理类
class StorageCache:
    def schema_storage(sub_agent_name: Optional[str] = None) -> SchemaStorage
    def schema_value_storage(sub_agent_name: Optional[str] = None) -> SchemaValueStorage
    def metrics_storage(sub_agent_name: Optional[str] = None) -> MetricStorage
    def semantic_storage(sub_agent_name: Optional[str] = None) -> SemanticModelStorage
    def reference_sql_storage(sub_agent_name: Optional[str] = None) -> ReferenceSqlStorage
```

---

## 增强元数据 (v1 Schema)

### 中文注释支持

DDL 定义会自动添加中文注释前缀，提升中文查询的语义搜索效果:

```python
def _enhance_definition_with_comments(
    definition: str,
    table_comment: str = "",
    column_comments: Optional[Dict[str, str]] = None,
) -> str:
    """增强 DDL 定义，添加中文注释"""
    enhanced_parts = []

    if table_comment:
        enhanced_parts.append(f"-- 表注释: {table_comment}")

    if column_comments:
        for col_name, col_comment in column_comments.items():
            enhanced_parts.append(f"-- 列 {col_name}: {col_comment}")

    enhanced_parts.append(definition)
    return "\n".join(enhanced_parts)
```

### 增强字段

| 字段 | 优先级 | 说明 |
|------|--------|------|
| `table_comment` | HIGH | 从 DDL COMMENT 提取 |
| `column_comments` | HIGH | 列注释 JSON |
| `business_tags` | HIGH | 业务领域标签 |
| `row_count` | MEDIUM | 表行数 |
| `sample_statistics` | MEDIUM | 列统计信息 |
| `relationship_metadata` | MEDIUM | 外键和关联路径 |
| `metadata_version` | - | 元数据版本 (0=旧版, 1=增强) |

---

## 搜索模式

### 1. 向量搜索

基于语义相似度的搜索:
```python
search(query_txt, select_fields, top_n, where, reranker)
```

### 2. 混合搜索

结合向量和全文搜索，可选重排序:
```python
search(
    query_txt,
    reranker=custom_reranker,  # 可选重排序器
    ...
)
```

### 3. 条件构建 DSL

类型安全的查询条件构建:
```python
from datus.storage.lancedb_conditions import and_, or_, eq, in_, like

# 构建复杂条件
where = and_(
    eq("database_name", "sales"),
    in_("table_type", ["table", "view"]),
    like("table_name", "customer%")
)
```

### 4. 主题过滤搜索

结合主题树的层次化过滤:
```python
search_with_subject_filter(
    query_text="销售指标",
    subject_path=["Finance", "Revenue"],  # 主题路径
    top_n=5
)
```

### 5. Schema Discovery 混合检索（向量 + FTS + Rerank）

Schema metadata 同时支持向量检索与 FTS（全文检索）。在 Text2SQL 的 Schema Discovery 中会进行融合排序，并可选
启用 LanceDB rerank（hybrid query + rerank）提升精度。满足 `hybrid_rerank_enabled=true`、候选表数量 ≥
`hybrid_rerank_min_tables`、模型文件存在且资源充足时触发；若模型缺失/资源不足或 reranker 初始化失败则自动跳过。

**配置参数（agent.yml）**:
```yaml
schema_discovery:
  hybrid_search_enabled: true
  hybrid_use_fts: true
  hybrid_vector_weight: 0.6
  hybrid_fts_weight: 0.3
  hybrid_row_count_weight: 0.2
  hybrid_tag_bonus: 0.1
  hybrid_comment_bonus: 0.05
  hybrid_rerank_enabled: false
  hybrid_rerank_weight: 0.2
  hybrid_rerank_min_tables: 20
  hybrid_rerank_top_n: 50
  hybrid_rerank_model: "./models/bge-reranker-large"
  hybrid_rerank_column: "definition"
  hybrid_rerank_min_cpu_count: 4
  hybrid_rerank_min_memory_gb: 8.0
```

**使用建议**:
- 短 Query 或命名依赖场景：提高 `hybrid_fts_weight`
- 表数量较大时启用 `hybrid_rerank_enabled`
- 若资源紧张，保持 rerank 关闭（默认）
- 启用 rerank 需要对应模型可用（如 `BAAI/bge-reranker-large`）

**权重校验规则**:
- `hybrid_*_weight`、`hybrid_tag_bonus`、`hybrid_comment_bonus` 必须在 `[0,1]`，否则回退默认值并记录 warning
- `hybrid_rerank_min_tables` 必须 `>= 0`，否则回退为 20
- `hybrid_rerank_top_n` 必须 `>= 1`，否则回退为 50
- `hybrid_rerank_model` / `hybrid_rerank_column` 用于配置 reranker 模型与输入列（`hybrid_rerank_model` 需为本地模型路径）
- `hybrid_rerank_min_cpu_count` / `hybrid_rerank_min_memory_gb` 用于资源门槛校验，不满足则禁用 rerank

**量大场景推荐组合**:

1) SQL 生成（高并发 + 覆盖优先）
```yaml
schema_discovery:
  hybrid_search_enabled: true
  hybrid_use_fts: true
  hybrid_vector_weight: 0.5
  hybrid_fts_weight: 0.4
  hybrid_row_count_weight: 0.2
  hybrid_tag_bonus: 0.1
  hybrid_comment_bonus: 0.05
  hybrid_rerank_enabled: false
```
建议：偏重 FTS 提升召回稳定性，关闭 rerank 降低算力成本。

2) SQL 审查（高精度 + 误报控制）
```yaml
schema_discovery:
  hybrid_search_enabled: true
  hybrid_use_fts: true
  hybrid_vector_weight: 0.6
  hybrid_fts_weight: 0.3
  hybrid_row_count_weight: 0.2
  hybrid_tag_bonus: 0.1
  hybrid_comment_bonus: 0.05
  hybrid_rerank_enabled: true
  hybrid_rerank_weight: 0.2
  hybrid_rerank_min_tables: 10
  hybrid_rerank_top_n: 50
  hybrid_rerank_model: "./models/bge-reranker-large"
  hybrid_rerank_column: "definition"
  hybrid_rerank_min_cpu_count: 4
  hybrid_rerank_min_memory_gb: 8.0
```
建议：审查场景更强调表/列精确度，开启 rerank 并降低触发门槛。

---

## 初始化命令

### bootstrap-kb 命令

```bash
datus-agent bootstrap-kb --namespace <your_namespace> --kb_update_strategy <strategy>
```

**策略选项**:
- `check`: 检查当前数据条目数量
- `overwrite`: 完全覆盖现有数据
- `incremental`: 增量更新 (变更更新 + 新增追加)

### 示例

```bash
# 检查状态
datus-agent bootstrap-kb --namespace my_database --kb_update_strategy check

# 完全重建
datus-agent bootstrap-kb --namespace my_database --kb_update_strategy overwrite

# 增量更新
datus-agent bootstrap-kb --namespace my_database --kb_update_strategy incremental
```

### 子代理 KB 初始化

```python
# 代码位置: datus/storage/sub_agent_kb_bootstrap.py
from datus.storage.sub_agent_kb_bootstrap import SubAgentKBBootstrap

# 初始化子代理知识库
bootstrap = SubAgentKBBootstrap(agent_config)
bootstrap.bootstrap_sub_agent_kb(sub_agent_name, kb_update_strategy)
```

---

## 最佳实践

### 1. 数据库配置

```yaml
# agent.yml
namespace:
  production_snowflake:
    type: snowflake
    account: ${SNOWFLAKE_ACCOUNT}
    username: ${SNOWFLAKE_USER}
    password: ${SNOWFLAKE_PASSWORD}
    database: ANALYTICS
```

### 2. 嵌入模型选择

```python
# 支持的嵌入模型
- bge-large-zh-v1.5   # 中文优先 (1024 维)
- bge-large-en-v1.5   # 英文优先 (1024 维)
- multilingual-e5     # 多语言支持
```

### 3. 索引创建时机

```python
# 数据加载完成后创建索引
def after_init():
    schema_store.create_indices()
    value_store.create_indices()
```

### 4. 批处理优化

```python
# 大数据量分批处理
batch_size = 500
for i in range(0, len(data), batch_size):
    batch = data[i:i + batch_size]
    store.store_batch(batch)
```

---

## 故障排查

### 常见问题

1. **权限错误**: 确保数据库用户有访问系统表/信息模式的权限
2. **连接超时**: 检查网络连接和数据库可用性
3. **嵌入模型失败**: 检查模型文件是否已下载到 `~/.datus/models/`
4. **存储路径冲突**: 子代理使用独立存储路径避免冲突

### 验证方法

```bash
# 检查配置
datus-agent --config-check

# 测试数据库连接
datus-agent --test-db-connection

# 验证工作流
datus-agent --validate-workflow
```

---

## API 参考

### SchemaStorage

```python
class SchemaStorage(BaseMetadataStorage):
    def search_similar(
        self,
        query_text: str,
        catalog_name: str = "",
        database_name: str = "",
        schema_name: str = "",
        top_n: int = 5,
        table_type: TABLE_TYPE = "table",
        reranker: Optional[Reranker] = None,
    ) -> pa.Table

    def do_search_similar(
        self,
        query_text: str,
        top_n: int = 5,
        where: WhereExpr = None,
        reranker: Optional[Reranker] = None,
    ) -> pa.Table

    def search_all(
        self,
        catalog_name: str = "",
        database_name: str = "",
        schema_name: str = "",
        table_type: TABLE_TYPE = "full",
        select_fields: Optional[List[str]] = None,
    ) -> pa.Table

    def create_indices(self) -> None
```

### MetricStorage

```python
class MetricStorage(BaseSubjectEmbeddingStore):
    def search_metrics(
        self,
        query_text: str = "",
        semantic_model_names: Optional[List[str]] = None,
        subject_path: Optional[List[str]] = None,
        top_n: int = 5,
    ) -> List[Dict[str, Any]]

    def search_all_metrics(
        self,
        semantic_model_names: Optional[List[str]] = None,
        subject_path: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]

    def batch_store_metrics(self, metrics: List[Dict[str, Any]]) -> None
```

### ReferenceSqlStorage

```python
class ReferenceSqlStorage(BaseSubjectEmbeddingStore):
    def search_reference_sql(
        self,
        query_text: Optional[str] = None,
        subject_path: Optional[List[str]] = None,
        top_n: Optional[int] = 5,
        selected_fields: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]

    def search_all_reference_sql(
        self,
        subject_path: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]

    def batch_store_sql(
        self,
        sql_items: List[Dict[str, Any]],
        subject_path_field: str = "subject_path",
    ) -> None
```

### StorageCache

```python
class StorageCache:
    def schema_storage(self, sub_agent_name: Optional[str] = None) -> SchemaStorage
    def schema_value_storage(self, sub_agent_name: Optional[str] = None) -> SchemaValueStorage
    def metrics_storage(self, sub_agent_name: Optional[str] = None) -> MetricStorage
    def semantic_storage(self, sub_agent_name: Optional[str] = None) -> SemanticModelStorage
    def reference_sql_storage(self, sub_agent_name: Optional[str] = None) -> ReferenceSqlStorage
```

### SubjectTreeStore

```python
class SubjectTreeStore:
    def create_node(
        self,
        parent_id: Optional[int],
        name: str,
        description: str = ""
    ) -> Dict[str, Any]

    def get_node(self, node_id: int) -> Optional[Dict[str, Any]]

    def get_node_by_path(self, path: List[str]) -> Optional[Dict[str, Any]]

    def find_or_create_path(self, path_components: List[str]) -> int

    def get_full_path(self, node_id: int) -> List[str]

    def rename(self, old_path: List[str], new_path: List[str]) -> bool

    def get_tree_structure(self) -> Dict[str, Any]
    def get_simple_tree_structure(self) -> Dict[str, Any]
```

---

## 架构演进

### v0 → v1 主要变化

1. **增强元数据**: 添加 `table_comment`, `column_comments`, `column_enums`, `business_tags` 等字段
2. **中文注释**: DDL 自动添加中文注释前缀
3. **主题树集成**: 指标和 SQL 支持层次化主题组织
4. **子代理隔离**: 支持独立的知识库作用域
5. **智能索引**: 根据数据量自动选择索引类型 (IVF_PQ / IVF_FLAT)
6. **缓存优化**: LRU 缓存提升性能
7. **新增字段**: `metadata_version`, `last_updated`, `relationship_metadata`

---

## 版本更新记录

### v2.1 (2026-01-23)
- 新增 `column_enums` 字段到 SchemaStorage (列枚举值)
- 新增 `last_updated` 字段 (更新时间戳)
- 新增 `relationship_metadata` 字段 (外键和关联路径)
- 新增 `do_search_similar` 内部搜索方法
- 新增 `search_all` 方法的 `select_fields` 参数
- 新增 ReferenceSqlStorage API 参考
- 新增 StorageCache API 参考
- 新增 SubAgentKBBootstrap 初始化说明
- 完善 SubjectTreeStore 数据库索引定义
- 修正缓存机制文档 (StorageCacheHolder, StorageCache)

### v2.0 (2026-01-22)
- 完整重写，基于最新代码架构
- 新增 LanceDB 向量存储核心实现
- 新增多层知识存储系统 (Schema, Metric, ReferenceSQL, Document)
- 新增主题树层次化组织
- 新增子代理作用域存储
- 新增智能索引策略
- 新增中文注释增强

---

## 相关资源

- **项目主页**: [https://datus.ai](https://datus.ai)
- **文档**: [https://docs.datus.ai](https://docs.datus.ai)
- **GitHub**: [https://github.com/Datus-ai/Datus-agent](https://github.com/Datus-ai/Datus-agent)
- **Slack 社区**: [https://join.slack.com/t/datus-ai](https://join.slack.com/t/datus-ai/shared_invite/zt-3g6h4fsdg-iOl5uNoz6A4GOc4xKKWUYg)
