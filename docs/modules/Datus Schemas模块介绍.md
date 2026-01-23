# Datus Schemas 模块介绍

> **文档版本**: v2.1
> **更新日期**: 2026-01-23
> **相关模块**: `datus/schemas/`
> **代码仓库**: [Datus Agent](https://github.com/Datus-ai/Datus-agent)

---

## 模块概述

### 核心功能

Datus Schemas 模块基于 **Pydantic v2** 提供类型安全的数据模型定义，为 Text2SQL 工作流的各个节点提供统一的输入输出验证、序列化和反序列化能力。

### 设计架构

```
┌─────────────────────────────────────────────────────────────┐
│                   应用层模型 (Node-Specific Models)          │
│  ┌───────────────┬───────────────┬───────────────┬─────────┐│
│  │GenSQLNodeInput│SchemaLinking  │ReasoningInput │DocSearch││
│  │               │Input/Result   │               │Input    ││
│  └───────────────┴───────────────┴───────────────┴─────────┘│
├─────────────────────────────────────────────────────────────┤
│                    核心节点模型 (Node Models)                │
│  ┌───────────────┬───────────────┬───────────────┬─────────┐│
│  │SqlTask        │TableSchema    │SQLContext     │Metric   ││
│  │GenerateSQL...│ExecuteSQL...  │Reflection...  │Context  ││
│  └───────────────┴───────────────┴───────────────┴─────────┘│
├─────────────────────────────────────────────────────────────┤
│                   基础模型 (Base Models)                     │
│  ┌───────────────┬───────────────┬───────────────┬─────────┐│
│  │BaseInput      │BaseResult     │CommonData     │TABLE_   ││
│  │               │               │               │TYPE     ││
│  └───────────────┴───────────────┴───────────────┴─────────┘│
├─────────────────────────────────────────────────────────────┤
│                   追踪与配置 (Action & Config)               │
│  ┌───────────────┬───────────────┬───────────────┬─────────┐│
│  │ActionHistory  │ActionHistory  │SubAgentConfig │Scoped   ││
│  │               │Manager        │               │Context  ││
│  └───────────────┴───────────────┴───────────────┴─────────┘│
└─────────────────────────────────────────────────────────────┘
```

---

## 文件结构

### 基础模型层

| 文件名 | 说明 |
|--------|------|
| `base.py` | 基础输入/输出模型、表类型枚举 |
| `action_history.py` | 动作历史追踪和管理 |
| `agent_models.py` | 子代理配置和作用域上下文 |

### 核心节点模型

| 文件名 | 说明 |
|--------|------|
| `node_models.py` | 核心节点数据模型 (SqlTask, TableSchema, SQLContext 等) |

### 专用节点模型

| 文件名 | 说明 |
|--------|------|
| `gen_sql_agentic_node_models.py` | GenSQL Agentic 节点模型 |
| `schema_linking_node_models.py` | 模式链接节点模型 |
| `reason_sql_node_models.py` | SQL 推理节点模型 |
| `search_metrics_node_models.py` | 指标搜索节点模型 |
| `doc_search_node_models.py` | 文档搜索节点模型 |
| `date_parser_node_models.py` | 日期解析节点模型 |
| `parallel_node_models.py` | 并行执行节点模型 |
| `compare_node_models.py` | 比较节点模型 |
| `fix_node_models.py` | 修复节点模型 |
| `subworkflow_node_models.py` | 子工作流节点模型 |
| `semantic_agentic_node_models.py` | 语义 Agentic 节点模型 |
| `chat_agentic_node_models.py` | 聊天 Agentic 节点模型 |
| `sql_summary_agentic_node_models.py` | SQL 摘要 Agentic 节点模型 |
| `tool_models.py` | 工具模型 (预留) |
| `visualization.py` | 可视化模型 |

---

## 核心数据模型详解

### 1. 基础模型 (base.py)

#### BaseInput

所有节点输入的基类，提供字典式访问和序列化能力。

```python
class BaseInput(BaseModel):
    class Config:
        extra = "allow"  # 允许额外字段以支持灵活的节点输入

    def get(self, key: str, default: Any = None) -> Any
    def __getitem__(self, key: str) -> Any
    def to_str(self) -> str
    @classmethod
    def from_str(cls, json_str: str) -> "BaseInput"
    def to_dict(self) -> Dict[str, Any]
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseInput"
```

**使用示例**:
```python
# 字典式访问
input_data["database_name"]
input_data.get("schema_name", "public")

# 序列化
json_str = input_data.to_str()
restored = BaseInput.from_str(json_str)
```

#### BaseResult

所有节点输出的基类，包含执行状态和错误信息。

```python
class BaseResult(BaseModel):
    success: bool                    # 操作是否成功
    error: Optional[str]             # 错误信息
    message: Optional[str]           # 通用消息或状态信息
    data: Optional[Dict[str, Any]]   # 通用数据负载

    # Agentic 节点专用字段
    action_history: Optional[List[dict]]    # 工具调用和动作的完整历史
    execution_stats: Optional[dict]         # 执行统计 (tokens, tools, duration 等)

    class Config:
        extra = "forbid"  # 禁止未定义的额外字段
```

#### CommonData

通用数据结构的基类，提供与 BaseInput 相同的访问接口。

#### TABLE_TYPE

表类型字面量类型:

```python
TABLE_TYPE = Literal["table", "view", "mv", "full"]

def parse_table_type_by_db(db_table_type: str) -> TABLE_TYPE:
    """将数据库类型转换为标准 TABLE_TYPE"""
```

---

### 2. 核心节点模型 (node_models.py)

#### SqlTask

SQL 任务定义模型，包含任务描述和数据库上下文。

```python
class SqlTask(BaseModel):
    id: str                              # 任务 ID
    database_type: str                   # 数据库类型
    task: str                            # SQL 任务描述或查询
    catalog_name: str                    # 目录名
    database_name: str                   # 数据库名
    schema_name: str                     # 模式名
    output_dir: str                      # 输出目录
    external_knowledge: str              # 外部知识
    tables: Optional[List[str]] = Field(default=[], description="List of table names to use")
    schema_linking_type: TABLE_TYPE      # 模式链接类型
    current_date: Optional[str]          # 当前日期引用
    date_ranges: str                     # 解析的日期范围
    subject_path: Optional[List[str]]    # 主题层次路径
```

**字段验证器**:
```python
@field_validator("task")
def validate_task(cls, v):
    if not v.strip():
        raise ValueError("'task' must not be empty")
    return v
```

#### TableSchema

表结构信息模型。

```python
class BaseTableSchema(BaseModel):
    identifier: str      # 唯一标识符
    catalog_name: str    # 目录名
    table_name: str      # 表名
    database_name: str   # 数据库名
    schema_name: str     # 模式名

class TableSchema(BaseTableSchema):
    definition: str      # DDL 模式文本
    table_type: str      # 模式类型

    def to_prompt(self, dialect: str = "snowflake", include_ddl: bool = True) -> str:
        """转换为 LLM prompt 字符串"""

    @classmethod
    def table_names_to_prompt(cls, schemas: List[TableSchema]) -> str:
        """仅返回表名列表"""

    @classmethod
    def list_to_prompt(cls, schemas: List[TableSchema], dialect: str = "snowflake") -> str:
        """批量转换为 prompt"""

    @classmethod
    def from_arrow(cls, table: pa.Table) -> List[TableSchema]:
        """从 PyArrow 表创建实例列表"""
```

**to_prompt 方法详解**:
- `include_ddl=True`: 返回完整 DDL (用于列/度量选择阶段)
- `include_ddl=False`: 仅返回表名和注释 (用于表选择阶段)

#### TableValue

表样本数据模型。

```python
class TableValue(BaseTableSchema):
    table_values: str     # 表的样本值
    table_type: str       # 模式类型

    def to_prompt(
        self,
        dialect: str = "snowflake",
        max_value_length: int = 500,
        max_text_mark_length: int = 16,
        processed_schemas: str = "",
    ) -> str:
        """转换为简洁的 LLM prompt 字符串"""

    @classmethod
    def from_arrow(cls, table: pa.Table) -> List[TableValue]:
        """从 PyArrow 表创建实例列表"""
```

**TEXT 列处理**: 对于 SQLite 数据库，可以自动将长 TEXT 列值替换为 `<TEXT>` 标记以节省 token。

#### Metric

业务指标模型。

```python
class Metric(BaseModel):
    name: str            # 指标名称
    llm_text: str        # LLM 友好的文本表示

    def to_prompt(self, dialect: str = "snowflake") -> str:
        return self.llm_text if self.llm_text else f"Metric: {self.name}"
```

#### ReferenceSql

参考 SQL 模型。

```python
class ReferenceSql(BaseModel):
    name: str            # SQL 名称
    sql: str             # SQL 查询
    comment: str         # 注释
    summary: str         # 摘要
    tags: str            # 标签
```

#### SQLContext

SQL 执行上下文模型。

```python
class SQLContext(BaseModel):
    sql_query: str                  # 生成的 SQL 查询
    explanation: Optional[str]      # SQL 查询解释
    sql_return: Any                 # SQL 执行结果
    sql_error: Optional[str]        # SQL 执行错误
    row_count: Optional[int]        # 返回行数
    reflection_strategy: Optional[str]     # 反思策略
    reflection_explanation: Optional[str]  # 反思解释

    def to_str(self, max_sql_return_length: int = 4294967296) -> str:
        """转换为字符串表示，支持截断"""

    def to_sample_str(self) -> str:
        """转换为示例字符串"""
```

#### Context

工作流上下文模型，聚合所有相关信息。

```python
class Context(BaseModel):
    sql_contexts: List[SQLContext]          # SQL 上下文列表
    table_schemas: List[TableSchema]        # 表结构
    table_values: List[TableValue]          # 表样本数据
    metrics: List[Metric]                   # 指标
    reference_sqls: List[ReferenceSql]      # 参考 SQL
    doc_search_keywords: List[str]          # 文档搜索关键词
    document_result: Optional[DocSearchResult]  # 文档搜索结果
    parallel_results: Optional[Dict[str, Any]]  # 并行执行结果
    last_selected_result: Optional[Any]     # 最后选择的结果
    selection_metadata: Optional[Dict[str, Any]]  # 选择过程元数据
    preflight_results: Optional[Dict[str, Any]] = Field(
        default=None, description="Results from preflight tool execution"
    )

    def update_schema_and_values(...)
    def update_last_sql_context(...)
    def update_metrics(...)
    def update_reference_sqls(...)
    def to_str(self) -> str:
```

**特殊警告**: `update_schema_and_values()` 方法调用后会:
1. `schema_linking_node` 将始终返回更新值而不在向量库中匹配
2. 值将在 `agentic_node` 中展开并拼接到用户问题中

#### GenerateSQLInput / GenerateSQLResult

SQL 生成节点的输入输出模型。

```python
class GenerateSQLInput(BaseInput):
    database_type: Optional[str]
    table_schemas: Union[List[TableSchema], str]
    data_details: Optional[List[TableValue]]
    metrics: Optional[List[Metric]]
    sql_task: SqlTask
    contexts: Optional[List[SQLContext]]
    external_knowledge: str
    prompt_version: str
    max_table_schemas_length: int = 4000
    max_data_details_length: int = 2000
    max_context_length: int = 8000
    max_value_length: int = 500
    max_text_mark_length: int = 16
    database_docs: Optional[str]
    include_schema_ddl: bool = False  # 是否在 prompt 中包含完整 DDL

class GenerateSQLResult(BaseResult):
    sql_query: str                    # 生成的 SQL 查询
    tables: List[str]                 # 查询中使用的表
    explanation: Optional[str]        # SQL 查询解释
```

#### ExecuteSQLInput / ExecuteSQLResult

SQL 执行节点的输入输出模型。

```python
class ExecuteSQLInput(BaseInput):
    database_name: str
    sql_query: str
    result_format: str = "csv"        # "csv" | "arrow" | "list"
    query_timeout_seconds: Optional[int]

    # 安全标志 (需要显式授权)
    allow_ddl: bool = False           # 允许 DDL 操作 (CREATE, ALTER, DROP)
    allow_dml: bool = False           # 允许 DML 操作 (INSERT, UPDATE, DELETE)
    require_explain_only: bool = False  # 仅允许 EXPLAIN 查询
    permission_reason: Optional[str]  # 权限原因 (用于审计日志)

class ExecuteSQLResult(BaseResult):
    sql_query: Optional[str]
    row_count: Optional[int]
    sql_return: Any                   # 执行结果 (支持多种格式)
    result_format: str

    def compact_result(self) -> str:
        """返回紧凑的执行结果字符串表示"""
```

#### SQLValidateInput / SQLValidateResult

SQL 验证节点的输入输出模型。

```python
class SQLValidateInput(BaseInput):
    sql_query: str
    dialect: Optional[str]           # 数据库方言
    check_table_existence: bool = True   # 检查表是否存在
    check_column_existence: bool = True  # 检查列是否存在
    check_dangerous_operations: bool = True  # 检查危险操作
```

#### ReflectionInput / ReflectionResult

反思节点的输入输出模型。

```python
class ReflectionInput(BaseInput):
    task_description: SqlTask
    sql_context: List[SQLContext]
    prompt_version: str = "2.1"
    sql_return_sample_line: int = 10  # -1 表示返回所有行

class StrategyType(str, Enum):
    SUCCESS = "SUCCESS"
    DOC_SEARCH = "DOC_SEARCH"
    SIMPLE_REGENERATE = "SIMPLE_REGENERATE"
    SCHEMA_LINKING = "SCHEMA_LINKING"
    REASONING = "REASONING"
    COLUMN_EXPLORATION = "COLUMN_EXPLORATION"
    UNKNOWN = "UNKNOWN"

class ReflectionResult(BaseResult):
    strategy: Optional[StrategyType]
    details: Dict[str, Union[str, List[str], Dict[str, Any]]]
```

---

### 3. 动作历史模型 (action_history.py)

#### ActionRole

动作角色枚举。

```python
class ActionRole(str, Enum):
    SYSTEM = "system"        # 系统 prompt 使用
    ASSISTANT = "assistant"  # AI 助手角色
    USER = "user"            # 用户角色
    TOOL = "tool"            # MCP 工具角色
    WORKFLOW = "workflow"    # 工作流角色
```

#### ActionStatus

动作状态枚举。

```python
class ActionStatus(str, Enum):
    PROCESSING = "processing"  # 处理中
    SUCCESS = "success"        # 成功
    FAILED = "failed"          # 失败
    SOFT_FAILED = "soft_failed"  # 软失败 (允许通过反思恢复)
```

#### ActionHistory

动作历史记录模型。

```python
class ActionHistory(BaseModel):
    action_id: str                 # 唯一标识符
    role: ActionRole               # 动作创建者角色
    messages: str                  # 思考或推理 (AI 或人类消息)
    action_type: str               # 动作类型 (NodeType / MCP tool name / message)
    input: Any                     # 输入数据
    output: Any                    # 输出数据
    status: ActionStatus           # 动作状态
    start_time: datetime           # 开始时间
    end_time: Optional[datetime]   # 结束时间

    @classmethod
    def create_action(
        cls,
        role: ActionRole,
        action_type: str,
        messages: str,
        input_data: dict,
        output_data: dict = None,
        output: dict = None,  # Backward compatibility
        status: ActionStatus = ActionStatus.PROCESSING,
    ) -> "ActionHistory"

    def is_done(self) -> bool:
        """检查动作是否完成"""
    def function_name(self) -> str:
        """获取函数名"""
```

#### ActionHistoryManager

动作历史管理器。

```python
class ActionHistoryManager:
    def __init__(self):
        self.actions: List[ActionHistory] = []
        self.current_action_id: Optional[str] = None

    def add_action(self, action: ActionHistory) -> None:
        """添加动作到历史 (通过 action_id 防止重复)"""

    def update_current_action(self, **kwargs) -> None:
        """更新当前动作"""

    def get_actions(self) -> List[ActionHistory]:
        """获取所有动作历史"""

    def clear(self) -> None:
        """清空所有动作"""

    def find_action_by_id(self, action_id: str) -> Optional[ActionHistory]:
        """通过 action_id 查找动作"""

    def update_action_by_id(self, action_id: str, **kwargs) -> bool:
        """通过 action_id 更新动作"""
```

---

### 4. 代理配置模型 (agent_models.py)

#### ScopedContextLists

作用域上下文列表 (规范化格式)。

```python
class ScopedContextLists(BaseModel):
    tables: List[str]    # 规范化的表标识符
    metrics: List[str]   # 规范化的指标标识符
    sqls: List[str]      # 规范化的 SQL 标识符

    def any(self) -> bool:
        """是否有任何作用域上下文"""
```

#### ScopedContext

作用域上下文配置。

```python
class ScopedContext(BaseModel):
    namespace: Optional[str]   # 对应数据源的命名空间
    tables: Optional[str]       # 子代理要使用的表 (逗号分隔)
    metrics: Optional[str]      # 子代理要使用的指标 (逗号分隔)
    sqls: Optional[str]         # 子代理要使用的参考 SQL (逗号分隔)

    @property
    def is_empty(self) -> bool:
        """是否为空"""

    def as_lists(self) -> ScopedContextLists:
        """转换为规范化列表格式"""
```

**字符串解析逻辑**:
- 支持逗号或换行符分隔
- 自动去重
- 去除空白字符

#### SubAgentConfig

子代理配置模型。

```python
class SubAgentConfig(BaseModel):
    system_prompt: str                      # 子代理名称
    agent_description: Optional[str]        # 子代理描述
    tools: str                              # 原生工具 (逗号分隔)
    mcp: str                                # MCP 工具 (逗号分隔)
    scoped_context: Optional[ScopedContext] # 作用域上下文
    rules: List[str]                        # 子代理规则列表
    prompt_version: str = "1.0"             # System Prompt 版本
    prompt_language: str = "en"             # System Prompt 语言
    scoped_kb_path: Optional[str]           # 作用域 KB 存储路径

    def has_scoped_context(self) -> bool:
        """是否有作用域上下文"""

    def has_scoped_context_by(self, attr_name: str) -> bool:
        """是否有特定的作用域上下文属性"""

    def is_in_namespace(self, namespace: str) -> bool:
        """是否在指定的命名空间中"""

    def as_payload(self, namespace: Optional[str] = None) -> Dict[str, Any]:
        """转换为负载格式"""

    @property
    def tool_list(self) -> List[str]:
        """获取工具列表"""
```

---

### 5. 专用节点模型

#### SchemaLinkingInput / SchemaLinkingResult

模式链接节点模型。

```python
class SchemaLinkingInput(BaseInput):
    input_text: str
    database_type: str
    catalog_name: str
    database_name: str
    schema_name: str
    matching_rate: Literal["fast", "medium", "slow", "from_llm"] = "fast"
    sql_context: Optional[SQLContext]
    prompt_version: str = "1.0"
    top_n: int = 5
    table_type: TABLE_TYPE = "table"

    def top_n_by_rate(self) -> int:
        """根据匹配率获取 top_n"""
        # fast → 5, medium → 10, slow/from_llm → 20

    @classmethod
    def from_sql_task(cls, sql_task: SqlTask, matching_rate: str = "fast") -> SchemaLinkingInput

class SchemaLinkingResult(BaseResult):
    table_schemas: List[TableSchema]
    schema_count: int
    table_values: List[TableValue]
    value_count: int

    def compact_result(self) -> str:
        """返回紧凑的结果字符串"""
```

#### SearchMetricsInput / SearchMetricsResult

指标搜索节点模型。

```python
class SearchMetricsInput(BaseInput):
    input_text: str
    sql_task: SqlTask
    database_type: str
    sql_contexts: Optional[List[SQLContext]]
    top_n: int = 5
    matching_rate: Literal["fast", "medium", "slow"] = "fast"

    def top_n_by_rate(self) -> int:
        # fast → 5, medium → 10, slow → 20

class SearchMetricsResult(BaseResult):
    sql_task: SqlTask
    metrics: List[Metric]
    metrics_count: int
```

#### DocSearchInput / DocSearchResult

文档搜索节点模型。

```python
class DocSearchInput(BaseInput):
    keywords: List[str]
    top_n: int = 5
    method: Literal["internal", "external", "llm"] = "internal"

class DocSearchResult(BaseResult):
    docs: Dict[str, List[str]]    # 关键词 → 文档文本列表
    doc_count: int
```

#### ReasoningInput / ReasoningResult

SQL 推理节点模型 (复用 GenerateSQLInput 和 ExecuteSQLResult)。

```python
class ReasoningInput(GenerateSQLInput):
    max_table_schemas_length: int = 4000
    max_data_details_length: int = 2000
    max_context_length: int = 8000
    max_value_length: int = 500
    max_sql_return_length: int = 1000
    max_text_mark_length: int = 16
    prompt_version: str = "1.0"
    max_turns: int = 20

class ReasoningResult(ExecuteSQLResult):
    sql_contexts: List[SQLContext]
```

#### GenSQLNodeInput / GenSQLNodeResult

GenSQL Agentic 节点模型。

```python
class GenSQLNodeInput(BaseInput):
    user_message: str
    catalog: Optional[str]
    database: Optional[str]
    db_schema: Optional[str]
    max_turns: int = 30
    external_knowledge: Optional[str] = ""
    workspace_root: Optional[str]    # 文件系统 MCP 服务器根目录
    prompt_version: Optional[str]
    prompt_language: Optional[str] = "en"
    schemas: Optional[list[TableSchema]]
    metrics: Optional[list[Metric]]
    reference_sql: Optional[list[ReferenceSql]]
    plan_mode: bool = False          # 启用计划模式
    auto_execute_plan: bool = False  # 自动执行计划

class GenSQLNodeResult(BaseResult):
    response: str                    # AI 助手响应
    sql: Optional[str]               # 生成或引用的 SQL 查询
    tokens_used: int = 0             # 使用的总 tokens
```

#### VisualizationInput / VisualizationOutput

可视化推荐模型。

```python
class VisualizationInput(BaseInput):
    data: DataLike                   # DataFrame, List[Dict], 或 PyArrow Table

class VisualizationOutput(BaseResult):
    chart_type: str                  # 图表类型
    x_col: str                       # X轴列名
    y_cols: list[str]                # Y轴列名列表
    reason: str                      # 推荐理由
```

---

## 数据流设计

### 输入验证流

```
用户输入 → Pydantic 验证 → BaseInput 标准化 → 节点处理
```

### 输出处理流

```
节点结果 → BaseResult 包装 → 序列化传输 → 客户端解析
```

### 动作追踪流

```
执行开始 → ActionHistory 记录 → 状态更新 → 执行结束 → 历史归档
```

### 上下文聚合流

```
各个节点输出 → Context 聚合 → 下一节点输入 → ...
```

---

## 设计特性

### 1. 类型安全

- 使用 Pydantic v2 提供运行时类型验证
- 自动类型转换和 coercion
- 详细的验证错误信息

### 2. 序列化支持

```python
# JSON 序列化
json_str = model.to_str()
restored = ModelClass.from_str(json_str)

# 字典序列化
data_dict = model.to_dict()
restored = ModelClass.from_dict(data_dict)

# PyArrow 支持
TableSchema.from_arrow(arrow_table)
TableValue.from_arrow(arrow_table)
```

### 3. 字典式访问

```python
# 所有模型支持字典式访问
value = model["field_name"]
value = model.get("field_name", default_value)
```

### 4. 灵活配置

```python
# BaseInput 允许额外字段
class Config:
    extra = "allow"

# BaseResult 禁止额外字段
class Config:
    extra = "forbid"
```

### 5. Prompt 生成

```python
# TableSchema 支持 LLM prompt 生成
schema.to_prompt(dialect="snowflake", include_ddl=True)
TableSchema.list_to_prompt(schemas, dialect="sqlite")

# Metric 支持 LLM prompt 生成
metric.to_prompt(dialect="snowflake")

# TableValue 支持简洁化输出
value.to_prompt(max_value_length=500, max_text_mark_length=16)
```

### 6. 字段验证器

```python
@field_validator("task")
def validate_task(cls, v):
    if not v.strip():
        raise ValueError("'task' must not be empty")
    return v

@field_validator("top_n")
def validate_top_n(cls, v):
    if v <= 0:
        raise ValueError("'top_n' must be greater than 0")
    return v
```

---

## 最佳实践

### 1. 模型继承

```python
# 创建专用输入模型
class MyNodeInput(BaseInput):
    custom_field: str = Field(..., description="Custom field")

# 创建专用结果模型
class MyNodeResult(BaseResult):
    custom_result: Any = Field(..., description="Custom result")
```

### 2. 字段定义

```python
class MyModel(BaseModel):
    # 必填字段
    required_field: str = Field(..., description="Required field")

    # 可选字段
    optional_field: Optional[str] = Field(None, description="Optional field")

    # 默认值字段
    default_field: str = Field(default="default", description="Default field")

    # 列表字段
    list_field: List[str] = Field(default_factory=list, description="List field")
```

### 3. 枚举使用

```python
class MyEnum(str, Enum):
    VALUE1 = "value1"
    VALUE2 = "value2"

class MyModel(BaseModel):
    enum_field: MyEnum = Field(default=MyEnum.VALUE1)

    class Config:
        use_enum_values = True  # 使用枚举值而非枚举对象
```

### 4. 上下文管理

```python
# 更新 Context
context.update_schema_and_values(table_schemas, table_values)
context.update_last_sql_context(sql_context)
context.update_metrics(metrics)
context.update_reference_sqls(reference_sqls)
```

### 5. 动作历史

```python
# 创建动作历史管理器
manager = ActionHistoryManager()

# 添加动作
action = ActionHistory.create_action(
    role=ActionRole.TOOL,
    action_type="search_table",
    messages="Searching for relevant tables",
    input_data={"query": "user query"},
    status=ActionStatus.PROCESSING
)
manager.add_action(action)

# 更新当前动作
manager.update_current_action(status=ActionStatus.SUCCESS, output_data={"tables": [...]})

# 获取历史
actions = manager.get_actions()
```

---

## API 参考

### BaseInput

```python
class BaseInput(BaseModel):
    def get(self, key: str, default: Any = None) -> Any
    def __getitem__(self, key: str) -> Any
    def to_str(self) -> str
    @classmethod
    def from_str(cls, json_str: str) -> "BaseInput"
    def to_dict(self) -> Dict[str, Any]
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseInput"
```

### BaseResult

```python
class BaseResult(BaseModel):
    success: bool
    error: Optional[str]
    message: Optional[str]
    data: Optional[Dict[str, Any]]
    action_history: Optional[List[dict]]
    execution_stats: Optional[dict]

    def get(self, key: str, default: Any = None) -> Any
    def __getitem__(self, key: str) -> Any
    def to_str(self) -> str
    def to_dict(self) -> Dict[str, Any]
    @classmethod
    def from_str(cls, json_str: str) -> "BaseResult"
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseResult"
```

### SqlTask

```python
class SqlTask(BaseModel):
    id: str
    database_type: str
    task: str
    catalog_name: str
    database_name: str
    schema_name: str
    output_dir: str
    external_knowledge: str
    tables: Optional[List[str]]
    schema_linking_type: TABLE_TYPE
    current_date: Optional[str]
    date_ranges: str
    subject_path: Optional[List[str]]
```

### TableSchema

```python
class TableSchema(BaseModel):
    identifier: str
    catalog_name: str
    table_name: str
    database_name: str
    schema_name: str
    definition: str
    table_type: str

    def to_prompt(self, dialect: str = "snowflake", include_ddl: bool = True) -> str
    @classmethod
    def list_to_prompt(cls, schemas: List[TableSchema], dialect: str = "snowflake") -> str
    @classmethod
    def from_arrow(cls, table: pa.Table) -> List[TableSchema]
```

### ActionHistory

```python
class ActionHistory(BaseModel):
    action_id: str
    role: ActionRole
    messages: str
    action_type: str
    input: Any
    output: Any
    status: ActionStatus
    start_time: datetime
    end_time: Optional[datetime]

    @classmethod
    def create_action(cls, role: ActionRole, action_type: str, messages: str, input_data: dict, output_data: dict = None, status: ActionStatus = ActionStatus.PROCESSING) -> "ActionHistory"
    def is_done(self) -> bool
    def function_name(self) -> str
```

### SubAgentConfig

```python
class SubAgentConfig(BaseModel):
    system_prompt: str
    agent_description: Optional[str]
    tools: str
    mcp: str
    scoped_context: Optional[ScopedContext]
    rules: List[str]
    prompt_version: str
    prompt_language: str
    scoped_kb_path: Optional[str]

    def has_scoped_context(self) -> bool
    def has_scoped_context_by(self, attr_name: str) -> bool
    def is_in_namespace(self, namespace: str) -> bool
    def as_payload(self, namespace: Optional[str] = None) -> Dict[str, Any]
    @property
    def tool_list(self) -> List[str]
```

---

## 版本更新记录

### v2.1 (2026-01-23)
- 新增 `table_names_to_prompt` 方法到 TableSchema
- 新增 `SQLValidateInput` 模型文档
- 新增 `VisualizationInput/Output` 可视化模型
- 修正 `tool_models.py` 为预留模型
- 修正 `preflight_results` 类型为 `Dict[str, Any]`

### v2.0 (2026-01-22)
- 完整重写，基于最新代码架构
- 新增核心节点模型 (SqlTask, TableSchema, SQLContext 等)
- 新增专用节点模型 (Agentic Node, Schema Linking 等)
- 新增动作历史和代理配置模型
- 新增 Pydantic v2 完整集成

---

## 相关资源

- **项目主页**: [https://datus.ai](https://datus.ai)
- **文档**: [https://docs.datus.ai](https://docs.datus.ai)
- **GitHub**: [https://github.com/Datus-ai/Datus-agent](https://github.com/Datus-ai/Datus-agent)
- **Slack 社区**: [https://join.slack.com/t/datus-ai](https://join.slack.com/t/datus-ai/shared_invite/zt-3g6h4fsdg-iOl5uNoz6A4GOc4xKKWUYg)
