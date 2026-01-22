# Datus Tools 模块介绍

> **文档版本**: v2.0
> **更新日期**: 2026-01-22
> **相关模块**: `datus/tools/`

---

## 模块概述

### 🏗️ 整体架构设计理念

Datus工具系统采用**分层插件化架构**，核心设计理念包括：

1. **统一抽象**：所有工具继承自 `BaseTool`，提供一致的接口和生命周期管理
2. **装饰器驱动**：使用 `@ToolAction` 装饰器声明工具能力，支持自动发现和调用
3. **异步优先**：原生支持异步操作和流式处理（SSE）
4. **标准化协议**：支持MCP（Model Context Protocol）服务器集成

### 📊 模块目录结构

```
datus/tools/
├── __init__.py              # 工具自动发现和注册
├── base.py                  # BaseTool 基类和 ToolAction 装饰器
├── func_tool/              # 函数式工具（异步包装器）
│   ├── base.py             # trans_to_function_tool 转换器
│   ├── database.py         # 数据库函数工具
│   └── enhanced_preflight_tools.py  # 增强预检工具
├── mcp_tools/              # MCP 服务器管理工具
│   ├── mcp_tool.py         # MCPTool 主类
│   ├── mcp_manager.py      # MCP 服务器管理器
│   └── mcp_config.py       # MCP 配置模型
├── db_tools/               # 数据库连接器
│   ├── base.py             # BaseSqlConnector 抽象类
│   ├── registry.py         # 连接器注册表
│   └── *_connector.py      # 各数据库连接器实现
├── search_tools/           # 文档搜索工具
│   └── search_tool.py      # SearchTool (内部/外部搜索)
├── llms_tools/             # LLM 相关工具
│   ├── reasoning_sql.py    # SQL 推理（支持流式）
│   ├── autofix_sql.py      # SQL 自动修复
│   ├── match_schema.py     # Schema 匹配
│   └── visualization_tool.py  # 可视化推荐
├── date_tools/             # 时间解析工具
│   └── date_parser.py      # DateParserTool
├── output_tools/           # 输出工具
│   └── output.py           # OutputTool (SQL/JSON/CSV)
└── lineage_graph_tools/    # 血缘图工具
    └── schema_lineage.py   # SchemaLineageTool
```

---

## 核心组件

### 1. BaseTool 基础架构

#### 1.1 BaseTool 抽象基类

```python
class BaseTool(ABC):
    """所有工具的抽象基类"""

    tool_name: str = "base_tool"
    tool_description: str = "Base tool class"

    def __init__(self, **kwargs):
        self.tool_params = kwargs
        self._actions = {}
        self._register_actions()

    def _register_actions(self):
        """注册所有 @ToolAction 装饰的方法"""
```

**关键字段：**
- `tool_name`: 工具名称标识
- `tool_description`: 工具描述
- `tool_params`: 工具初始化参数
- `_actions`: 注册的动作方法字典
- `tool_ctx`: `ContextVar` 类型的工具上下文

**核心方法：**
| 方法 | 说明 |
|------|------|
| `set_tool_context(tool_context)` | 设置工具上下文 |
| `get_actions()` | 获取所有可用动作 |
| `call_action(action_name, *args, **kwargs)` | 调用指定动作 |
| `get_tool_manifest()` | 获取工具清单（MCP注册用） |

#### 1.2 ToolAction 装饰器

```python
class ToolAction:
    """用于标记工具动作方法的装饰器"""

    def __init__(self, name: Optional[str] = None, description: str = ""):
        self.name = name
        self.description = description

    def __call__(self, func: Callable):
        wrapper.is_tool_action = True
        wrapper.action_name = action_name
        wrapper.description = self.description
        wrapper.signature = inspect.signature(func)
        return wrapper
```

**使用示例：**
```python
class MCPTool(BaseTool):
    @ToolAction(description="添加MCP服务器配置")
    def add_server(self, name: str, type: str, **config_params) -> BaseToolExecResult:
        # 实现逻辑
        pass
```

#### 1.3 BaseToolExecResult 结果模型

```python
@dataclass
class BaseToolExecResult:
    result: Any = field(init=True, default=None)
    success: bool = field(init=True, default=True)
    message: str = field(init=True, default="")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
```

---

### 2. 函数式工具 (func_tool)

#### 2.1 trans_to_function_tool 转换器

将类的绑定方法转换为 `FunctionTool`，支持异步调用：

```python
def trans_to_function_tool(bound_method: Callable) -> FunctionTool:
    """
    将绑定方法转换为函数工具。
    解决 '@function_tool' 只能用于静态方法的问题。
    """
    tool_template = function_tool(bound_method)

    # 移除 self 参数
    corrected_schema = json.loads(json.dumps(tool_template.params_json_schema))
    if "self" in corrected_schema.get("properties", {}):
        del corrected_schema["properties"]["self"]

    # 创建异步调用器
    async_invoker = create_async_invoker(bound_method)

    return FunctionTool(
        name=tool_template.name,
        description=tool_template.description,
        params_json_schema=corrected_schema,
        on_invoke_tool=async_invoker,
    )
```

**关键特性：**
- 自动移除 `self` 参数
- 支持同步和异步方法
- 统一返回 `FuncToolResult` 格式

#### 2.2 FuncToolResult 标准结果

```python
class FuncToolResult(BaseModel):
    success: int = Field(default=1, description="1=成功, 0=失败")
    error: Optional[str] = Field(default=None, description="错误信息")
    result: Optional[Any] = Field(default=None, description="执行结果")
```

#### 2.3 数据库函数工具 (DBFuncTool)

支持 Sub-Agent 作用域的数据库函数工具：

```python
class DBFuncTool:
    def __init__(
        self,
        agent_config: AgentConfig,
        sub_agent_name: Optional[str] = None,
        **kwargs
    ):
        self.sub_agent_name = sub_agent_name
        self.schema_rag = SchemaWithValueRAG(agent_config, sub_agent_name)
        self.metrics_rag = SemanticMetricsRAG(agent_config, sub_agent_name)
```

**提供的工具函数：**
| 函数名 | 描述 |
|--------|------|
| `search_table` | 搜索相关表 |
| `describe_table` | 获取表结构 |
| `search_reference_sql` | 搜索参考SQL |
| `parse_temporal_expressions` | 解析时间表达式 |

#### 2.4 增强预检工具 (EnhancedPreflightTools)

v2.4 引入的高级SQL分析工具：

```python
class EnhancedPreflightTools:
    async def analyze_query_plan(self, sql: str, ...) -> FuncToolResult:
        """分析查询执行计划"""

    async def check_table_conflicts(self, table_name: str, ...) -> FuncToolResult:
        """检查表结构冲突和重复构建"""

    async def validate_partitioning(self, table_name: str, ...) -> FuncToolResult:
        """验证分区策略并提供建议"""
```

---

### 3. MCP 工具 (mcp_tools)

#### 3.1 MCPTool 主类

```python
class MCPTool(BaseTool):
    """MCP 服务器管理工具"""

    tool_name = "mcp_tool"
    tool_description = "Management tool for MCP (Model Context Protocol) servers"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.manager = MCPManager()
```

#### 3.2 MCP 工具动作

| 动作 | 描述 | 签名 |
|------|------|------|
| `add_server` | 添加MCP服务器配置 | `add_server(name, type, **config_params)` |
| `remove_server` | 删除MCP服务器配置 | `remove_server(name: str)` |
| `list_servers` | 列出MCP服务器配置 | `list_servers(server_type: Optional[str] = None)` |
| `get_server` | 获取MCP服务器配置 | `get_server(name: str)` |
| `check_connectivity` | 检查服务器连接 | `check_connectivity(name: str)` |
| `list_tools` | 列出服务器可用工具 | `list_tools(server_name: str, apply_filter: bool = True)` |
| `call_tool` | 调用服务器工具 | `call_tool(server_name, tool_name, arguments)` |
| `set_tool_filter` | 设置工具过滤器 | `set_tool_filter(server_name, allowed_tools, blocked_tools, enabled)` |
| `get_tool_filter` | 获取工具过滤器 | `get_tool_filter(server_name: str)` |
| `remove_tool_filter` | 移除工具过滤器 | `remove_tool_filter(server_name: str)` |

#### 3.3 MCP 配置解析

```python
def parse_command_string(s: str) -> Tuple[str, Optional[str], Dict[str, Any]]:
    """
    解析命令行字符串为结构化信息。

    返回: (transport_type, name, payload)
    - 'studio'/'stdio': payload = {"command": str, "args": [...], "env": {...}}
    - 'sse'/'http':    payload = {"url": str, "headers": {...}, "timeout": float}
    """
```

---

### 4. 数据库工具 (db_tools)

#### 4.1 BaseSqlConnector 抽象连接器

```python
class BaseSqlConnector(ABC):
    """数据库连接器抽象基类"""

    # Text2SQL 工作流允许的SQL类型（只读操作）
    ALLOWED_SQL_TYPES = {
        SQLType.SELECT,         # 数据查询
        SQLType.EXPLAIN,        # 查询执行计划
        SQLType.METADATA_SHOW,  # 元数据查询
    }

    def __init__(self, config: ConnectionConfig, dialect: str):
        self.config = config
        self.timeout_seconds = config.timeout_seconds
        self.connection = None
        self.dialect = dialect
```

**上下文管理器支持：**
```python
def __enter__(self):
    self.connect()
    return self

def __exit__(self, exc_type, exc_val, exc_tb):
    if exc_type:
        self._safe_rollback()
    self.close()
    return False
```

#### 4.2 核心方法

| 方法 | 说明 |
|------|------|
| `execute(input_params, result_format)` | 执行SQL查询（支持格式：csv/arrow/pandas/list） |
| `execute_query(sql, result_format)` | 执行SELECT查询 |
| `execute_explain(sql, result_format)` | 执行EXPLAIN查询 |
| `execute_ddl(sql)` | 执行DDL语句 |
| `get_databases(catalog_name, include_sys)` | 获取数据库列表 |
| `get_tables(catalog_name, database_name, schema_name)` | 获取表列表 |
| `get_views(...)` | 获取视图列表 |
| `get_tables_with_ddl(...)` | 获取表及其DDL |
| `get_sample_rows(tables, top_n, ...)` | 获取表样本数据 |
| `switch_context(catalog_name, database_name, schema_name)` | 切换上下文 |
| `full_name(...)` | 获取表完整名称 |
| `identifier(...)` | 获取SQL标识符 |

#### 4.3 SQL类型安全检查

```python
def execute(self, input_params: Any, result_format: str = "csv") -> ExecuteSQLResult:
    sql_type = parse_sql_type(sql_query, self.dialect)

    # 安全检查：验证SQL类型是否允许
    if sql_type not in self.ALLOWED_SQL_TYPES:
        allow_ddl = getattr(input_params, "allow_ddl", False)
        allow_dml = getattr(input_params, "allow_dml", False)

        if sql_type == SQLType.DDL and not allow_ddl:
            return ExecuteSQLResult(
                success=False,
                error="DDL operations not allowed in text2sql workflow"
            )
```

#### 4.4 支持的数据库连接器

| 数据库 | 连接器类 | 模块 |
|--------|----------|------|
| StarRocks | StarRocksConnector | `starrocks_connector.py` |
| SQLite | SQLiteConnector | `sqlite_connector.py` |
| DuckDB | DuckDBConnector | `duckdb_connector.py` |
| MySQL | MySQLConnector | 外部适配器 |
| Snowflake | SnowflakeConnector | 外部适配器 |

---

### 5. 搜索工具 (search_tools)

#### 5.1 SearchTool 类

```python
class SearchTool(BaseTool):
    """使用各种方法搜索文档的工具"""

    tool_name = "search"
    tool_description = "Search for documents using various methods (internal, external, llm)"

    def execute(self, input_data: DocSearchInput) -> DocSearchResult:
        if input_data.method == "internal":
            return self._search_internal(input_data)
        elif input_data.method == "external":
            return search_by_tavily(input_data.keywords, input_data.top_n)
        elif input_data.method == "llm":
            return DocSearchResult(success=False, error="LLM search not implemented")
```

**搜索方法：**
- `internal`: 内部文档搜索（使用 DocumentStore）
- `external`: 外部搜索（使用 Tavily API）
- `llm`: LLM 搜索（待实现）

---

### 6. LLM 工具 (llms_tools)

#### 6.1 reasoning_sql - SQL推理（支持流式）

```python
async def reasoning_sql_with_mcp_stream(
    model: LLMBaseModel,
    input_data: ReasoningInput,
    tool_config: Dict[str, Any],
    tools: List[Tool],
    action_history_manager: Optional[ActionHistoryManager] = None,
) -> AsyncGenerator[ActionHistory, None]:
    """使用流式支持生成SQL推理"""
```

**特性：**
- 支持 SSE 流式输出
- 集成 MCP 工具调用
- 自动提取 SQLContext
- 记录 ActionHistory

#### 6.2 autofix_sql - SQL自动修复

```python
@optional_traceable()
def autofix_sql(
    model: LLMBaseModel,
    input_data: FixInput,
    docs: list[str]
) -> FixResult:
    """使用LLM自动修复SQL错误"""
```

#### 6.3 MatchSchemaTool - Schema匹配

```python
class MatchSchemaTool(BaseTool):
    def execute(self, input_data: SchemaLinkingInput) -> SchemaLinkingResult:
        """使用LLM匹配schema"""
        table_metadata = self.storage.search_all(database_name=input_data.database_name)
        all_tables = gen_all_table_dict(table_metadata)
        match_result = self.match_schema(input_data, table_metadata, all_tables)
```

**支持 Map-Reduce 模式**：
- 当表数量 > 200 时自动使用 map-reduce
- 并行处理 schema 匹配任务
- 汇总子任务结果

#### 6.4 VisualizationTool - 可视化推荐

```python
class VisualizationTool(BaseTool):
    tool_name = "visualization_tool"
    tool_description = "Recommend a chart configuration for a dataset"

    def execute(self, input_data: VisualizationInput) -> VisualizationOutput:
        """使用LLM或启发式方法推荐可视化配置"""
```

**支持的图表类型：**
- Bar Chart（柱状图）
- Line Chart（折线图）
- Scatter Plot（散点图）
- Pie Chart（饼图）

---

### 7. 时间工具 (date_tools)

#### 7.1 DateParserTool 类

```python
class DateParserTool(BaseTool):
    """使用LLM解析文本中的时间表达式"""

    tool_name = "date_parser_tool"
    tool_description = "Tool for extracting and parsing temporal expressions from natural language"

    def __init__(self, language: str = "en", **kwargs):
        super().__init__(**kwargs)
        self.language = language

    def execute(self, task_text: str, current_date: str, model: LLMBaseModel) -> List[ExtractedDate]:
        """执行日期解析操作"""
```

**ExtractedDate 模型：**
```python
class ExtractedDate(BaseModel):
    original_text: str          # 原始文本
    parsed_date: Optional[str]   # 解析的单个日期
    start_date: Optional[str]    # 范围开始日期
    end_date: Optional[str]      # 范围结束日期
    date_type: str              # specific/range/relative
    confidence: float           # 置信度
```

---

### 8. 输出工具 (output_tools)

#### 8.1 OutputTool 类

```python
class OutputTool(BaseTool):
    def execute(
        self,
        input_data: OutputInput,
        sql_connector: BaseSqlConnector,
        model: Optional[LLMBaseModel] = None,
    ) -> OutputResult:
        """执行输出操作"""
```

**支持的输出格式：**
- `sql`: SQL文件
- `json`: JSON文件
- `csv`: CSV文件
- 默认: 同时生成 SQL、JSON、CSV

#### 8.2 check_sql - SQL结果检查

```python
@optional_traceable()
def check_sql(
    self,
    input_data: OutputInput,
    sql_connector: BaseSqlConnector,
    model: Optional[LLMBaseModel] = None,
) -> Tuple[str, str]:
    """检查SQL执行结果是否正确，必要时修正SQL"""
```

---

### 9. 血缘图工具 (lineage_graph_tools)

#### 9.1 SchemaLineageTool 类

```python
class SchemaLineageTool(BaseTool):
    """用于管理和查询schema血缘信息的工具"""

    def __init__(
        self,
        storage: Optional[SchemaWithValueRAG] = None,
        agent_config: Optional[AgentConfig] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if storage:
            self.store = storage
        else:
            self.store = SchemaWithValueRAG(agent_config)
```

**核心方法：**
| 方法 | 说明 |
|------|------|
| `execute(input_param, model)` | 执行schema血缘操作 |
| `_search_similar_schemas(input_param, top_n)` | 搜索相似schema |
| `search_similar_schemas_by_schema(input_param, top_n)` | 在所有schema中搜索最相似的 |
| `get_schems_by_db(connector, input_param)` | 从数据库获取schema |

---

## 工具注册系统

### 自动发现机制

```python
def get_tool_types() -> List[str]:
    """通过扫描工具目录获取所有可用的工具类型"""
    tools_dir = os.path.dirname(__file__)

    tool_types = [
        d.replace("_tools", "")
        for d in os.listdir(tools_dir)
        if os.path.isdir(os.path.join(tools_dir, d))
        and d.endswith("_tools")
        and not d.startswith("__")
    ]

    return tool_types
```

**命名约定：**
- 工具目录必须以 `_tools` 结尾
- 工具类需在 `__init__.py` 的 `__all__` 中声明
- 工具类型 = 目录名去除 `_tools` 后缀

### 工具获取

```python
def get_tool(tool_type: str, **kwargs) -> Optional[BaseTool]:
    """通过类型获取工具实现"""
    tool_dir = f"{tool_type}_tools"
    module = importlib.import_module(f"tools.{tool_dir}")

    for tool_name in module.__all__:
        tool_class = getattr(module, tool_name)
        return tool_class(**kwargs)

    return None
```

---

## 架构特性

### 1. 插件化架构

**优势：**
- 零配置集成：新工具只需放在对应目录
- 动态加载：支持运行时加载第三方插件
- 热插拔：工具可独立更新

### 2. 异步优先

**流式处理示例：**
```python
async def reasoning_sql_with_mcp_stream(...) -> AsyncGenerator[ActionHistory, None]:
    async for action in base_mcp_stream(...):
        yield action
```

### 3. 上下文管理

**Sub-Agent 作用域支持：**
```python
class DBFuncTool:
    def __init__(
        self,
        agent_config: AgentConfig,
        sub_agent_name: Optional[str] = None,
        **kwargs
    ):
        self.sub_agent_name = sub_agent_name
        # 使用 sub_agent_name 创建隔离的存储
```

### 4. 标准化接口

**统一返回格式：**
- `BaseToolExecResult`: BaseTool 动作返回
- `FuncToolResult`: 函数工具返回
- `ExecuteSQLResult`: SQL执行结果
- `SchemaLinkingResult`: Schema链接结果

---

## 使用示例

### 创建自定义工具

```python
from datus.tools.base import BaseTool, ToolAction, BaseToolExecResult
from typing import Optional, Dict, Any

class MyCustomTool(BaseTool):
    """自定义工具示例"""

    tool_name = "my_custom_tool"
    tool_description = "My custom tool for demonstration"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 初始化逻辑

    @ToolAction(description="执行自定义操作")
    def do_something(
        self,
        param1: str,
        param2: Optional[int] = None
    ) -> BaseToolExecResult:
        """执行自定义操作"""
        try:
            # 实现逻辑
            result = self._process(param1, param2)

            return BaseToolExecResult(
                success=True,
                message="操作成功",
                result=result
            )
        except Exception as e:
            return BaseToolExecResult(
                success=False,
                message=f"操作失败: {e}"
            )

    def _process(self, param1: str, param2: Optional[int]) -> Dict[str, Any]:
        # 实际处理逻辑
        return {"status": "completed", "value": param1}
```

### 使用函数工具包装器

```python
from datus.tools.func_tool.base import trans_to_function_tool, FuncToolResult

class MyDatabaseTool:
    def __init__(self, connection_string: str):
        self.connection_string = connection_string

    def query_data(self, sql: str, limit: int = 100) -> FuncToolResult:
        """查询数据库"""
        try:
            # 执行查询
            results = self._execute_query(sql, limit)

            return FuncToolResult(
                success=1,
                result={"rows": results, "count": len(results)}
            )
        except Exception as e:
            return FuncToolResult(
                success=0,
                error=str(e)
            )

# 转换为 FunctionTool
tool_instance = MyDatabaseTool("connection_string")
function_tool = trans_to_function_tool(tool_instance.query_data)
```

### 使用MCP工具

```python
from datus.tools.mcp_tools.mcp_tool import MCPTool

# 创建MCP工具实例
mcp_tool = MCPTool()

# 添加服务器
result = mcp_tool.add_server(
    name="my_server",
    type="stdio",
    command="python",
    args=["-m", "my_mcp_server"]
)

# 列出工具
tools_result = mcp_tool.list_tools("my_server", apply_filter=True)

# 调用工具
call_result = mcp_tool.call_tool(
    server_name="my_server",
    tool_name="my_function",
    arguments={"param1": "value1"}
)
```

---

## 版本更新记录

### v2.0 (2026-01-22)
- 完整重写，基于最新代码架构
- 新增 BaseTool/ToolAction 装饰器模式
- 新增函数工具异步包装器
- 新增增强预检工具（v2.4）
- 新增 Sub-Agent 作用域支持
- 新增 MCP 工具集成
- 完善数据库连接器抽象
- 添加流式处理支持

### v1.0 (2026-01-05)
- 初始版本
- 高层次架构概述
