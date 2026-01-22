# Datus Utils 模块介绍

> **文档版本**: v2.0
> **更新日期**: 2026-01-22
> **相关模块**: `datus/utils/`

---

## 模块概述

### 🏗️ 整体架构设计理念

**"数据工程基础设施工具集"** - 通过标准化工具集合实现数据工程应用的完整基础设施支持

Datus Utils模块采用**分层工具架构**，核心设计理念包括：

1. **基础设施抽象**：将系统级操作抽象为标准接口
2. **数据处理流水线**：提供端到端的从数据到结果的处理能力
3. **环境自适应**：根据运行环境自动调整行为和配置
4. **可靠性保障**：内置错误处理、监控和性能优化

### 📊 模块目录结构

```
datus/utils/
├── __init__.py               # 模块初始化
├── constants.py             # 常量定义（数据库、LLM、SQL类型）
├── exceptions.py            # 异常体系（ErrorCode、DatusException）
├── async_utils.py           # 异步运行时管理
├── loggings.py             # 结构化日志系统
├── path_manager.py         # 集中式路径管理
├── sub_agent_manager.py    # Sub-Agent 管理
├── sql_utils.py             # SQL 处理工具（DDL解析、验证）
├── json_utils.py            # JSON 数据处理
├── token_utils.py           # Token 计算工具
├── compress_utils.py        # 数据压缩工具
├── benchmark_utils.py       # SQL 基准测试工具
├── error_handling.py        # 统一错误处理
├── traceable_utils.py      # 可追踪装饰器
├── text_utils.py            # 文本清理工具
├── device_utils.py          # 设备检测
├── env.py                   # 环境变量管理
├── csv_utils.py             # CSV 处理工具
├── pyarrow_utils.py         # PyArrow 工具
├── schema_utils.py          # Schema 处理工具
├── time_utils.py            # 时间处理工具
├── typing_fix.py            # 类型兼容修复
└── context_lock.py          # 上下文锁
```

---

## 核心组件

### 1. 常量定义 (constants.py)

```python
class DBType(str, Enum):
    """支持的数据库类型"""
    SQLITE = "sqlite"
    DUCKDB = "duckdb"
    MYSQL = "mysql"
    POSTGRESQL = "postgresql"
    POSTGRES = "postgres"
    SNOWFLAKE = "snowflake"
    CLICKHOUSE = "clickhouse"
    BIGQUERY = "bigquery"
    STARROCKS = "starrocks"
    SQLSERVER = "sqlserver"
    MSSQL = "mssql"
    ORACLE = "oracle"
    HIVE = "hive"
    CLICKZETTA = "clickzetta"

    @classmethod
    def support_catalog(cls, db_type: str) -> bool
    @classmethod
    def support_database(cls, db_type: str) -> bool
    @classmethod
    def support_schema(cls, db_type: str) -> bool


class LLMProvider(str, Enum):
    """支持的 LLM 提供商"""
    OPENAI = "openai"
    CLAUDE = "claude"
    DEEPSEEK = "deepseek"
    QWEN = "qwen"
    GLM = "glm"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    LLAMA = "llama"
    GPT = "gpt"


class SQLType(str, Enum):
    """SQL 语句类型"""
    SELECT = "select"
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    MERGE = "merge"
    DDL = "ddl"
    METADATA_SHOW = "metadata"
    EXPLAIN = "explain"
    CONTENT_SET = "context_set"
    UNKNOWN = "unknown"


# 系统内置 Sub-Agent
SYS_SUB_AGENTS = {"gen_semantic_model", "gen_metrics", "gen_sql_summary"}
```

**数据库方言支持：**
- **SUPPORT_CATALOG_DIALECTS**: StarRocks, Snowflake, BigQuery
- **SUPPORT_DATABASE_DIALECTS**: 除 SQLite 外的所有数据库
- **SUPPORT_SCHEMA_DIALECTS**: Snowflake, BigQuery, MSSQL, Oracle, DuckDB, PostgreSQL

---

### 2. 异常体系 (exceptions.py)

#### 2.1 ErrorCode 枚举

```python
class ErrorCode(Enum):
    """7位错误码体系：类别(2位) + 子类(2位) + 序号(3位)"""

    # 通用错误 (10xxxx)
    COMMON_UNKNOWN = ("1000000", "Unknown error occurred")
    COMMON_FIELD_INVALID = ("1000001", "{field_name} invalid")
    COMMON_FILE_NOT_FOUND = ("100002", "{config_name} not found: {file_name}")
    COMMON_FIELD_REQUIRED = ("100003", "Missing required field: {field_name}")

    # 节点执行错误 (20xxxx)
    NODE_EXECUTION_FAILED = ("200001", "Node execution failed")
    NODE_NO_SQL_CONTEXT = ("200002", "No SQL context available")

    # 模型错误 (30xxxx)
    MODEL_REQUEST_FAILED = ("300001", "LLM request failed")
    MODEL_TIMEOUT = ("300003", "Model request timeout")
    MODEL_AUTHENTICATION_ERROR = ("300011", "Authentication failed (HTTP 401)")
    MODEL_PERMISSION_ERROR = ("300012", "API key lacks permissions (HTTP 403)")
    MODEL_RATE_LIMIT = ("300015", "Rate limit exceeded (HTTP 429)")

    # 工具错误 (40xxxx)
    TOOL_EXECUTION_FAILED = ("400001", "Tool execution failed")

    # 存储错误 (41xxxx)
    STORAGE_CONNECTION_FAILED = ("410001", "Failed to connect to vector database")
    STORAGE_SEARCH_FAILED = ("410004", "Vector search failed")

    # 数据库错误 (50xxxx)
    DB_CONNECTION_FAILED = ("500001", "Failed to establish connection")
    DB_EXECUTION_TIMEOUT = ("500007", "Query execution timed out")
```

#### 2.2 DatusException 类

```python
class DatusException(Exception):
    """Datus 自定义异常"""

    def __init__(
        self,
        code: ErrorCode,
        message: Optional[str] = None,
        message_args: Optional[dict[str, Any]] = None,
        *args: object,
    ):
        self.code = code
        self.message_args = message_args or {}
        self.message = self.build_msg(message, message_args)
        super().__init__(self.message, *args)

    def build_msg(self, message: Optional[str], message_args: Optional[dict]) -> str:
        """构建错误消息"""
        if message:
            final_message = message
        elif message_args:
            final_message = self.code.desc.format(**message_args)
        else:
            final_message = self.code.desc
        return f"error_code={self.code.code}, error_message={final_message}"
```

**使用示例：**
```python
# 使用预定义错误码
raise DatusException(
    ErrorCode.COMMON_FIELD_REQUIRED,
    message_args={"field_name": "api_key"}
)
# 输出: error_code=100003, error_message=Missing required field: api_key

# 自定义消息
raise DatusException(
    ErrorCode.DB_CONNECTION_FAILED,
    message="Custom connection error message"
)
```

#### 2.3 全局异常处理器

```python
def setup_exception_handler(console_logger=None, prefix_wrap_func=None):
    """设置全局异常处理器

    自动捕获所有异常并记录到日志系统
    """
    sys.excepthook = global_exception_handler
```

---

### 3. 异步运行时 (async_utils.py)

```python
def run_async(coro: Awaitable[T]) -> T:
    """
    在同步上下文中运行异步协程

    智能检测运行环境并选择执行策略：
    - 在异步上下文中：使用线程池执行
    - 在同步上下文中：创建新事件循环
    """
    if loop and loop.is_running():
        # 使用线程池
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()
    else:
        # 直接运行
        return asyncio.run(coro)


async def await_cancellable(coro: Awaitable[T], timeout: Optional[float] = None) -> T:
    """
    等待可取消的协程（不屏蔽取消信号）
    """
    if timeout:
        return await asyncio.wait_for(coro, timeout=timeout)
    else:
        return await coro
```

---

### 4. 日志系统 (loggings.py)

#### 4.1 DynamicLogManager

```python
class DynamicLogManager:
    """支持运行时切换输出目标的动态日志管理器"""

    def __init__(self, debug=False, log_dir=None):
        # 自动检测日志目录
        if log_dir is None:
            if _is_source_environment():
                log_dir = "./logs"
            else:
                log_dir = str(get_path_manager().logs_dir)

    def set_output_target(self, target: Literal["both", "file", "console", "none"]):
        """设置日志输出目标"""

    @contextmanager
    def temporary_output(self, target: Literal["both", "file", "console", "none"]):
        """临时切换输出目标的上下文管理器"""
```

**使用示例：**
```python
# 临时只输出到文件
with log_context("file"):
    logger.info("此日志只会输出到文件")

# 配置日志
configure_logging(debug=True, log_dir="./logs", console_output=True)
```

#### 4.2 结构化日志

使用 `structlog` 配置：
- 自动添加代码位置 (`fileno`)
- 异常信息追踪 (`exc_info`)
- 彩色控制台输出（文件中自动移除颜色代码）

---

### 5. SQL 处理工具 (sql_utils.py)

#### 5.1 预编译正则模式

```python
# 性能优化：模块级预编译所有正则表达式
_TABLE_NAME_RE = re.compile(
    r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?[`"]?([\w.]+)[`"]?\s*\(',
    re.IGNORECASE
)
_COLUMN_COMMENT_RE = re.compile(
    r"COMMENT\s*(?:=\s*)?['\"]([^'\"]+)['\"]",
    re.IGNORECASE
)
```

#### 5.2 输入验证

```python
def validate_sql_input(sql: Any, max_length: int = MAX_SQL_LENGTH) -> Tuple[bool, str]:
    """
    验证 SQL 输入的安全性和正确性

    防止：
    - ReDoS 攻击（通过括号深度限制）
    - 内存耗尽（通过长度限制）
    - NULL 字节注入
    """
    # 类型验证
    if not isinstance(sql, str):
        return False, f"SQL must be a string, got {type(sql).__name__}"

    # 长度验证
    if len(sql) > max_length:
        return False, f"SQL length ({len(sql)}) exceeds maximum ({max_length})"

    # 括号深度检查（防止 ReDoS）
    paren_depth = 0
    max_paren_depth = 100
```

#### 5.3 中文注释保护

```python
def _preserve_chinese_comments(sql: str) -> tuple:
    """保护中文注释文本在 DDL 清理过程中不被移除

    StarRocks 使用双引号包裹中文 COMMENT 文本
    """
    pattern = re.compile(r'COMMENT\s*(?:=\s*)?"([^"]*)"', re.IGNORECASE)

    def replace_with_placeholder(match):
        placeholder = f"__CHINESE_COMMENT_{len(protected)}__"
        protected[placeholder] = match.group(0)
        return f'COMMENT "{placeholder}"'

    return pattern.sub(replace_with_placeholder, sql), protected
```

#### 5.4 DDL 解析

```python
def parse_metadata_from_ddl(sql: str, dialect: str = DBType.SNOWFLAKE) -> Dict[str, Any]:
    """解析 CREATE TABLE 语句

    使用 sqlglot 进行主要解析，正则表达式作为回退
    """
    dialect = parse_dialect(dialect)
    parsed = sqlglot.parse_one(sql.strip(), dialect=dialect, error_level=sqlglot.ErrorLevel.IGNORE)

    # 提取表信息、列信息、主键、外键、索引等


def extract_enhanced_metadata_from_ddl(sql: str, dialect: str) -> Dict[str, Any]:
    """从 DDL 中提取完整的元数据（多策略解析）"""
    # Strategy 1: 原始 SQL
    # Strategy 2: 清理后的 SQL
    # Strategy 3: 不同方言
    # Fallback: 正则表达式解析
```

#### 5.5 SQL 类型检测

```python
def parse_sql_type(sql: str, dialect: str = DBType.SNOWFLAKE) -> SQLType:
    """
    解析 SQL 语句类型

    支持：SELECT, INSERT, UPDATE, DELETE, MERGE, DDL, EXPLAIN, SHOW, DESCRIBE, USE, SET
    """
    first_keyword_match = re.match(r'^\s*([A-Za-z]+)', sql_clean, re.IGNORECASE)
    first_keyword = first_keyword_match.group(1).upper()

    keyword_map = {
        'SELECT': SQLType.SELECT,
        'INSERT': SQLType.INSERT,
        'CREATE': SQLType.DDL,
        'SHOW': SQLType.METADATA_SHOW,
        'EXPLAIN': SQLType.EXPLAIN,
        # ...
    }
```

#### 5.6 元数据标识符

```python
def metadata_identifier(
    dialect: str,
    catalog_name: str = "",
    database_name: str = "",
    schema_name: str = "",
    table_name: str = ""
) -> str:
    """
    创建数据库表的唯一标识符

    格式：catalog.database.schema.table
    空组件会创建连续的点（例如："catalog.database..table"）
    """
    parts = []
    if table_name:
        parts.append(table_name)
    if schema_name:
        parts.insert(0, schema_name)
    if database_name:
        parts.insert(0, database_name)
    if catalog_name:
        parts.insert(0, catalog_name)

    return ".".join(parts)
```

---

### 6. JSON 数据处理 (json_utils.py)

#### 6.1 LLM 结果解析

```python
def llm_result2json(llm_str: str, expected_type: type[Dict | List] = dict) -> Union[Dict, List, None]:
    """
    将 LLM 输出字符串转换为 JSON 对象或数组

    支持格式：
    1. 纯 JSON 字符串
    2. ```json ... ``` 代码块
    3. ``` ... ``` 代码块

    自动修复损坏的 JSON
    """
    try:
        cleaned_string = strip_json_str(llm_str)
        result = json_repair.loads(cleaned_string)

        # 验证有意义的内容
        if isinstance(result, dict):
            metadata_fields = {"fallback", "error", "traceback", "raw_response"}
            has_any_content = any(
                _has_content(result.get(key))
                for key in result.keys()
                if key not in metadata_fields
            )
            if not has_any_content:
                return None

        return result
    except (json.JSONDecodeError, ValueError, AttributeError, TypeError):
        return None


def llm_result2sql(llm_str: str) -> Optional[str]:
    """
    从 LLM 输出中提取 SQL

    查找 ```sql ... ``` 或 ```SQL ... ``` 代码块
    回退：查找包含 SQL 关键字的代码块
    """
    sql_pattern = r"```(?:sql|SQL)\s*\n?(.*?)\n?```"
    match = re.search(sql_pattern, llm_str, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
```

#### 6.2 数据格式转换

```python
def json2csv(result: Any, columns: Optional[List[str]] = None) -> str:
    """将 JSON 数据转换为 CSV 格式"""
    if isinstance(result, str):
        if result.strip().startswith("[") or result.strip().startswith("{"):
            result = json_repair.loads(result)

    df = pd.DataFrame(result)
    output = StringIO()
    df.to_csv(output, index=False, columns=columns)
    return output.getvalue()


def json_list2markdown_table(json_list: List[Dict[str, Any]]) -> str:
    """将字典列表转换为 Markdown 表格"""
    df = pd.DataFrame(json_list)
    return df.to_markdown()
```

#### 6.3 数据规范化

```python
def _normalize_for_json(data: Any) -> Any:
    """
    将各种 Python/Pydantic/pandas/NumPy 对象转换为 JSON 可序列化结构
    """
    # 支持的类型：
    # - datetime, date, time → ISO 格式字符串
    # - Decimal → 字符串
    # - UUID → 字符串
    # - Pydantic BaseModel → model_dump()
    # - pandas DataFrame → 字典列表
    # - NumPy 数组 → 列表
    # - dataclass → asdict()
```

---

### 7. 路径管理 (path_manager.py)

```python
class DatusPathManager:
    """集中式 .datus 目录路径管理器"""

    def __init__(self, datus_home: Optional[str] = None):
        if datus_home:
            self._datus_home = Path(datus_home).expanduser().resolve()
        else:
            self._datus_home = Path.home() / ".datus"

    @property
    def conf_dir(self) -> Path:
        """配置目录: ~/.datus/conf"""
        return self._datus_home / "conf"

    @property
    def data_dir(self) -> Path:
        """数据目录: ~/.datus/data"""
        return self._datus_home / "data"

    @property
    def logs_dir(self) -> Path:
        """日志目录: ~/.datus/logs"""
        return self._datus_home / "logs"

    @property
    def sessions_dir(self) -> Path:
        """会话目录: ~/.datus/sessions"""
        return self._datus_home / "sessions"

    # ... 更多目录属性

    def rag_storage_path(self, namespace: str) -> Path:
        """RAG 存储路径"""
        return self.data_dir / f"datus_db_{namespace}"

    def sub_agent_path(self, agent_name: str) -> Path:
        """Sub-Agent 存储路径"""
        return self.data_dir / "sub_agents" / agent_name
```

**全局单例：**
```python
def get_path_manager(datus_home: Optional[Path] = None) -> DatusPathManager:
    """获取全局路径管理器实例（线程安全单例）"""
    global _path_manager
    if _path_manager is None:
        with _path_manager_lock:
            if _path_manager is None:
                _path_manager = DatusPathManager(datus_home)
    return _path_manager
```

---

### 8. Sub-Agent 管理 (sub_agent_manager.py)

```python
class SubAgentManager:
    """Sub-Agent 配置和提示管理操作封装"""

    def list_agents(self) -> Dict[str, Dict[str, Any]]:
        """列出所有 Sub-Agent 配置"""

    def get_agent(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """获取特定 Sub-Agent 配置"""

    def save_agent(self, config: SubAgentConfig, previous_name: Optional[str] = None) -> Dict[str, Any]:
        """持久化 Sub-Agent 配置

        处理：
        - 作用域知识库创建/重命名/清除
        - 提示模板复制/移除
        """

    def remove_agent(self, agent_name: str) -> bool:
        """删除 Sub-Agent"""

    def bootstrap_agent(
        self,
        config: SubAgentConfig,
        *,
        components: Optional[Sequence[str]] = None,
        strategy: SubAgentBootstrapStrategy = "overwrite",
    ) -> BootstrapResult:
        """引导 Sub-Agent（创建作用域知识库）"""
```

---

### 9. 数据压缩 (compress_utils.py)

```python
class DataCompressor:
    """NL2SQL Agent 查询结果数据压缩器"""

    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        token_threshold: int = 1024,
        tolerance_ratio: float = 0.1,
        output_format: Literal["table", "csv"] = "csv",
    ):
        """
        初始化数据压缩器

        model_name: 支持最新 (o200k_base) 和生产 (cl100k_base) 模型
        """
        try:
            self.tokenizer = tiktoken.encoding_for_model(model_name)
        except Exception:
            self.tokenizer = None

    def compress(self, data: Union[List[Dict], pd.DataFrame, pa.Table]) -> Dict:
        """
        压缩数据并返回结果

        压缩策略：
        - 行压缩：>20 行时，取前 10 行和后 10 行
        - 列压缩：保留 ID 和时间列，从中间移除其他列
        - 混合压缩：同时应用行和列压缩
        """

        return {
            "original_rows": original_rows,
            "original_columns": original_columns,
            "is_compressed": is_compressed,
            "compressed_data": compressed_data,
            "removed_columns": removed_columns,
            "compression_type": compression_type,
        }

    @classmethod
    def quick_compress(
        cls,
        data: Union[List[Dict], pd.DataFrame, pa.Table],
        model_name: str = "gpt-3.5-turbo",
        token_threshold: int = 1024,
        output_format: Literal["table", "csv"] = "csv",
    ) -> str:
        """快速压缩方法（一次性使用）"""
```

---

### 10. 基准测试 (benchmark_utils.py)

```python
@dataclass
class WorkflowArtifacts:
    """工作流产物数据类"""
    files: list[str]
    reference_sqls: list[str]
    reference_sql_names: list[str]
    semantic_models: list[str]
    metrics_names: list[str]


@dataclass
class ComparisonOutcome:
    """SQL 执行比较结果"""
    match_rate: float = 0.0
    matched_columns: list[tuple[str, str]] = field(default_factory=list)
    column_match_details: list[dict] = field(default_factory=list)
    value_match_details: list[dict] = field(default_factory=list)


class SQLComparator:
    """SQL 执行结果比较器"""

    def compare_csv_results(
        self,
        gold_standard: pd.DataFrame,
        actual_result: pd.DataFrame,
        ignore_order: bool = True,
        ignore_case: bool = True,
    ) -> ComparisonOutcome:
        """
        比较 CSV 结果与金标准

        列级匹配分析：
        - 精确列匹配：名称和数据类型都匹配
        - 值匹配：数值容差比较、字符串忽略大小写
        """

    def calculate_accuracy_metrics(
        self,
        outcomes: List[ComparisonOutcome]
    ) -> Dict[str, float]:
        """计算准确性指标（准确率、召回率、F1 分数）"""
```

---

### 11. 统一错误处理 (error_handling.py)

```python
class NodeErrorResult(BaseResult):
    """统一节点错误结果"""

    def __init__(
        self,
        success: bool = False,
        error_code: str = "",
        error_message: str = "",
        error_details: Optional[Dict[str, Any]] = None,
        node_context: Optional[Dict[str, Any]] = None,
        retryable: bool = False,
        recovery_suggestions: Optional[List[str]] = None,
    ):
        # 错误码、消息、详情、上下文、可重试性、恢复建议


def unified_error_handler(node_type: str, operation: str):
    """
    统一错误处理装饰器

    自动处理：
    - DatusException：记录并重新抛出
    - JSONDecodeError：创建标准化错误结果
    - ConnectionError/TimeoutError：标记为可重试
    - 其他异常：记录堆栈跟踪
    """
```

---

### 12. 可追踪装饰器 (traceable_utils.py)

```python
def optional_traceable(name: str = "", run_type: RUN_TYPE_T = "chain"):
    """
    可选的可追踪装饰器

    当 LangSmith 可用时自动包装函数
    """
    def decorator(func):
        if not HAS_LANGSMITH:
            return func
        try:
            from langsmith import traceable

            trace_name = name or getattr(func, "__name__", "agent_operation")
            return traceable(name=trace_name, run_type=run_type)(func)
        except ImportError:
            return func

    return decorator


def create_openai_client(
    cls: Type[Union[OpenAI, AsyncOpenAI]],
    api_key: str,
    base_url: str,
    default_headers: Union[dict[str, str], None] = None,
    timeout: float = 300.0,
) -> Union[OpenAI, AsyncOpenAI]:
    """创建 OpenAI 客户端（禁用内置重试）"""
```

---

### 13. 其他工具模块

#### 13.1 文本清理 (text_utils.py)

```python
def clean_text(text: str) -> str:
    """
    清理文本（Unicode 规范化、移除不可见字符、统一换行符）
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00a0", " ").replace("\u200b", "").replace("\ufeff", "")
    text = re.sub(r"[\x00-\x08\x0B-\x1F\x7F]", "", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.strip()


def strip_markdown_code_block(text: str) -> str:
    """移除 Markdown 代码块标记"""
```

#### 13.2 Token 工具 (token_utils.py)

```python
def get_encoding():
    """获取 tiktoken 编码器（cl100k_base）"""
    global _encoding
    if _encoding is None:
        _encoding = tiktoken.get_encoding("cl100k_base")
    return _encoding


def cal_task_size(count: int, step: int) -> int:
    """计算任务分割数量"""
    return int(round(count / step + 0.5, 0))


def cal_gpt_tokens(text, encoding=None) -> int:
    """计算 GPT token 数量"""
```

---

## 架构特性

### 1. 环境感知

**自动检测源码环境：**
```python
def _is_source_environment() -> bool:
    """检查是否从源码目录运行"""
    has_pyproject = os.path.exists(os.path.join(project_root, "pyproject.toml"))
    has_datus_dir = os.path.exists(os.path.join(project_root, "datus"))
    return has_pyproject and has_datus_dir
```

### 2. 多策略解析

**DDL 解析多策略回退：**
1. 主要策略：sqlglot 解析
2. 回退策略：预编译正则表达式解析
3. 清理策略：移除错误消息片段
4. 中文保护：保留中文注释

### 3. 线程安全

**路径管理器双检锁：**
```python
def get_path_manager() -> DatusPathManager:
    global _path_manager
    if _path_manager is None:
        with _path_manager_lock:
            if _path_manager is None:
                _path_manager = DatusPathManager()
    return _path_manager
```

### 4. 类型安全

**SQL 类型安全检查：**
- `ALLOWED_SQL_TYPES` 白名单机制
- 输入验证（长度、类型、括号平衡）
- 方言映射和规范化

### 5. 可扩展性

**装饰器驱动：**
```python
@unified_error_handler("ExecuteSQLNode", "sql_execution")
def execute(self, input_data: ExecuteSQLInput) -> ExecuteSQLResult:
    # 自动错误处理
    pass

@optional_traceable(name="custom_name", run_type="chain")
def custom_function():
    # 自动 LangSmith 追踪
    pass
```

---

## 使用示例

### SQL DDL 解析

```python
from datus.utils.sql_utils import parse_metadata_from_ddl

ddl = """
CREATE TABLE users (
    id INT PRIMARY KEY COMMENT '用户ID',
    name VARCHAR(100) COMMENT '用户名',
    created_at TIMESTAMP COMMENT '创建时间'
) COMMENT='用户表'
"""

result = parse_metadata_from_ddl(ddl, dialect=DBType.STARROCKS)
# 返回:
# {
#     "table": {"name": "users", "comment": "用户表"},
#     "columns": [
#         {"name": "id", "type": "INT", "comment": "用户ID"},
#         {"name": "name", "type": "VARCHAR(100)", "comment": "用户名"},
#         {"name": "created_at", "type": "TIMESTAMP", "comment": "创建时间"}
#     ],
#     "primary_keys": ["id"]
# }
```

### LLM 结果解析

```python
from datus.utils.json_utils import llm_result2json

llm_output = '''
Here's the result:
```json
{
    "sql": "SELECT * FROM users",
    "explanation": "Get all users"
}
```
'''

result = llm_result2json(llm_output)
# 返回: {"sql": "SELECT * FROM users", "explanation": "Get all users"}
```

### 路径管理

```python
from datus.utils.path_manager import get_path_manager

pm = get_path_manager()

# 获取各种路径
config_path = pm.agent_config_path()        # ~/.datus/conf/agent.yml
logs_dir = pm.logs_dir                    # ~/.datus/logs
rag_path = pm.rag_storage_path("default") # ~/.datus/data/datus_db_default

# 创建保存目录
save_dir = pm.save_run_dir("namespace", "run_id")
# 自动创建目录并返回路径
```

### 数据压缩

```python
from datus.utils.compress_utils import DataCompressor

compressor = DataCompressor(
    model_name="gpt-4o",
    token_threshold=2000,
    output_format="csv"
)

# 压缩大型查询结果
result = compressor.compress(large_dataframe)
# 返回:
# {
#     "original_rows": 10000,
#     "is_compressed": True,
#     "compressed_data": "...",  # 前10行 + ... + 后10行
#     "compression_type": "rows"
# }
```

### 异步执行

```python
from datus.utils.async_utils import run_async

async def async_function():
    await asyncio.sleep(1)
    return "async result"

# 在同步上下文中运行异步函数
result = run_async(async_function())
```

### 错误处理

```python
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.error_handling import unified_error_handler

@unified_error_handler("MyNode", "my_operation")
def my_operation():
    # 自动错误处理
    raise DatusException(
        ErrorCode.DB_CONNECTION_FAILED,
        message="Database connection timeout"
    )
```

---

## 版本更新记录

### v2.0 (2026-01-22)
- 完整重写，基于最新代码架构
- 新增 28 个工具模块详细说明
- 新增 ErrorCode 7 位错误码体系
- 新增 DatusException 标准化异常
- 新增 DynamicLogManager 动态日志系统
- 新增 DatusPathManager 集中式路径管理
- 新增 SubAgentManager Sub-Agent 管理
- 新增 DataCompressor 数据压缩工具
- 新增 SQL 工具完整文档（DDL 解析、中文注释保护）
- 新增 JSON 工具完整文档（LLM 结果解析）
- 新增统一错误处理装饰器
- 新增可追踪装饰器支持
- 完善异步运行时文档

### v1.0 (2026-01-05)
- 初始版本
- 高层次架构概述
