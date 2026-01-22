# Datus CLI 模块介绍

> **文档版本**: v1.0
> **更新日期**: 2026-01-06
> **相关模块**: `datus/cli/`

---

基于对 `datus/cli` 模块的深度代码分析，下面详细介绍这个 CLI 系统的设计和功能。

### 设计哲学与架构模式

**Datus-CLI 采用分层架构设计**，遵循以下设计原则：

1. **命令驱动架构**：基于前缀区分的命令系统 (`!` 工具命令、`@` 上下文命令、`/` 聊天命令、`.` 内部命令)
2. **REPL 交互模式**：Read-Eval-Print Loop 交互式命令行界面
3. **模块化命令处理器**：每个功能领域都有独立的命令处理器类
4. **状态管理**：通过 `CliContext` 维护会话状态和历史记录
5. **异步执行支持**：支持后台 Agent 初始化和异步命令执行

### 核心组件分析

#### 1. **main.py** (160行) - CLI 入口点

**核心职责**：
- 命令行参数解析和验证
- Web 界面启动支持
- 数据库连接参数处理
- 应用生命周期管理

**关键特性**：
```python
class Application:
    def run(self):
        # 支持两种模式：CLI 和 Web 界面
        if args.web:
            self._run_web_interface(args)
        else:
            cli = DatusCLI(args)
            cli.run()
```

**为何如此设计**：
- **双模式支持**：既支持传统的命令行界面，也支持现代的 Web 界面
- **参数验证**：确保必需的命名空间参数存在
- **环境隔离**：通过命名空间实现多环境配置支持

#### 2. **repl.py** (1087行) - 核心 REPL 引擎

**核心功能**：
- 交互式命令循环处理
- 命令解析和路由分发
- 键盘绑定和自动补全集成
- 数据库连接管理和 SQL 执行
- 错误处理和异常管理

**命令类型系统**：
```python
class CommandType(Enum):
    SQL = "sql"        # 直接 SQL 执行
    TOOL = "tool"      # ! 前缀工具命令
    CONTEXT = "context"  # @ 前缀上下文命令  
    CHAT = "chat"      # / 前缀聊天命令
    INTERNAL = "internal"  # . 前缀内部命令
```

**命令解析逻辑**：
```python
def _parse_command(self, text: str) -> Tuple[CommandType, str, str]:
    # 基于前缀和内容智能判断命令类型
    # 支持 SQL 自动检测和聊天回退
```

**设计优势**：
- **智能命令解析**：通过 SQL 语法分析自动识别命令类型
- **键盘增强**：Shift+Tab 切换计划模式，Tab 补全，Ctrl+O 显示详情
- **异步初始化**：Agent 在后台线程初始化，避免阻塞 UI
- **状态跟踪**：通过 ActionHistory 记录所有操作

#### 3. **agent_commands.py** (1032行) - Agent 工作流命令

**主要功能**：
- 工作流节点执行管理
- 自然语言查询处理 (`!darun`)
- 模式链接 (`!sl`) 和 SQL 生成 (`!gen`)
- 结果保存和输出管理

**核心方法分析**：

**cmd_schema_linking**：
```python
def cmd_schema_linking(self, args: str):
    # 基于向量搜索找到相关表结构
    # 使用 SchemaWithValueRAG 进行语义搜索
    # 显示表定义和示例数据
```

**cmd_darun_screen**：
```python
def cmd_darun_screen(self, args: str, task: SqlTask = None):
    # 完整的 AI 驱动 SQL 生成工作流
    # 包含状态显示和管理
    # 支持并行执行和选择
```

**为何如此设计**：
- **工作流抽象**：将复杂的 AI 工作流封装为简单命令
- **渐进式交互**：支持从简单 SQL 到复杂 AI 驱动查询的平滑过渡
- **状态可视化**：通过屏幕显示提供工作流执行状态

#### 4. **chat_commands.py** (790行) - AI 聊天命令

**核心功能**：
- 多类型 Agentic 节点支持 (Chat, GenSQL, Semantic, SQLSummary)
- 会话管理和切换
- 计划模式支持
- 上下文感知输入生成

**Agentic 节点架构**：
```python
# 支持四种类型的对话节点
CHAT_NODE = "chat"           # 通用对话
GENSQL_NODE = "gensql"       # SQL 生成对话
SEMANTIC_NODE = "semantic"   # 语义模型对话  
SQL_SUMMARY_NODE = "sql_summary"  # SQL 摘要对话
```

**节点切换逻辑**：
```python
def _should_create_new_node(self, subagent_name: str = None) -> bool:
    # 智能判断是否需要创建新节点
    # 支持从通用聊天切换到专用子代理
```

**设计优势**：
- **多模态对话**：不同类型的对话节点满足不同需求
- **会话压缩**：自动管理对话历史长度，避免 token 限制
- **上下文集成**：自动注入数据库上下文和历史查询

#### 5. **autocomplete.py** (1087行) - 智能自动补全

**补全类型**：
- **SQL 关键字补全**：标准 SQL 语法关键字
- **表名列名补全**：基于当前数据库模式
- **命令补全**：CLI 命令和子命令
- **@引用补全**：表、指标、SQL 引用

**补全器架构**：
```python
class SQLCompleter(Completer):
    # SQL 语法补全
    
class AtReferenceCompleter(Completer):  
    # @引用补全 (表、指标、SQL)
    
class SubagentCompleter(Completer):
    # 子代理名称补全
```

**为何如此设计**：
- **上下文感知**：补全结果基于当前数据库连接和会话状态
- **多级补全**：组合使用多个补全器提供全面建议
- **性能优化**：懒加载和缓存机制避免重复计算

#### 6. **plan_hooks.py** (5901行) - 计划模式执行引擎

**核心功能**：
- 拦截 Agent 执行流程
- 任务分类和路由
- 执行状态管理和显示
- 阻塞式输入处理

**任务分类系统**：
```python
class TaskType:
    TOOL_EXECUTION = "tool_execution"  # 需要外部工具调用
    LLM_ANALYSIS = "llm_analysis"     # 需要 LLM 推理
    HYBRID = "hybrid"                # 可能需要工具和分析
```

**执行钩子架构**：
```python
class PlanHooks(AgentHooks):
    # 继承 Agent 生命周期钩子
    # 拦截和控制执行流程
    # 提供用户交互界面
```

**为何如此设计**：
- **透明拦截**：在不修改 Agent 核心逻辑的情况下增强功能
- **智能路由**：基于任务内容自动选择执行策略
- **用户控制**：允许用户干预和指导执行流程

#### 7. **sub_agent_wizard.py** (1855行) - 子代理创建向导

**核心功能**：
- 交互式子代理配置创建
- 分步骤配置流程
- 实时预览和验证
- 模板管理和自定义

**向导步骤**：
1. **基本信息**：名称、描述、系统提示
2. **作用域上下文**：表、指标、SQL 选择
3. **规则配置**：自定义规则和约束
4. **MCP 工具集成**：外部工具配置

**为何如此设计**：
- **用户友好**：图形化界面降低配置复杂度
- **模板支持**：提供预定义模板加速配置
- **实时反馈**：边配置边预览最终效果

#### 8. **metadata_commands.py** (362行) - 元数据管理命令

**核心功能**：
- 数据库、表、模式信息查询
- 数据库切换和管理
- 索引和约束信息显示

**数据库适配**：
```python
# 自动适配不同数据库类型
if db_type in (DBType.SQLITE, DBType.DUCKDB):
    # 文件数据库特殊处理
else:
    # 传统 RDBMS 处理
```

#### 9. **context_commands.py** (66行) - 上下文浏览命令

**核心功能**：
- 数据库目录树浏览 (`@catalog`)
- 语义模型和指标展示 (`@subject`)

**可视化界面**：
```python
# 使用 Textual 框架提供丰富的 TUI 界面
show_catalog_screen(...)
show_subject_screen(...)
```

#### 10. **cli_context.py** (228行) - 上下文状态管理

**状态管理**：
- 数据库连接上下文
- 最近使用的表、指标、SQL
- 当前任务状态

**历史记录管理**：
```python
@dataclass
class CliContext:
    recent_tables: deque = field(default_factory=lambda: deque(maxlen=20))
    recent_metrics: deque = field(default_factory=lambda: deque(maxlen=20))  
    recent_sql_contexts: deque = field(default_factory=lambda: deque(maxlen=20))
```

**为何如此设计**：
- **状态持久化**：在会话间保持上下文
- **智能缓存**：LRU 缓存最近使用的资源
- **去重机制**：避免重复添加相同项目

### 命令系统架构

#### 命令分类体系

| 命令类型 | 前缀 | 功能领域 | 示例 |
|---------|------|---------|------|
| **工具命令** | `!` | AI 工作流执行 | `!sl`, `!gen`, `!save` |
| **上下文命令** | `@` | 数据浏览探索 | `@catalog`, `@subject` |
| **聊天命令** | `/` | AI 对话交互 | `/explain this query` |
| **内部命令** | `.` | CLI 控制管理 | `.databases`, `.help` |

#### 命令处理器架构

```python
class DatusCLI:
    def __init__(self, args):
        # 初始化各个命令处理器
        self.agent_commands = AgentCommands(self, self.cli_context)
        self.chat_commands = ChatCommands(self)
        self.context_commands = ContextCommands(self)
        self.metadata_commands = MetadataCommands(self)
        self.sub_agent_commands = SubAgentCommands(self)
        
        # 命令路由表
        self.commands = {
            "!sl": self.agent_commands.cmd_schema_linking,
            "@catalog": self.context_commands.cmd_catalog,
            "/": self._execute_chat_command,
            ".help": self._cmd_help,
            # ...
        }
```

### 数据库连接架构

#### 多数据库支持

**命名空间隔离**：
```python
# 通过命名空间实现环境隔离
namespace:
  production:
    type: snowflake
    account: ${PROD_ACCOUNT}
  development:  
    type: sqlite
    uri: dev.db
```

**连接管理**：
```python
class DbManager:
    def get_conn(self, namespace: str, db_name: str):
        # 统一连接接口，支持多种数据库类型
```

### AI 集成架构

#### Agentic 节点系统

**四种节点类型**：
1. **ChatAgenticNode**：通用对话能力
2. **GenSQLAgenticNode**：SQL 生成专用
3. **SemanticAgenticNode**：语义模型生成
4. **SqlSummaryAgenticNode**：SQL 摘要分析

**统一接口设计**：
```python
class BaseAgenticNode:
    def setup_tools(self):
        # 配置工具链
        
    def run_async(self, input_data):
        # 异步执行逻辑
```

### 错误处理与日志

#### 统一异常处理

```python
setup_exception_handler(
    console_logger=self.console.print,
    prefix_wrap_func=lambda x: f"[bold red]{x}[/bold red]"
)
```

#### 操作历史跟踪

```python
class ActionHistoryManager:
    def add_action(self, action: ActionHistory):
        # 记录所有 CLI 操作用于审计和调试
```

### 性能优化策略

1. **异步初始化**：Agent 在后台线程初始化
2. **懒加载**：按需加载补全数据和工具
3. **缓存机制**：缓存表结构和元数据信息
4. **分页显示**：智能表格显示避免终端阻塞

### 扩展性设计

#### 插件化架构

- **新命令类型**：轻松添加新的命令前缀和处理器
- **自定义节点**：通过继承 BaseAgenticNode 添加新 AI 能力
- **数据库适配器**：通过统一接口支持新数据库类型
- **工具集成**：MCP 协议支持外部工具无缝集成

### 适用场景分析

**数据工程师日常工作**：
- 快速 SQL 原型开发和测试
- 数据库模式探索和理解
- AI 辅助的复杂查询生成

**数据分析师交互分析**：
- 自然语言到 SQL 的转换
- 结果解释和优化建议
- 历史查询管理和重用

**开发团队协作**：
- 标准化的数据访问接口
- 查询结果的版本控制和共享
- 环境隔离的开发/测试/生产切换

**教学和培训**：
- 交互式 SQL 学习环境
- AI 辅助的查询理解
- 逐步的复杂性递进

这个 CLI 系统体现了现代数据工具的设计理念，既保留了命令行的效率和可脚本化，又融入了 AI 能力的智能化和用户友好的交互体验。通过模块化的架构设计，它能够灵活地适应不同的使用场景和扩展需求。