# Datus Agent Node 模块介绍

> **文档版本**: v2.1
> **更新日期**: 2026-01-23
> **相关模块**: `datus/agent/node/`, `datus/configuration/node_type.py`

---

## 目录

1. [系统概述](#1-系统概述)
2. [架构设计理念](#2-架构设计理念)
3. [节点类型定义](#3-节点类型定义)
4. [节点基类与工厂](#4-节点基类与工厂)
5. [节点实现详解](#5-节点实现详解)
6. [节点继承关系](#6-节点继承关系)
7. [预检编排器](#7-预检编排器)
8. [工作流集成](#8-工作流集成)
9. [Text2SQL完整流程](#9-text2sql完整流程)
10. [输入输出模型](#10-输入输出模型)

---

## 1. 系统概述

### 1.1 什么是Node模块

Datus Agent Node 模块是整个数据工程智能代理系统的**核心执行引擎**。它实现了**可编排的工作流节点执行引擎**，通过标准化节点接口实现复杂数据工程工作流的模块化执行。

```
┌─────────────────────────────────────────────────────────────────┐
│                     Datus Agent 系统                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐         ┌──────────────┐         ┌─────────┐ │
│  │   Workflow   │────────▶│   Node Engine│────────▶│  Output │ │
│  │   Definition │ 编排    │   Execution  │ 执行    │  Result │ │
│  └──────────────┘         └──────────────┘         └─────────┘ │
│                                                                  │
│  ┌──────────────┐         ┌──────────────┐         ┌─────────┐ │
│  │   Workflow   │◀────────│   Node Store │◀────────│  Debug  │ │
│  │   Runtime    │ 状态    │   History    │ 追溯    │  Logs   │ │
│  └──────────────┘         └──────────────┘         └─────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 核心设计原则

| 原则 | 描述 | 实现方式 |
|------|------|----------|
| **标准化接口** | 所有节点继承统一的Node基类 | 抽象方法 `execute()` / `execute_stream()` |
| **流式执行** | 支持同步和异步流式两种模式 | `execute()` vs `execute_stream()` |
| **工具集成** | 原生支持MCP协议和传统工具调用 | AgenticNode基类 + Tool抽象 |
| **事件驱动** | 通过ActionHistory实现节点间通信 | `action_history_manager.add_action()` |
| **工厂模式** | 统一的节点创建入口 | `Node.new_instance()` |

---

## 2. 架构设计理念

### 2.1 架构层次结构

```
┌─────────────────────────────────────────────────────────────────┐
│                    工作流层 (Workflow Layer)                     │
│         负责工作流定义、节点编排、状态管理、执行控制              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────┐    ┌─────────────────────┐            │
│  │   控制流节点         │    │   功能执行节点       │            │
│  │   Control Flow      │    │   Functional Nodes  │            │
│  │                     │    │                     │            │
│  │  BeginNode          │    │  IntentAnalysisNode │            │
│  │  ReflectNode        │    │  SchemaDiscoveryNode│            │
│  │  ParallelNode       │    │  GenerateSQLNode    │            │
│  │  SelectionNode      │    │  ExecuteSQLNode     │            │
│  │  SubworkflowNode    │    │  ...                │            │
│  └─────────────────────┘    └─────────────────────┘            │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                    代理节点层 (Agentic Layer)                    │
│         提供会话管理、工具集成、MCP协议支持                      │
│                                                                  │
│  ┌─────────────────────┐    ┌─────────────────────┐            │
│  │   ChatAgenticNode   │    │   GenSQLAgenticNode │            │
│  │   (多用途对话)      │    │   (SQL生成对话)      │            │
│  └─────────────────────┘    └─────────────────────┘            │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                    基础节点层 (Base Layer)                       │
│                    Node抽象基类 + 工具抽象                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    Node (抽象基类)                        │   │
│  │   - update_context()    - setup_input()                  │   │
│  │   - execute()           - execute_stream()               │   │
│  │   - start()/complete()/fail()                            │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 节点分类体系

| 类别 | 数量 | 功能描述 | 继承关系 |
|------|------|----------|----------|
| **控制流节点** | 6个 | 工作流流程控制（开始、并行、选择、回退等） | 直接继承Node |
| **功能执行节点** | 18个 | 具体业务功能（意图分析、SQL生成、执行等） | 直接继承Node |
| **代理节点** | 4个 | 会话管理、工具集成、MCP支持 | 继承AgenticNode |

### 节点类型快速参考表

| 节点类型 | 功能 | 说明 |
|---------|------|------|
| **控制流节点** | | |
| `start` | 开始 | 工作流起点 |
| `hitl` | 人工介入 | Human in the loop |
| `reflect` | 反思 | 评估和自我反思 |
| `parallel` | 并行 | 并行执行子节点 |
| `selection` | 选择 | 从多个候选中选择最佳结果 |
| `subworkflow` | 子工作流 | 执行嵌套工作流 |
| **SQL 工作流节点** | | |
| `intent_analysis` | 意图分析 | 使用启发式分析查询意图 |
| `intent_clarification` | 意图澄清 | 修复错别字、澄清歧义、提取实体 |
| `schema_discovery` | Schema 发现 | 发现相关 Schema 和表 |
| `schema_validation` | Schema 验证 | 验证 Schema 充分性 |
| `schema_linking` | Schema 链接 | 理解查询并找到相关 Schema |
| `generate_sql` | SQL 生成 | 生成 SQL 查询 |
| `sql_validate` | SQL 验证 | 验证 SQL 语法和语义 |
| `execute_sql` | SQL 执行 | 执行 SQL 查询 |
| `result_validation` | 结果验证 | 验证 SQL 执行结果质量 |
| `output` | 输出 | 向用户返回结果 |
| `reasoning` | 推理分析 | 推理分析 |
| `fix` | 修复 | 修复 SQL 查询 |
| `search_metrics` | 搜索指标 | 搜索业务指标 |
| `compare` | 比较 | 与预期比较 SQL |
| `date_parser` | 日期解析 | 解析查询中的时间表达式 |
| `knowledge_enhancement` | 知识增强 | 统一和丰富知识 |
| `doc_search` | 文档搜索 | 搜索相关文档 |
| **Agentic 节点** | | |
| `chat` | 对话 AI | 带工具调用的对话交互 |
| `gensql` | 生成 SQL | 带工具调用的 SQL 生成 |
| `semantic` | 语义模型 | 带工具调用的语义模型生成 |
| `sql_summary` | SQL 摘要 | 带工具调用的 SQL 摘要生成 |

---

## 3. 节点类型定义

### 3.1 控制流节点类型 (`datus/configuration/node_type.py`)

```python
# 控制流类型
TYPE_BEGIN = "start"              # 工作流起点
TYPE_HITL = "hitl"                # 人类参与节点
TYPE_REFLECT = "reflect"          # 智能评估和反思
TYPE_PARALLEL = "parallel"        # 并行执行节点
TYPE_SELECTION = "selection"      # 智能选择节点
TYPE_SUBWORKFLOW = "subworkflow"  # 嵌套工作流节点
```

### 3.2 SQL工作流动作类型

```python
# 意图处理
TYPE_INTENT_ANALYSIS = "intent_analysis"           # 意图分析
TYPE_INTENT_CLARIFICATION = "intent_clarification" # 意图澄清

# 模式处理
TYPE_SCHEMA_DISCOVERY = "schema_discovery"         # 模式发现
TYPE_SCHEMA_LINKING = "schema_linking"             # 模式链接（旧版）
TYPE_SCHEMA_VALIDATION = "schema_validation"       # 模式验证

# SQL处理
TYPE_GENERATE_SQL = "generate_sql"                 # SQL生成
TYPE_SQL_VALIDATE = "sql_validate"                 # SQL验证
TYPE_EXECUTE_SQL = "execute_sql"                   # SQL执行
TYPE_RESULT_VALIDATION = "result_validation"       # 结果验证
TYPE_REASONING = "reasoning"                       # SQL推理

# 辅助功能
TYPE_FIX = "fix"                                   # SQL修复
TYPE_COMPARE = "compare"                           # 结果比较
TYPE_OUTPUT = "output"                             # 输出处理
TYPE_DOC_SEARCH = "doc_search"                     # 文档搜索
TYPE_DATE_PARSER = "date_parser"                   # 日期解析
TYPE_SEARCH_METRICS = "search_metrics"             # 指标搜索
TYPE_KNOWLEDGE_ENHANCEMENT = "knowledge_enhancement" # 知识增强
```

### 3.3 代理节点类型

```python
TYPE_CHAT = "chat"           # 对话式AI节点
TYPE_GENSQL = "gensql"       # 智能SQL生成节点
TYPE_SEMANTIC = "semantic"   # 语义模型生成节点
TYPE_SQL_SUMMARY = "sql_summary"  # SQL摘要生成节点
```

---

## 4. 节点基类与工厂

### 4.1 抽象基类 `Node`

```python
class Node(ABC):
    """所有节点的抽象基类"""

    # ===== 实例属性 =====
    id: str                    # 节点唯一标识
    description: str           # 人类可读描述
    type: str                  # 节点类型 (node_type.py中定义)
    input: BaseInput           # 输入数据
    status: NodeStatus         # pending/running/completed/failed
    result: BaseResult         # 执行结果
    start_time: float          # 开始时间戳
    end_time: float            # 结束时间戳
    dependencies: List[str]    # 依赖的节点ID列表
    metadata: Dict[str, Any]   # 元数据
    agent_config: AgentConfig  # 代理配置
    model: LLMBaseModel        # LLM模型实例
    tools: List[Tool]          # 可用工具列表
    workflow: Workflow         # 关联的工作流

    # ===== 抽象方法 =====
    @abstractmethod
    def update_context(self, workflow: "Workflow") -> Dict[str, Any]:
        """更新工作流上下文"""

    @abstractmethod
    def setup_input(self, workflow: "Workflow") -> Dict[str, Any]:
        """设置节点输入"""

    @abstractmethod
    def execute(self) -> BaseResult:
        """同步执行节点逻辑"""

    @abstractmethod
    async def execute_stream(
        self,
        action_history_manager: Optional[ActionHistoryManager]
    ) -> AsyncGenerator[ActionHistory, None]:
        """异步流式执行，返回ActionHistory事件流"""
```

### 4.2 公共方法

| 方法 | 功能 | 返回值 |
|------|------|--------|
| `start()` | 标记节点开始执行 | None |
| `complete(result)` | 标记节点完成 | None |
| `fail(error)` | 标记节点失败 | None |
| `run()` | 同步执行节点 | BaseResult |
| `run_async()` | 异步执行 | asyncio.Task |
| `run_stream()` | 流式执行 | AsyncGenerator |
| `execute_with_standardized_handling()` | 标准化执行（含错误处理） | BaseResult |
| `add_dependency(node_id)` | 添加依赖 | None |
| `_sql_connector(database_name)` | 获取数据库连接器 | BaseConnector |

### 4.3 节点工厂方法

```python
@classmethod
def new_instance(
    cls,
    node_id: str,
    description: str,
    node_type: str,
    input_data: Dict[str, Any],
    agent_config: AgentConfig,
    tools: List[Tool],
    node_name: Optional[str] = None
) -> "Node":
    """根据node_type创建对应的节点实例

    Args:
        node_id: 节点唯一标识
        description: 节点描述
        node_type: 节点类型 (来自node_type.py常量)
        input_data: 输入数据
        agent_config: 代理配置
        tools: 可用工具列表
        node_name: 可选节点名称

    Returns:
        对应类型的节点实例
    """
```

**工厂映射示例**：

| node_type | 节点类 |
|-----------|--------|
| `"start"` | BeginNode |
| `"intent_analysis"` | IntentAnalysisNode |
| `"intent_clarification"` | IntentClarificationNode |
| `"schema_discovery"` | SchemaDiscoveryNode |
| `"generate_sql"` | GenerateSQLNode |
| `"execute_sql"` | ExecuteSQLNode |
| `"reflect"` | ReflectNode |
| `"chat"` | ChatAgenticNode |
| ... | ... |

### 4.4 序列化支持

```python
def to_dict(self) -> Dict[str, Any]:
    """将节点序列化为字典"""
    return {
        "id": self.id,
        "type": self.type,
        "description": self.description,
        "status": self.status.value,
        "metadata": self.metadata,
    }

@classmethod
def from_dict(cls, data: Dict[str, Any], workflow: "Workflow") -> "Node":
    """从字典反序列化节点"""
    # 使用工厂方法创建节点
    return cls.new_instance(...)
```

---

## 5. 节点实现详解

### 5.1 控制流节点 (6个)

#### BeginNode - 工作流起点

```python
class BeginNode(Node):
    """工作流起始节点，无操作"""
    TYPE_BEGIN = "start"

    def execute(self) -> BaseResult:
        # 初始化工作流上下文
        # 设置任务参数
        return BaseResult(success=True)

    async def execute_stream(self, action_history_manager) -> AsyncGenerator:
        yield ActionHistory(...)
```

#### ReflectNode - 智能评估和反思

```python
class ReflectNode(Node):
    """SQL质量控制节点，基于执行结果进行策略选择"""
    TYPE_REFLECT = "reflect"

    # 支持的反思策略
    STRATEGIES = {
        "schema_linking": {"max_iterations": 2, "desc": "重新分析数据库模式"},
        "simple_regenerate": {"max_iterations": 3, "desc": "简单重新生成SQL"},
        "reasoning": {"max_iterations": 3, "desc": "深度推理分析"},
        "doc_search": {"max_iterations": 1, "desc": "搜索相关文档"},
    }

    def execute(self) -> BaseResult:
        # 1. 评估SQL执行结果质量
        # 2. 根据评估结果选择策略
        # 3. 返回下一步指令（继续/重试/回退/终止）
```

#### ParallelNode - 并行执行节点

```python
class ParallelNode(Node):
    """并行执行多个子节点，提高成功率"""
    TYPE_PARALLEL = "parallel"

    def execute(self) -> BaseResult:
        # 使用ThreadPoolExecutor并发执行
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(node.execute) for node in self.child_nodes]
            results = [f.result() for f in futures]

        # 聚合结果
        return ParallelResult(results=results)
```

#### SelectionNode - 智能选择节点

```python
class SelectionNode(Node):
    """从多个候选结果中选择最佳结果"""
    TYPE_SELECTION = "selection"

    def execute(self) -> BaseResult:
        # LLM-based选择：分析候选结果并选择最佳
        # Rule-based选择：基于预定义规则选择
        pass
```

#### SubworkflowNode - 嵌套工作流节点

```python
class SubworkflowNode(Node):
    """执行嵌套的子工作流"""
    TYPE_SUBWORKFLOW = "subworkflow"

    def execute(self) -> BaseResult:
        # 1. 加载子工作流定义
        # 2. 传递上下文参数
        # 3. 递归执行子工作流
        # 4. 收集结果并返回
```

#### HitlNode - 人类参与节点

```python
class HitlNode(Node):
    """支持人工干预的工作流暂停和交互"""
    TYPE_HITL = "hitl"

    def execute(self) -> BaseResult:
        # 1. 暂停工作流执行
        # 2. 等待人工输入（审批/拒绝/修改）
        # 3. 根据人工输入继续执行
```

### 5.2 功能执行节点 (17+)

#### IntentAnalysisNode - 意图分析节点

```python
class IntentAnalysisNode(Node):
    """双阶段意图检测：快速启发式 + LLM回退"""
    TYPE_INTENT_ANALYSIS = "intent_analysis"

    SUPPORTED_INTENTS = ["text2sql", "sql_review", "data_analysis"]

    def execute(self) -> BaseResult:
        # 阶段1: 启发式检测（关键词 + 正则）
        intent, confidence = self._heuristic_detection(self.input.task)

        # 阶段2: LLM回退（confidence < 0.7时）
        if confidence < 0.7:
            intent, confidence = self._llm_classification(self.input.task)

        # 输出到workflow.metadata
        workflow.metadata["detected_intent"] = intent
        workflow.metadata["intent_confidence"] = confidence
```

#### IntentClarificationNode - 意图澄清节点

```python
class IntentClarificationNode(Node):
    """LLM-based业务意图澄清：纠错、消歧、实体提取"""
    TYPE_INTENT_CLARIFICATION = "intent_clarification"

    def execute(self) -> BaseResult:
        # 1. Typo correction: "华山" → "华南"
        # 2. Entity extraction: business_terms, time_range, dimensions, metrics
        # 3. Query normalization: 纠正语法错误，标准化表达

        # 输出到workflow.metadata
        workflow.metadata["clarified_task"] = clarified_task
        workflow.metadata["intent_clarification"] = {...}
```

#### SchemaDiscoveryNode - 模式发现节点

```python
class SchemaDiscoveryNode(Node):
    """自动数据库模式探索和表发现"""
    TYPE_SCHEMA_DISCOVERY = "schema_discovery"

    def execute(self) -> BaseResult:
        # 使用clarified_task进行搜索
        clarified_task = workflow.metadata.get("clarified_task")

        # 发现策略
        strategies = [
            self._semantic_search,      # 语义搜索
            self._keyword_matching,     # 关键词匹配
            self._llm_inference,        # LLM推理
            self._progressive_matching, # 渐进式匹配（基于reflection_round）
        ]

        # 输出: workflow.context.table_schemas
```

#### SchemaValidationNode - 模式验证节点

```python
class SchemaDiscoveryNode(Node):
    """验证SQL模式约束和完整性"""
    TYPE_SCHEMA_VALIDATION = "schema_validation"

    def execute(self) -> BaseResult:
        # 验证项
        validations = [
            self._validate_column_exists,    # 列存在性验证
            self._validate_table_relations,  # 表关系完整性
            self._validate_metadata,         # 模式元数据验证
        ]
```

#### GenerateSQLNode - SQL生成节点

```python
class GenerateSQLNode(Node):
    """LLM-based SQL查询生成"""
    TYPE_GENERATE_SQL = "generate_sql"

    def execute(self) -> BaseResult:
        # 1. 使用llm_result2json()标准化解析
        from datus.utils.json_utils import llm_result2json
        result = llm_result2json(response_text, expected_type=dict)

        # 2. 生成SQL
        sql = result.get("sql_query")

        # 3. 缓存优化（1小时TTL）
        cache_key = self._generate_cache_key()
```

#### SQLValidateNode - SQL验证节点

```python
class SQLValidateNode(Node):
    """SQL执行前综合验证"""
    TYPE_SQL_VALIDATE = "sql_validate"

    DANGEROUS_PATTERNS = {
        "DROP", "ALTER", "CREATE", "TRUNCATE",
        "DELETE", "UPDATE", "INSERT"
    }

    def execute(self) -> BaseResult:
        # 验证层级
        validations = [
            self._syntax_validation,     # 语法验证 (sqlglot)
            self._schema_validation,     # 模式验证
            self._dangerous_ops_check,   # 危险操作检测
            self._naming_check,          # 命名规范检查
        ]
```

#### ExecuteSQLNode - SQL执行节点

```python
class ExecuteSQLNode(Node):
    """数据库查询执行和结果处理"""
    TYPE_EXECUTE_SQL = "execute_sql"

    async def execute_stream(self, action_history_manager) -> AsyncGenerator:
        # 异步执行SQL
        async for event in self._execute_sql_async():
            yield event
```

#### ResultValidationNode - 结果验证节点

```python
class ResultValidationNode(Node):
    """SQL执行结果质量验证"""
    TYPE_RESULT_VALIDATION = "result_validation"

    def execute(self) -> BaseResult:
        # 质量检查
        checks = [
            self._check_empty_result,    # 空结果检测
            self._check_anomalies,       # 异常值检测
            self._check_integrity,       # 数据完整性检查
        ]

        # HTML预览生成
        html_preview = self._generate_html_preview(df)
```

#### 其他功能节点

| 节点 | 类型 | 功能 |
|------|------|------|
| `SchemaLinkingNode` | schema_linking | 理解查询并找到相关模式（旧版） |
| `ReasonSQLNode` | reasoning | SQL查询推理和增强 |
| `FixNode` | fix | 修复问题SQL查询 |
| `CompareNode` | compare | SQL结果与期望比较 |
| `KnowledgeEnhancementNode` | knowledge_enhancement | 统一知识处理和增强 |
| `DateParserNode` | date_parser | 解析自然语言时间表达式 |
| `DocSearchNode` | doc_search | 在知识库中搜索相关文档 |
| `SearchMetricsNode` | search_metrics | 搜索和匹配相关的业务指标 |
| `OutputNode` | output | 格式化和输出执行结果 |

### 5.3 代理节点 (3+)

#### AgenticNode - 代理节点基类

```python
class AgenticNode(Node):
    """提供会话管理和工具集成的代理节点基类"""

    def __init__(self):
        # 会话管理
        self.session_id: Optional[str] = None
        self.session_db_path: Optional[str] = None

        # 工具集成
        self.available_tools: List[Tool] = []
        self.mcp_servers: List[MCP Server] = []

    async def execute_stream(self, action_history_manager) -> AsyncGenerator:
        # 流式执行，支持工具调用
        async for event in self._execute_with_tools():
            yield event

    async def _manual_compact(self) -> dict:
        """手动上下文压缩"""
        pass

    async def _auto_compact(self) -> bool:
        """自动上下文压缩（90% tokens阈值）"""
        pass
```

#### ChatAgenticNode - 对话式AI节点

```python
class ChatAgenticNode(Node):
    """多模式智能对话交互"""
    TYPE_CHAT = "chat"

    EXECUTION_MODES = ["text2sql", "data_analysis", "sql_review"]

    async def execute_stream(self, action_history_manager) -> AsyncGenerator:
        # 1. 预检编排（自动执行必要工具）
        # 2. 计划执行（结构化执行计划）
        # 3. 实时事件流输出
```

#### GenSQLAgenticNode - 智能SQL生成节点

```python
class GenSQLAgenticNode(Node):
    """基于会话的智能SQL生成"""
    TYPE_GENSQL = "gensql"

    # 继承AgenticNode能力
    # - 多轮对话上下文保持
    # - 自动上下文压缩
    # - MCP服务器支持
```

#### CompareAgenticNode - 比较分析代理节点

```python
class CompareAgenticNode(Node):
    """SQL比较的代理实现"""
    TYPE_COMPARE = "compare"
```

---

## 6. 节点继承关系

```
Node (抽象基类)
│
├── BeginNode
├── IntentAnalysisNode
├── IntentClarificationNode
├── KnowledgeEnhancementNode
├── SchemaDiscoveryNode
├── SchemaLinkingNode
├── SchemaValidationNode
├── GenerateSQLNode
├── ExecuteSQLNode
├── SQLValidateNode
├── ResultValidationNode
├── ReflectNode
├── ReasonSQLNode
├── OutputNode
├── DocSearchNode
├── FixNode
├── HitlNode
├── DateParserNode
├── SearchMetricsNode
├── CompareNode
├── ParallelNode
├── SelectionNode
├── SubworkflowNode
│
└── AgenticNode (继承自Node的代理基类)
    ├── ChatAgenticNode
    ├── GenSQLAgenticNode
    └── CompareAgenticNode
```

---

## 7. 预检编排器

### 7.1 PreflightOrchestrator

```python
class PreflightOrchestrator(Node):
    """协调执行预检工具，为Text2SQL准备上下文"""

    SUPPORTED_TOOLS = [
        "search_table",              # 表搜索
        "describe_table",            # 表描述
        "search_reference_sql",      # 参考SQL搜索
        "parse_temporal_expressions", # 时间表达式解析
    ]

    def execute(self) -> BaseResult:
        # 1. 智能缓存（基于参数生成缓存键）
        # 2. 顺序执行工具
        # 3. 错误处理和降级
        # 4. 性能监控

        # 输出: workflow.context.preflight_results
```

### 7.2 SqlReviewPreflightOrchestrator

```python
class SqlReviewPreflightOrchestrator(Node):
    """SQL审查专用预检工具编排"""

    # 工具分类
    CRITICAL_TOOLS = [
        "describe_table",           # 表结构分析
        "search_external_knowledge", # 规则检索
    ]

    ENHANCED_TOOLS = [
        "analyze_query_plan",       # 查询计划分析
        "check_table_conflicts",    # 表冲突检测
        "validate_partitioning",    # 分区验证
    ]

    # 强制执行顺序
    REQUIRED_TOOL_SEQUENCE = [
        "describe_table",
        "search_external_knowledge",
        "read_query",
        "get_table_ddl",
        "analyze_query_plan",
        "check_table_conflicts",
        "validate_partitioning",
    ]
```

---

## 8. 工作流集成

### 8.1 Workflow类结构

```python
class Workflow:
    name: str                      # 工作流名称
    task: SqlTask                  # SQL任务
    nodes: Dict[str, Node]         # 节点映射 {node_id: Node}
    node_order: List[str]          # 节点执行顺序 [node_id1, node_id2, ...]
    current_node_index: int        # 当前节点索引
    status: WorkflowStatus         # pending/running/completed/failed/paused
    metadata: Dict[str, Any]       # 工作流元数据
    context: Context               # 执行上下文
    reflection_round: int          # 反思轮次
```

### 8.2 Context结构

```python
class Context:
    table_schemas: List[TableSchema]      # 表模式
    table_values: List[TableValue]         # 表数据
    sql_contexts: List[SQLContext]         # SQL上下文链
    metrics: List[Metric]                  # 指标
    document_result: DocumentResult        # 文档结果
    parallel_results: Dict                 # 并行执行结果
    preflight_results: List[Dict]          # 预检工具结果 (v2.0新增)
```

### 8.3 工作流管理方法

| 方法 | 功能 |
|------|------|
| `add_node(node, position)` | 添加节点 |
| `remove_node(node_id)` | 移除节点 |
| `move_node(node_id, new_position)` | 移动节点 |
| `get_node(node_id)` | 获取节点 |
| `get_current_node()` | 获取当前节点 |
| `get_next_node()` | 获取下一节点 |
| `advance_to_next_node()` | 推进到下一节点 |
| `get_last_node_by_type(node_type)` | 获取最后特定类型节点 |
| `adjust_nodes(suggestions)` | 根据建议调整节点 |

---

## 9. Text2SQL完整流程

### 9.1 默认工作流定义

```yaml
text2sql:
  # 阶段1: 意图理解
  - intent_analysis:        # 快速启发式任务类型检测
  - intent_clarification:   # LLM意图澄清（修正拼写歧义）

  # 阶段2: 模式发现
  - schema_discovery:       # 发现相关表和模式
  - schema_validation:      # 验证模式是否充足

  # 阶段3: SQL生成
  - generate_sql:           # 生成SQL查询

  # 阶段4: SQL验证与执行
  - sql_validate:           # 验证SQL语法、模式和危险操作
  - execute_sql:            # 执行SQL查询

  # 阶段5: 结果验证
  - result_validation:      # 验证结果质量

  # 阶段6: 反思与输出
  - reflect:                # 评估与自反思（可选，可回退）
  - output:                 # 输出结果
```

### 9.2 数据流图

```
┌─────────────────────────────────────────────────────────────────┐
│ BeginNode                                                       │
│ - 初始化任务参数                                                  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│ IntentAnalysisNode                                              │
│ - 快速意图检测 (关键词 + LLM回退)                                │
│ → workflow.metadata["detected_intent"]                         │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│ IntentClarificationNode                                         │
│ - 业务意图澄清 (纠错、消歧、实体提取)                              │
│ → workflow.metadata["clarified_task"]                          │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│ PreflightOrchestrator                                           │
│ - 执行预检工具 (表搜索、描述、参考SQL、时间解析)                     │
│ → workflow.context.preflight_results                           │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│ SchemaDiscoveryNode                                             │
│ - 使用clarified_task进行模式发现                                 │
│ → workflow.context.table_schemas                               │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│ GenerateSQLNode                                                 │
│ - 基于模式和任务生成SQL                                           │
│ → output.sql_query                                              │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│ ExecuteSQLNode                                                  │
│ - 执行SQL并获取结果                                               │
│ → output.sql_result                                             │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│ OutputNode                                                      │
│ - 格式化输出结果 (SQL Generation Report v2.8)                   │
└─────────────────────────────────────────────────────────────────┘

        ↖─────────────────────────────────────────────────┐
        │ ReflectNode (可选回退)                           │
        │ - 策略: schema_linking/simple_regenerate/reasoning │
        │ - 可回退到: SchemaDiscoveryNode/GenerateSQLNode  │
        └─────────────────────────────────────────────────┘
```

---

## 10. 输入输出模型

### 10.1 节点输入模型示例

| 节点类型 | 输入模型 | 关键字段 |
|----------|----------|----------|
| generate_sql | GenerateSQLInput | task, table_schemas, hint_knowledge |
| execute_sql | ExecuteSQLInput | sql_query, database_name |
| reflect | ReflectionInput | sql_query, sql_result, error_info |
| schema_discovery | SchemaDiscoveryInput | task, top_n, reflection_round |
| chat | ChatNodeInput | messages, execution_mode, plan_mode |

### 10.2 节点输出模型示例

| 节点类型 | 输出模型 | 关键字段 |
|----------|----------|----------|
| generate_sql | GenerateSQLResult | sql_query, sqlplanation |
| execute_sql | ExecuteSQLResult | result, row_count, execution_time |
| reflect | ReflectionResult | status, next_action, suggestions |
| schema_discovery | SchemaDiscoveryResult | table_schemas, matched_tables |
| chat | ChatNodeResult | response, tool_calls, messages |

---

## 附录A: 节点类型快速参考

### 控制流节点 (6个)

| 类型 | 节点类 | 功能 |
|------|--------|------|
| `"start"` | BeginNode | 工作流起点 |
| `"hitl"` | HitlNode | 人类参与 |
| `"reflect"` | ReflectNode | 智能评估和反思 |
| `"parallel"` | ParallelNode | 并行执行 |
| `"selection"` | SelectionNode | 智能选择 |
| `"subworkflow"` | SubworkflowNode | 嵌套工作流 |

### SQL处理节点 (10+个)

| 类型 | 节点类 | 功能 |
|------|--------|------|
| `"intent_analysis"` | IntentAnalysisNode | 意图分析 |
| `"intent_clarification"` | IntentClarificationNode | 意图澄清 |
| `"schema_discovery"` | SchemaDiscoveryNode | 模式发现 |
| `"schema_linking"` | SchemaLinkingNode | 模式链接 |
| `"schema_validation"` | SchemaValidationNode | 模式验证 |
| `"generate_sql"` | GenerateSQLNode | SQL生成 |
| `"sql_validate"` | SQLValidateNode | SQL验证 |
| `"execute_sql"` | ExecuteSQLNode | SQL执行 |
| `"result_validation"` | ResultValidationNode | 结果验证 |
| `"reasoning"` | ReasonSQLNode | SQL推理 |

### 辅助功能节点 (6个)

| 类型 | 节点类 | 功能 |
|------|--------|------|
| `"fix"` | FixNode | SQL修复 |
| `"compare"` | CompareNode | 结果比较 |
| `"knowledge_enhancement"` | KnowledgeEnhancementNode | 知识增强 |
| `"date_parser"` | DateParserNode | 日期解析 |
| `"doc_search"` | DocSearchNode | 文档搜索 |
| `"search_metrics"` | SearchMetricsNode | 指标搜索 |
| `"output"` | OutputNode | 输出处理 |

### 代理节点 (3+个)

| 类型 | 节点类 | 功能 |
|------|--------|------|
| `"chat"` | ChatAgenticNode | 对话式AI |
| `"gensql"` | GenSQLAgenticNode | SQL生成对话 |
| `"semantic"` | SemanticAgenticNode | 语义模型生成 |
| `"sql_summary"` | SqlSummaryAgenticNode | SQL摘要生成 |

---

## 附录B: 关键文件路径

| 组件 | 文件路径 |
|------|----------|
| 节点类型定义 | `datus/configuration/node_type.py` |
| 节点基类/工厂 | `datus/agent/node/node.py` |
| 节点导出 | `datus/agent/node/__init__.py` |
| 工作流定义 | `datus/agent/workflow.py` |
| 所有节点实现 | `datus/agent/node/*.py` |
| 输入输出模型 | `datus/schemas/input.py`, `datus/schemas/output.py` |

---

**文档维护**: 本文档应随 `datus/agent/node/` 和 `datus/configuration/node_type.py` 的修改同步更新。

---

## 版本更新记录

### v2.1 (2026-01-23)
- 合并 Datus内置工具详细清单.md 内容
- 新增节点类型快速参考表（28个节点类型）
- 调整节点分类数量（控制流6个、功能执行18个、Agentic 4个）
- 新增 SQL Summary 节点类型

### v2.0 (2026-01-22)
- 完整重写，基于最新代码架构
- 新增节点基类和工厂模式
- 新增流式执行支持
- 新增 Preflight 编排器
- 新增 Agentic Node 代理节点
- 新增完整 Text2SQL 流程图
