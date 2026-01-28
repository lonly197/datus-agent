# Datus Agent Workflow 模块介绍

> **文档版本**: v2.1
> **更新日期**: 2026-01-28
> **相关模块**: `datus/agent/`, `datus/agent/runner/`
> **代码仓库**: [Datus Agent](https://github.com/Datus-ai/Datus-agent)

---

基于对 Datus 代码仓库的详细分析，本文档提供 Agent Workflow 模块的全面架构和技术说明。

## 核心组件

### 1. Workflow 类

**位置**: `datus/agent/workflow.py`

工作流是 AI 和人类协作的节点序列执行框架。

**核心属性**:
```python
class Workflow:
    name: str                      # 工作流名称
    task: Optional[SqlTask]        # SQL 任务对象
    nodes: Dict[str, Node]         # 节点映射
    node_order: List[str]          # 节点执行顺序
    current_node_index: int        # 当前执行位置
    status: str                    # pending/running/completed/failed/paused
    context: Context               # 执行上下文
    metadata: Dict                 # 元数据（用于 cancellation 等）
```

**核心方法**:
| 方法 | 功能 |
|------|------|
| `add_node(node, position)` | 添加节点到工作流 |
| `remove_node(node_id)` | 移除节点 |
| `move_node(node_id, new_position)` | 移动节点位置 |
| `get_current_node()` | 获取当前执行节点 |
| `advance_to_next_node()` | 推进到下一节点 |
| `get_last_node_by_type(node_type)` | 获取指定类型的最后节点 |
| `adjust_nodes(suggestions)` | 根据建议调整工作流 |
| `save/load(file_path)` | 持久化工作流 |
| `ensure_not_cancelled()` | 检查并触发协作取消 |

### 2. WorkflowRunner 类

**位置**: `datus/agent/runner/`

封装工作流生命周期管理，支持独立执行。采用模块化设计，拆分为以下子模块：

| 文件 | 职责 |
|------|------|
| `workflow_executor.py` | 同步/流式执行核心引擎 |
| `workflow_lifecycle.py` | 初始化、恢复、ActionHistory 管理 |
| `workflow_navigator.py` | 节点跳转逻辑（reflect/output） |
| `workflow_termination.py` | 终止处理、错误报告、输出保证 |

**核心属性**:
```python
class WorkflowExecutor:
    workflow: Optional[Workflow]   # 工作流实例
    workflow_ready: bool           # 就绪状态
    run_id: str                    # 执行 ID (格式: YYYYMMDD_HHMMSS)
    initial_metadata: Dict         # 初始元数据
    _completed_nodes_count: int    # 已完成节点计数（性能优化）
```

**核心方法**:
| 方法 | 功能 |
|------|------|
| `run(sql_task, check_storage)` | 同步执行工作流 |
| `run_stream(...)` | 流式执行工作流（AsyncGenerator） |

### 3. WorkflowTerminationStatus 枚举

**位置**: `datus/agent/workflow_status.py`

工作流终止状态定义：

```python
class WorkflowTerminationStatus:
    CONTINUE = "continue"              # 继续执行
    SKIP_TO_REFLECT = "skip_to_reflect"  # 跳转到反思节点
    PROCEED_TO_OUTPUT = "proceed_to_output"  # 继续到输出节点（生成报告）
    TERMINATE_WITH_ERROR = "terminate_with_error"  # 终止并报错
    TERMINATE_SUCCESS = "terminate_success"  # 成功终止
```

### 4. Agent 类

**位置**: `datus/agent/agent.py`

主入口点，负责初始化、工作流管理和执行循环。

**核心功能**:
```python
class Agent:
    def __init__(self, args, agent_config, db_manager=None)
    def _initialize_model(self) -> LLMBaseModel  # 初始化 LLM
    def _check_storage_modules(self)              # 检查存储模块
    def create_workflow_runner(...) -> WorkflowExecutor  # 创建工作流运行器
```

---

## 工作流类型

**配置文件**: `datus/agent/workflow.yml`

### 1. text2sql (推荐)

完整的 Text2SQL 工作流，包含意图分析、Schema 发现、SQL 生成、验证和反思：

```
intent_analysis → intent_clarification → schema_discovery → schema_validation
    → generate_sql → sql_validate → execute_sql → result_validation
    → reflect → output
```

**适用场景**: 生产环境的复杂查询任务

### 2. reflection

带反思的 SQL 生成工作流：

```
schema_discovery → generate_sql → execute_sql → reflect → output
```

**适用场景**: 需要自我评估和改进的复杂查询

### 3. fixed

固定工作流，无反思环节：

```
schema_discovery → generate_sql → execute_sql → output
```

**适用场景**: 标准 SQL 生成和执行任务

### 4. dynamic

动态工作流，支持反思节点触发工作流重构：

```
schema_discovery → generate_sql → execute_sql → reflect → output
```

**适用场景**: 支持动态调整的查询处理

### 5. metric_to_sql

指标到 SQL 工作流：

```
schema_discovery → search_metrics → date_parser → generate_sql → execute_sql → output
```

**适用场景**: 基于业务指标的查询生成

### 6. chat_agentic

交互式聊天工作流：

```
chat_agentic → execute_sql → output
```

**适用场景**: 交互式对话查询

### 7. chat_agentic_plan

纯规划工作流（无执行）：

```
chat_agentic → output
```

**适用场景**: 纯规划和对话任务

### 8. gensql_agentic

专注 SQL 生成工作流：

```
sql_chatbot → execute_sql → output
```

**适用场景**: 增强上下文的 SQL 生成

---

## 节点类型

**配置文件**: `datus/configuration/node_type.py`

### 控制流节点 (6个)

| 节点类型 | 功能 | 说明 |
|---------|------|------|
| `start` | 开始 | 工作流起点 |
| `hitl` | 人工介入 | Human-in-the-Loop |
| `reflect` | 反思 | 评估和自我反思 |
| `parallel` | 并行 | 并行执行子节点 |
| `selection` | 选择 | 从多个候选中选择最佳结果 |
| `subworkflow` | 子工作流 | 执行嵌套工作流 |

### SQL 工作流节点 (18个)

| 节点类型 | 功能 | 说明 |
|---------|------|------|
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

### Agentic 节点 (4个)

| 节点类型 | 功能 | 说明 |
|---------|------|------|
| `chat` | 对话 AI | 带工具调用的对话交互 |
| `gensql` | 生成 SQL | 带工具调用的 SQL 生成 |
| `semantic` | 语义模型 | 带工具调用的语义模型生成 |
| `sql_summary` | SQL 摘要 | 带工具调用的 SQL 摘要生成 |

---

## Preflight Orchestrator

**位置**: `datus/agent/node/preflight_orchestrator.py`

证据驱动的生成架构，SQL 生成前预执行工具收集证据。

### 必需工具（Text2SQL 默认执行）

| 工具名称 | 功能 | 输入 | 输出 |
|---------|------|------|------|
| `search_table` | 语义搜索表 | query_text, catalog, database, schema | 匹配的表列表 |
| `describe_table` | 获取表的详细列信息 | table_name | DDL、列信息、样例数据 |
| `search_reference_sql` | 搜索参考 SQL 查询 | query | 相似查询及 SQL |
| `parse_temporal_expressions` | 解析时间表达式 | query | 时间范围、日期函数 |

### 增强工具（v2.4 新增）

| 工具名称 | 功能 | 用途 |
|---------|------|------|
| `validate_sql_syntax` | SQL 语法验证 | 检查 SQL 语法正确性 |
| `analyze_query_plan` | 查询执行计划分析 | EXPLAIN 性能分析 |
| `check_table_conflicts` | 表结构冲突检测 | 检测相似表和命名冲突 |
| `validate_partitioning` | 分区策略验证 | 检查分区表配置 |

---

## 工作流执行机制

### 1. 执行流程

```python
# 1. 初始化工作流
workflow_runner = agent.create_workflow_runner()
workflow_runner.initialize_workflow(sql_task)

# 2. 执行节点循环
while not workflow_runner.is_complete():
    current_node = workflow_runner.workflow.get_current_node()
    result = current_node.execute()
    workflow_runner.workflow.advance_to_next_node()

# 3. 获取最终结果
final_result = workflow_runner.workflow.get_final_result()
```

### 2. 节点依赖管理

```python
# 节点可以声明依赖关系
node.dependencies = ["other_node_id"]

# 执行时确保依赖节点先完成
for dep_id in node.dependencies:
    dep_node = workflow.nodes[dep_id]
    assert dep_node.status == "completed"
```

### 3. 工作流调整

```python
# 根据建议调整工作流
suggestions = [
    {"action": "add", "node": {...}, "position": 0},
    {"action": "remove", "node_id": "old_node"},
    {"action": "move", "node_id": "node_id", "position": 2},
    {"action": "modify", "node_id": "node_id", "modifications": {...}},
]
workflow.adjust_nodes(suggestions)
```

### 4. 协作取消

```python
# 检查取消标志
workflow.ensure_not_cancelled()
# 或者使用 CancellationRegistry
from datus.agent.cancellation_registry import is_cancelled_sync
if is_cancelled_sync(task_id):
    raise asyncio.CancelledError()
```

---

## 子代理 (Sub-Agents)

**位置**: `datus/schemas/agent_models.py`

### 系统内置子代理

| 子代理名称 | 功能 |
|-----------|------|
| `gen_semantic_model` | 生成语义模型 |
| `gen_metrics` | 生成指标定义 |
| `gen_sql_summary` | 生成 SQL 摘要 |

### 子代理配置

```python
class SubAgentConfig:
    system_prompt: str           # 系统提示词
    agent_description: str       # 代理描述
    tools: str                   # 工具列表（逗号分隔）
    mcp: str                     # MCP 工具
    scoped_context: ScopedContext  # 作用域上下文
    rules: List[str]             # 规则列表
    prompt_version: str          # 提示词版本
    prompt_language: str         # 提示词语言
    scoped_kb_path: str          # 知识库路径
```

### 作用域上下文

```python
class ScopedContext:
    namespace: str              # 命名空间
    tables: Optional[str]       # 表列表
    metrics: Optional[str]      # 指标列表
    sqls: Optional[str]        # SQL 列表
```

---

## 错误处理

**位置**: `datus/agent/error_handling.py`

### 错误码体系

```python
class ErrorCode:
    # 节点错误
    NODE_NO_SQL_CONTEXT = "NODE_NO_SQL_CONTEXT"
    NODE_EXECUTION_FAILED = "NODE_EXECUTION_FAILED"

    # 存储错误
    STORAGE_SEARCH_FAILED = "STORAGE_SEARCH_FAILED"
    STORAGE_SAVE_FAILED = "STORAGE_SAVE_FAILED"

    # 模型错误
    MODEL_EMBEDDING_ERROR = "MODEL_EMBEDDING_ERROR"

    # 工作流错误
    WORKFLOW_INVALID_NODE = "WORKFLOW_INVALID_NODE"
    WORKFLOW_CANCELLED = "WORKFLOW_CANCELLED"
```

### 反射节点可达性检查

```python
from datus.utils.error_handler import check_reflect_node_reachable

# 检查反射节点是否可达
if not check_reflect_node_reachable(workflow, current_idx):
    # 跳过反思，直接到输出
    workflow.metadata["skip_reflect"] = True
```

---

## 版本更新记录

### v2.1 (2026-01-28)
- WorkflowRunner 重构为 `datus/agent/runner/` 子模块
- 新增 `workflow_executor.py`：同步/流式执行核心引擎
- 新增 `workflow_lifecycle.py`：初始化、恢复、ActionHistory 管理
- 新增 `workflow_navigator.py`：节点跳转逻辑
- 新增 `workflow_termination.py`：终止处理、错误报告
- 原 `workflow_runner.py` 保留为向后兼容入口

### v2.0 (2026-01-23)
- 完整重写，基于最新代码架构
- 新增 text2sql 工作流（10节点完整流程）
- 新增 8 个节点类型（intent_analysis, intent_clarification, schema_discovery, schema_validation, sql_validate, result_validation, knowledge_enhancement）
- 新增 Preflight Orchestrator 证据驱动生成架构
- 新增 WorkflowRunner 终止状态机制
- 新增协作取消机制（CancellationRegistry）
- 新增工作流调整（adjust_nodes）方法
- 新增工作流保存/加载功能
- 新增节点类型注册表模式

### v1.0 (2025-12-30)
- 初始版本
- 基于代码库分析的高层次架构概述

---

## 相关资源

- **项目主页**: [https://datus.ai](https://datus.ai)
- **文档**: [https://docs.datus.ai](https://docs.datus.ai)
- **GitHub**: [https://github.com/Datus-ai/Datus-agent](https://github.com/Datus-ai/Datus-agent)
- **Slack 社区**: [https://join.slack.com/t/datus-ai](https://join.slack.com/t/datus-ai/shared_invite/zt-3g6h4fsdg-iOl5uNoz6A4GOc4xKKWUYg)
