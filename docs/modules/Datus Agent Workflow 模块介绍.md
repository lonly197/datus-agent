# Datus Agent Workflow 模块介绍

> **文档版本**: v2.1
> **更新日期**: 2026-01-28
> **相关模块**: `datus/agent/`, `datus/agent/runner/`
> **代码仓库**: [Datus Agent](https://github.com/Datus-ai/Datus-agent)

---

基于对 Datus 代码仓库的详细分析，本文档提供 Agent Workflow 模块的全面架构和技术说明。

## 模块架构总览

```
datus/agent/
├── workflow.py              # 工作流核心模型
├── workflow_status.py       # 终止状态枚举
├── workflow_runner.py       # 向后兼容入口（已重构）
├── plan.py                  # 工作流计划生成
├── evaluate.py              # 节点结果评估
├── agent.py                 # 主入口点
├── error_handling.py        # 错误处理
├── cancellation_registry.py # 协作取消注册表
└── runner/                  # 执行器子模块（v2.1 新增）
    ├── __init__.py
    ├── workflow_executor.py      # 同步/流式执行引擎
    ├── workflow_lifecycle.py     # 生命周期管理
    ├── workflow_navigator.py     # 节点跳转导航
    └── workflow_termination.py   # 终止与错误处理
```

## 核心组件

### 1. Workflow 类

**位置**: `datus/agent/workflow.py`

工作流核心模型，管理节点序列、执行状态和上下文。

**核心属性**:
```python
class Workflow:
    name: str                      # 工作流名称
    task: Optional[SqlTask]        # SQL 任务对象
    nodes: Dict[str, Node]         # 节点映射 {node_id: Node}
    node_order: List[str]          # 节点执行顺序 [id1, id2, ...]
    current_node_index: int        # 当前执行位置索引
    status: str                    # pending/running/completed/failed/paused
    context: Context               # 执行上下文（存储 table_schemas, sql_contexts 等）
    metadata: Dict                 # 元数据（cancellation, task_id, termination_status 等）
    reflection_round: int          # 反思轮次计数
```

**核心方法**:
| 方法 | 功能 | 返回 |
|------|------|------|
| `add_node(node, position)` | 添加节点 | node_id |
| `remove_node(node_id)` | 移除节点 | bool |
| `move_node(node_id, new_position)` | 移动节点位置 | bool |
| `get_current_node()` | 获取当前节点 | Optional[Node] |
| `advance_to_next_node()` | 推进到下一节点 | Optional[Node] |
| `get_last_node_by_type(type)` | 获取指定类型最后节点 | Optional[Node] |
| `is_complete()` | 检查是否完成 | bool |
| `save(file_path)` | 持久化到 YAML | None |
| `load(file_path)` | 从 YAML 加载 | Workflow |
| `ensure_not_cancelled()` | 检查协作取消 | None (raise CancelledError) |
| `reset()` | 重置工作流状态 | None |

### 2. WorkflowExecutor 类

**位置**: `datus/agent/runner/workflow_executor.py`

工作流执行引擎，支持同步和流式两种执行模式。

**核心属性**:
```python
class WorkflowExecutor:
    args: argparse.Namespace       # 命令行参数
    global_config: AgentConfig     # 全局配置
    workflow: Optional[Workflow]   # 工作流实例
    run_id: str                    # 执行 ID (格式: YYYYMMDD_HHMMSS)
    initial_metadata: Dict         # 初始元数据
    _completed_nodes_count: int    # 已完成节点计数（O(1) 性能优化）
    workflow_task: Optional[asyncio.Task]  # 异步任务引用
```

**核心方法**:
| 方法 | 功能 | 模式 |
|------|------|------|
| `run(sql_task, check_storage)` | 同步执行工作流 | 同步 |
| `run_stream(...)` | 流式执行（AsyncGenerator） | 异步 |

### 3. WorkflowLifecycle 类

**位置**: `datus/agent/runner/workflow_lifecycle.py`

工作流生命周期管理：初始化、恢复、先决条件检查。

**核心方法**:
| 方法 | 功能 |
|------|------|
| `initialize_workflow(sql_task, metadata)` | 生成新工作流计划 |
| `resume_workflow(config, global_config)` | 从检查点恢复工作流 |
| `init_or_load_workflow(sql_task, workflow, metadata)` | 初始化或加载工作流 |
| `prepare_first_node(workflow)` | 准备第一个执行节点 |
| `check_prerequisites(sql_task, workflow, check_storage)` | 检查先决条件 |
| `create_action_history(...)` | 创建 ActionHistory 条目 |
| `update_action_status(...)` | 更新 ActionHistory 状态 |
| `generate_failure_suggestions(workflow)` | 生成失败建议 |

### 4. WorkflowNavigator 类

**位置**: `datus/agent/runner/workflow_navigator.py`

工作流节点导航：支持跳转到 reflect/output 节点。

**核心方法**:
| 方法 | 功能 |
|------|------|
| `jump_to_reflect_node()` | 跳转到下一个 reflect 节点 |
| `jump_to_output_node()` | 跳转到 output 节点 |
| `find_output_node()` | 查找 output 节点 |
| `find_output_node_index(output_node_id)` | 查找 output 节点索引 |

### 5. WorkflowTerminationManager 类

**位置**: `datus/agent/runner/workflow_termination.py`

工作流终止管理：节点失败处理、错误报告、任务取消。

**核心方法**:
| 方法 | 功能 |
|------|------|
| `handle_node_failure(node, action)` | 处理节点失败（软/硬失败） |
| `terminate_workflow(status, message)` | 终止工作流 |

### 6. OutputNodeExecutor 类

**位置**: `datus/agent/runner/workflow_termination.py`

确保输出节点执行：即使工作流提前退出，也保证生成报告。

**核心方法**:
| 方法 | 功能 |
|------|------|
| `ensure_output_node_execution(metadata)` | 确保 output 节点执行 |
| `check_parallel_node_success(node)` | 检查并行节点子项成功 |

### 7. WorkflowTerminationStatus 枚举

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

---

## 模块交互关系

### 执行流程图

```
┌─────────────────────────────────────────────────────────────────┐
│                         Agent.create_workflow_runner()          │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   WorkflowExecutor.__init__()                   │
│  - 初始化 args, global_config, run_id, initial_metadata         │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│               WorkflowExecutor.run() / run_stream()              │
└─────────────────────────────┬───────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
    ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
    │ Lifecycle   │  │ Navigator   │  │ Termination │
    │ Management  │  │ Navigation  │  │ Manager     │
    └──────┬──────┘  └──────┬──────┘  └──────┬──────┘
           │                │                │
           └────────────────┼────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Workflow                                   │
│  - nodes, node_order, current_node_index                        │
│  - context (table_schemas, sql_contexts)                        │
│  - metadata (termination_status, cancellation)                  │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Node.run() / Node.run_stream()              │
│  - 各类型节点执行具体逻辑                                         │
│  - 返回结果到 context                                            │
└─────────────────────────────────────────────────────────────────┘
```

### 子模块交互

```python
# workflow_executor.py 中的协作模式

class WorkflowExecutor:
    def run_stream(self, sql_task, ...):
        # 1. 初始化子模块
        lifecycle = WorkflowLifecycle(self.args, self.global_config)
        navigator = WorkflowNavigator(self.workflow)
        termination_mgr = WorkflowTerminationManager(self.workflow)
        output_executor = OutputNodeExecutor(self.workflow)

        # 2. 生命周期管理：初始化/恢复工作流
        workflow = lifecycle.init_or_load_workflow(sql_task, ...)

        # 3. 设置子模块工作流引用
        navigator.set_workflow(workflow)
        termination_mgr.set_workflow(workflow)
        output_executor.set_workflow(workflow)

        # 4. 执行循环
        while not workflow.is_complete():
            # 节点导航
            if jump_to_reflect:
                navigator.jump_to_reflect_node()

            # 节点失败处理
            status = termination_mgr.handle_node_failure(current_node)

            # 确保输出节点执行
            output_executor.ensure_output_node_execution(metadata)
```

### 与外部模块的交互

| 外部模块 | 交互方式 |
|----------|----------|
| `plan.py` | `generate_workflow()` 生成工作流结构 |
| `evaluate.py` | `evaluate_result()` 评估节点结果 |
| `node/__init__.py` | `Node.run()` / `Node.run_stream()` 执行节点 |
| `storage/` | `global_config.check_init_storage_config()` 检查存储 |
| `cancellation_registry.py` | `is_cancelled_sync()` 检查取消状态 |
| `schemas/` | `SqlTask`, `ActionHistory`, `Context` 数据结构 |

---

## 工作流类型

**配置文件**: `datus/agent/workflow.yml`

| 类型 | 节点序列 | 适用场景 |
|------|----------|----------|
| `text2sql` | intent_analysis → intent_clarification → schema_discovery → schema_validation → generate_sql → sql_validate → execute_sql → result_validation → reflect → output | 生产环境的复杂查询任务 |
| `reflection` | schema_discovery → generate_sql → execute_sql → reflect → output | 需要自我评估和改进的复杂查询 |
| `fixed` | schema_discovery → generate_sql → execute_sql → output | 标准 SQL 生成和执行任务 |
| `dynamic` | schema_discovery → generate_sql → execute_sql → reflect → output | 支持动态调整的查询处理 |
| `metric_to_sql` | schema_discovery → search_metrics → date_parser → generate_sql → execute_sql → output | 基于业务指标的查询生成 |
| `chat_agentic` | chat_agentic → execute_sql → output | 交互式对话查询 |
| `chat_agentic_plan` | chat_agentic → output | 纯规划和对话任务 |
| `gensql_agentic` | sql_chatbot → execute_sql → output | 增强上下文的 SQL 生成 |

---

## 节点类型

**配置文件**: `datus/configuration/node_type.py`

### 控制流节点

| 类型 | 功能 |
|------|------|
| `start` | 工作流起点 |
| `hitl` | Human-in-the-Loop 人工介入 |
| `reflect` | 评估和自我反思 |
| `parallel` | 并行执行子节点 |
| `selection` | 从多个候选中选择最佳结果 |
| `subworkflow` | 执行嵌套工作流 |

### SQL 工作流节点

| 类型 | 功能 |
|------|------|
| `intent_analysis` | 使用启发式分析查询意图 |
| `intent_clarification` | 修复错别字、澄清歧义、提取实体 |
| `schema_discovery` | 发现相关 Schema 和表 |
| `schema_validation` | 验证 Schema 充分性 |
| `schema_linking` | 理解查询并找到相关 Schema |
| `generate_sql` | 生成 SQL 查询 |
| `sql_validate` | 验证 SQL 语法和语义 |
| `execute_sql` | 执行 SQL 查询 |
| `result_validation` | 验证 SQL 执行结果质量 |
| `output` | 向用户返回结果 |
| `reasoning` | 推理分析 |
| `fix` | 修复 SQL 查询 |
| `search_metrics` | 搜索业务指标 |
| `compare` | 与预期比较 SQL |
| `date_parser` | 解析查询中的时间表达式 |
| `knowledge_enhancement` | 统一和丰富知识 |
| `doc_search` | 搜索相关文档 |

### Agentic 节点

| 类型 | 功能 |
|------|------|
| `chat` | 带工具调用的对话交互 |
| `gensql` | 带工具调用的 SQL 生成 |
| `semantic` | 带工具调用的语义模型生成 |
| `sql_summary` | 带工具调用的 SQL 摘要生成 |

---

## 执行机制

### 同步执行流程

```python
def run(self, sql_task=None, check_storage=False) -> Dict:
    # 1. 检查先决条件
    if not lifecycle.check_prerequisites(sql_task, self.workflow, check_storage):
        return {}

    # 2. 准备第一个节点
    lifecycle.prepare_first_node(self.workflow)

    # 3. 执行循环
    while not self.workflow.is_complete() and step_count < max_steps:
        current_node = self.workflow.get_current_node()
        current_node.run()

        # 4. 处理失败
        if current_node.status == "failed":
            status = termination_mgr.handle_node_failure(current_node)
            if status == TERMINATE_WITH_ERROR:
                break

        # 5. 评估结果
        evaluation = evaluate_result(current_node, self.workflow)
        if evaluation["success"]:
            self.workflow.advance_to_next_node()
        else:
            # 软失败：跳转到 reflect
            navigator.jump_to_reflect_node()

    # 6. 确保输出节点执行
    output_executor.ensure_output_node_execution({})

    # 7. 保存工作流
    return self._finalize_workflow(step_count)
```

### 流式执行流程

```python
async def run_stream(self, sql_task, action_history_manager) -> AsyncGenerator[ActionHistory]:
    # Yield: 初始化完成
    yield init_action

    while self.workflow and not self.workflow.is_complete():
        # 1. 检查取消
        self.workflow.ensure_not_cancelled()

        # 2. 创建节点执行 Action
        node_action = create_action_history(f"node_{current_node.id}", ...)
        yield node_action

        # 3. 流式执行节点
        async for action in current_node.run_stream(action_history_manager):
            yield action

        # 4. 处理失败
        if current_node.status == "failed":
            status = termination_mgr.handle_node_failure(current_node, node_action)

        # 5. 评估并推进
        evaluation = evaluate_result(current_node, self.workflow)
        if evaluation["success"]:
            self.workflow.advance_to_next_node()

    # 6. 确保输出节点执行
    output_executor.ensure_output_node_execution({})

    # 7. Yield: 完成
    yield completion_action
```

### 协作取消机制

```python
# 检查取消标志
workflow.ensure_not_cancelled()

# 或使用 CancellationRegistry
from datus.agent.cancellation_registry import is_cancelled_sync
if is_cancelled_sync(task_id):
    raise asyncio.CancelledError()
```

---

## 错误处理

### 错误码体系

```python
class ErrorCode:
    NODE_NO_SQL_CONTEXT = "NODE_NO_SQL_CONTEXT"
    NODE_EXECUTION_FAILED = "NODE_EXECUTION_FAILED"
    STORAGE_SEARCH_FAILED = "STORAGE_SEARCH_FAILED"
    STORAGE_SAVE_FAILED = "STORAGE_SAVE_FAILED"
    MODEL_EMBEDDING_ERROR = "MODEL_EMBEDDING_ERROR"
    WORKFLOW_INVALID_NODE = "WORKFLOW_INVALID_NODE"
    WORKFLOW_CANCELLED = "WORKFLOW_CANCELLED"
```

### 软失败与硬失败

- **软失败 (SOFT_FAILED)**: 存在可达的 reflect 节点，跳转到 reflect 进行恢复
- **硬失败 (FAILED)**: 无可达 reflect 节点，立即终止工作流

```python
# 节点失败处理流程
status = termination_mgr.handle_node_failure(current_node)

if status == SKIP_TO_REFLECT:
    navigator.jump_to_reflect_node()  # 软失败：跳转
elif status == TERMINATE_WITH_ERROR:
    break  # 硬失败：终止
```

---

## Preflight Orchestrator

**位置**: `datus/agent/node/preflight_orchestrator.py`

证据驱动的生成架构，SQL 生成前预执行工具收集证据。

### 必需工具

| 工具 | 功能 |
|------|------|
| `search_table` | 语义搜索表 |
| `describe_table` | 获取表的详细列信息 |
| `search_reference_sql` | 搜索参考 SQL 查询 |
| `parse_temporal_expressions` | 解析时间表达式 |

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
- 新增 Preflight Orchestrator 证据驱动生成架构
- 新增 WorkflowRunner 终止状态机制
- 新增协作取消机制（CancellationRegistry）

---

## 相关文档

本文档与其他文档存在以下引用和依赖关系：

### 依赖文档（本文档引用的文档）

| 文档 | 路径 | 引用内容 |
|------|------|----------|
| 节点模块介绍 | [docs/modules/Datus Agent Node模块介绍.md](./Datus%20Agent%20Node模块介绍.md) | 节点类型定义、节点基类与工厂 |
| 节点使用指南 | [docs/workflow/nodes.md](./workflow/nodes.md) | 各类节点的详细功能说明 |
| 编排指南 | [docs/workflow/orchestration.md](./workflow/orchestration.md) | workflow.yml 配置说明 |
| 任务处理流程 | [docs/Datus Text2SQL 任务处理流程介绍.md](./Datus%20Text2SQL%20任务处理流程介绍.md) | Text2SQL 完整流程 |
| SQL Review 流程 | [docs/Datus SQL Review 任务处理流程介绍.md](./Datus%20SQL%20Review%20任务处理流程介绍.md) | SQL Review 工作流 |
| 节点配置 | [docs/configuration/nodes.md](./configuration/nodes.md) | 节点类型注册表配置 |
| 工作流配置 | [docs/configuration/workflow.md](./configuration/workflow.md) | 工作流类型配置 |

### 被依赖文档（引用本文档的文档）

| 文档 | 路径 | 引用内容 |
|------|------|----------|
| CLI 执行命令 | [docs/cli/execution_command.md](./cli/execution_command.md) | `datus run` 命令与工作流执行 |
| CLI 聊天命令 | [docs/cli/chat_command.md](./cli/chat_command.md) | `datus chat` 命令与聊天工作流 |
| 工作流介绍 | [docs/workflow/introduction.md](./workflow/introduction.md) | 工作流基本概念与类型 |
| 快速开始 | [docs/getting_started/Quickstart.md](./getting_started/Quickstart.md) | 工作流执行示例 |
| 发布说明 | [docs/release_notes.md](./release_notes.md) | 工作流相关功能更新 |

### 架构相关文档

| 文档 | 路径 | 关系 |
|------|------|------|
| 架构设计 | [docs/develop/Architecture.md](./develop/Architecture.md) | 上层架构设计，与工作流模块互补 |
| 目录结构迁移 | [docs/develop/directory_structure_migration.md](./develop/directory_structure_migration.md) | `datus/agent/runner/` 目录说明 |

### 文档层级关系

```
docs/
├── modules/
│   ├── Datus Agent Workflow 模块介绍.md  ← 本文
│   ├── Datus Agent Node模块介绍.md       ← 节点定义
│   ├── Datus Storage模块介绍.md          ← 上下文存储
│   └── Datus Schemas模块介绍.md          ← 数据模型
│
├── workflow/
│   ├── introduction.md                   ← 工作流入门
│   ├── nodes.md                          ← 节点详情
│   ├── orchestration.md                  ← 编排配置
│   └── api.md                            ← API 集成
│
└── configuration/
    ├── nodes.md                          ← 节点配置
    └── workflow.md                       ← 工作流配置
```
