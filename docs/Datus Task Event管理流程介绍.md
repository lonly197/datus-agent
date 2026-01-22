# Datus Task Event 管理流程介绍

> **文档版本**: v2.0
> **更新日期**: 2026-01-22
> **相关模块**: `datus/api/event_converter.py`, `datus/api/models.py`, `datus/agent/cancellation_registry.py`

---

## 目录

1. [系统概述](#1-系统概述)
2. [核心概念](#2-核心概念)
3. [ID 类型与职责分离](#3-id-类型与职责分离)
4. [事件类型与关联策略](#4-事件类型与关联策略)
5. [EventConverter 架构](#5-eventconverter-架构)
6. [任务生命周期管理](#6-任务生命周期管理)
7. [任务取消机制](#7-任务取消机制)
8. [事件流程示例](#8-事件流程示例)
9. [最佳实践](#9-最佳实践)

---

## 1. 系统概述

### 1.1 Event 系统的作用

Datus Task Event 系统是连接后端工作流执行和前端用户界面的核心桥梁：

```
┌─────────────────────────────────────────────────────────────────┐
│                     Datus Event System                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐         ┌──────────────┐         ┌─────────┐ │
│  │   Workflow   │────────▶│ EventConverter│────────▶│  Front  │ │
│  │   Execution  │  Action  │              │  Event  │   End   │ │
│  └──────────────┘ History └──────────────┘ Stream  └─────────┘ │
│                                                                  │
│  ┌──────────────┐         ┌──────────────┐         ┌─────────┐ │
│  │  Task Store  │◀────────│     API      │◀────────│  Tasks  │ │
│  │  (SQLite)    │  CRUD   │   Service    │  Status │  List   │ │
│  └──────────────┘         └──────────────┘         └─────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**核心职责**：
1. **实时状态同步**: 将工作流执行进度实时推送给前端
2. **工具调用可见**: 让用户看到系统调用了哪些工具（schema_discovery, execute_sql 等）
3. **Preflight工具跟踪**: 展示预检工具调用（搜索表、描述表、验证SQL等）
4. **任务进度追踪**: 通过 PlanUpdateEvent 更新 TodoList 状态
5. **错误反馈**: 通过 ErrorEvent/SqlExecutionErrorEvent 及时通知用户执行错误
6. **SQL执行细粒度事件**: Start/Progress/Result/Error 四个阶段

### 1.2 为什么需要 Event 系统

**问题背景**：
- Text2SQL 查询是**长时间运行的任务**（可能需要 10-60 秒）
- 用户需要**实时看到进度**，而不是一直等待最终结果
- 前端需要**关联事件到具体的计划**，以便正确组织显示
- Preflight工具需要**细粒度的事件跟踪**以支持证据驱动生成

---

## 2. 核心概念

### 2.1 ActionHistory

**定义**: 工作流中每个原子操作的记录

```python
class ActionHistory:
    action_id: str           # 唯一标识 (UUID)
    role: ActionRole         # SYSTEM/ASSISTANT/USER/TOOL/WORKFLOW
    messages: str            # 思考/推理内容
    action_type: str         # 操作类型（节点类型/MCP工具名/preflight工具名）
    input: Any               # 输入参数
    output: Any              # 输出结果
    status: ActionStatus     # PROCESSING/SUCCESS/FAILED/SOFT_FAILED
    start_time: datetime     # 开始时间
    end_time: Optional[datetime]  # 结束时间
```

**ActionRole 类型**：

| 角色 | 用途 |
|------|------|
| `SYSTEM` | 系统提示词 |
| `ASSISTANT` | AI助手响应（工具调用、思考过程） |
| `USER` | 用户消息 |
| `TOOL` | MCP工具调用（包括preflight工具） |
| `WORKFLOW` | 工作流控制（初始化、完成、节点执行） |

**ActionStatus 状态**：

| 状态 | 用途 | 后续流转 |
|------|------|----------|
| `PROCESSING` | 处理中 | → SUCCESS/FAILED/SOFT_FAILED |
| `SUCCESS` | 成功完成 | 终止状态 |
| `FAILED` | 失败 | 终止状态，可能触发错误事件 |
| `SOFT_FAILED` | 软失败 | 可通过反思节点恢复执行 |

**生命周期**：
```
创建 → PROCESSING → SUCCESS/FAILED/SOFT_FAILED
```

### 2.2 ActionHistoryManager

**定义**: ActionHistory 的管理器，支持流式执行中的历史记录

```python
class ActionHistoryManager:
    actions: List[ActionHistory]     # 动作历史列表
    current_action_id: Optional[str]  # 当前动作ID

    def add_action(action)           # 添加动作（防重复）
    def update_current_action(**kwargs)  # 更新当前动作
    def get_actions() -> List        # 获取所有动作
    def find_action_by_id(action_id) # 按ID查找
    def update_action_by_id(action_id, **kwargs)  # 按ID更新
    def clear()                      # 清空历史
```

### 2.3 DeepResearchEvent

**定义**: 前端可消费的事件模型（基类）

```python
class BaseEvent(BaseModel):
    event: DeepResearchEventType    # 事件类型标识
    id: str                          # 事件唯一 ID
    planId: Optional[str]            # 关联的计划 ID（关键字段！）
    timestamp: int                   # 时间戳（毫秒）
```

**事件类型层次结构**：

```
DeepResearchEvent (Union Type)
├── ChatEvent                    # 聊天消息
├── PlanUpdateEvent              # 计划更新（TodoList）
├── ToolCallEvent                # 工具调用开始
├── ToolCallResultEvent          # 工具调用结果
├── SqlExecutionStartEvent       # SQL执行开始 (v2.0新增)
├── SqlExecutionProgressEvent    # SQL执行进度 (v2.0新增)
├── SqlExecutionResultEvent      # SQL执行结果 (v2.0新增)
├── SqlExecutionErrorEvent       # SQL执行错误 (v2.0新增)
├── ErrorEvent                   # 错误事件
├── CompleteEvent                # 完成事件
└── ReportEvent                  # 报告事件
```

### 2.4 Task Store

**定义**: SQLite数据库存储任务信息

```python
class TaskStore:
    """SQLite-based task storage"""

    # 任务表结构
    | 字段 | 类型 | 说明 |
    |------|------|------|
    | id | INTEGER | 自增主键 |
    | task_id | TEXT | 任务唯一标识 (UNIQUE) |
    | task_query | TEXT | 用户原始任务描述 |
    | sql_query | TEXT | 生成的SQL查询 |
    | sql_result | TEXT | SQL执行结果 |
    | status | TEXT | 任务状态 (running/completed/failed/cancelled) |
    | user_feedback | TEXT | 用户反馈 (success/failed) |
    | created_at | TEXT | 创建时间 |
    | updated_at | TEXT | 更新时间 |
```

**核心方法**：

| 方法 | 功能 |
|------|------|
| `create_task(task_id, task_query)` | 创建新任务记录 |
| `update_task(task_id, **kwargs)` | 更新任务信息（sql_query, sql_result, status） |
| `get_task(task_id)` | 获取单个任务详情 |
| `get_all_feedback()` | 获取所有任务反馈 |
| `record_feedback(task_id, status)` | 记录用户反馈 |
| `cleanup_old_tasks(hours=24)` | 清理过期任务（默认24小时） |
| `delete_task(task_id)` | 删除任务 |

---

## 3. ID 类型与职责分离

### 3.1 ID 类型概览

| ID 变量 | 类型 | 用途 | 生命周期 | 稳定性 | 前端可见 |
|---------|------|------|----------|--------|----------|
| **virtual_plan_id** | UUID | **整体计划关联标识** | 实例级，不变 | ✅ 稳定 | ✅ 是 |
| **virtual_step_id** | 字符串 | **步骤级事件关联标识** | 会话级，对应特定步骤 | ✅ 稳定 | ✅ 是 |
| **active_virtual_step_id** | 字符串 | **内部状态跟踪** | 会话级，动态变化 | ❌ 变化 | ❌ 否 |
| **failed_virtual_steps** | Set[str] | **失败步骤跟踪** | 会话级，动态变化 | ❌ 变化 | ✅ 是 (ERROR状态) |
| **todo_id** | 字符串 | **特定任务标识** | 单次 action，可能为空 | ⚠️ 不确定 | ✅ 是 |
| **event_id** | 字符串 | **事件唯一标识** | 单次 event | ✅ 唯一 | ✅ 是 |

### 3.2 ID 职责分离原则

**核心原则**: 明确区分"内部状态跟踪"、"整体计划关联"和"步骤级事件关联"

```
┌─────────────────────────────────────────────────────────────────┐
│                    ID 职责分离架构                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  virtual_plan_id (整体计划关联标识)                        │  │
│  │  - 用途: PlanUpdateEvent.id，整体计划的唯一标识             │  │
│  │  - 生命周期: 实例创建时生成，永不变化                        │  │
│  │  - 前端可见: ✅ 是 (用于整体计划关联)                       │  │
│  └──────────────────────────────────────────────────────────┘  │
│                           │                                     │
│                           ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  PlanUpdateEvent                                          │  │
│  │    .id = virtual_plan_id                                   │  │
│  │    .todos[] = TodoItem(id="step_schema", ...)              │  │
│  │    .todos[] = TodoItem(id="step_exec", ...)                │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  virtual_step_id (步骤级事件关联标识)                      │  │
│  │  - 用途: ToolCallEvent.planId，绑定到特定 TodoItem          │  │
│  │  - 来源: _get_virtual_step_id(action.action_type)          │  │
│  │  - 映射: schema_discovery → "step_schema"                  │  │
│  │         preflight_search_table → "step_schema"             │  │
│  │         preflight_validate_sql_syntax → "step_exec"        │  │
│  │  - 前端可见: ✅ 是 (用于步骤级事件关联)                    │  │
│  └──────────────────────────────────────────────────────────┘  │
│                           │                                     │
│                           ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  ToolCallEvent (带 preflight)                              │  │
│  │    .planId = "step_schema"  (virtual_step_id)              │  │
│  │    .toolName = "preflight_search_table"                    │  │
│  │    ↓                                                        │  │
│  │  绑定到 TodoItem(id="step_schema")                         │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  failed_virtual_steps (失败步骤跟踪)                       │  │
│  │  - 用途: 标记哪些步骤执行失败                              │  │
│  │  - 生命周期: 随工作流执行变化                              │  │
│  │  - 前端可见: ✅ 是 (TodoItem.status = ERROR)              │  │
│  └──────────────────────────────────────────────────────────┘  │
│                           │                                     │
│                           ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  PlanUpdateEvent.todos[]                                   │  │
│  │    TodoItem(id="step_exec", status=ERROR)  ← 失败步骤     │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 为什么不能混淆使用

**错误的混用**（修复前）：
```python
# ❌ 错误：将内部状态 ID 暴露给前端
if not todo_id and self.active_virtual_step_id:
    todo_id = self.active_virtual_step_id  # 导致事件 planId 随步骤变化

# 结果：
# - 步骤 1 (intent): ToolCallEvent.planId = "step_intent"
# - 步骤 2 (schema): ToolCallEvent.planId = "step_schema"
# - PlanUpdateEvent.id = "550e8400..." (virtual_plan_id)
# → 前端无法关联事件！
```

**正确的分离**（当前版本 v2.0）：
```python
# ✅ 正确：使用虚拟步骤 ID 或稳定的 virtual_plan_id
plan_id = self._get_unified_plan_id(action, force_associate=True)
# 返回优先级: todo_id > virtual_step_id > virtual_plan_id > None

# Text2SQL workflow 结果：
# - schema_discovery: ToolCallEvent.planId = "step_schema" (virtual_step_id)
# - preflight_search_table: ToolCallEvent.planId = "step_schema" (virtual_step_id)
# - execute_sql: ToolCallEvent.planId = "step_exec" (virtual_step_id)
# - Agentic: ToolCallEvent.planId = "todo_1" (extracted todo_id)
# → 每个事件正确绑定到对应的 TodoItem！
```

---

## 4. 事件类型与关联策略

### 4.1 VIRTUAL_STEPS 完整定义 (v2.0)

```python
VIRTUAL_STEPS = [
    {
        "id": "step_intent",
        "content": "分析查询意图",
        "node_types": [
            "intent_analysis",        # 意图分析节点
            "intent_clarification",   # 意图clarification节点
        ],
    },
    {
        "id": "step_schema",
        "content": "发现数据库模式",
        "node_types": [
            # 核心节点
            "schema_discovery",
            "schema_validation",
            # Preflight 工具 - Schema Discovery
            "preflight_search_table",
            "preflight_describe_table",
            "preflight_get_table_ddl",
            "preflight_search_reference_sql",
            "preflight_parse_temporal_expressions",
            "preflight_check_table_exists",
        ],
    },
    {
        "id": "step_sql",
        "content": "生成SQL查询",
        "node_types": [
            "generate_sql",     # SQL生成节点
            "sql_generation",   # SQL生成（旧名称）
        ],
    },
    {
        "id": "step_exec",
        "content": "执行SQL并验证结果",
        "node_types": [
            # 核心节点
            "execute_sql",
            "sql_execution",
            "sql_validate",
            "result_validation",
            # Preflight 工具 - SQL Execution & Validation
            "preflight_validate_sql_syntax",
            "preflight_analyze_query_plan",
            "preflight_check_table_conflicts",
            "preflight_validate_partitioning",
        ],
    },
    {
        "id": "step_reflect",
        "content": "自我纠正与优化",
        "node_types": [
            "reflect",              # 反思节点
            "reflection_analysis",  # 反射分析（旧名称）
        ],
    },
    {
        "id": "step_output",
        "content": "生成结果报告",
        "node_types": [
            "output",              # 输出节点
            "output_generation",   # 输出生成（旧名称）
        ],
    },
]
```

### 4.2 Preflight 工具映射表

**Preflight 工具 → Virtual Step 映射**：

| Preflight 工具 | 映射到 | 对应阶段 |
|----------------|--------|----------|
| `preflight_search_table` | step_schema | Schema Discovery |
| `preflight_describe_table` | step_schema | Schema Discovery |
| `preflight_get_table_ddl` | step_schema | Schema Discovery |
| `preflight_search_reference_sql` | step_schema | Schema Discovery |
| `preflight_parse_temporal_expressions` | step_schema | Schema Discovery |
| `preflight_check_table_exists` | step_schema | Schema Discovery |
| `preflight_validate_sql_syntax` | step_exec | SQL Execution |
| `preflight_analyze_query_plan` | step_exec | SQL Execution |
| `preflight_check_table_conflicts` | step_exec | SQL Execution |
| `preflight_validate_partitioning` | step_exec | SQL Execution |

### 4.3 事件类型分类

#### A. 计划关联事件 (force_associate=True)

这些事件**必须**关联到 planId，因为它们是执行过程中的关键操作：

| 事件类型 | 用途 | planId 来源 |
|---------|------|-------------|
| `ToolCallEvent` (Text2SQL) | 工具调用开始 | `virtual_step_id` (如 "step_schema") |
| `ToolCallEvent` (Agentic) | 工具调用开始 | `todo_id` (从 action 提取) |
| `ToolCallResultEvent` | 工具调用结果 | 同对应的 ToolCallEvent |
| `SqlExecutionStartEvent` | SQL执行开始 | `virtual_step_id` ("step_exec") |
| `SqlExecutionProgressEvent` | SQL执行进度 | `virtual_step_id` ("step_exec") |
| `SqlExecutionResultEvent` | SQL执行结果 | `virtual_step_id` ("step_exec") |
| `SqlExecutionErrorEvent` | SQL执行错误 | `virtual_step_id` ("step_exec") |
| SQL验证 `ChatEvent` | SQL验证结果 | `virtual_step_id` (如 "step_exec") |

#### B. 任务特定事件 (使用 todo_id)

这些事件关联到**特定的 TodoItem**：

| 事件类型 | 用途 | planId 来源 |
|---------|------|-------------|
| Agentic `ToolCallEvent` | 执行特定 todo | 从 action.input 提取的 `todo_id` |
| `todo_update` 事件 | 更新特定 todo | 从 action.output 提取的 `todo_id` |

#### C. 一般消息事件 (force_associate=False)

这些事件**不需要**关联到 planId：

| 事件类型 | 用途 | planId 来源 |
|---------|------|-------------|
| 一般 `ChatEvent` | 助手思考过程 | `None` (除非有 todo_id) |
| `raw_stream` | 流式 Token | `None` (除非有 todo_id) |
| `CompleteEvent` | 任务完成 | `None` |

#### D. 计划管理事件 (特殊处理)

| 事件类型 | 用途 | ID 策略 |
|---------|------|---------|
| `PlanUpdateEvent` | 计划状态更新 | `.id = virtual_plan_id`, `.planId = None` |
| `PlanUpdateEvent` (失败状态) | 标记失败步骤 | `.id = virtual_plan_id`, `failed_virtual_steps` |
| workflow 触发的 `plan_update` | 工作流计划更新 | 使用 `virtual_plan_id` 确保一致性 |

### 4.4 统一 planId 获取策略

```python
def _get_unified_plan_id(
    self,
    action: ActionHistory,
    force_associate: bool = False
) -> Optional[str]:
    """
    统一的 planId 获取策略，确保事件关联一致性。

    Args:
        action: 要转换的 action
        force_associate:
            - True: 事件需要关联（ToolCall, ToolCallResult, SQL execution）
            - False: 事件不需要关联（一般 ChatEvent）

    Returns:
        planId 优先级: todo_id > virtual_step_id > virtual_plan_id > None
    """
    # 1. 优先从 action 提取 todo_id（Agentic workflows，最精确）
    todo_id = self._extract_todo_id_from_action(action)
    if todo_id:
        return todo_id

    # 2. 映射 action_type 到 virtual_step_id（步骤级关联，支持 preflight）
    #    schema_discovery → "step_schema"
    #    preflight_search_table → "step_schema"
    #    execute_sql → "step_exec"
    #    preflight_validate_sql_syntax → "step_exec"
    virtual_step_id = self._get_virtual_step_id(action.action_type)
    if virtual_step_id:
        return virtual_step_id

    # 3. 对于需要强制关联的事件，使用 virtual_plan_id（整体计划关联）
    if force_associate:
        return self.virtual_plan_id

    # 4. 其他情况返回 None
    return None
```

**优先级说明**：

| 优先级 | 检查项 | 返回值 | 使用场景 |
|-------|-------|--------|----------|
| 1 | `todo_id` extraction | 提取的 todo_id | Agentic workflows（有明确 todo_id） |
| 2 | `virtual_step_id` mapping | "step_schema", "step_exec" 等 | Text2SQL workflow 工具事件 + preflight |
| 3 | `virtual_plan_id` fallback | UUID (如 "550e8400...") | 整体计划关联（当无步骤 ID 时） |
| 4 | `None` | None | 不需要关联的事件 |

---

## 5. EventConverter 架构

### 5.1 类结构

```python
class DeepResearchEventConverter:
    """ActionHistory → DeepResearchEvent 转换器"""

    # ===== 类常量 =====
    VIRTUAL_STEPS: List[Dict]  # 虚拟步骤定义（含 preflight 工具映射）

    # ===== 实例变量 =====
    virtual_plan_id: str                    # 稳定的事件关联 ID（UUID）
    active_virtual_step_id: Optional[str]   # 内部状态跟踪 ID（如 "step_schema"）
    virtual_plan_emitted: bool              # 是否已发送初始 PlanUpdateEvent
    tool_call_map: Dict[str, str]           # action_id → tool_call_id 映射
    completed_virtual_steps: Set[str]       # 已完成的虚拟步骤集合
    failed_virtual_steps: Set[str]          # 失败的虚拟步骤集合 (v2.0新增)
    active_todo_item_id: Optional[str]      # Agentic workflow 当前 todo ID
    todo_item_action_map: Dict[str, str]    # action_id → todo_id 映射

    # ===== 核心方法 =====
    def _get_virtual_step_id(node_type: str) -> Optional[str]
    def _generate_virtual_plan_update(node_type: Optional[str]) -> PlanUpdateEvent
    def _extract_todo_id_from_action(action: ActionHistory) -> Optional[str]
    def _get_unified_plan_id(action: ActionHistory, force_associate: bool) -> Optional[str]
    async def convert_stream_to_events(stream: AsyncGenerator) -> AsyncGenerator[str]
```

### 5.2 核心方法详解

#### `_get_virtual_step_id(node_type: str) -> Optional[str]`

```python
def _get_virtual_step_id(self, node_type: str) -> Optional[str]:
    """Map node type to virtual step ID.

    Supports:
    - Core nodes: schema_discovery, generate_sql, execute_sql, etc.
    - Preflight tools: preflight_search_table, preflight_validate_sql_syntax, etc.
    """
    for step in self.VIRTUAL_STEPS:
        if node_type in step["node_types"]:
            return str(step["id"])
    return None
```

#### `_generate_virtual_plan_update(current_node_type: Optional[str]) -> PlanUpdateEvent`

```python
def _generate_virtual_plan_update(self, current_node_type: Optional[str] = None) -> PlanUpdateEvent:
    """Generate PlanUpdateEvent based on current progress.

    Status priority: ERROR > COMPLETED > IN_PROGRESS > PENDING
    This ensures failed steps are never incorrectly marked as COMPLETED.

    Updates:
    - self.active_virtual_step_id: 当前活跃步骤
    - self.completed_virtual_steps: 已完成步骤集合
    - self.failed_virtual_steps: 失败步骤集合 (v2.0新增)
    """
```

**状态优先级**：
```
ERROR > COMPLETED > IN_PROGRESS > PENDING
```

### 5.3 转换流程

```
┌─────────────────────────────────────────────────────────────────┐
│              convert_action_to_event(action, seq_num)           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. 提取 todo_id                                                │
│     └─ todo_id = _extract_todo_id_from_action(action)          │
│                                                                  │
│  2. 确定 active_virtual_step_id                                 │
│     └─ active_virtual_step_id = _get_virtual_step_id(action_type)│
│                                                                  │
│  3. 根据 action.role 和 action_type 分发到具体处理逻辑          │
│     ├─ WORKFLOW role → PlanUpdateEvent                          │
│     ├─ ASSISTANT role → ChatEvent / ToolCallEvent               │
│     ├─ TOOL role → ToolCallEvent / ToolCallResultEvent          │
│     └─ 特定 action_type → preflight工具、sql_execution等        │
│                                                                  │
│  4. 获取 planId（关键！）                                       │
│     └─ planId = _get_unified_plan_id(action, force_associate)   │
│                                                                  │
│  5. 构建 DeepResearchEvent 并返回                               │
│     └─ return [ChatEvent, ToolCallEvent, SqlExecutionEvent...]  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 5.4 SQL执行事件 (v2.0新增)

```python
# SQL执行开始事件
class SqlExecutionStartEvent(BaseEvent):
    event: DeepResearchEventType = SQL_EXECUTION_START
    sqlQuery: str                  # 执行的SQL语句
    databaseName: Optional[str]    # 目标数据库
    estimatedRows: Optional[int]   # 预估行数

# SQL执行进度事件
class SqlExecutionProgressEvent(BaseEvent):
    event: DeepResearchEventType = SQL_EXECUTION_PROGRESS
    sqlQuery: str                  # 执行的SQL语句
    progress: float                # 进度 (0.0 - 1.0)
    currentStep: str               # 当前执行步骤描述
    elapsedTime: Optional[int]     # 已耗时（毫秒）

# SQL执行结果事件
class SqlExecutionResultEvent(BaseEvent):
    event: DeepResearchEventType = SQL_EXECUTION_RESULT
    sqlQuery: str                  # 执行的SQL语句
    rowCount: int                  # 返回行数
    executionTime: int             # 执行时间（毫秒）
    data: Optional[Any]            # 结果数据（预览）
    hasMoreData: bool              # 是否有更多数据
    dataPreview: Optional[str]     # 文本预览

# SQL执行错误事件
class SqlExecutionErrorEvent(BaseEvent):
    event: DeepResearchEventType = SQL_EXECUTION_ERROR
    sqlQuery: str                  # 失败的SQL语句
    error: str                     # 错误信息
    errorType: str                 # 错误类型（syntax/permission/timeout等）
    suggestions: List[str]         # 修复建议
    canRetry: bool                 # 是否可重试
```

---

## 6. 任务生命周期管理

### 6.1 任务状态流转

```
┌─────────────────────────────────────────────────────────────────┐
│                    Task Status Flow                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐    运行中     ┌──────────┐                       │
│  │ pending  │────────────▶│ running  │                       │
│  └──────────┘              └────┬─────┘                       │
│                                │                              │
│           ┌────────────────────┼────────────────────┐         │
│           ▼                    ▼                    ▼         │
│    ┌──────────┐         ┌──────────┐         ┌──────────┐    │
│    │completed │         │ cancelled│         │  failed  │    │
│    └──────────┘         └──────────┘         └──────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  SOFT_FAILED 状态（可恢复）                               │  │
│  │                                                          │  │
│  │  running ──失败但可恢复──▶ SOFT_FAILED                   │  │
│  │      ▲                            │                      │  │
│  │      │                            ▼                      │  │
│  │      └────── 反思节点恢复 ─────── running               │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 RunningTask 数据结构

```python
@dataclass
class RunningTask:
    """API层运行中任务的数据结构"""
    task_id: str                      # 任务唯一标识
    task: asyncio.Task[Any]           # 实际运行的 asyncio 任务
    created_at: datetime              # 创建时间
    status: str                       # running/cancelled/completed/failed
    meta: Optional[Dict[str, Any]]    # 元数据
        # meta 结构:
        # {
        #     "type": "workflow_stream" | "chat_research",
        #     "request": RunWorkflowRequest,
        #     "client": client_id,
        #     "completion_event": asyncio.Event,
        #     "cancelled": False,  # 取消标志
        # }
```

### 6.3 任务生命周期流程

```
┌─────────────────────────────────────────────────────────────────┐
│                    Task Lifecycle                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. 接收请求                                                    │
│     POST /workflows/run 或 /workflows/chat_research             │
│     ↓                                                          │
│     生成/验证 task_id                                           │
│     ↓                                                          │
│     2. 创建任务记录                                             │
│        TaskStore.create_task(task_id, task_query)              │
│        ↓                                                        │
│        3. 注册运行中任务                                        │
│           RunningTaskRegistry.register_running_task()           │
│           ↓                                                      │
│           4. 创建 asyncio Task                                  │
│              asyncio.create_task(_run_xxx_workflow_task())      │
│              ↓                                                  │
│              5. 执行工作流                                       │
│                 WorkflowRunner.run_stream()                     │
│                 ↓                                               │
│                 yield ActionHistory                             │
│                 ↓                                               │
│              6. 转换事件                                        │
│                 EventConverter.convert_stream_to_events()       │
│                 ↓                                               │
│                 yield DeepResearchEvent → SSE                   │
│                 ↓                                               │
│              7. 完成/取消                                        │
│                 - 正常完成: status = "completed"                │
│                 - 取消: status = "cancelled"                    │
│                 - 错误: status = "failed"                       │
│                 ↓                                               │
│              8. 清理资源                                        │
│                 TaskStore.update_task()                         │
│                 RunningTaskRegistry.unregister_running_task()   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 6.4 Task Store 操作示例

```python
# 创建任务
task_id = "client_abc_20260122094530"
task_store.create_task(task_id, "查询最近30天销售额")

# 更新任务（SQL生成后）
task_store.update_task(
    task_id,
    sql_query="SELECT SUM(amount) FROM sales WHERE date >= ...",
    status="running"
)

# 更新任务（SQL执行后）
task_store.update_task(
    task_id,
    sql_result="1000000.00\n5000",
    status="completed"
)

# 记录用户反馈
task_store.record_feedback(task_id, "success")  # 或 "failed"

# 清理24小时前的任务
cleaned_count = task_store.cleanup_old_tasks(hours=24)
```

---

## 7. 任务取消机制

### 7.1 Cancellation Registry 架构

```python
# datus/agent/cancellation_registry.py

# 模块级注册表（进程内共享）
_cancellation_registry: Dict[str, bool] = {}
_registry_lock = asyncio.Lock()
```

### 7.2 异步 API

```python
async def is_cancelled(task_id: str) -> bool:
    """检查任务是否已取消"""
    async with _registry_lock:
        return _cancellation_registry.get(task_id, False)

async def mark_cancelled(task_id: str) -> None:
    """标记任务为已取消"""
    async with _registry_lock:
        _cancellation_registry[task_id] = True

async def clear_cancelled(task_id: str) -> None:
    """清除取消状态"""
    async with _registry_lock:
        _cancellation_registry.pop(task_id, None)
```

### 7.3 同步 API（用于同步上下文）

```python
def is_cancelled_sync(task_id: str) -> bool:
    """同步检查任务是否已取消"""
    return _cancellation_registry.get(task_id, False)

def mark_cancelled_sync(task_id: str) -> None:
    """同步标记任务为已取消"""
    _cancellation_registry[task_id] = True

def clear_cancelled_sync(task_id: str) -> None:
    """同步清除取消状态"""
    _cancellation_registry.pop(task_id, None)
```

### 7.4 取消流程

```
┌─────────────────────────────────────────────────────────────────┐
│                    Task Cancellation Flow                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. 用户请求取消                                                │
│     DELETE /workflows/tasks/{task_id}                           │
│     ↓                                                          │
│  2. API 层处理                                                  │
│     - 验证任务归属（client_id检查）                              │
│     - 设置 meta["cancelled"] = True                             │
│     - 调用 mark_cancelled_sync(task_id)                         │
│     - 调用 running_task.task.cancel()                           │
│     ↓                                                          │
│  3. Generator 检测取消                                           │
│     - Service 层检查 meta["cancelled"]                          │
│     - Workflow 层检查 is_cancelled_sync()                       │
│     - 抛出 asyncio.CancelledError                               │
│     ↓                                                          │
│  4. 任务清理                                                    │
│     - TaskStore.update_task(status="cancelled")                │
│     - clear_cancelled_sync(task_id)                             │
│     - unregister_running_task(task_id)                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 7.5 取消事件流

```python
# Generator 中检测取消
async for event in service.run_workflow_stream(req, current_client, task_id):
    running_task = await service.get_running_task(task_id)
    if running_task and running_task.meta and running_task.meta.get("cancelled"):
        # 发送取消事件
        yield f"data: {to_str({'cancelled': True, 'message': 'Task was cancelled'})}\n\n"
        raise asyncio.CancelledError("Task cancelled by API")
```

---

## 8. 事件流程示例

### 8.1 Text2SQL Workflow 完整流程（带 Preflight）

```
用户查询: "查询最近30天的销售总额"

┌─────────────────────────────────────────────────────────────────┐
│ 1. workflow_init (ActionRole.WORKFLOW)                          │
├─────────────────────────────────────────────────────────────────┤
│ 输出: PlanUpdateEvent                                           │
│   .id = "550e8400-..." (virtual_plan_id)                        │
│   .planId = None                                                │
│   .todos = [                                                     │
│     {id: "step_intent", status: IN_PROGRESS, ...},              │
│     {id: "step_schema", status: PENDING, ...},                  │
│     {id: "step_sql", status: PENDING, ...},                     │
│     {id: "step_exec", status: PENDING, ...},                    │
│     {id: "step_reflect", status: PENDING, ...},                 │
│     {id: "step_output", status: PENDING, ...},                  │
│   ]                                                              │
│                                                                  │
│ 内部状态: active_virtual_step_id = "step_intent"                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. intent_analysis → intent_clarification (ActionRole.WORKFLOW)│
├─────────────────────────────────────────────────────────────────┤
│ 输出: PlanUpdateEvent                                           │
│   .todos = [                                                     │
│     {id: "step_intent", status: COMPLETED, ...},                │
│     {id: "step_schema", status: IN_PROGRESS, ...},              │
│     ...                                                          │
│   ]                                                              │
│                                                                  │
│ 内部状态: active_virtual_step_id = "step_schema"                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. preflight_search_table (ActionRole.TOOL)                     │
├─────────────────────────────────────────────────────────────────┤
│ 输入: {"query": "最近30天的销售总额"}                            │
│                                                                  │
│ 输出: ToolCallEvent                                              │
│   .planId = "step_schema"  ✅ virtual_step_id                   │
│   .toolName = "preflight_search_table"                          │
│                                                                  │
│ 输出: ToolCallResultEvent                                        │
│   .planId = "step_schema"                                       │
│   .data = {tables: ["sales", "orders"], ...}                     │
│                                                                  │
│ 前端关联: 绑定到 TodoItem(id="step_schema")                      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. preflight_describe_table (ActionRole.TOOL)                   │
├─────────────────────────────────────────────────────────────────┤
│ 输出: ToolCallEvent                                              │
│   .planId = "step_schema"  ✅                                   │
│   .toolName = "preflight_describe_table"                        │
│                                                                  │
│ 输出: ToolCallResultEvent                                        │
│   .planId = "step_schema"                                       │
│   .data = {schema: {...}}                                        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 5. generate_sql (ActionRole.ASSISTANT)                          │
├─────────────────────────────────────────────────────────────────┤
│ 输出: ToolCallEvent                                              │
│   .planId = "step_sql"  ✅                                      │
│   .toolName = "generate_sql"                                    │
│                                                                  │
│ 输出: ToolCallResultEvent                                        │
│   .planId = "step_sql"                                          │
│   .data = {sql_query: "SELECT SUM(amount) FROM ..."}            │
│                                                                  │
│ 前端关联: 绑定到 TodoItem(id="step_sql")                         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 6. preflight_validate_sql_syntax (ActionRole.TOOL)              │
├─────────────────────────────────────────────────────────────────┤
│ 输出: SqlExecutionStartEvent                                    │
│   .planId = "step_exec"  ✅                                     │
│   .sqlQuery = "SELECT SUM(amount) FROM ..."                     │
│   .estimatedRows = None                                         │
│                                                                  │
│ 输出: SqlExecutionProgressEvent                                  │
│   .planId = "step_exec"                                         │
│   .progress = 0.5                                               │
│   .currentStep = "验证SQL语法"                                  │
│                                                                  │
│ 输出: SqlExecutionResultEvent                                    │
│   .planId = "step_exec"                                         │
│   .sqlQuery = "SELECT SUM(amount) FROM ..."                     │
│   .rowCount = 0                                                 │
│   .executionTime = 150                                          │
│   .data = None                                                  │
│                                                                  │
│ 前端关联: 绑定到 TodoItem(id="step_exec")                         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 7. output_generation (ActionRole.WORKFLOW)                      │
├─────────────────────────────────────────────────────────────────┤
│ 输出: ChatEvent (SQL Generation Report v2.8)                    │
│   .planId = None                                                │
│   .content = "## 📋 SQL生成报告..."                              │
│                                                                  │
│ 输出: PlanUpdateEvent                                           │
│   .id = "550e8400-..."                                          │
│   .todos = [                                                     │
│     {id: "step_intent", status: COMPLETED, ...},                │
│     {id: "step_schema", status: COMPLETED, ...},                │
│     {id: "step_sql", status: COMPLETED, ...},                   │
│     {id: "step_exec", status: COMPLETED, ...},                  │
│     {id: "step_reflect", status: COMPLETED, ...},               │
│     {id: "step_output", status: COMPLETED, ...},                │
│   ]                                                              │
└─────────────────────────────────────────────────────────────────┘
```

### 8.2 带错误恢复的流程

```
执行过程中某步骤失败：

┌─────────────────────────────────────────────────────────────────┐
│ preflight_validate_sql_syntax (ActionRole.TOOL, FAILED)         │
├─────────────────────────────────────────────────────────────────┤
│ 输出: ToolCallResultEvent (error=True)                          │
│   .planId = "step_exec"                                         │
│   .error = True                                                 │
│   .data = {"error": "SQL语法错误: 缺少GROUP BY子句"}             │
│                                                                  │
│ 内部状态: failed_virtual_steps.add("step_exec")                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ reflect (ActionRole.WORKFLOW)                                   │
├─────────────────────────────────────────────────────────────────┤
│ 输出: PlanUpdateEvent                                           │
│   .todos = [                                                     │
│     {id: "step_intent", status: COMPLETED, ...},                │
│     {id: "step_schema", status: COMPLETED, ...},                │
│     {id: "step_sql", status: COMPLETED, ...},                   │
│     {id: "step_exec", status: ERROR, ...},  ← 标记为ERROR       │
│     {id: "step_reflect", status: IN_PROGRESS, ...},             │
│     {id: "step_output", status: PENDING, ...},                  │
│   ]                                                              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 反思后重试 → 生成正确SQL → 执行成功                              │
├─────────────────────────────────────────────────────────────────┤
│ 输出: PlanUpdateEvent (最终结果)                                │
│   .todos = [                                                     │
│     ...                                                          │
│     {id: "step_exec", status: COMPLETED, ...},  ← 恢复为完成    │
│     ...                                                          │
│   ]                                                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 9. 最佳实践

### 9.1 添加新事件类型时的 checklist

当在 `convert_action_to_event()` 中添加新的事件处理时：

```python
elif action.action_type == "new_action_type":
    # ✅ 1. 确定事件类型
    #    - 是计划关联事件？ → force_associate=True
    #    - 是一般消息？ → force_associate=False
    #    - 是特定 todo？ → 优先使用提取的 todo_id

    # ✅ 2. 确定是否需要映射到虚拟步骤
    #    - 如果是新阶段 → 更新 VIRTUAL_STEPS
    #    - 如果是现有阶段 → 检查 node_types 映射

    # ✅ 3. 获取 planId
    new_plan_id = self._get_unified_plan_id(
        action,
        force_associate=True  # 或 False
    )

    # ✅ 4. 创建事件
    events.append(
        NewEvent(
            id=event_id,
            planId=new_plan_id,  # ✅ 始终使用 _get_unified_plan_id()
            timestamp=timestamp,
            ...
        )
    )
```

### 9.2 添加 Preflight 工具时的 checklist

```python
# 1. 在 VIRTUAL_STEPS 中添加工具映射
VIRTUAL_STEPS = [
    ...
    {
        "id": "step_exec",  # 或对应的步骤
        "node_types": [
            ...
            "preflight_my_new_tool",  # 新增
        ],
    },
]

# 2. 验证映射
assert converter._get_virtual_step_id("preflight_my_new_tool") == "step_exec"
```

### 9.3 禁止模式

```python
# ❌ 禁止 1: 直接使用 active_virtual_step_id 作为 planId
planId=self.active_virtual_step_id

# ❌ 禁止 2: 手动 fallback 到 active_virtual_step_id
planId = todo_id if todo_id else self.active_virtual_step_id

# ❌ 禁止 3: 使用 self.plan_id（每次 converter 创建时生成）
planId=self.plan_id

# ❌ 禁止 4: 硬编码步骤 ID
planId="step_schema"

# ❌ 禁止 5: 在 VIRTUAL_STEPS 外部使用 hardcoded 映射
if action_type == "custom_tool":
    planId = "step_custom"  # 应该添加到 VIRTUAL_STEPS
```

### 9.4 推荐模式

```python
# ✅ 推荐 1: 需要事件关联（ToolCall, ToolCallResult, SQL execution）
planId=self._get_unified_plan_id(action, force_associate=True)

# ✅ 推荐 2: 不需要事件关联（一般 ChatEvent, CompleteEvent）
planId=self._get_unified_plan_id(action, force_associate=False)

# ✅ 推荐 3: 明确使用 todo_id（如果已提取）
if todo_id:
    planId=todo_id
else:
    planId=self._get_unified_plan_id(action, force_associate=True)

# ✅ 推荐 4: PlanUpdateEvent.id 使用 virtual_plan_id
PlanUpdateEvent(id=self.virtual_plan_id, planId=None, ...)

# ✅ 推荐 5: 添加 preflight 工具到 VIRTUAL_STEPS
VIRTUAL_STEPS = [
    {
        "id": "step_schema",
        "node_types": [..., "preflight_my_tool"],
    },
]
```

### 9.5 调试技巧

#### 查看事件 planId 分发
```bash
# 查看所有 planId 赋值
grep -n "planId=" datus/api/event_converter.py

# 验证 active_virtual_step_id 不用于 planId
grep -n "active_virtual_step_id" datus/api/event_converter.py | grep -v "self.active_virtual_step_id ="
```

#### 验证 Preflight 工具映射
```python
# 测试映射
converter = DeepResearchEventConverter()
assert converter._get_virtual_step_id("preflight_search_table") == "step_schema"
assert converter._get_virtual_step_id("preflight_validate_sql_syntax") == "step_exec"
```

#### 添加调试日志
```python
def _get_unified_plan_id(self, action: ActionHistory, force_associate: bool) -> Optional[str]:
    todo_id = self._extract_todo_id_from_action(action)
    if todo_id:
        self.logger.debug(f"[planId] Using todo_id={todo_id} for {action.action_type}")
        return todo_id

    virtual_step_id = self._get_virtual_step_id(action.action_type)
    if virtual_step_id:
        self.logger.debug(f"[planId] Using virtual_step_id={virtual_step_id} for {action.action_type}")
        return virtual_step_id

    if force_associate:
        self.logger.debug(f"[planId] Using virtual_plan_id={self.virtual_plan_id} for {action.action_type}")
        return self.virtual_plan_id

    self.logger.debug(f"[planId] No planId for {action.action_type}")
    return None
```

---

## 附录

### A. 相关文件

| 文件 | 说明 |
|------|------|
| `datus/api/event_converter.py` | Event 转换核心逻辑（2200+行） |
| `datus/api/models.py` | Event 模型定义（含 SQL 执行事件） |
| `datus/schemas/action_history.py` | ActionHistory 模型 |
| `datus/agent/cancellation_registry.py` | 任务取消注册表 |
| `datus/agent/workflow_runner.py` | 工作流执行器 |
| `datus/storage/task/store.py` | Task Store 实现 |

### B. 关键常量

```python
# Virtual Steps (Text2SQL, v2.0 含 Preflight)
VIRTUAL_STEPS = [
    {"id": "step_intent", "content": "分析查询意图", "node_types": [...]},
    {"id": "step_schema", "content": "发现数据库模式", "node_types": [...]},  # 含 preflight
    {"id": "step_sql", "content": "生成SQL查询", "node_types": [...]},
    {"id": "step_exec", "content": "执行SQL并验证结果", "node_types": [...]},  # 含 preflight
    {"id": "step_reflect", "content": "自我纠正与优化", "node_types": [...]},
    {"id": "step_output", "content": "生成结果报告", "node_types": [...]},
]

# Action Roles
class ActionRole(str, Enum):
    SYSTEM = "system"
    ASSISTANT = "assistant"
    USER = "user"
    TOOL = "tool"
    WORKFLOW = "workflow"

# Action Status
class ActionStatus(str, Enum):
    PROCESSING = "processing"
    SUCCESS = "success"
    FAILED = "failed"
    SOFT_FAILED = "soft_failed"  # 可通过反思恢复

# Deep Research Event Types
class DeepResearchEventType(str, Enum):
    CHAT = "chat"
    PLAN_UPDATE = "plan_update"
    TOOL_CALL = "tool_call"
    TOOL_CALL_RESULT = "tool_call_result"
    SQL_EXECUTION_START = "sql_execution_start"       # v2.0新增
    SQL_EXECUTION_PROGRESS = "sql_execution_progress" # v2.0新增
    SQL_EXECUTION_RESULT = "sql_execution_result"     # v2.0新增
    SQL_EXECUTION_ERROR = "sql_execution_error"       # v2.0新增
    COMPLETE = "complete"
    REPORT = "report"
    ERROR = "error"
```

### C. 修改历史

| 日期 | 版本 | 修改内容 |
|------|------|---------|
| 2026-01-17 | v1.0 | 初始版本，记录 plan_id 一致性修复 |
| 2026-01-17 | v1.1 | 新增 `_get_unified_plan_id()` 方法 |
| 2026-01-17 | v1.2 | 新增虚拟步骤 ID 映射 |
| 2026-01-22 | v2.0 | 新增 Preflight 工具映射、SQL执行事件、失败步骤跟踪、任务取消机制 |

---

**文档维护**: 本文档应随 `event_converter.py`、`models.py`、`cancellation_registry.py` 的修改同步更新。
