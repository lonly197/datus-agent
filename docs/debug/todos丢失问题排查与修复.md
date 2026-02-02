# Reflect 节点重试时 Todos 丢失问题排查与修复

> **文档版本**: v1.0
> **更新日期**: 2026-02-02
> **问题描述**: Reflect 节点重试时，todos 数组被清空而非追加

---

## 问题现象

在 Text2SQL 工作流中，当 Reflect 节点重试时，前序步骤的 todos 状态丢失，表现为：
1. 发送空的 `PlanUpdateEvent`（`todos: []`）
2. 已完成的步骤状态被重置
3. 新的 steps 被追加到空列表，而非在已有状态上追加

---

## 根因分析

### 根因 1: 空 todos 的 PlanUpdateEvent 被误发送

**位置**: `datus/api/event_converter/core.py` 第 254-282 行

**问题描述**:
当 `plan_update` action 的 output 中没有 `todo_list` 或 `todos` 字段时，代码仍然发送了空的 `PlanUpdateEvent`。

```python
# 原代码逻辑
if action.action_type == "plan_update" and action.output:
    todos = []
    # ... 从 output 中提取 todos
    if todos:
        # 只有 todos 非空时才更新状态
    events.append(PlanUpdateEvent(id=..., todos=todos))  # 空列表也被发送
    return events
```

**影响**: 当 Reflect 节点发送没有 todos 的 `plan_update` action 时，会覆盖已有状态为空。

---

### 根因 2: Reflect 节点的 plan_update action 不包含 todos

**位置**: `datus/agent/node/reflect_node.py` 第 351-372 行

**问题描述**:
Reflect 节点的 `plan_update` action 只包含 `plan_adjustment` 信息，不包含完整的 todos 列表。

```python
plan_update_action = ActionHistory(
    action_id="reflection_plan_update",
    role=ActionRole.WORKFLOW,
    action_type="plan_update",
    # ...
    output={
        "plan_adjustment": {
            "strategy": result.strategy,
            "explanation": result.details.get("explanation", ""),
            "nodes_added": result.details.get("nodes_added", [])
        }
    },
)
```

**影响**: 事件转换器没有足够的上下文来维护 todos 状态。

---

### 根因 3: VirtualStepManager 未确保 output 节点在最后

**位置**: `datus/api/event_converter/virtual_steps.py` 第 95-156 行

**问题描述**:
`generate_virtual_plan_update` 方法按照 `_seen_steps` 顺序生成 todos，但 Reflect 节点重试时可能导致 output 节点被插入到中间位置。

```python
# 原代码
def generate_virtual_plan_update(self, current_node_type: Optional[str] = None):
    # _seen_steps 可能包含 output 在中间位置
    for step_id in self._seen_steps:
        todos.append(TodoItem(id=step_id, ...))
```

**影响**: output 节点可能不在 todo 列表的最后，导致报告展示异常。

---

## 修复方案

### 修复 1: 空 todos 时不发送 PlanUpdateEvent

**文件**: `datus/api/event_converter/core.py`

```python
# 0. Handle explicit plan updates
if action.action_type == "plan_update" and action.output:
    todos = []
    todo_data_source = None
    new_todo_ids = []

    if isinstance(action.output, dict):
        if "todo_list" in action.output and isinstance(action.output["todo_list"], dict):
            todo_data_source = action.output["todo_list"].get("items", [])
        elif "todos" in action.output and isinstance(action.output["todos"], list):
            todo_data_source = action.output["todos"]

        if todo_data_source:
            for todo_data in todo_data_source:
                # ... 构建 todos

    # If no todo_data_source provided, skip sending empty plan_update
    if todo_data_source is None:
        return events  # 不发送空事件

    if todos:
        is_append = bool(new_todo_ids) and not self._todo_state_manager.get_todo_state_list()
        self._update_todo_state(todos, replace_order=is_append)
        todos = self._get_todo_state_list()
        events.append(PlanUpdateEvent(id=..., todos=todos))
    else:
        events.append(PlanUpdateEvent(id=..., todos=[]))  # 空列表显式发送
    return events
```

**关键改动**:
- 跟踪 `todo_data_source` 是否存在
- 如果没有 todos 数据，不发送任何 PlanUpdateEvent
- 只有明确发送空 todos 时才发送 `todos: []`

---

### 修复 2: VirtualStepManager 确保 output 在最后

**文件**: `datus/api/event_converter/virtual_steps.py`

```python
def generate_virtual_plan_update(self, current_node_type: Optional[str] = None) -> Optional[PlanUpdateEvent]:
    current_step_id = self.get_virtual_step_id(current_node_type) if current_node_type else None
    if current_step_id:
        self.active_virtual_step_id = current_step_id
        if current_step_id not in self._seen_steps:
            self._seen_steps.append(current_step_id)

    if not self._seen_steps:
        return None

    # Build the final plan with output step at the end
    non_output_steps = [s for s in self._seen_steps if s != "step_output"]
    all_steps = non_output_steps + ["step_output"]  # output 始终在最后

    # ... 其余逻辑不变
```

**关键改动**:
- 分离非 output 步骤和 output 步骤
- 拼接时确保 output 在最后

---

### 修复 3: 添加工具调用事件生成

**文件**: `datus/api/event_converter/core.py`

```python
# Handle SQL Execution
elif action.action_type == "sql_execution":
    # ...
    if action.status in (ActionStatus.SUCCESS, ActionStatus.FAILED):
        # Generate ToolCallEvent if not already generated
        if action.status != ActionStatus.PROCESSING:
            events.append(ToolCallEvent(...))
        events.append(ToolCallResultEvent(...))

# Handle Preflight Tool Execution
elif action.action_type.startswith("preflight_"):
    # ...
    if action.status in [ActionStatus.SUCCESS, ActionStatus.FAILED]:
        if action.status != ActionStatus.PROCESSING:
            events.append(ToolCallEvent(...))
        events.append(ToolCallResultEvent(...))
```

**关键改动**:
- SUCCESS/FAILED 状态时也生成 ToolCallEvent
- 确保工具调用事件完整性

---

### 修复 4: 添加向后兼容的属性别名

**文件**: `datus/api/event_converter/core.py`

```python
@property
def active_virtual_step_id(self) -> Optional[str]:
    """Backward-compatible alias for _virtual_step_manager.active_virtual_step_id."""
    return self._virtual_step_manager.active_virtual_step_id

@active_virtual_step_id.setter
def active_virtual_step_id(self, value: Optional[str]) -> None:
    self._virtual_step_manager.active_virtual_step_id = value
```

---

### 修复 5: 添加 reflect 和 output 节点处理

**文件**: `datus/api/event_converter/core.py`

```python
# 1. Handle chat/assistant messages
if action.role == ActionRole.ASSISTANT:
    # ...
    # Handle reflect node
    if action.action_type in ("reflect", "reflection_analysis"):
        if action.status == ActionStatus.SUCCESS:
            plan_update = self._generate_virtual_plan_update("reflect")
            if plan_update and plan_update.todos:
                events.append(plan_update)
            content = "🔄 **Executing Step**: 自我纠正与优化"
            events.append(ChatEvent(id=event_id, planId=..., content=content))

    # Handle output node
    elif action.action_type in ("output", "output_generation"):
        if action.status == ActionStatus.SUCCESS:
            plan_update = self._generate_virtual_plan_update("output")
            if plan_update and plan_update.todos:
                events.append(plan_update)
            content = "📄 **Executing Step**: 生成结果报告"
            events.append(ChatEvent(id=event_id, planId=..., content=content))
```

### 修复 6: 输出节点触发策略与元数据链路补全

**文件**:
- `datus/agent/runner/workflow_executor.py`
- `datus/agent/runner/workflow_termination.py`
- `datus/agent/node/output_node.py`

**问题描述**:
SSE 流式模式下，`output` 节点可能在 `max_steps` 分支和 `finally` 分支被重复触发，
导致 `output_generation_start`/`completion` 重复发送，前端出现报告闪现/覆盖。
此外，`output_generation` 的 ActionHistory 中缺少必要的 metadata，
导致 SQL 失败/重试等信息无法正确展示。

**修复内容**:
1. 仅在 output 未成功执行时触发 output（避免重复执行）；
2. output 成功后设置 `_output_executed`，失败时不设置，允许 `finally` 重试；
3. `output_generation` 输出补齐 `metadata/sql_query_final/sql_result_final/row_count`。

**关键改动**:
```python
# workflow_executor.py
if not self.workflow.metadata.get("_output_executed"):
    async for output_action in output_executor.run_stream(...):
        yield output_action

# workflow_termination.py
if self.workflow.metadata.get("_output_executed"):
    return
...
if output_status == "completed":
    self.workflow.metadata["_output_executed"] = True
```

### 修复 7: text2sql workflow metadata 链路补齐

**文件**:
- `datus/agent/node/intent_analysis_node.py`
- `datus/agent/node/schema_discovery/core.py`
- `datus/agent/node/result_validation_node.py`
- `datus/agent/node/execute_sql_node.py`
- `datus/agent/node/output_node.py`

**问题描述**:
部分节点未写入关键 metadata，导致 output 节点报告信息缺失。
例如：
- `intent_analysis` 未写 `intent_analysis`（output 读取该 key）；
- `schema_discovery` 未写 `discovered_tables`，导致 `schema_validation` 中 `candidate_tables_count` 恒为 0；
- `result_validation` 未传到 output；
- `execute_sql` 失败未标注 `failure_stage`。

**修复内容**:
1. `intent_analysis` 补写 `intent_analysis`；
2. `schema_discovery` 补写 `discovered_tables`；
3. `result_validation` 失败补写 `failure_stage=result_validation`；
4. `execute_sql` 失败补写 `failure_stage=execute_sql`（不覆盖已有值）；
5. `output_node` 输出补充 `result_validation`。

---

## 测试验证

### 测试结果

```
tests/unit_tests/test_event_converter.py
  test_schema_discovery_generates_tool_events: PASSED
  test_sql_execution_generates_tool_events: PASSED
  test_workflow_completion_generates_final_plan_update: PASSED
  test_virtual_plan_state_transitions: PASSED
  test_event_flow_validation: PASSED
  test_preflight_tool_binding: PASSED
  test_tool_prefixed_preflight_events_bind_to_virtual_steps: PASSED
  test_output_node_binding_to_step_output: PASSED
  test_reflect_node_binding_to_step_reflect: PASSED

============================== 9 passed in 0.67s ===============================
```

### 验证测试

1. **VirtualStepManager 确保 output 在最后**: 通过
2. **TodoStateManager append 模式**: 通过
3. **plan_update 无 todos 时不发送空事件**: 通过
4. **output 节点生成 ChatEvent**: 通过

---

## 修复总结

| 修复项 | 文件 | 状态 |
|--------|------|------|
| 空 todos 时不发送 PlanUpdateEvent | core.py | ✅ |
| output 步骤始终在最后 | virtual_steps.py | ✅ |
| 工具调用事件完整性 | core.py | ✅ |
| 向后兼容属性别名 | core.py | ✅ |
| reflect/output 节点处理 | core.py | ✅ |

---

## 后续建议

1. **监控**: 在生产环境中监控 `PlanUpdateEvent` 的 todos 长度，确保不会出现异常清空
2. **测试**: 添加集成测试，验证完整 Text2SQL 工作流（包括重试）的 todos 状态
3. **文档**: 完善 `plan_update` action 的规范文档，明确 todos 字段的必要性
