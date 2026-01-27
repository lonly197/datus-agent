# SQL Review 调试测试指南

## 问题背景

基于 Gemini 对 SQL Review 任务执行日志的分析，声称预检流程失效。但经过代码审查，发现预检代码实际上存在且正确。真正的根因需要通过调试日志确认。

## 已完成的修改

### 1. 添加调试日志

**文件**: `datus/agent/node/chat_agentic_node.py`

**修改位置**:
- Line 1372-1379: 方法入口日志，记录 workflow 状态
- Line 1440-1450: 预检检查日志，记录条件判断结果

**新增日志**:
```
ChatAgenticNode.execute_stream() called for node: {node_id}
self.workflow: {workflow}
self.workflow.metadata: {metadata}
required_tool_sequence: {sequence}
Preflight check: has_workflow={bool}, has_metadata={bool}, has_required_tools={bool}
```

### 2. 创建测试脚本

**文件**: `scripts/test_sql_review.sh`

自动化测试脚本，用于发送 SQL Review 请求并保存日志。

## 如何运行测试

### 步骤 1: 启动 Datus 服务

```bash
cd /Users/lpp/workspace/lonly/datus-agent
source .venv/bin/activate

# 启动服务
python -m datus.api.server \
  --config ~/.datus/conf/agent.yml \
  --namespace test \
  --workflow chat_agentic \
  --port 8080 \
  --host 0.0.0.0
```

### 步骤 2: 运行测试脚本

在**另一个终端**中运行:

```bash
cd /Users/lpp/workspace/lonly/datus-agent
source .venv/bin/activate

# 运行测试
./scripts/test_sql_review.sh
```

### 步骤 3: 查看日志输出

测试脚本会生成两个文件:
- `test_output/sql_review_YYYYMMDD_HHMMSS.log` - 完整日志
- `test_output/sql_review_YYYYMMDD_HHMMSS_messages.txt` - SSE 消息流

### 步骤 4: 分析关键日志

在日志中查找以下关键信息:

#### 4.1 确认方法被调用
```bash
grep "ChatAgenticNode.execute_stream() called" test_output/sql_review_*.log
```

**期望输出**:
```
ChatAgenticNode.execute_stream() called for node: node_1
```

**如果不存在**: 说明 ChatAgenticNode.execute_stream() 未被调用，可能是代码路径问题。

#### 4.2 检查 workflow 状态
```bash
grep "self.workflow:" test_output/sql_review_*.log
```

**期望输出**:
```
self.workflow: <datus.agent.workflow.Workflow object at 0x...>
```

**如果显示 None**: 说明 workflow 未正确传递到 ChatAgenticNode。

#### 4.3 检查 metadata
```bash
grep "self.workflow.metadata:" test_output/sql_review_*.log
```

**期望输出**: 应该显示完整的 metadata 字典，包含 `required_tool_sequence`。

**如果显示 None 或缺失**: 说明 metadata 在传递过程中丢失。

#### 4.4 检查 required_tool_sequence
```bash
grep "required_tool_sequence:" test_output/sql_review_*.log
```

**期望输出**:
```
required_tool_sequence: ['describe_table', 'search_external_knowledge', 'read_query', 'get_table_ddl', 'analyze_query_plan', 'check_table_conflicts', 'validate_partitioning']
```

**如果显示 None 或空列表**: 说明 required_tool_sequence 未正确配置。

#### 4.5 检查预检条件
```bash
grep "Preflight check:" test_output/sql_review_*.log
```

**期望输出**:
```
Preflight check: has_workflow=True, has_metadata=True, has_required_tools=True
```

**如果 has_required_tools=False**: 说明预检条件未满足。

#### 4.6 确认预检工具执行
```bash
grep "Executing required preflight tool sequence" test_output/sql_review_*.log
```

**期望输出**: 应该看到此消息，后面跟着预检工具的执行日志。

**如果不存在**: 说明预检代码块未执行。

## 预期结果

### 场景 A: workflow 为 None

**日志特征**:
```
ChatAgenticNode.execute_stream() called for node: node_1
self.workflow: None
WARNING: self.workflow is None in ChatAgenticNode.execute_stream()
Preflight check: has_workflow=False, has_metadata=False, has_required_tools=False
```

**根因**: workflow 未正确传递到 ChatAgenticNode

**修复方向**: 检查 Node 初始化和 workflow 赋值逻辑

### 场景 B: metadata 缺失

**日志特征**:
```
ChatAgenticNode.execute_stream() called for node: node_1
self.workflow: <Workflow object>
self.workflow.metadata: None 或 {}
required_tool_sequence: None
Preflight check: has_workflow=True, has_metadata=False, has_required_tools=False
```

**根因**: workflow.metadata 在执行时丢失

**修复方向**: 检查 metadata 在 workflow 初始化和节点执行之间的传递

### 场景 C: required_tool_sequence 缺失

**日志特征**:
```
ChatAgenticNode.execute_stream() called for node: node_1
self.workflow: <Workflow object>
self.workflow.metadata: {...其他字段...}
required_tool_sequence: None
Preflight check: has_workflow=True, has_metadata=True, has_required_tools=False
```

**根因**: metadata 存在但缺少 required_tool_sequence

**修复方向**: 检查 service.py 中 metadata 的构建逻辑

### 场景 D: 一切正常

**日志特征**:
```
ChatAgenticNode.execute_stream() called for node: node_1
self.workflow: <Workflow object>
self.workflow.metadata: {'required_tool_sequence': [...], ...}
required_tool_sequence: [...]
Preflight check: has_workflow=True, has_metadata=True, has_required_tools=True
Executing required preflight tool sequence...
Tool sequence: [...]
```

**根因**: 代码逻辑正常，预检工具应该执行

**后续**: 检查预检工具执行日志，确认是否完成

## 下一步行动

根据测试结果:

1. **如果场景 A 或 B**: 修复 workflow 传递逻辑
2. **如果场景 C**: 修复 service.py metadata 构建逻辑
3. **如果场景 D**: 检查预检工具执行是否成功，如果不成功，检查预检编排器

## 相关文件

- **修改**: `datus/agent/node/chat_agentic_node.py`
- **测试脚本**: `scripts/test_sql_review.sh`
- **配置**: `datus/api/service.py` (Line 736-751)
- **日志**: `test_output/sql_review_*.log`

## 联系方式

如有问题，请查看完整分析报告:
`/Users/lpp/.claude/plans/witty-humming-ocean.md`
