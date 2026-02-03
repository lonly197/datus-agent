# Datus SQL 审查任务处理流程介绍

> **文档版本**: v3.1
> **更新日期**: 2026-02-03
> **相关模块**: `datus/agent/node/`, `datus/tools/`, `datus/api/service.py`
> **相关文档**: [Text2SQL 任务处理流程](Datus%20Text2SQL%20任务处理流程介绍.md)

---

## 概述

本文档描述 Datus SQL 审查任务的处理流程。SQL 审查任务使用专用的 `chat_agentic_plan` 工作流，结合强制预检工具序列和专业的 SQL 审查提示词，实现多维度的 SQL 质量评估和优化建议。

本文档整合了原 Text2SQL 文档中与 SQL 审查相关的内容，包括任务识别、工作流配置、强制预检工具序列与对比说明。

`★ Insight ─────────────────────────────────────`
- **任务识别**：基于关键词自动识别 SQL 审查任务（区分大小写）
- **工作流设计**：`chat_agentic_plan` 专为对话式 AI 设计，支持工具调用和流式事件
- **强制预检**：7 个工具序列确保审查基于实证数据，而非 LLM 猜测
`─────────────────────────────────────────────────`

## 1. 任务识别与分类

### 1.1 智能任务识别 (`datus/api/service.py:_identify_task_type`)

**识别逻辑**：基于关键词匹配自动识别任务类型。

```python
def _identify_task_type(self, task_text: str) -> str:
    task_lower = task_text.lower()

    # SQL审查任务特征
    review_keywords = [
        "审查", "review", "检查", "check", "审核", "audit",
        "质量", "quality", "评估", "evaluate", "分析sql", "analyze sql",
    ]
    if any(keyword in task_lower for keyword in review_keywords):
        return "sql_review"

    # 数据分析任务特征
    analysis_keywords = [
        "分析", "analysis", "对比", "compare", "趋势", "trend",
        "统计", "statistics", "汇总", "summary", "报告", "report",
    ]
    if any(keyword in task_lower for keyword in analysis_keywords):
        return "data_analysis"

    # 默认Text2SQL
    return "text2sql"
```

### 1.2 执行模式覆盖

支持通过 `execution_mode` 参数显式指定任务类型：

| execution_mode | 工作流 | 说明 |
|----------------|--------|------|
| `text2sql` | `text2sql` | Text2SQL 转换（10 步结构化流程） |
| `sql_review` | `chat_agentic_plan` | SQL 审查（7 个强制预检工具） |
| `data_analysis` | `chat_agentic_plan` | 数据分析（Plan 模式） |
| `deep_analysis` | `chat_agentic_plan` | 深度分析（手动确认执行） |

## 2. SQL 审查工作流配置

### 2.1 工作流定义 (`datus/agent/workflow.yml`)

```yaml
# Plan 模式工作流 - SQL 审查使用
chat_agentic_plan:
  - chat_agentic  # 对话式 AI 交互，支持工具调用
  - output        # 结果输出
```

### 2.2 任务处理配置 (`datus/api/service.py`)

```python
if task_type == "sql_review":
    return {
        "workflow": "chat_agentic_plan",
        "plan_mode": False,               # 禁用传统 plan 模式
        "auto_execute_plan": False,       # 禁用自动执行
        "system_prompt": "sql_review",    # 使用专用 SQL 审查提示词
        "output_format": "markdown",      # Markdown 格式输出
        "required_tool_sequence": [
            "describe_table",             # 表结构分析
            "search_external_knowledge",  # StarRocks 规则检索
            "read_query",                 # SQL 语法验证
            "get_table_ddl",              # DDL 定义获取
            "analyze_query_plan",         # 查询计划分析
            "check_table_conflicts",      # 表冲突检测
            "validate_partitioning",      # 分区验证
        ],
    }
```

### 2.3 强制预检工具序列 ⭐

SQL 审查任务在 LLM 推理前**强制执行**7 个预检工具，确保审查基于实证数据，而非 LLM 猜测：

| 序号 | 工具名称 | 功能 | 数据用途 |
|------|----------|------|----------|
| 1 | `describe_table` | 获取表结构信息 | 字段类型、索引分析 |
| 2 | `search_external_knowledge` | 检索审查规则 | 规范合规性检查 |
| 3 | `read_query` | 执行 SQL 验证 | 语法正确性验证 |
| 4 | `get_table_ddl` | 获取表 DDL 定义 | 深入结构分析 |
| 5 | `analyze_query_plan` | 查询执行计划分析 | 性能评估 |
| 6 | `check_table_conflicts` | 表结构冲突检测 | 重复建设风险 |
| 7 | `validate_partitioning` | 分区策略验证 | 分区优化建议 |

`★ Insight ─────────────────────────────────────`
- **强制执行保证**：系统级约束确保 7 个工具必须全部执行，收集完整的审查证据
- **数据驱动审查**：所有审查结论基于工具执行结果，而非 LLM 猜测
- **容错设计**：部分工具失败不影响整体，失败原因会注入上下文供 LLM 参考
`─────────────────────────────────────────────────`

## 3. 预检工具执行机制

### 3.1 工具执行顺序

预检工具按固定顺序执行，每个工具的结果注入上下文供后续工具和 LLM 参考：

`★ Insight ─────────────────────────────────────`
- **工具序列设计**：从表结构→规则→SQL执行→架构分析→性能评估，层层递进
- **容错机制**：部分工具失败不影响整体，失败原因会注入上下文
- **缓存优化**：查询计划分析和表冲突检测结果可缓存 30分钟-2小时
`─────────────────────────────────────────────────`

### 3.2 工具执行入口 (`datus/agent/node/chat_agentic_node.py`)

```python
async def run_preflight_tools(self, workflow, action_history_manager):
    """在 execute_stream 开始前强制执行预检工具"""
    for tool_name in required_tool_sequence:
        # 1. 发送工具调用开始事件
        await self._send_tool_call_event(tool_name, tool_call_id, input_data)

        # 2. 执行工具并记录结果
        result = await self._execute_preflight_tool(tool_name, sql_query, ...)

        # 3. 发送工具调用结果事件
        await self._send_tool_call_result_event(tool_call_id, result, ...)

        # 4. 注入结果到上下文
        self._inject_tool_result_into_context(workflow, tool_name, result)
```

### 3.3 智能缓存支持 (`datus/cli/plan_hooks.py:QueryCache`)

```yaml
plan_hooks:
  enable_query_caching: true
  cache_ttl_seconds:
    describe_table: 1800         # 30分钟
    search_external_knowledge: 3600  # 1小时
    read_query: 300              # 5分钟
    get_table_ddl: 3600          # 1小时
    analyze_query_plan: 1800     # 30分钟
    check_table_conflicts: 3600  # 1小时
    validate_partitioning: 7200  # 2小时
```

## 4. 预检工具详解

### 4.1 `describe_table` - 表结构分析

```python
def describe_table(self, table_name, catalog, database, schema_name):
    """获取表的字段定义、索引信息、数据类型"""
    return {
        "success": True,
        "columns": [...],
        "indexes": [...],
        "table_comment": "线索事实表",
    }
```

### 4.2 `search_external_knowledge` - 规则检索

```python
def search_external_knowledge(self, query_text, domain, layer1, layer2, top_n):
    """检索 StarRocks SQL 审查规则和最佳实践"""
    return {"result": [{"terminology": "...", "explanation": "..."}]}
```

### 4.3 `read_query` - SQL 语法验证

```python
def read_query(self, sql):
    """执行 SQL 查询，验证语法正确性"""
    return {
        "success": True,
        "result": [...],
        "row_count": 100,
    }
```

### 4.4 `get_table_ddl` - DDL 定义获取

```python
def get_table_ddl(self, table_name, catalog, database, schema_name):
    """获取表的完整 DDL 定义"""
    return {
        "success": True,
        "ddl": "CREATE TABLE ...",
    }
```

### 4.5 `analyze_query_plan` - 查询计划分析

**功能**：执行 `EXPLAIN` 分析查询执行计划，识别性能热点。

```python
def analyze_query_plan(self, sql, catalog, database, schema_name):
    """分析 SQL 执行计划"""
    return {
        "success": True,
        "plan_text": "EXPLAIN output...",
        "estimated_rows": 1000,
        "estimated_cost": 150.5,
        "hotspots": [...],           # 性能热点
        "join_analysis": {...},       # JOIN 效率分析
        "index_usage": {...},         # 索引使用情况
    }
```

### 4.6 `check_table_conflicts` - 表冲突检测

**功能**：检测表结构相似性和重复建设风险。

```python
def check_table_conflicts(self, table_name, catalog, database, schema_name):
    """检测表结构冲突"""
    return {
        "success": True,
        "exists_similar": True,
        "matches": [...],
        "duplicate_build_risk": "medium",
        "layering_violations": [...],
    }
```

### 4.7 `validate_partitioning` - 分区验证

**功能**：验证分区策略的合理性和优化空间。

```python
def validate_partitioning(self, table_name, catalog, database, schema_name):
    """验证表分区设计"""
    return {
        "success": True,
        "partitioned": True,
        "partition_info": {...},
        "validation_results": {...},
        "issues": [...],
        "recommended_partition": {...},
    }
```

## 5. SQL 审查提示词模板

### 5.1 模板文件

**位置**: `datus/prompts/prompt_templates/sql_review_system_1.0.j2`

### 5.2 审查框架

```jinja2
你是一个专业的SQL质量审查专家，负责对StarRocks数据库的SQL语句进行全面的质量审查和优化建议。

## 强制审查步骤清单
{# MACHINE_READABLE: 以下是SQL审查必须执行的强制步骤 #}
{# STEP_1: describe_table - 获取待审查SQL中涉及的表结构信息 #}
{# STEP_2: search_external_knowledge - 检索StarRocks审查规则 #}
{# STEP_3: read_query - 执行待审查SQL进行验证 #}
{# STEP_4: get_table_ddl - 获取表DDL定义 #}
{# STEP_5: analyze_query_plan - 分析查询执行计划 #}
{# STEP_6: check_table_conflicts - 检测表冲突 #}
{# STEP_7: validate_partitioning - 验证分区策略 #}

## 审查维度

1. **规范合规性检查**
   - SELECT * 禁止检查
   - 分区裁剪验证
   - 命名规范检查

2. **性能优化评估**
   - 执行计划分析
   - 索引使用情况
   - JOIN 效率评估

3. **数据一致性验证**
   - 业务逻辑正确性
   - 数据质量检查

4. **架构和设计审查**
   - 分区策略合理性
   - 数据仓库分层规范
   - 重复建设风险检测

5. **整改建议**
   - 优化后的 SQL 代码
   - 预期性能提升
```

## 6. 上下文注入机制

### 6.1 预检结果注入

预检工具执行结果通过 `preflight_results` 字段注入到 LLM 上下文：

```python
workflow.context.preflight_results = {
    "describe_table": {...},
    "external_knowledge": {...},
    "read_query": {...},
    "ddl": {...},
    "query_plan_analysis": {...},
    "table_conflicts": {...},
    "partitioning_validation": {...},
}
```

### 6.2 智能错误事件分发

```python
async def _dispatch_error_event(self, error_type, sql_query, error_desc, tool_name, table_names):
    if error_type == "permission_error":
        await self._send_permission_error_event(sql_query, error_desc)
    elif error_type == "timeout_error":
        await self._send_timeout_error_event(sql_query, error_desc)
    elif error_type == "table_not_found":
        await self._send_table_not_found_error_event(table_name, error_desc)
    elif error_type == "connection_error":
        await self._send_db_connection_error_event(sql_query, error_desc)
```

## 7. 审查报告结构

按照 `sql_review_system_1.0.j2` 模板生成：

```markdown
### 📋 审查概览
[简要总结审查结果，是否通过审查，主要问题点]

### 🔍 审查规则
[列出使用的审查规则和标准]

### 📊 执行计划分析
[基于查询执行计划的性能分析]

### 🏗️ 表结构与分区评估
[基于表冲突检测和分区验证结果的架构分析]

### ⚠️ 发现问题
[列出所有发现的问题，按严重程度排序]

### 💡 优化建议
[具体的改进措施和优化方案]

### 🛠️ 优化后的SQL
[优化后的 SQL 代码]

### 📈 预期效果
[性能提升和改进效果说明]
```

## 8. 事件流处理

通过 `ChatAgenticNode` 转换为 SSE 事件流：

| 事件类型 | 说明 |
|----------|------|
| `PlanUpdateEvent` | 预检计划更新 |
| `ToolCallEvent` | 工具调用开始 |
| `ToolCallResultEvent` | 工具调用结果 |
| `ErrorEvent` | 细粒度错误事件 |
| `ChatEvent` | 对话事件 |
| `CompletedEvent` | 任务完成 |

## 9. 相关工作流对比

### 9.1 Text2SQL 工作流 (`text2sql`)

SQL 审查使用 `chat_agentic_plan`，而 Text2SQL 使用专用的 `text2sql` 工作流：

```yaml
text2sql:
  - intent_analysis         # 意图分析（任务类型识别）
  - intent_clarification    # 意图澄清（错别字、歧义、实体提取）
  - schema_discovery        # Schema 发现（三阶段混合召回）
  - schema_validation       # Schema 充分性验证
  - generate_sql            # SQL 生成
  - sql_validate            # SQL 语法和语义验证
  - execute_sql             # SQL 执行
  - result_validation       # 结果质量验证
  - reflect                 # 反思与纠错
  - output                  # 结果输出
```

### 9.2 核心差异

| 特性 | SQL 审查 | Text2SQL |
|------|----------|----------|
| 工作流 | `chat_agentic_plan` | `text2sql` |
| 执行模式 | 强制工具序列 | Preflight Orchestrator |
| 输出格式 | Markdown 报告 | JSON 数据 |
| 反思机制 | 无 | Reflect 节点 |
| 验证节点 | 无 | sql_validate, result_validation |

## 10. 配置示例

### 10.1 API 请求示例

```bash
curl --location --request POST 'http://localhost:8000/workflows/chat_research' \
  --header 'Accept: text/event-stream' \
  --header 'Content-Type: application/json' \
  --data-raw '{
    "namespace": "test",
    "database_name": "test",
    "task": "审查以下SQL：SELECT * FROM dwd_assign_dlr_clue_fact_di WHERE clue_create_time >= '\''2025-12-24'\''",
    "ext_knowledge": "使用StarRocks 3.3 SQL审查规则",
    "execution_mode": "sql_review"
  }'
```

### 10.2 agent.yml 配置

```yaml
plan_hooks:
  enable_query_caching: true
  cache_ttl_seconds:
    describe_table: 1800
    search_external_knowledge: 3600
    read_query: 300
    get_table_ddl: 3600
    analyze_query_plan: 1800
    check_table_conflicts: 3600
    validate_partitioning: 7200
```

## 11. 架构优势

1. **强制执行保证**：系统级约束确保数据收集的可靠性
2. **数据驱动审查**：审查结论基于实际工具执行结果
3. **实时事件流**：前端实时展示执行进度
4. **多维度分析**：性能、架构、分区多角度评估
5. **容错设计**：部分失败不影响整体执行

## 12. 版本历史

### v3.1 (2026-02-03) - 文档结构优化

- **内容补充**：添加强制预检工具序列概览（从 Text2SQL 文档迁移）
- **消除重复**：删除重复的工具序列表格
- **交叉引用完善**：强化与 Text2SQL 文档的互联

### v3.0 (2026-01-26)

- 修正 `text2sql` 工作流定义（10 步流程）
- 更新与 Text2SQL 文档的一致性
- 移除已废弃的 `text2sql_standard` 描述
- 优化文档结构，减少冗余

### v2.5 (2025-12-31)

- Text2SQL 工作流统一架构
- 整合 v1.0 至 v2.4 版本内容
- 新增三个增强预检工具

### v2.4 (2025-12-xx)

- 新增 `analyze_query_plan`、`check_table_conflicts`、`validate_partitioning`
- 实现 PreflightOrchestrator 统一调度
- 增强缓存系统和批量处理
