# Datus 5 大场景改进方案

> **文档版本**: v1.0
> **更新日期**: 2026-01-17

---

## 文档概述

基于对 Datus 最新代码库（`/Users/lonlyhuang/workspace/git/Datus-agent/datus`）的深入分析，结合之前完成的节点模块文档，本文档提供针对5大业务场景的详细改进方案和实施路线图。

**核心发现:**
- **智能问数**: ✅ 完整实现（两阶段意图处理 + 9步Text2SQL工作流）
- **SQL生成**: ✅ 完整实现（多种策略 + 反射机制）
- **SQL审查**: ✅ 完整实现（7种预检工具 + 3种增强工具）
- **深度分析**: ⚠️ 部分实现（缺少专用工作流和高级分析模式）
- **数据质检**: ❌ 有限实现（仅基础结果验证，缺少全面质检能力）

**最新代码变化 (2026-01-17):**
- ✅ IntentAnalysisNode（启发式意图检测 + LLM回退）
- ✅ IntentClarificationNode（纠错 + 消歧 + 实体提取）
- ✅ EnhancedPreflightTools（query_plan分析 + 冲突检测 + 分区验证）
- ✅ SQL预验证（validate_and_suggest_sql_fixes）
- ✅ DDL/DML safeguards（安全防护）
- ✅ **SchemaDiscoveryNode增强** - 集成SchemaLinkingNode能力（渐进式匹配、外部知识增强、LLM Schema匹配）

**🚨 P0 优化任务 (进行中):**
- ⚠️ **SchemaLinkingNode 和 SchemaDiscoveryNode 统一迁移** - Phase 0.1 已完成
- 📋 **已完成代码实现** - feature/schema-discovery-enhancement 分支
- 📋 **待完成**: workflow.yml 更新、测试套件、主分支合并

---

## 第一部分：场景现状分析

### 场景1: 智能问数 ✅ 完整实现

**目标**: 根据业务分析需求提供查询SQL/数据/数据解读

**当前实现**:
```python
# Text2SQL 工作流 (9步完整流程)
text2sql:
  - intent_analysis       # 启发式意图检测 (关键词 + LLM回退)
  - intent_clarification  # LLM意图澄清 (纠错 + 消歧 + 实体提取)
  - schema_discovery      # 语义搜索 + 关键词匹配 + LLM推理
  - schema_validation     # 模式验证
  - generate_sql          # SQL生成 (llm_result2json标准化)
  - sql_validate          # 语法验证 + 模式验证 + DDL/DML safeguards
  - execute_sql           # 异步执行 + 连接池
  - result_validation     # 结果质量验证
  - reflect               # 4策略反思机制
  - output                # 结果输出
```

**关键文件**:
- `datus/agent/node/intent_analysis_node.py` - 启发式意图检测
- `datus/agent/node/intent_clarification_node.py` - LLM意图澄清
- `datus/agent/node/schema_discovery_node.py` - 模式发现
- `datus/agent/node/generate_sql_node.py` - SQL生成
- `datus/agent/node/sql_validate_node.py` - SQL验证
- `datus/agent/node/execute_sql_node.py` - SQL执行
- `datus/agent/node/reflect_node.py` - 反思机制

**能力亮点**:
1. **两阶段意图处理**:
   - IntentAnalysisNode: 快速启发式检测（关键词匹配）+ LLM回退（confidence < 0.7）
   - IntentClarificationNode: 纠错（"华山" → "华南"）+ 消歧 + 实体提取

2. **智能模式发现**:
   - 语义搜索（向量相似度）
   - 关键词匹配
   - LLM推理

3. **标准化JSON解析**:
   - `llm_result2json()` 统一处理所有LLM响应
   - 支持markdown、截断JSON、格式错误修复

4. **SQL预验证**:
   - `validate_and_suggest_sql_fixes()` 语法验证
   - DDL/DML safeguards 防止未授权schema修改

**增强建议**:
1. DataInterpretationNode - 数据洞察生成
2. ConversationMemoryNode - 长期对话记忆
3. 多轮对话优化

---

### 场景2: SQL生成 ✅ 完整实现

**目标**: 根据业务需求生成满足规范的高性能SQL

**当前实现**:
```python
# 多种SQL生成策略
1. Text2SQL 工作流 (9步)
2. metric_to_sql 工作流 (指标驱动)
3. gensql_agentic 工作流 (会话式生成)
4. reflection 工作流 (反思式生成)
```

**关键特性**:
- `llm_result2json()` 标准化JSON解析
- 1小时TTL缓存
- 多种生成策略 (plan-based, schema-based, direct)
- 反射机制 (4种策略)

**反射策略详解**:
```python
# ReflectNode 支持的4种策略
strategies = {
    "schema_linking": {"max_iterations": 2, "focus": "表链接优化"},
    "simple_regenerate": {"max_iterations": 3, "focus": "简单重新生成"},
    "reasoning": {"max_iterations": 3, "focus": "推理式生成"},
    "doc_search": {"max_iterations": 1, "focus": "文档搜索"}
}
```

**优化建议**:
1. PerformanceOptimizationNode - SQL性能分析
2. SQLRewriteNode - 等价SQL重写
3. ExecutionPlanAnalysisNode - 深度执行计划分析

---

### 场景3: SQL审查 ✅ 完整实现

**目标**: 从多维度审查SQL的正确性、合理性、规范性

**当前实现**:
```python
# SQL审查预检编排器 (7种工具)
Legacy Tools:
  - describe_table           # 表结构描述
  - search_external_knowledge  # 外部知识检索
  - read_query               # SQL查询读取
  - get_table_ddl            # DDL获取

Enhanced Tools (v2.4):
  - analyze_query_plan       # 执行计划分析 (with fallback)
  - check_table_conflicts    # 表冲突检测
  - validate_partitioning    # 分区验证
```

**关键文件**:
- `datus/agent/node/sql_review_preflight_orchestrator.py` - SQL审查预检编排
- `datus/tools/func_tool/enhanced_preflight_tools.py` - 增强预检工具
- `datus/agent/node/chat_agentic_node.py` - 对话式审查 (sql_review模式)

**增强特性**:
1. **SQL预验证**: `validate_and_suggest_sql_fixes()` 语法验证 + 修复建议
2. **关键工具 vs 辅助工具分类**:
   - 关键工具: `describe_table`, `search_external_knowledge`（必须成功）
   - 辅助工具: `read_query`, `get_table_ddl`, `analyze_query_plan`, `check_table_conflicts`, `validate_partitioning`（可选）
3. **Fallback规则分析**: 当EXPLAIN失败时，使用静态规则分析SQL性能
4. **分区策略验证**: 检测分区键选择、分区类型、时间分区建议
5. **表冲突检测**: 相似表结构检测、分层违规检测、重复建设风险评估

**Fallback规则分析示例**:
```python
# enhanced_preflight_tools.py 中的fallback分析
hotspots = []
if "SELECT *" in sql:
    hotspots.append({
        "reason": "select_star",
        "severity": "medium",
        "recommendation": "Specify only needed columns"
    })
if "LIKE '%...%'" in sql:
    hotspots.append({
        "reason": "leading_wildcard_like",
        "severity": "medium",
        "recommendation": "Leading wildcards prevent index usage"
    })
```

**优化建议**:
1. BusinessRuleValidationNode - 业务规则验证
2. SecurityAuditNode - SQL安全审计
3. BestPracticeCheckNode - 最佳实践检查
4. ReviewReportNode - 审查报告生成

---

### 场景4: 深度分析 ⚠️ 部分实现

**目标**: 探索式数据分析，生成详细的分析报告

**当前实现**:
```python
# 现有深度分析能力
- ChatAgenticNode (data_analysis模式)
- ExecuteSQLNode (数据查询)
- ReflectNode (有限反思)
- PreflightOrchestrator (数据准备)
```

**能力差距**:
1. ❌ 探索式分析工作流缺失
2. ❌ ReAct推理循环缺失
3. ❌ 统计分析能力缺失
4. ❌ 假设生成和验证缺失
5. ❌ 可视化生成缺失
6. ❌ 报告生成缺失

**新增工作流设计**:
```yaml
# workflow.yml 新增
deep_analysis:
  - intent_analysis
  - intent_clarification
  - exploratory_analysis    # 新增：探索式分析
  - statistical_analysis    # 新增：统计分析
  - hypothesis_testing      # 新增：假设验证
  - visualization           # 新增：可视化生成
  - report_generation       # 新增：报告生成
  - output
```

---

### 场景5: 数据质检 ❌ 有限实现

**目标**: 分析表的建表规范、字段规范、索引使用、数据质量

**当前实现**:
```python
# 现有质检能力
- ResultValidationNode (基础结果验证)
- SQLValidateNode (SQL语法验证)
- EnhancedPreflightTools (部分质量检查)
```

**能力差距**:
1. ❌ 建表规范检查缺失
2. ❌ 字段规范检查缺失
3. ❌ 索引使用分析缺失
4. ❌ 数据质量检查缺失（完整性、一致性、准确性）
5. ❌ 数据漂移检测缺失
6. ❌ 质检报告生成缺失

**新增工作流设计**:
```yaml
# workflow.yml 新增
data_quality:
  - schema_standards_check   # 新增：规范检查
  - data_profiling           # 新增：数据画像
  - data_quality_check       # 新增：质量检查
  - index_analysis           # 新增：索引分析
  - quality_report           # 新增：质检报告
  - output
```

---

## 第二部分：最新代码实现分析

### IntentAnalysisNode 实现

**文件**: `datus/agent/node/intent_analysis_node.py`

**核心功能**:
```python
class IntentAnalysisNode(Node):
    """
    启发式意图检测节点

    功能:
    - 关键词检测（快速）
    - LLM 回退机制（confidence < 0.7）
    - 跳过逻辑（当 execution_mode 预设时）
    """

    async def _detect_intent(self, task_text: str) -> IntentResult:
        # 1. 启发式检测（关键词匹配）
        heuristic_result = detector.detect_sql_intent_by_keyword(task_text)
        is_sql_intent, metadata = heuristic_result

        # 2. 计算confidence
        confidence = min(total_matches * 0.2, 0.8)
        if has_patterns:
            confidence += 0.2

        # 3. LLM回退（如果confidence < 0.7）
        if confidence < 0.7 and use_llm_fallback:
            llm_result = await detector.classify_intent_with_llm(task_text, model)
```

**输出**:
- `workflow.metadata["detected_intent"]` - 检测到的意图类型
- `workflow.metadata["intent_confidence"]` - 置信度 (0-1)
- `workflow.metadata["intent_metadata"]` - 元数据（匹配的关键词、模式等）

---

### IntentClarificationNode 实现

**文件**: `datus/agent/node/intent_clarification_node.py`

**核心功能**:
```python
class IntentClarificationNode(Node, LLMMixin):
    """
    业务意图澄清节点

    功能:
    - 纠错 (e.g., "华山" → "华南")
    - 消歧 (e.g., "最近的销售" → "最近30天的销售数据")
    - 实体提取 (business_terms, time_range, dimensions, metrics)
    """

    async def _clarify_intent(self, task_text: str, ext_knowledge: str) -> Dict:
        prompt = """你是一个专业的数据分析助手。请分析用户的查询意图...

        输出JSON格式：
        {
            "clarified_task": "澄清和规范化后的查询",
            "entities": {
                "business_terms": ["业务术语"],
                "time_range": "时间范围",
                "dimensions": ["数据维度"],
                "metrics": ["指标名称"]
            },
            "corrections": {
                "typos_fixed": ["纠正的错别字"],
                "ambiguities_resolved": ["澄清的模糊表述"]
            },
            "confidence": 0.95
        }"""

        # 使用 llm_call_with_retry 和1小时TTL缓存
        response = await self.llm_call_with_retry(
            prompt=prompt,
            operation_name="intent_clarification",
            cache_key=f"intent_clarification:{hash(task_text)}",
            max_retries=3
        )

        # 使用 llm_result2json 标准化解析
        clarification_result = llm_result2json(response_text, expected_type=dict)
```

**输出**:
- `workflow.metadata["clarified_task"]` - 澄清后的任务
- `workflow.metadata["original_task"]` - 原始任务
- `workflow.metadata["intent_clarification"]` - 完整澄清结果

---

### EnhancedPreflightTools 实现

**文件**: `datus/tools/func_tool/enhanced_preflight_tools.py`

**三个增强工具**:

#### 1. analyze_query_plan - 执行计划分析
```python
async def analyze_query_plan(self, sql: str, catalog: str, database: str, schema: str):
    # 1. 执行 EXPLAIN 查询
    explain_sql = f"EXPLAIN {sql}"
    result = connector.execute_arrow(explain_sql)

    # 2. 分析执行计划
    analysis = self._analyze_plan_text(plan_text, db_type)

    # 3. 返回分析结果
    return {
        "estimated_cost": analysis["estimated_cost"],
        "estimated_rows": analysis["estimated_rows"],
        "hotspots": analysis["hotspots"],  # 性能热点
        "join_analysis": analysis["join_analysis"],
        "index_usage": analysis["index_usage"],
        "recommendations": analysis["recommendations"]
    }
```

**Fallback分析** (当EXPLAIN失败时):
```python
def _fallback_query_analysis(self, sql: str, error: str):
    hotspots = []
    # 规则1: SELECT * 检测
    if re.search(r'SELECT\s+\*\s+FROM', sql):
        hotspots.append({"reason": "select_star", "severity": "medium"})

    # 规则2: LIKE '%...%' 检测
    if re.search(r'LIKE\s+[\'"]?%\w+%[\'"]?', sql):
        hotspots.append({"reason": "leading_wildcard_like", "severity": "medium"})

    # 规则3: 函数索引列检测
    # 规则4: JOIN无ON条件检测
    # 规则5: ORDER BY无LIMIT检测
```

#### 2. check_table_conflicts - 表冲突检测
```python
async def check_table_conflicts(self, table_name: str, catalog: str, database: str, schema: str):
    # 1. 获取表信息
    table_info = self.schema_rag.get_table_schema(table_name, catalog, database, schema)

    # 2. 查找相似表结构
    similar_tables = self._find_similar_tables(table_info, catalog, database, schema)

    # 3. 分析分层违规
    layering_violations = self._analyze_layering_violations(table_name, similar_tables)

    # 4. 评估重复建设风险
    duplicate_risk = self._assess_duplicate_risk(similar_tables, layering_violations)

    return {
        "exists_similar": len(similar_tables) > 0,
        "matches": similar_tables,  # 相似度 > 60% 的表
        "duplicate_build_risk": duplicate_risk,  # high/medium/low
        "layering_violations": layering_violations,
        "recommendations": self._generate_conflict_recommendations(...)
    }
```

**分层违规检测**:
```python
def _analyze_layering_violations(self, table_name: str, similar_tables: List[Dict]):
    violations = []
    # 检测分层模式
    ods_patterns = ["ods_", "origin_", "raw_"]
    dwd_patterns = ["dwd_", "dim_", "fact_"]
    dws_patterns = ["dws_", "summary_", "agg_"]
    ads_patterns = ["ads_", "report_", "dashboard_"]

    # 检查表名是否同时匹配多个分层模式
    detected_layers = []
    for layer_name, patterns in layer_patterns:
        if any(pattern in table_lower for pattern in patterns):
            detected_layers.append(layer_name)

    if len(detected_layers) > 1:
        violations.append(f"表名暗示多个分层: {detected_layers}")
```

#### 3. validate_partitioning - 分区验证
```python
async def validate_partitioning(self, table_name: str, catalog: str, database: str, schema: str):
    # 1. 获取表DDL
    ddl_result = connector.get_table_ddl(table_name, catalog, database, schema)

    # 2. 解析分区信息
    partition_info = self._parse_partition_info(ddl_text)
    # 示例: {"is_partitioned": True, "partition_type": "RANGE", "partition_key": "date_column"}

    # 3. 验证分区策略
    validation_results = self._validate_partition_strategy(partition_info, table_name)

    # 4. 生成分区建议
    recommendations = self._generate_partition_recommendations(partition_info, validation_results)

    return {
        "partitioned": partition_info.get("is_partitioned", False),
        "partition_info": partition_info,
        "validation_results": validation_results,  # {"is_valid": bool, "score": 0-100}
        "issues": validation_results.get("issues", []),
        "recommended_partition": recommendations,
        "performance_impact": performance_impact
    }
```

**分区策略验证**:
```python
def _validate_partition_strategy(self, partition_info: Dict, table_name: str):
    validation_results = {"is_valid": True, "issues": [], "warnings": [], "score": 100}

    if not partition_info.get("is_partitioned"):
        # 大表应考虑分区
        if any(keyword in table_lower for keyword in ["fact", "log", "event", "metric"]):
            validation_results["warnings"].append("大表应考虑分区以提升性能")
            validation_results["score"] -= 20
        return validation_results

    # 检查分区键质量
    partition_key = partition_info.get("partition_key")
    time_indicators = ["date", "time", "timestamp", "created_at", "updated_at"]
    has_time_key = any(indicator in partition_key.lower() for indicator in time_indicators)

    if not has_time_key:
        validation_results["warnings"].append("建议使用时间型分区键")
        validation_results["score"] -= 15

    # 避免高基数键
    if any(indicator in partition_key.lower() for indicator in ["id", "uuid", "hash"]):
        validation_results["warnings"].append("高基数分区键可能导致性能问题")
        validation_results["score"] -= 10
```

---

### Text2SQL 工作流实现

**文件**: `datus/agent/workflow.yml`

**完整流程**:
```yaml
text2sql:
  - intent_analysis       # Step 1: 任务类型检测 (text2sql/sql_review/data_analysis)
  - intent_clarification  # Step 2: 业务意图澄清 (纠错+消歧+实体提取)
  - schema_discovery      # Step 3: 模式发现 (语义+关键词+LLM)
  - schema_validation     # Step 4: 模式验证 (列存在性+模糊匹配)
  - generate_sql          # Step 5: SQL生成 (llm_result2json标准化)
  - sql_validate          # Step 6: SQL验证 (语法+模式+DDL/DML safeguards)
  - execute_sql           # Step 7: SQL执行 (异步+连接池)
  - result_validation     # Step 8: 结果验证 (质量检查)
  - reflect               # Step 9: 反思机制 (4策略自纠错)
  - output                # Step 10: 结果输出
```

**关键数据流**:
```
IntentAnalysisNode
  ↓ workflow.metadata["detected_intent"]
IntentClarificationNode
  ↓ workflow.metadata["clarified_task"]
SchemaDiscoveryNode (使用clarified_task进行语义搜索)
  ↓ workflow.context.table_schemas
SchemaValidationNode (验证schema充分性)
  ↓ 验证通过的schema
GenerateSQLNode (生成SQL)
  ↓ workflow.context.generated_sql
SQLValidateNode (验证SQL)
  ↓ 验证通过的SQL
ExecuteSQLNode (执行SQL)
  ↓ SQL执行结果
ResultValidationNode (验证结果质量)
  ↓ 验证通过的结果
ReflectNode (反思和自纠错)
  ↓ 可能重新生成SQL
OutputNode (输出最终结果)
```

---

### 安全防护实现

**DDL/DML Safeguards**:

```python
# sql_validate_node.py 中的安全检查
def _validate_ddl_dml_safety(self, sql: str, database_type: str):
    """
    验证DDL/DML操作的安全性

    检查项:
    1. DDL操作 (CREATE/ALTER/DROP TABLE) - 需要显式授权
    2. DML操作 (UPDATE/DELETE) - 需要WHERE条件
    3. 危险函数 (DROP DATABASE, TRUNCATE) - 禁止
    """
    sql_upper = sql.strip().upper()

    # 1. 检测DDL操作
    ddl_keywords = ["CREATE TABLE", "ALTER TABLE", "DROP TABLE"]
    has_ddl = any(keyword in sql_upper for keyword in ddl_keywords)

    if has_ddl:
        # 检查是否获得授权
        if not self._has_ddl_permission():
            return {
                "safe": False,
                "error": "DDL操作需要显式授权",
                "suggestion": "请在agent_config中启用allow_ddl=true"
            }

    # 2. 检测DML操作
    dml_keywords = ["UPDATE", "DELETE"]
    has_dml = any(sql_upper.startswith(keyword) for keyword in dml_keywords)

    if has_dml:
        # 检查是否有WHERE条件
        if "WHERE" not in sql_upper:
            return {
                "safe": False,
                "error": "DML操作缺少WHERE条件",
                "suggestion": "请添加WHERE条件限制影响范围"
            }

    # 3. 检测危险操作
    dangerous_keywords = ["DROP DATABASE", "DROP SCHEMA", "TRUNCATE TABLE"]
    has_dangerous = any(keyword in sql_upper for keyword in dangerous_keywords)

    if has_dangerous:
        return {
            "safe": False,
            "error": "检测到危险操作",
            "suggestion": "此操作被禁止，请检查SQL"
        }

    return {"safe": True}
```

---

## 第三部分：场景能力差距分析

### 深度分析能力差距

**缺失能力清单**:

| 能力 | 描述 | 优先级 |
|------|------|--------|
| 探索式分析 | 自动多维度分析、趋势识别、异常检测 | P0 |
| 统计分析 | 描述性统计、假设检验、相关性分析、回归分析 | P0 |
| 假设验证 | 自动生成假设、统计验证、结果解释 | P0 |
| 可视化 | 自动推荐图表类型、生成Plotly/Matplotlib图表 | P0 |
| 报告生成 | HTML/Markdown报告、多模板、包含洞察 | P0 |
| ReAct推理 | 循环推理、工具调用、观察-推理-行动 | P1 |

---

### 数据质检能力差距

**缺失能力清单**:

| 能力 | 描述 | 优先级 |
|------|------|--------|
| 规范检查 | 命名规范、建表规范、字段类型规范、分区规范、注释规范 | P0 |
| 数据画像 | 统计信息、分布分析、基数分析、数据采样 | P0 |
| 质量检查 | 完整性、一致性、准确性、时效性、唯一性 | P0 |
| 索引分析 | 索引使用情况、冗余检测、缺失建议、效果评估 | P0 |
| 数据漂移 | 分布漂移、模式漂移、趋势变化 | P1 |
| 质检报告 | 汇总所有结果、质量评分、改进建议 | P0 |

---

### 智能问数增强空间

**增强建议**:

| 能力 | 描述 | 优先级 |
|------|------|--------|
| 数据解读 | 分析查询结果、生成业务洞察、趋势解读 | P1 |
| 对话记忆 | 长期对话历史持久化、向量检索、语义搜索 | P1 |
| 多轮优化 | 上下文压缩、摘要管理、引用消解 | P2 |

---

### SQL生成优化空间

**优化建议**:

| 能力 | 描述 | 优先级 |
|------|------|--------|
| 性能优化 | SQL性能分析、执行计划深度分析、瓶颈识别 | P2 |
| SQL重写 | 等价SQL重写、连接优化、子查询优化、聚合优化 | P2 |
| 执行计划 | 深度执行计划分析、成本估算、索引建议 | P2 |

---

### SQL审查增强空间

**增强建议**:

| 能力 | 描述 | 优先级 |
|------|------|--------|
| 业务规则验证 | 业务逻辑检查、数据一致性验证、业务约束 | P2 |
| 安全审计 | SQL注入检测、权限检查、敏感数据识别 | P2 |
| 最佳实践 | SQL最佳实践检查、命名规范、代码风格 | P2 |
| 审查报告 | 汇总所有审查结果、改进建议、优先级排序 | P2 |

---

## 第四部分：改进方案设计

### 新节点设计 (20个节点)

#### 深度分析节点 (5个)

##### 1. ExploratoryAnalysisNode
```python
class ExploratoryAnalysisNode(Node):
    """
    探索式数据分析节点

    功能:
    - 多维度自动分析
    - 趋势识别和异常检测
    - 相关性分析
    - 数据分布分析

    输入:
    - SQL查询结果或表名
    - 分析维度配置

    输出:
    - 数据概况
    - 多维度分析结果
    - 趋势和异常
    - 相关性矩阵
    """

    async def run(self):
        # 1. 数据概况分析
        data_overview = self._analyze_overview(data)

        # 2. 多维度切分分析
        dimensional_analysis = self._analyze_dimensions(data, dimensions)

        # 3. 趋势和模式识别
        trends = self._identify_trends(data)

        # 4. 异常检测
        anomalies = self._detect_anomalies(data)

        # 5. 相关性矩阵
        correlations = self._compute_correlations(data)
```

##### 2. StatisticalAnalysisNode
```python
class StatisticalAnalysisNode(Node):
    """
    统计分析节点

    功能:
    - 描述性统计 (均值、中位数、标准差、分位数)
    - 假设检验 (t-test, chi-square, ANOVA)
    - 相关性分析 (Pearson, Spearman)
    - 回归分析
    - 时间序列分析

    输出:
    - 统计摘要
    - 假设检验结果 (p值, 统计量)
    - 相关性矩阵
    - 回归模型
    """

    async def run(self):
        # 1. 描述性统计
        descriptive_stats = self._compute_descriptive_stats(data)

        # 2. 分布检验
        distribution_tests = self._test_distributions(data)

        # 3. 相关性分析
        correlations = self._compute_correlations(data)

        # 4. 假设检验
        hypothesis_tests = self._perform_hypothesis_tests(data)
```

##### 3. HypothesisTestingNode
```python
class HypothesisTestingNode(Node):
    """
    假设生成和验证节点

    功能:
    - 自动生成分析假设
    - 统计验证假设
    - 结果解释

    输出:
    - 生成的假设列表
    - 假设检验结果
    - p值和统计显著性解释
    """

    async def run(self):
        # 1. 基于数据特征生成假设
        hypotheses = await self._generate_hypotheses(data)

        # 2. 选择合适的统计检验方法
        test_methods = self._select_test_methods(hypotheses)

        # 3. 执行假设检验
        test_results = self._perform_tests(data, hypotheses, test_methods)

        # 4. 解释p值和统计显著性
        interpretations = self._interpret_results(test_results)
```

##### 4. VisualizationNode
```python
class VisualizationNode(Node):
    """
    数据可视化节点

    功能:
    - 自动推荐图表类型
    - 生成 Plotly/Matplotlib 图表
    - 支持交互式图表

    输出:
    - 图表配置 (JSON)
    - 图表HTML
    - 图表描述
    """

    async def run(self):
        # 1. 分析数据特征
        data_features = self._analyze_features(data)

        # 2. 推荐合适的图表类型
        chart_types = self._recommend_chart_types(data_features)

        # 3. 生成可视化代码
        chart_configs = self._generate_charts(data, chart_types)

        # 4. 渲染图表 (JSON/HTML)
        charts = self._render_charts(chart_configs)
```

##### 5. ReportGenerationNode
```python
class ReportGenerationNode(Node):
    """
    分析报告生成节点

    功能:
    - 生成 HTML/Markdown 报告
    - 多种报告模板
    - 包含图表和洞察

    输出:
    - 报告URL
    - 报告摘要
    """

    async def run(self):
        # 1. 收集所有分析结果
        analysis_results = self._collect_results()

        # 2. 选择合适的报告模板
        template = self._select_template(analysis_results)

        # 3. 生成洞察和结论
        insights = await self._generate_insights(analysis_results)

        # 4. 渲染最终报告
        report = self._render_report(template, analysis_results, insights)
```

---

#### 数据质检节点 (6个)

##### 1. SchemaStandardsCheckNode
```python
class SchemaStandardsCheckNode(Node):
    """
    模式规范检查节点

    检查项:
    - 命名规范 (表名、字段名)
    - 建表规范 (主键、外键、索引)
    - 字段类型规范
    - 分区规范
    - 注释规范

    输出:
    - 规范检查报告
    - 违规项列表
    - 改进建议
    """

    async def run(self):
        # 1. 获取表DDL
        ddl = self._get_table_ddl(table_name)

        # 2. 应用命名规范规则
        naming_violations = self._check_naming_standards(ddl)

        # 3. 应用建表规范规则
        structure_violations = self._check_structure_standards(ddl)

        # 4. 应用字段类型规范
        type_violations = self._check_type_standards(ddl)

        # 5. 生规范检查报告
        report = self._generate_report(naming_violations, structure_violations, type_violations)
```

##### 2. DataProfilingNode
```python
class DataProfilingNode(Node):
    """
    数据画像节点

    功能:
    - 统计信息 (行数、列数、数据类型)
    - 分布分析 (直方图、分位数)
    - 基数分析 (唯一值、NULL值)
    - 数据采样

    输出:
    - 数据画像报告
    - 统计摘要
    - 采样数据
    """

    async def run(self):
        # 1. 分析表结构
        table_info = self._analyze_table_structure(table_name)

        # 2. 计算统计信息
        statistics = self._compute_statistics(table_name)

        # 3. 采样数据
        samples = self._sample_data(table_name, sample_size=1000)

        # 4. 生成数据画像
        profile = self._generate_profile(table_info, statistics, samples)
```

##### 3. DataQualityCheckNode
```python
class DataQualityCheckNode(Node):
    """
    数据质量检查节点

    检查维度:
    - 完整性 (NULL值、缺失值)
    - 一致性 (外键约束、数据类型)
    - 准确性 (格式验证、范围验证)
    - 时效性 (数据新鲜度)
    - 唯一性 (重复数据)

    输出:
    - 质量检查报告
    - 质量评分 (0-100)
    - 问题列表
    """

    async def run(self):
        # 1. 完整性检查
        completeness = self._check_completeness(table_name)

        # 2. 一致性检查
        consistency = self._check_consistency(table_name)

        # 3. 准确性检查
        accuracy = self._check_accuracy(table_name)

        # 4. 时效性检查
        timeliness = self._check_timeliness(table_name)

        # 5. 唯一性检查
        uniqueness = self._check_uniqueness(table_name)

        # 6. 生成质量评分
        quality_score = self._calculate_quality_score(
            completeness, consistency, accuracy, timeliness, uniqueness
        )
```

##### 4. IndexAnalysisNode
```python
class IndexAnalysisNode(Node):
    """
    索引分析节点

    功能:
    - 索引使用情况分析
    - 冗余索引检测
    - 缺失索引建议
    - 索引效果评估

    输出:
    - 索引分析报告
    - 优化建议
    """

    async def run(self):
        # 1. 获取表索引信息
        indexes = self._get_indexes(table_name)

        # 2. 分析查询模式
        query_patterns = self._analyze_query_patterns(table_name)

        # 3. 评估索引效率
        effectiveness = self._evaluate_effectiveness(indexes, query_patterns)

        # 4. 生成索引优化建议
        recommendations = self._generate_recommendations(indexes, effectiveness)
```

##### 5. DataDriftDetectionNode
```python
class DataDriftDetectionNode(Node):
    """
    数据漂移检测节点

    功能:
    - 分布漂移检测
    - 模式漂移检测
    - 趋势变化检测

    输出:
    - 漂移检测报告
    - 漂移指标 (KL散度、PSI)
    """

    async def run(self):
        # 1. 获取历史数据分布
        historical_dist = self._get_historical_distribution(table_name)

        # 2. 获取当前数据分布
        current_dist = self._get_current_distribution(table_name)

        # 3. 计算漂移指标 (KL散度、PSI)
        drift_metrics = self._calculate_drift_metrics(historical_dist, current_dist)

        # 4. 生成漂移报告
        report = self._generate_drift_report(drift_metrics)
```

##### 6. QualityReportNode
```python
class QualityReportNode(Node):
    """
    质检报告生成节点

    功能:
    - 汇总所有质检结果
    - 生成 HTML/Markdown 报告
    - 提供改进建议

    输出:
    - 质检报告URL
    - 综合质量评分
    """

    async def run(self):
        # 1. 收集所有质检节点结果
        quality_results = self._collect_quality_results()

        # 2. 计算综合质量评分
        overall_score = self._calculate_overall_score(quality_results)

        # 3. 生成改进建议
        recommendations = self._generate_recommendations(quality_results)

        # 4. 渲染最终报告
        report = self._render_report(quality_results, overall_score, recommendations)
```

---

#### 智能问数节点 (2个)

##### 1. DataInterpretationNode
```python
class DataInterpretationNode(Node):
    """
    数据解读节点

    功能:
    - 分析查询结果
    - 生成业务洞察
    - 趋势解读

    输出:
    - 数据洞察
    - 趋势解读
    - 业务建议
    """

    async def run(self):
        # 1. 分析查询结果
        result_analysis = self._analyze_result(data)

        # 2. 生成业务洞察
        insights = await self._generate_insights(result_analysis, business_knowledge)

        # 3. 趋势解读
        trend_interpretation = self._interpret_trends(result_analysis)

        # 4. 业务建议
        recommendations = self._generate_recommendations(insights, trend_interpretation)
```

##### 2. ConversationMemoryNode
```python
class ConversationMemoryNode(Node):
    """
    对话记忆节点

    功能:
    - 管理长期对话历史
    - 向量存储
    - 语义检索

    输出:
    - 相关历史对话
    - 上下文摘要
    """

    async def run(self):
        # 1. 存储当前对话
        self._store_conversation(current_conversation)

        # 2. 语义检索相关历史
        relevant_history = self._retrieve_relevant_history(query)

        # 3. 生成上下文摘要
        context_summary = await self._generate_summary(relevant_history)

        # 4. 返回相关对话和摘要
        return {
            "relevant_history": relevant_history,
            "context_summary": context_summary
        }
```

---

#### SQL优化节点 (3个)

##### 1. PerformanceOptimizationNode
```python
class PerformanceOptimizationNode(Node):
    """
    SQL性能优化节点

    功能:
    - SQL性能分析
    - 瓶颈识别
    - 优化建议

    输出:
    - 性能分析报告
    - 优化建议
    """

    async def run(self):
        # 1. 执行计划分析
        execution_plan = await self._analyze_execution_plan(sql)

        # 2. 性能瓶颈识别
        bottlenecks = self._identify_bottlenecks(execution_plan)

        # 3. 生成优化建议
        recommendations = self._generate_optimization_recommendations(bottlenecks)
```

##### 2. SQLRewriteNode
```python
class SQLRewriteNode(Node):
    """
    SQL重写节点

    功能:
    - 等价SQL重写
    - 连接优化
    - 子查询优化
    - 聚合优化

    输出:
    - 重写后的SQL
    - 优化说明
    """

    async def run(self):
        # 1. 分析SQL结构
        sql_structure = self._analyze_structure(sql)

        # 2. 应用重写规则
        rewritten_sql = self._apply_rewrite_rules(sql, sql_structure)

        # 3. 验证等价性
        equivalence = self._verify_equivalence(sql, rewritten_sql)

        # 4. 生成优化说明
        optimization_notes = self._generate_optimization_notes(sql, rewritten_sql)
```

##### 3. ExecutionPlanAnalysisNode
```python
class ExecutionPlanAnalysisNode(Node):
    """
    执行计划分析节点

    功能:
    - 深度执行计划分析
    - 成本估算
    - 索引建议

    输出:
    - 执行计划分析报告
    - 索引建议
    """

    async def run(self):
        # 1. 解析执行计划
        execution_plan = await self._parse_execution_plan(sql)

        # 2. 成本估算
        cost_estimation = self._estimate_cost(execution_plan)

        # 3. 索引建议
        index_recommendations = self._generate_index_recommendations(execution_plan)

        # 4. 生成分析报告
        report = self._generate_report(execution_plan, cost_estimation, index_recommendations)
```

---

#### SQL审查节点 (4个)

##### 1. BusinessRuleValidationNode
```python
class BusinessRuleValidationNode(Node):
    """
    业务规则验证节点

    功能:
    - 业务逻辑检查
    - 数据一致性验证
    - 业务约束验证

    输出:
    - 业务规则验证报告
    - 违规项列表
    """

    async def run(self):
        # 1. 加载业务规则
        business_rules = self._load_business_rules()

        # 2. 验证业务逻辑
        logic_violations = self._validate_business_logic(sql, business_rules)

        # 3. 验证数据一致性
        consistency_violations = self._validate_data_consistency(sql, business_rules)

        # 4. 生成验证报告
        report = self._generate_validation_report(logic_violations, consistency_violations)
```

##### 2. SecurityAuditNode
```python
class SecurityAuditNode(Node):
    """
    SQL安全审计节点

    功能:
    - SQL注入检测
    - 权限检查
    - 敏感数据识别

    输出:
    - 安全审计报告
    - 风险等级
    """

    async def run(self):
        # 1. SQL注入检测
        injection_risks = self._detect_sql_injection(sql)

        # 2. 权限检查
        permission_issues = self._check_permissions(sql)

        # 3. 敏感数据识别
        sensitive_data = self._identify_sensitive_data(sql)

        # 4. 生成安全审计报告
        report = self._generate_security_report(injection_risks, permission_issues, sensitive_data)
```

##### 3. BestPracticeCheckNode
```python
class BestPracticeCheckNode(Node):
    """
    最佳实践检查节点

    功能:
    - SQL最佳实践检查
    - 命名规范检查
    - 代码风格检查

    输出:
    - 最佳实践检查报告
    - 改进建议
    """

    async def run(self):
        # 1. 加载最佳实践规则
        best_practices = self._load_best_practices()

        # 2. 检查SQL最佳实践
        practice_violations = self._check_best_practices(sql, best_practices)

        # 3. 检查命名规范
        naming_violations = self._check_naming_conventions(sql)

        # 4. 检查代码风格
        style_violations = self._check_code_style(sql)

        # 5. 生成检查报告
        report = self._generate_check_report(practice_violations, naming_violations, style_violations)
```

##### 4. ReviewReportNode
```python
class ReviewReportNode(Node):
    """
    审查报告生成节点

    功能:
    - 汇总所有审查结果
    - 生成 HTML/Markdown 报告
    - 优先级排序

    输出:
    - 审查报告URL
    - 问题汇总
    """

    async def run(self):
        # 1. 收集所有审查结果
        review_results = self._collect_review_results()

        # 2. 优先级排序
        prioritized_issues = self._prioritize_issues(review_results)

        # 3. 生成报告
        report = self._generate_review_report(review_results, prioritized_issues)

        # 4. 返回报告URL
        return {"report_url": report.url, "issues": prioritized_issues}
```

---

### 新工作流设计

#### 深度分析工作流
```yaml
# workflow.yml 新增
deep_analysis:
  - intent_analysis         # 任务类型确认
  - intent_clarification    # 业务意图澄清
  - exploratory_analysis    # 探索式分析
  - statistical_analysis    # 统计分析
  - hypothesis_testing      # 假设验证
  - visualization           # 可视化生成
  - report_generation       # 报告生成
  - output                  # 输出结果
```

#### 数据质检工作流
```yaml
# workflow.yml 新增
data_quality:
  - schema_standards_check  # 规范检查
  - data_profiling          # 数据画像
  - data_quality_check      # 质量检查
  - index_analysis          # 索引分析
  - quality_report          # 质检报告
  - output                  # 输出结果
```

#### 智能问数增强工作流
```yaml
# workflow.yml 修改
text2sql_enhanced:
  - intent_analysis
  - intent_clarification
  - conversation_memory     # 新增：对话记忆
  - schema_discovery
  - schema_validation
  - generate_sql
  - sql_validate
  - execute_sql
  - data_interpretation     # 新增：数据解读
  - result_validation
  - reflect
  - output
```

---

### API 增强

#### RunWorkflowRequest 扩展
```python
# datus/schemas/api_models.py

class RunWorkflowRequest(BaseModel):
    # 现有字段...
    workflow: str
    namespace: str
    task: str
    database_name: Optional[str] = None
    domain: Optional[str] = None
    layer1: Optional[str] = None
    layer2: Optional[str] = None
    ext_knowledge: Optional[str] = None
    current_date: Optional[str] = None
    plan_mode: Optional[bool] = False

    # 新增字段
    output_format: Optional[str] = Field(
        "json",
        description="Output format: json/markdown/html"
    )
    analysis_depth: Optional[str] = Field(
        "standard",
        description="Analysis depth: basic/standard/deep"
    )
    include_visualization: Optional[bool] = Field(
        False,
        description="Include data visualization in output"
    )
    include_insights: Optional[bool] = Field(
        False,
        description="Include AI-generated insights in output"
    )
    max_execution_time: Optional[int] = Field(
        300,
        description="Maximum execution time in seconds"
    )
```

---

## 第五部分：实施路线图

### Phase 1: 深度分析能力建设 (4-6周)

**优先级: P0 (最高)**

**目标**: 构建完整的深度分析能力，支持探索式数据分析

**任务列表**:

#### Week 1-2: 核心分析节点
- [ ] ExploratoryAnalysisNode 实现
  - 数据概况分析
  - 多维度切分分析
  - 趋势识别和异常检测
  - 相关性矩阵计算
- [ ] StatisticalAnalysisNode 实现
  - 描述性统计计算
  - 分布检验 (正态性、偏度、峰度)
  - 相关性分析 (Pearson, Spearman)
  - 假设检验 (t-test, chi-square, ANOVA)
- [ ] HypothesisTestingNode 实现
  - 自动假设生成 (基于数据特征)
  - 统计检验方法选择
  - p值计算和解释
  - 统计显著性判断

#### Week 3-4: 可视化和报告
- [ ] VisualizationNode 实现
  - 数据特征分析
  - 图表类型推荐算法
  - Plotly 图表生成
  - 交互式图表支持
- [ ] ReportGenerationNode 实现
  - 报告模板库建设
  - 洞察生成算法
  - HTML/Markdown 报告渲染
  - 报告样式定制

#### Week 5-6: 工作流集成
- [ ] deep_analysis 工作流配置
- [ ] Prompt 模板开发
  - `deep_analysis_system_1.0.j2`
  - `statistical_analysis_system_1.0.j2`
  - `exploratory_analysis_system_1.0.j2`
- [ ] 测试和优化
  - 单元测试 (每个节点)
  - 集成测试 (端到端工作流)
  - 性能优化

**关键文件**:
- `datus/agent/node/exploratory_analysis_node.py`
- `datus/agent/node/statistical_analysis_node.py`
- `datus/agent/node/hypothesis_testing_node.py`
- `datus/agent/node/visualization_node.py`
- `datus/agent/node/report_generation_node.py`
- `datus/prompts/deep_analysis_system_1.0.j2`
- `datus/prompts/statistical_analysis_system_1.0.j2`

**验收标准**:
- [ ] 支持端到端的深度分析工作流
- [ ] 自动生成分析报告 (HTML格式)
- [ ] 包含可视化图表 (至少3种类型)
- [ ] 通过10+测试用例
- [ ] 统计检验准确率 > 90%

---

### Phase 2: 数据质检能力建设 (3-4周)

**优先级: P0 (最高)**

**目标**: 构建全面的数据质检能力，支持规范检查和质量监控

**任务列表**:

#### Week 1: 规范检查节点
- [ ] SchemaStandardsCheckNode 实现
  - 命名规范检查 (表名、字段名)
  - 建表规范检查 (主键、外键、索引)
  - 字段类型规范检查
  - 分区规范检查
  - 注释规范检查
- [ ] 规则库建设
  - `datus/rules/schema_standards.yaml`
  - 可配置规则引擎
  - 规则优先级和严重程度
- [ ] 配置化支持
  - 规则热更新
  - 自定义规则支持

#### Week 2: 质量检查节点
- [ ] DataProfilingNode 实现
  - 表结构分析
  - 统计信息计算
  - 分布分析 (直方图、分位数)
  - 基数分析 (唯一值、NULL值)
  - 数据采样
- [ ] DataQualityCheckNode 实现
  - 完整性检查 (NULL值、缺失值)
  - 一致性检查 (外键约束、数据类型)
  - 准确性检查 (格式验证、范围验证)
  - 时效性检查 (数据新鲜度)
  - 唯一性检查 (重复数据)
- [ ] 质量评分算法
  - 多维度加权评分
  - 阈值配置
  - 趋势跟踪

#### Week 3: 分析节点
- [ ] IndexAnalysisNode 实现
  - 索引使用情况分析
  - 冗余索引检测
  - 缺失索引建议
  - 索引效果评估
- [ ] DataDriftDetectionNode 实现
  - 历史数据分布获取
  - 当前数据分布计算
  - KL散度、PSI计算
  - 漂移趋势分析

#### Week 4: 报告和集成
- [ ] QualityReportNode 实现
  - 收集所有质检结果
  - 综合质量评分
  - 改进建议生成
  - 报告渲染
- [ ] data_quality 工作流配置
- [ ] 测试和优化

**关键文件**:
- `datus/agent/node/schema_standards_check_node.py`
- `datus/agent/node/data_profiling_node.py`
- `datus/agent/node/data_quality_check_node.py`
- `datus/agent/node/index_analysis_node.py`
- `datus/agent/node/data_drift_detection_node.py`
- `datus/agent/node/quality_report_node.py`
- `datus/rules/schema_standards.yaml`
- `datus/rules/data_quality_rules.yaml`

**验收标准**:
- [ ] 支持端到端的数据质检工作流
- [ ] 自动生成质检报告 (HTML格式)
- [ ] 质量评分准确性 > 90%
- [ ] 通过10+测试用例
- [ ] 规范检查覆盖率 > 95%

---

### Phase 3: 智能问数增强 (2-3周)

**优先级: P1 (高)**

**目标**: 增强智能问数能力，提供更好的数据洞察

**任务列表**:

#### Week 1: 数据洞察节点
- [ ] DataInterpretationNode 实现
  - 查询结果分析
  - 业务洞察生成 (LLM-based)
  - 趋势解读
  - 业务建议生成
- [ ] 洞察生成算法
  - 结果摘要算法
  - 趋势识别算法
  - 异常检测算法
- [ ] 业务知识库集成
  - 业务术语映射
  - 指标库集成

#### Week 2: 对话记忆节点
- [ ] ConversationMemoryNode 实现
  - 对话历史存储 (SQLite/PostgreSQL)
  - 向量化 (Embedding)
  - 语义检索 (向量相似度)
  - 上下文压缩
- [ ] 向量存储集成
  - 向量数据库选择 (Milvus/Qdrant)
  - Embedding模型选择
  - 向量索引优化
- [ ] 语义检索
  - 相关对话检索
  - 上下文相关性评分

#### Week 3: 集成和优化
- [ ] text2sql 工作流增强
  - 集成 DataInterpretationNode
  - 集成 ConversationMemoryNode
- [ ] 多轮对话优化
  - 引用消解
  - 上下文一致性
  - 对话状态管理
- [ ] 测试和优化

**关键文件**:
- `datus/agent/node/data_interpretation_node.py`
- `datus/agent/node/conversation_memory_node.py`
- `datus/storage/conversation_memory/`
- `datus/prompts/data_interpretation_system_1.0.j2`

**验收标准**:
- [ ] 自动生成数据洞察
- [ ] 支持10+轮对话记忆
- [ ] 洞察相关性 > 80%
- [ ] 对话上下文准确性 > 90%

---

### Phase 4: SQL生成/审查优化 (2-3周)

**优先级: P2 (中)**

**目标**: 优化SQL生成和审查能力，提供更好的性能和安全性

**任务列表**:

#### Week 1: 性能优化节点
- [ ] PerformanceOptimizationNode 实现
  - 执行计划深度分析
  - 性能瓶颈识别
  - 优化建议生成
- [ ] SQLRewriteNode 实现
  - SQL结构分析
  - 重写规则引擎
  - 等价性验证
- [ ] ExecutionPlanAnalysisNode 实现
  - 执行计划解析
  - 成本估算
  - 索引建议

#### Week 2: 安全和规范节点
- [ ] BusinessRuleValidationNode 实现
  - 业务规则加载
  - 业务逻辑验证
  - 数据一致性验证
- [ ] SecurityAuditNode 实现
  - SQL注入检测
  - 权限检查
  - 敏感数据识别
- [ ] BestPracticeCheckNode 实现
  - 最佳实践规则库
  - 命名规范检查
  - 代码风格检查

#### Week 3: 集成和优化
- [ ] 工作流增强
  - text2sql 工作流集成性能优化
  - sql_review 工作流集成安全审计
- [ ] 规则库建设
  - `datus/rules/best_practices.yaml`
  - `datus/rules/business_rules.yaml`
  - `datus/rules/security_rules.yaml`
- [ ] 测试和优化

**关键文件**:
- `datus/agent/node/performance_optimization_node.py`
- `datus/agent/node/sql_rewrite_node.py`
- `datus/agent/node/execution_plan_analysis_node.py`
- `datus/agent/node/business_rule_validation_node.py`
- `datus/agent/node/security_audit_node.py`
- `datus/agent/node/best_practice_check_node.py`
- `datus/agent/node/review_report_node.py`
- `datus/prompts/performance_optimization_system_1.0.j2`
- `datus/prompts/security_audit_system_1.0.j2`
- `datus/prompts/best_practice_check_system_1.0.j2`

**验收标准**:
- [ ] SQL性能优化建议准确率 > 85%
- [ ] SQL重写等价性 > 95%
- [ ] 安全审计覆盖10+风险类型
- [ ] 最佳实践检查覆盖率 > 90%

---

### Phase 5: 架构优化和生态建设 (4-6周)

**优先级: P3 (低)**

**目标**: 优化架构，建设生态，提供更好的扩展性

**任务列表**:

#### Week 1-2: 插件化架构
- [ ] 节点插件系统实现
  - 插件加载机制
  - 插件生命周期管理
  - 插件依赖管理
- [ ] 动态加载机制
  - 热加载支持
  - 版本兼容性检查
  - 错误隔离
- [ ] 节点注册表自动发现
  - 自动扫描插件目录
  - 节点类型自动注册
  - 插件元数据管理

#### Week 3-4: 配置简化
- [ ] 可视化工作流编辑器
  - 拖拽式节点编辑
  - 连线可视化
  - 实时预览
- [ ] 配置向导和模板
  - 场景模板库
  - 配置向导
  - 快速开始指南
- [ ] 智能默认值
  - 基于场景的默认配置
  - 自动推荐节点
  - 配置优化建议

#### Week 5-6: 监控和告警
- [ ] 实时性能监控
  - 节点执行时间监控
  - 内存使用监控
  - 并发监控
- [ ] 节点执行追踪
  - 执行链路追踪
  - 性能瓶颈识别
  - 错误追踪
- [ ] 异常告警系统
  - 告警规则配置
  - 多渠道告警 (邮件/钉钉/企微)
  - 告警聚合和降噪

**关键文件**:
- `datus/agent/node/plugin_system.py`
- `datus/configuration/node_registry.py`
- `datus/api/workflow_editor.py`
- `datus/monitoring/performance_monitor.py`
- `datus/monitoring/execution_tracker.py`
- `datus/monitoring/alert_system.py`

**验收标准**:
- [ ] 支持第三方节点插件
- [ ] 插件热加载 < 5秒
- [ ] 可视化编辑器响应时间 < 100ms
- [ ] 监控覆盖所有节点
- [ ] 告警响应时间 < 1分钟

---

## 附录

### 关键文件清单

#### 需要创建的新文件 (48个)

**深度分析节点 (5个)**:
- `datus/agent/node/exploratory_analysis_node.py`
- `datus/agent/node/statistical_analysis_node.py`
- `datus/agent/node/hypothesis_testing_node.py`
- `datus/agent/node/visualization_node.py`
- `datus/agent/node/report_generation_node.py`

**数据质检节点 (6个)**:
- `datus/agent/node/schema_standards_check_node.py`
- `datus/agent/node/data_profiling_node.py`
- `datus/agent/node/data_quality_check_node.py`
- `datus/agent/node/index_analysis_node.py`
- `datus/agent/node/data_drift_detection_node.py`
- `datus/agent/node/quality_report_node.py`

**智能问数节点 (2个)**:
- `datus/agent/node/data_interpretation_node.py`
- `datus/agent/node/conversation_memory_node.py`

**SQL优化节点 (3个)**:
- `datus/agent/node/performance_optimization_node.py`
- `datus/agent/node/sql_rewrite_node.py`
- `datus/agent/node/execution_plan_analysis_node.py`

**SQL审查节点 (4个)**:
- `datus/agent/node/business_rule_validation_node.py`
- `datus/agent/node/security_audit_node.py`
- `datus/agent/node/best_practice_check_node.py`
- `datus/agent/node/review_report_node.py`

**Schema 模型 (20个)**:
- `datus/schemas/exploratory_analysis_node_models.py`
- `datus/schemas/statistical_analysis_node_models.py`
- `datus/schemas/hypothesis_testing_node_models.py`
- `datus/schemas/visualization_node_models.py`
- `datus/schemas/report_generation_node_models.py`
- `datus/schemas/schema_standards_check_node_models.py`
- `datus/schemas/data_profiling_node_models.py`
- `datus/schemas/data_quality_check_node_models.py`
- `datus/schemas/index_analysis_node_models.py`
- `datus/schemas/data_drift_detection_node_models.py`
- `datus/schemas/quality_report_node_models.py`
- `datus/schemas/data_interpretation_node_models.py`
- `datus/schemas/conversation_memory_node_models.py`
- `datus/schemas/performance_optimization_node_models.py`
- `datus/schemas/sql_rewrite_node_models.py`
- `datus/schemas/execution_plan_analysis_node_models.py`
- `datus/schemas/business_rule_validation_node_models.py`
- `datus/schemas/security_audit_node_models.py`
- `datus/schemas/best_practice_check_node_models.py`
- `datus/schemas/review_report_node_models.py`

**Prompt 模板 (8个)**:
- `datus/prompts/deep_analysis_system_1.0.j2`
- `datus/prompts/statistical_analysis_system_1.0.j2`
- `datus/prompts/exploratory_analysis_system_1.0.j2`
- `datus/prompts/data_quality_check_system_1.0.j2`
- `datus/prompts/data_interpretation_system_1.0.j2`
- `datus/prompts/performance_optimization_system_1.0.j2`
- `datus/prompts/security_audit_system_1.0.j2`
- `datus/prompts/best_practice_check_system_1.0.j2`

**配置文件 (5个)**:
- `datus/rules/schema_standards.yaml`
- `datus/rules/data_quality_rules.yaml`
- `datus/rules/best_practices.yaml`
- `datus/rules/business_rules.yaml`
- `datus/rules/security_rules.yaml`

**存储模块 (1个)**:
- `datus/storage/conversation_memory/__init__.py`

#### 需要修改的现有文件 (5个)

- `datus/configuration/node_type.py` - 新增20个节点类型
- `datus/agent/node/__init__.py` - 导出20个新节点
- `datus/agent/node/node.py` - 添加20个新节点工厂方法
- `datus/agent/workflow.yml` - 新增2个工作流配置
- `datus/schemas/api_models.py` - API模型扩展 (5个新字段)

---

### 测试场景

#### 场景1: 深度分析 E2E 测试

```python
def test_deep_analysis_workflow():
    """
    测试完整的深度分析工作流
    """
    request = RunWorkflowRequest(
        workflow="deep_analysis",
        namespace="test_namespace",
        task="分析销售数据的趋势、异常和相关性，生成可视化报告",
        database_name="sales_db",
        analysis_depth="deep",
        include_visualization=True,
        include_insights=True
    )

    response = client.post("/workflows/run", json=request.dict())

    assert response.status_code == 200
    result = response.json()

    # 验证工作流完成
    assert result["status"] == "success"

    # 验证包含探索式分析结果
    assert "exploratory_analysis" in result["data"]
    assert result["data"]["exploratory_analysis"]["trends"] is not None

    # 验证包含统计分析结果
    assert "statistical_analysis" in result["data"]
    assert "correlations" in result["data"]["statistical_analysis"]

    # 验证包含可视化
    assert "visualizations" in result["data"]
    assert len(result["data"]["visualizations"]) > 0

    # 验证包含报告
    assert "report_url" in result["data"]
```

#### 场景2: 数据质检 E2E 测试

```python
def test_data_quality_workflow():
    """
    测试完整的数据质检工作流
    """
    request = RunWorkflowRequest(
        workflow="data_quality",
        namespace="test_namespace",
        task="检查用户表的建表规范和数据质量",
        database_name="user_db",
        table_name="users"
    )

    response = client.post("/workflows/run", json=request.dict())

    assert response.status_code == 200
    result = response.json()

    # 验证工作流完成
    assert result["status"] == "success"

    # 验证包含规范检查结果
    assert "schema_standards_check" in result["data"]
    assert "violations" in result["data"]["schema_standards_check"]

    # 验证包含数据画像
    assert "data_profiling" in result["data"]
    assert "statistics" in result["data"]["data_profiling"]

    # 验证包含质量检查结果
    assert "data_quality_check" in result["data"]
    assert "quality_score" in result["data"]["data_quality_check"]
    assert 0 <= result["data"]["data_quality_check"]["quality_score"] <= 100

    # 验证包含索引分析
    assert "index_analysis" in result["data"]

    # 验证包含质检报告
    assert "quality_report_url" in result["data"]
```

#### 场景3: 智能问数 E2E 测试

```python
def test_smart_qa_with_memory_workflow():
    """
    测试智能问数增强工作流 (带对话记忆)
    """
    # 第一轮对话
    request1 = RunWorkflowRequest(
        workflow="text2sql_enhanced",
        namespace="test_namespace",
        task="查询最近30天的销售额",
        database_name="sales_db",
        include_insights=True
    )

    response1 = client.post("/workflows/run", json=request1.dict())
    assert response1.status_code == 200
    result1 = response1.json()

    # 验证包含数据解读
    assert "data_interpretation" in result1["data"]

    # 第二轮对话 (引用上一轮结果)
    request2 = RunWorkflowRequest(
        workflow="text2sql_enhanced",
        namespace="test_namespace",
        task="按地区分组展示",  # 引用上一轮的销售额数据
        database_name="sales_db",
        include_insights=True
    )

    response2 = client.post("/workflows/run", json=request2.dict())
    assert response2.status_code == 200
    result2 = response2.json()

    # 验证对话记忆起作用
    assert "conversation_memory" in result2["data"]
    assert len(result2["data"]["conversation_memory"]) > 0
```

---

### 验收标准总结

#### Phase 1: 深度分析能力建设
- [ ] 支持端到端的深度分析工作流
- [ ] 自动生成分析报告 (HTML格式)
- [ ] 包含可视化图表 (至少3种类型: 折线图、柱状图、饼图)
- [ ] 通过10+测试用例
- [ ] 统计检验准确率 > 90%

#### Phase 2: 数据质检能力建设
- [ ] 支持端到端的数据质检工作流
- [ ] 自动生成质检报告 (HTML格式)
- [ ] 质量评分准确性 > 90%
- [ ] 通过10+测试用例
- [ ] 规范检查覆盖率 > 95%

#### Phase 3: 智能问数增强
- [ ] 自动生成数据洞察
- [ ] 支持10+轮对话记忆
- [ ] 洞察相关性 > 80%
- [ ] 对话上下文准确性 > 90%

#### Phase 4: SQL生成/审查优化
- [ ] SQL性能优化建议准确率 > 85%
- [ ] SQL重写等价性 > 95%
- [ ] 安全审计覆盖10+风险类型
- [ ] 最佳实践检查覆盖率 > 90%

#### Phase 5: 架构优化和生态建设
- [ ] 支持第三方节点插件
- [ ] 插件热加载 < 5秒
- [ ] 可视化编辑器响应时间 < 100ms
- [ ] 监控覆盖所有节点
- [ ] 告警响应时间 < 1分钟

---

## 总结

本文档提供了针对 Datus 5 大场景的详细改进方案，包括:

1. **场景现状分析**: 详细分析了5个场景的当前实现状态
2. **最新代码实现分析**: 深入解析了 IntentAnalysisNode、IntentClarificationNode、EnhancedPreflightTools 等最新实现
3. **场景能力差距分析**: 识别了深度分析和数据质检的能力差距
4. **改进方案设计**: 设计了20个新节点、2个新工作流、API增强和Prompt模板增强
5. **实施路线图**: 提供了5个Phase的详细实施计划 (15-22周)

**关键成果**:
- **智能问数**: ✅ 已完整实现，建议增加数据解读和对话记忆
- **SQL生成**: ✅ 已完整实现，建议增加性能优化和SQL重写
- **SQL审查**: ✅ 已完整实现，建议增加业务规则验证和安全审计
- **深度分析**: ⚠️ 部分实现，需要新增5个节点 (ExploratoryAnalysis、StatisticalAnalysis、HypothesisTesting、Visualization、ReportGeneration)
- **数据质检**: ❌ 有限实现，需要新增6个节点 (SchemaStandardsCheck、DataProfiling、DataQualityCheck、IndexAnalysis、DataDriftDetection、QualityReport)

**实施优先级**:
- **P0 (最高)**: Phase 1 深度分析 + Phase 2 数据质检 (7-10周)
- **P1 (高)**: Phase 3 智能问数增强 (2-3周)
- **P2 (中)**: Phase 4 SQL生成/审查优化 (2-3周)
- **P3 (低)**: Phase 5 架构优化和生态建设 (4-6周)

通过实施本改进方案，Datus 将从基础的SQL处理平台，升级为支持复杂业务场景的智能化数据分析平台。
