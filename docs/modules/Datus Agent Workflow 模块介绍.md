# Datus Agent Workflow 模块介绍

> **文档版本**: v1.0
> **更新日期**: 2025-12-30
> **相关模块**: `datus/agent/`

---

基于对Datus代码库的详细分析，我为您提供一份全面的系统架构和技术说明文档。

## Datus系统概述

Datus是一个开源的数据工程智能代理系统，由三个主要组件构成：

1. **Datus-CLI**: AI驱动的命令行接口，为数据工程师提供类Claude Code的交互体验
2. **Datus-Chat**: 基于Web的聊天机器人，支持多轮对话和反馈机制
3. **Datus-API**: REST API接口，为其他代理或应用提供稳定的数据服务

系统通过自动构建数据系统的语义语境图谱，结合元数据、指标、参考SQL和外部知识，实现数据查询的上下文感知。

## 代理类型 (Agent Types)

### 1. 主代理 (Main Agent)
- **位置**: `datus/agent/agent.py`
- **功能**: 系统核心控制器，负责初始化、工作流管理和执行循环
- **职责**:
  - 数据库连接管理
  - 知识库引导 (metadata, metrics, reference_sql)
  - 基准测试执行
  - 评估和数据集生成

### 2. 子代理 (Sub-Agents)
- **位置**: `datus/schemas/agent_models.py`, `datus/cli/sub_agent_commands.py`
- **类型**:
  - **用户自定义子代理**: 通过配置创建的领域特定代理
  - **系统内置子代理**: `gen_semantic_model`, `gen_metrics`, `gen_sql_summary`
- **特点**:
  - 作用域上下文限制 (scoped context)
  - 专用知识库 (scoped knowledge base)
  - 自定义工具和MCP服务器
  - 命名空间隔离

### 3. 会话代理 (Chat Agents)
- **位置**: `datus/agent/node/chat_agentic_node.py`
- **功能**: 支持多轮对话的交互式代理
- **特点**:
  - 工具调用能力
  - 流式响应
  - 会话管理
  - 数据库和文件系统工具集成

## 工作流类型 (Workflow Types)

### 1. 核心工作流类型
基于 `datus/agent/workflow.yml` 的配置：

#### **reflection** (反思工作流)
```
schema_linking → generate_sql → execute_sql → reflect → output
```
- **适用场景**: 需要自我评估和改进的复杂查询
- **特点**: 包含反思节点，可根据执行结果调整后续行为

#### **fixed** (固定工作流)
```
schema_linking → generate_sql → execute_sql → output
```
- **适用场景**: 标准SQL生成和执行任务
- **特点**: 简单直接，无反思环节

#### **dynamic** (动态工作流)
```
schema_linking → generate_sql → execute_sql → reflect → output
```
- **适用场景**: 支持动态调整的查询处理
- **特点**: 反思节点可触发工作流重构

#### **metric_to_sql** (指标到SQL工作流)
```
schema_linking → search_metrics → date_parser → generate_sql → execute_sql → output
```
- **适用场景**: 基于业务指标的查询生成
- **特点**: 专门处理指标相关的语义查询

#### **chat_agentic** (聊天代理工作流)
```
chat → execute_sql → output
```
- **适用场景**: 交互式对话查询
- **特点**: 聊天节点处理所有逻辑，执行SQL并输出结果

#### **chat_agentic_plan** (聊天规划工作流)
```
chat → output
```
- **适用场景**: 纯规划和对话任务
- **特点**: 聊天节点内部处理所有工具调用

### 2. 工作流节点类型
基于 `datus/configuration/node_type.py` 的定义：

#### **控制节点 (Control Types)**:
- **start**: 工作流开始
- **reflect**: 评估和自我反思
- **hitl**: 人工干预 (Human-in-the-Loop)
- **parallel**: 并行执行子节点
- **selection**: 从多个候选结果中选择最佳
- **subworkflow**: 执行嵌套工作流

#### **动作节点 (Action Types)**:
- **schema_linking**: 理解查询并查找相关schema
- **generate_sql**: 生成SQL查询
- **execute_sql**: 执行SQL查询
- **output**: 返回结果给用户
- **reasoning**: 推理分析
- **doc_search**: 搜索相关文档
- **fix**: 修复SQL查询
- **search_metrics**: 搜索指标
- **compare**: 比较SQL与预期
- **date_parser**: 解析时间表达式

#### **代理节点 (Agentic Types)**:
- **chat**: 对话式AI交互
- **gensql**: SQL生成与对话AI
- **semantic**: 语义模型生成
- **sql_summary**: SQL摘要生成

## 工具集成 (Tool Integration)

### 1. 函数工具 (Function Tools)
位置: `datus/tools/func_tool/`

#### **数据库工具**:
- **DBFuncTool**: 数据库操作工具
  - 数据库连接管理
  - 查询执行
  - 模式检查
  - 结果处理

#### **上下文搜索工具**:
- **ContextSearchTools**: 知识库搜索
  - 元数据搜索
  - 指标搜索
  - 参考SQL搜索
  - 文档搜索

#### **日期解析工具**:
- **DateParsingTools**: 时间表达式处理
  - 自然语言日期解析
  - 时间范围计算
  - 格式标准化

#### **文件系统工具**:
- **FilesystemFuncTool**: 文件操作
  - 文件读取/写入
  - 目录操作
  - 路径管理

#### **生成工具**:
- **GenerationTools**: 内容生成
  - SQL生成
  - 文档生成
  - 摘要生成

#### **规划工具**:
- **PlanTool**: 任务规划
  - 会话TODO存储
  - 任务分解
  - 执行跟踪

### 2. MCP (Model Context Protocol) 工具
位置: `datus/tools/mcp_tools/`

#### **支持的通信协议**:
- **stdio**: 标准输入输出 (本地命令)
- **sse**: 服务器发送事件 (HTTP流式)
- **http**: HTTP通信

#### **核心功能**:
- 多协议MCP服务器管理
- 工具过滤 (allowlist/blocklist)
- 服务器连接测试
- 工具调用代理

#### **典型应用**:
- MetricFlow服务器 (指标查询)
- 文件系统服务器 (文件操作)
- API服务器 (外部服务集成)

### 3. 数据库工具
位置: `datus/tools/db_tools/`

#### **支持的数据库类型**:
- SQLite, DuckDB (轻量级)
- Snowflake, BigQuery (云数据仓库)
- PostgreSQL, MySQL (关系型)
- ClickHouse, StarRocks (分析型)
- SQL Server, Oracle (企业级)

#### **核心组件**:
- **DBManager**: 数据库连接管理器
- **连接器**: 各数据库专用连接器
- **方言配置**: SQL方言特定处理

### 4. LLM工具
位置: `datus/tools/llms_tools/`

#### **推理工具**:
- **ReasoningSQL**: SQL推理和优化
- **AutoFixSQL**: SQL自动修复
- **MatchSchema**: Schema匹配

#### **可视化工具**:
- **VisualizationTool**: 查询结果可视化

## 知识库系统 (Knowledge Base)

### 1. 存储组件
位置: `datus/storage/`

#### **元数据存储 (Schema Metadata)**:
- 数据库schema信息
- 表结构、列信息
- 关系和约束

#### **指标存储 (Metrics)**:
- 业务指标定义
- 语义模型
- 成功故事案例

#### **参考SQL存储 (Reference SQL)**:
- 示例查询
- 最佳实践SQL
- 领域特定查询模板

#### **外部知识存储 (External Knowledge)**:
- 文档和外部知识
- 上下文增强信息

### 2. 子代理知识库
- **作用域限制**: 基于命名空间和数据库的知识过滤
- **组件支持**: metadata, metrics, reference_sql
- **构建策略**: overwrite, plan模式

## 模型集成 (Model Integration)

### 1. 支持的LLM提供商
基于 `datus/models/`:

- **OpenAI**: GPT系列模型
- **Claude**: Anthropic Claude模型
- **Gemini**: Google Gemini模型
- **DeepSeek**: 深度求索模型
- **Qwen**: 阿里通义千问模型
- **OpenAI兼容**: 兼容OpenAI API的模型

### 2. 核心功能
- **统一接口**: 标准化生成和工具调用
- **流式响应**: 支持实时输出
- **会话管理**: 多轮对话上下文
- **令牌计数**: 使用量跟踪
- **JSON输出**: 结构化响应支持

## 接口系统 (Interface System)

### 1. CLI接口
位置: `datus/cli/`

#### **核心功能**:
- REPL交互模式
- Web界面集成 (Streamlit)
- 子代理管理
- 命名空间操作
- 知识库引导

#### **命令类型**:
- 数据库操作
- 代理管理
- 知识库管理
- 基准测试
- 评估和分析

### 2. API接口
位置: `datus/api/`

#### **服务类型**:
- **FastAPI服务器**: REST API服务
- **流式响应**: 实时结果推送
- **会话管理**: 多用户并发支持
- **守护进程**: 后台服务模式

#### **端点功能**:
- 任务执行
- 工作流管理
- 结果检索
- 配置管理

### 3. Web界面
位置: `datus/cli/web/`

#### **组件**:
- **聊天执行器**: 多轮对话处理
- **聊天机器人**: 用户界面
- **配置管理器**: 界面配置
- **会话加载器**: 会话状态管理

## 配置系统 (Configuration System)

### 1. 配置文件结构
- **agent.yml**: 主配置文件
- **命名空间**: 数据库环境隔离
- **模型配置**: LLM提供商设置
- **工作流配置**: 执行流程定义
- **工具配置**: 外部工具集成

### 2. 动态配置
- **子代理配置**: 用户自定义代理
- **MCP服务器配置**: 外部服务集成
- **工具过滤**: 安全性控制

## 基准测试和评估 (Benchmarking & Evaluation)

### 1. 支持的基准平台
- **Spider2**: 跨领域文本到SQL
- **BIRD**: 大规模指令微调
- **Semantic Layer**: 语义层查询

### 2. 评估功能
- **自动评估**: 结果准确性检查
- **性能基准**: 执行时间和资源使用
- **数据集生成**: 从轨迹文件生成训练数据

## 架构特点 (Architecture Highlights)

### 1. 模块化设计
- **插件化架构**: 易于扩展新组件
- **协议标准化**: MCP协议集成
- **工具抽象**: 统一工具接口

### 2. 可扩展性
- **自定义节点**: 用户定义工作流节点
- **自定义工具**: 扩展工具生态
- **自定义模型**: 支持新LLM提供商

### 3. 生产就绪
- **错误处理**: 完善的异常处理机制
- **日志系统**: 结构化日志记录
- **监控集成**: 性能和使用量跟踪

### 4. 安全性
- **工具过滤**: MCP服务器工具访问控制
- **命名空间隔离**: 数据环境安全隔离
- **配置验证**: Pydantic模型验证

这份文档涵盖了Datus系统的核心架构、功能特性和技术实现，为您提供了全面的技术参考。系统通过模块化设计实现了高度的可扩展性，能够适应不同规模和复杂度的数据工程需求。