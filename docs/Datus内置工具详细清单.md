# Datus 内置工具详细清单

> **文档版本**: v1.0
> **更新日期**: 2025-12-29
> **相关模块**: `datus/tools/`

---

基于对Datus代码仓库的详细分析，当前支持的所有内置工具清单。Datus作为一个数据工程代理系统，提供了丰富的工具集来支持数据分析、查询生成、文件操作、计划管理等各种功能。

### 1. **数据库功能工具 (DBFuncTool)**
**位置**: `datus/tools/func_tool/database.py`

**核心功能**:
- **search_table**: 基于语义相似度搜索表结构和示例数据
- **list_databases**: 列出可用的数据库
- **list_schemas**: 列出数据库中的schema
- **list_tables**: 列出表、视图和物化视图
- **describe_table**: 获取表的详细列信息和语义模型
- **read_query**: 执行任意SQL查询
- **get_table_ddl**: 获取表的DDL定义

**使用场景**:
- 数据探索和发现
- 表结构分析
- SQL查询执行
- 语义模型管理
- 适用于数据工程师进行数据库结构分析和查询开发

### 2. **上下文搜索工具 (ContextSearchTools)**
**位置**: `datus/tools/func_tool/context_search.py`

**核心功能**:
- **list_domain_layers_tree**: 展示度量、参考SQL和外部知识的领域层级结构
- **search_metrics**: 搜索业务指标和KPI
- **search_reference_sql**: 搜索参考SQL查询
- **search_external_knowledge**: 搜索外部知识库

**使用场景**:
- 业务指标发现
- SQL查询参考查找
- 领域知识检索
- 适用于需要理解业务逻辑和查找相似查询的数据分析师

### 3. **文件系统工具 (FilesystemFuncTool)**
**位置**: `datus/tools/func_tool/filesystem_tool.py`

**核心功能**:
- **read_file**: 读取文件内容
- **read_multiple_files**: 批量读取多个文件
- **write_file**: 创建或覆盖文件
- **edit_file**: 精确编辑文件内容
- **create_directory**: 创建目录
- **list_directory**: 列出目录内容
- **directory_tree**: 显示目录树结构
- **move_file**: 移动或重命名文件/目录
- **search_files**: 递归搜索匹配模式的文件

**使用场景**:
- 文件内容读取和编辑
- 项目文件管理
- 代码文件搜索
- 配置管理
- 适用于需要处理文件系统操作的数据工程师和开发者

### 4. **日期解析工具 (DateParsingTools)**
**位置**: `datus/tools/date_tools/date_parser.py`

**核心功能**:
- **extract_and_parse_dates**: 从文本中提取和解析时间表达式
- **parse_temporal_expression**: 解析单个时间表达式
- **generate_date_context**: 为SQL生成生成日期上下文

**使用场景**:
- 时间相关的查询理解
- 日期范围解析
- 相对时间表达式处理
- 适用于处理时间序列数据和时间相关的业务查询

### 5. **生成工具 (GenerationTools)**
**位置**: `datus/tools/func_tool/generation_tools.py`

**核心功能**:
- **check_semantic_model_exists**: 检查语义模型是否存在
- **check_metric_exists**: 检查指标是否存在
- **end_generation**: 完成生成过程
- **generate_sql_summary_id**: 生成SQL摘要的唯一ID

**使用场景**:
- 语义模型管理工作流
- 指标定义管理
- 生成过程控制
- ID生成和重复检查

### 6. **计划工具 (PlanTool)**
**位置**: `datus/tools/func_tool/plan_tools.py`

**核心功能**:
- **todo_read**: 读取任务列表
- **todo_write**: 创建或更新任务列表
- **todo_update**: 更新任务状态

**使用场景**:
- 多步骤任务规划和管理
- 工作流执行跟踪
- 任务状态管理
- 适用于复杂的数据分析和处理工作流

### 7. **LLM增强工具**
**位置**: `datus/tools/llms_tools/`

**核心功能**:
- **autofix_sql**: 使用LLM自动修复SQL错误
- **match_schema**: 模式匹配工具
- **reasoning_sql**: SQL推理工具
- **visualization_tool**: 数据可视化工具

**使用场景**:
- SQL错误自动修复
- 模式匹配和推理
- 数据可视化
- 适用于需要LLM辅助的复杂SQL生成和调试

### 8. **数据库连接器**
**位置**: `datus/tools/db_tools/`

**支持的数据库类型**:
- SQLite (内置，无依赖)
- DuckDB (轻量级，少量依赖)
- Snowflake (可选扩展)
- StarRocks (可选扩展)

**使用场景**:
- 多数据库环境支持
- 连接管理和适配
- 适用于企业级数据库集成

### 9. **MCP工具**
**位置**: `datus/tools/mcp_tools/`

**核心功能**:
- MCP服务器管理
- 多种通信协议支持 (stdio, sse, streamable)
- 工具扩展机制

**使用场景**:
- 外部工具集成
- 自定义工具开发
- 协议兼容性

### 10. **输出工具 (OutputTool)**
**位置**: `datus/tools/output_tools/`

**核心功能**:
- 结构化输出格式化
- 结果展示和管理

**使用场景**:
- 结果格式化输出
- 报告生成

### 11. **搜索工具 (SearchTool)**
**位置**: `datus/tools/search_tools/`

**核心功能**:
- Tavily搜索引擎集成
- 外部信息检索

**使用场景**:
- 外部数据源搜索
- 网络信息检索

### 12. **血缘图工具 (SchemaLineageTool)**
**位置**: `datus/tools/lineage_graph_tools/`

**核心功能**:
- 数据库模式血缘分析
- 表关系图构建

**使用场景**:
- 数据血缘追踪
- 依赖关系分析
- 数据治理

## 工具架构特点

1. **统一接口**: 所有工具都继承自 `BaseTool`，提供一致的接口和注册机制

2. **MCP兼容**: 支持Model Context Protocol，便于工具扩展

3. **语义搜索**: 集成LanceDB进行高效的语义搜索和知识管理

4. **多数据库支持**: 通过适配器模式支持多种数据库系统

5. **工作流管理**: 内置任务计划和执行管理机制

6. **安全沙箱**: 文件系统工具提供路径沙箱保护

这些工具共同构成了Datus强大的数据工程能力，使其能够处理从简单查询到复杂数据分析的各种任务。每个工具都针对特定场景优化，提供专业级的数据工程解决方案。