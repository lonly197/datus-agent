# Datus API 模块介绍

> **文档版本**: v2.0
> **更新日期**: 2026-01-22
> **相关模块**: `datus/api/`

---

## 模块概述

### 🏗️ 整体架构设计理念

**"流式AI工作流API服务架构"** - 通过FastAPI构建的数据工程AI工作流的RESTful和流式API服务

Datus API模块采用了**异步流式API架构**，核心设计理念是：

1. **RESTful + 流式**：支持传统RESTful API和实时流式响应
2. **多租户隔离**：通过命名空间实现不同数据库环境的隔离
3. **认证授权**：基于JWT的客户端认证和授权机制
4. **事件驱动**：通过Server-Sent Events (SSE)实现实时工作流状态更新

### 📊 架构层次结构

```
┌─────────────────────────────────────────┐
│            客户端层 (Web/App)            │
├─────────────────────────────────────────┤
│        API网关层 (FastAPI + Uvicorn)     │
├─────────────────────────────────────────┤
│        认证授权层 (JWT + 客户端验证)      │
├─────────────────────────────────────────┤
│        业务逻辑层 (Agent + Workflow)      │
├─────────────────────────────────────────┤
│        数据持久层 (Task Store + Cache)    │
└─────────────────────────────────────────┘
```

### 🎯 核心设计特性

#### 1. **异步流式架构**
```python
# 流式工作流执行
@app.post("/workflows/chat_research")
async def chat_research_endpoint(
    request: ChatResearchRequest,
    current_client: str = Depends(get_current_client)
) -> StreamingResponse:
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )
```

**设计优势**：
- **实时交互**：支持工作流执行过程中的实时状态更新
- **资源效率**：避免长时间阻塞和内存累积
- **用户体验**：提供渐进式结果展示和进度反馈

#### 2. **多租户架构**
```python
# 命名空间隔离
agent = service.get_or_create_agent(
    namespace=request.namespace,
    catalog_name=request.catalog_name,
    database_name=request.database_name
)
```

**设计优势**：
- **环境隔离**：不同数据库环境完全隔离
- **资源复用**：相同配置的Agent实例复用
- **动态扩展**：运行时创建新的命名空间环境

#### 3. **事件驱动通信**
```python
# Server-Sent Events (SSE)
async def event_generator():
    async for event in converter.convert_events(action_history):
        yield f"data: {event.model_dump_json()}\n\n"
```

**设计优势**：
- **标准化协议**：基于SSE的标准化实时通信
- **类型安全**：Pydantic模型确保事件数据结构一致性
- **可扩展性**：支持添加新的事件类型而不破坏现有客户端

#### 4. **JWT认证机制**
```python
# 客户端凭据验证
@app.post("/auth/token")
async def token_endpoint(
    grant_type: str = Form(...),
    client_id: str = Form(...),
    client_secret: str = Form(...)
):
    if auth_service.validate_client_credentials(client_id, client_secret):
        return auth_service.generate_access_token(client_id)
```

**设计优势**：
- **无状态认证**：JWT令牌无需服务器端存储会话状态
- **标准化协议**：遵循OAuth 2.0客户端凭据流
- **安全可配置**：支持环境变量覆盖和自定义JWT配置

### 🔄 详细功能分析

#### **1. service.py - 核心服务逻辑** (1806行)

**核心功能**：
- **工作流执行引擎**：管理同步和异步工作流执行
- **任务生命周期管理**：跟踪任务状态、取消和清理
- **多Agent实例管理**：按命名空间缓存和管理Agent实例
- **流式响应处理**：将ActionHistory转换为SSE事件流
- **任务ID唯一性验证**：防止客户端重复提交
- **任务取消机制**：通过cancellation_registry实现优雅取消
- **执行模式配置**：支持text2sql/sql_review/data_analysis/deep_analysis

**适用场景**：
- 复杂的数据分析工作流执行
- 需要实时反馈的长运行任务
- 多用户并发访问的API服务

**设计优势**：
```python
# 智能任务管理
running_task = RunningTask(task_id=task_id, task=asyncio_task, ...)
await service.register_running_task(task_id, asyncio_task, meta)

# 优雅的任务取消
if running_task and not running_task.task.done():
    running_task.task.cancel()
    await service.unregister_running_task(task_id)
```

**执行模式配置** (service.py:753-830)：
```python
def _configure_task_processing_by_execution_mode(self, execution_mode: str, request: ChatResearchRequest) -> dict:
    """根据执行模式配置处理参数"""

    predefined_configs = {
        "text2sql": {
            "workflow": "text2sql",
            "plan_mode": False,
            "auto_execute_plan": False,
            "system_prompt": "text2sql_system",
            "required_tool_sequence": [
                "search_table", "describe_table",
                "search_reference_sql", "parse_temporal_expressions",
            ],
        },
        "sql_review": {
            "workflow": "chat_agentic_plan",
            "plan_mode": False,
            "auto_execute_plan": False,
            "system_prompt": "sql_review",
            "required_tool_sequence": [
                "describe_table", "search_external_knowledge",
                "read_query", "get_table_ddl",
                "analyze_query_plan", "check_table_conflicts",
                "validate_partitioning",
            ],
        },
        "data_analysis": {
            "workflow": "chat_agentic_plan",
            "plan_mode": True,
            "auto_execute_plan": True,
            "system_prompt": "plan_mode",
        },
        "deep_analysis": {
            "workflow": "chat_agentic_plan",
            "plan_mode": True,
            "auto_execute_plan": False,
            "system_prompt": "deep_analysis_system",
        },
    }
```

#### **2. models.py - API数据模型体系** (281行)

**核心功能**：
- **请求响应模型**：RunWorkflowRequest/Response, ChatResearchRequest
- **事件模型体系**：DeepResearchEvent及其子类
- **任务状态管理**：TodoStatus, TodoItem
- **SQL执行事件**：SqlExecutionStartEvent/ProgressEvent/ResultEvent/ErrorEvent
- **认证模型**：TokenResponse, HealthResponse

**核心模型更新 (v2.0)**：

| 模型 | 用途 | 新增字段 |
|------|------|----------|
| `ChatResearchRequest` | 聊天研究请求 | domain, layer1, layer2, prompt, prompt_mode, execution_mode, auto_execute_plan |
| `TodoStatus` | 计划项状态 | PENDING, IN_PROGRESS, COMPLETED, FAILED |
| `TodoItem` | 计划执行项 | id, content, status |
| `TaskStartResponse` | 任务开始确认 | task_id, status, message, estimated_start_time |
| `SqlExecutionStartEvent` | SQL执行开始 | sqlQuery, databaseName, estimatedRows |
| `SqlExecutionProgressEvent` | SQL执行进度 | sqlQuery, progress, currentStep, elapsedTime |
| `SqlExecutionResultEvent` | SQL执行结果 | sqlQuery, rowCount, executionTime, data, hasMoreData, dataPreview |
| `SqlExecutionErrorEvent` | SQL执行错误 | sqlQuery, error, errorType, suggestions, canRetry |

**适用场景**：
- API接口的数据序列化和验证
- 流式事件的类型安全传输
- 前后端数据契约定义

**设计优势**：
```python
# 联合类型确保类型安全
DeepResearchEvent = Union[
    ChatEvent,
    PlanUpdateEvent,
    ToolCallEvent,
    ToolCallResultEvent,
    SqlExecutionStartEvent,
    SqlExecutionProgressEvent,
    SqlExecutionResultEvent,
    SqlExecutionErrorEvent,
    CompleteEvent,
    ReportEvent,
    ErrorEvent,
]

# 自动JSON序列化
event.model_dump_json()
```

#### **3. auth.py - 认证和授权系统** (136行)

**核心功能**：
- **JWT令牌管理**：生成、验证和刷新访问令牌
- **客户端凭据验证**：OAuth 2.0客户端凭据流实现
- **依赖注入**：FastAPI依赖注入的认证中间件
- **环境变量支持**：JWT_SECRET_KEY环境变量覆盖
- **配置路径管理**：通过path_manager动态确定配置路径

**适用场景**：
- 需要安全访问控制的API服务
- 多客户端应用程序的身份验证
- 企业级应用的访问授权

**设计优势**：
```python
# FastAPI依赖注入
def get_current_client(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> str:
    token = credentials.credentials
    payload = auth_service.validate_token(token)
    return payload.get("client_id")

# 环境变量覆盖
self.jwt_secret = os.getenv("JWT_SECRET_KEY", jwt_config.get("secret_key"))
```

#### **4. event_converter.py - 事件转换器** (2200+行)

**核心功能**：
- **ActionHistory到DeepResearchEvent转换**：将内部动作历史转换为前端友好的事件格式
- **虚拟步骤计划跟踪**：为Text2SQL工作流维护虚拟TodoList
- **SQL报告生成**：v2.8版本6段式结构化报告
- **DDL解析与注释提取**：从DDL中解析表和字段注释
- **SQL结构分析**：分析SQL的JOIN、聚合、窗口函数等

**适用场景**：
- 工作流执行状态的实时监控
- 前端界面的事件驱动更新
- 开发者友好的SQL报告生成

**虚拟步骤跟踪 (event_converter.py:44-84)**：
```python
VIRTUAL_STEPS = [
    {"id": "step_intent", "content": "分析查询意图", "node_types": ["intent_analysis", "intent_clarification"]},
    {"id": "step_schema", "content": "发现数据库模式", "node_types": ["schema_discovery", "schema_validation", ...]},
    {"id": "step_sql", "content": "生成SQL查询", "node_types": ["generate_sql", "sql_generation"]},
    {"id": "step_exec", "content": "执行SQL并验证结果", "node_types": ["execute_sql", "sql_execution", ...]},
    {"id": "step_reflect", "content": "自我纠正与优化", "node_types": ["reflect", "reflection_analysis"]},
    {"id": "step_output", "content": "生成结果报告", "node_types": ["output", "output_generation"]},
]
```

**SQL生成报告v2.8结构**：
1. SQL设计概述 - 任务理解、设计逻辑、数据规模
2. 表和字段详情 - 表清单、字段清单、关联关系
3. 带注释的SQL - 业务逻辑注释
4. SQL验证结果 - 语法、表存在性、列存在性、危险操作
5. 执行验证结果 - 结果预览、统计信息
6. 优化建议 - 性能和质量建议

#### **5. server.py - 服务器生命周期管理**

**核心功能**：
- **守护进程管理**：支持前台运行、后台守护和进程控制
- **配置管理**：命令行参数解析和服务器配置
- **信号处理**：优雅的启动、停止和重启流程
- **生命周期管理**：启动初始化、关闭时任务取消

**适用场景**：
- 生产环境的服务部署和管理
- 需要高可用性的长期运行服务
- 系统集成和自动化部署

**设计优势**：
```python
# 守护进程生命周期
def _daemon_worker(args, agent_args, pid_file, log_file):
    # 1. 设置进程会话
    os.setsid()
    # 2. 写入PID文件
    _write_pid_file(pid_file, os.getpid())
    # 3. 重定向标准IO
    _redirect_stdio(log_file)
    # 4. 运行服务器
    _run_server(args, agent_args)
```

---

## API 端点参考

### 认证端点

| 端点 | 方法 | 认证 | 描述 |
|------|------|------|------|
| `/auth/token` | POST | 无 | OAuth2 令牌获取 (客户端凭据流) |

### 健康检查端点

| 端点 | 方法 | 认证 | 描述 |
|------|------|------|------|
| `/` | GET | 无 | 根端点，返回API信息 |
| `/health` | GET | 无 | 健康检查，返回数据库和LLM状态 |

### 工作流端点

| 端点 | 方法 | 认证 | 描述 |
|------|------|------|------|
| `/workflows/run` | POST | JWT | 执行工作流 (sync/async) |
| `/workflows/chat_research` | POST | JWT | 聊天研究流式响应 (DeepResearchEvent) |
| `/workflows/feedback` | POST | JWT | 记录任务反馈 |
| `/workflows/tasks` | GET | JWT | 列出运行中和最近的任务 |
| `/workflows/tasks/{task_id}` | GET | JWT | 获取任务详情 |
| `/workflows/tasks/{task_id}` | DELETE | JWT | 取消运行中的任务 |

### SSE 事件类型

#### 工作流执行事件 (`/workflows/run`)
- `started` - 任务开始
- `sql_generated` - SQL生成完成
- `execution_complete` - SQL执行完成
- `output_ready` - 输出生成完成
- `done` - 任务完成
- `cancelled` - 任务取消
- `error` - 执行错误
- `progress` - 总体进度
- `node_progress` - 节点执行进度
- `node_detail` - 节点详情
- `node_stream` - 节点流式输出
- `chat_response` - 聊天响应
- `chat_thinking` - 思考过程流式输出
- `generic_action` - 通用动作

#### 深度研究事件 (`/workflows/chat_research`)
- `chat` - 聊天消息事件
- `plan_update` - 计划更新事件 (TodoItem状态)
- `tool_call` - 工具调用事件
- `tool_call_result` - 工具调用结果事件
- `sql_execution_start` - SQL执行开始
- `sql_execution_progress` - SQL执行进度
- `sql_execution_result` - SQL执行结果
- `sql_execution_error` - SQL执行错误
- `complete` - 任务完成
- `report` - 报告生成
- `error` - 错误事件

---

## 架构创新点

#### 1. **工作流执行的流式API设计**
```
传统方式: 同步阻塞API → 长时间等待 → 返回完整结果
Datus方式: 流式SSE API → 实时事件流 → 渐进式结果展示
```

#### 2. **多租户的动态Agent管理**
```
传统方式: 单Agent实例 → 硬编码配置 → 静态环境
Datus方式: 多Agent实例池 → 运行时配置 → 动态环境隔离
```

#### 3. **事件驱动的前端集成模式**
```
传统方式: 轮询状态API → 频繁请求 → 资源浪费
Datus方式: SSE事件流 → 推送更新 → 实时响应
```

#### 4. **JWT + 客户端凭据的认证架构**
```
传统方式: API密钥直接传递 → 安全风险 → 管理复杂
Datus方式: JWT短期令牌 → 安全可控 → 标准化协议
```

#### 5. **虚拟步骤计划跟踪**
```
传统方式: 一次性返回结果 → 无过程透明度
Datus方式: PlanUpdateEvent → 实时TodoList → 过程可视化
```

#### 6. **执行模式动态配置**
```
传统方式: 固定工作流 → 单一处理逻辑
Datus方式: execution_mode参数 → 场景化配置 → 灵活适配
```

---

## 设计哲学总结

Datus API模块的设计哲学可以总结为：

**"以流式交互为核心，通过标准化协议实现AI工作流的实时Web服务化"**

1. **实时性优先**：通过SSE流式响应实现工作流的实时交互体验
2. **标准化第一**：采用RESTful API + JWT + SSE的标准化协议栈
3. **多租户原生**：内置命名空间隔离支持企业级多环境部署
4. **事件驱动架构**：通过事件流实现前后端的松耦合通信
5. **开发者友好**：提供结构化的SQL报告，支持虚拟步骤跟踪

这种设计使得Datus不仅是一个数据工程工具，更是一个**企业级的AI工作流服务平台**，能够为复杂的数据分析应用提供实时、可靠、可扩展的API服务基础设施。API模块的设计确保了系统能够同时支持传统的同步调用和现代的实时流式交互，为不同类型的客户端应用提供了灵活的集成方式。

---

## 关键文件路径

| 文件 | 绝对路径 | 行数 |
|------|----------|------|
| 服务主文件 | `/Users/lonlyhuang/workspace/git/Datus-agent/datus/api/service.py` | 1806 |
| 数据模型 | `/Users/lonlyhuang/workspace/git/Datus-agent/datus/api/models.py` | 281 |
| 认证模块 | `/Users/lonlyhuang/workspace/git/Datus-agent/datus/api/auth.py` | 136 |
| 启动脚本 | `/Users/lonlyhuang/workspace/git/Datus-agent/datus/api/server.py` | 371 |
| 事件转换 | `/Users/lonlyhuang/workspace/git/Datus-agent/datus/api/event_converter.py` | 2200+ |
| API 文档 | `/Users/lonlyhuang/workspace/git/Datus-agent/datus/api/README.md` | - |
