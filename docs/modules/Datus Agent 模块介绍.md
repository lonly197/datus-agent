# Datus Agent 模块介绍

> **文档版本**: v2.2
> **更新日期**: 2026-01-28
> **相关模块**: `datus/agent/`
> **代码仓库**: [Datus Agent](https://github.com/Datus-ai/Datus-agent)

---

基于对 Datus 代码仓库的详细分析，本文档提供 Agent 模块的全面架构和技术说明。

## 模块架构总览

```
datus/agent/
├── agent.py                 # 主入口点（v2.2 已重构）
├── bootstrap/               # 知识库初始化子模块（v2.2 新增）
│   ├── __init__.py
│   ├── knowledge_base_bootstrapper.py  # 知识库引导程序
│   └── sub_agent_refresher.py          # 子代理知识库刷新
├── benchmark/               # 基准测试子模块（v2.2 新增）
│   ├── __init__.py
│   └── benchmark_engine.py  # 基准测试引擎
├── dataset/                 # 数据集生成子模块（v2.2 新增）
│   ├── __init__.py
│   └── trajectory_dataset_generator.py  # 轨迹数据集生成器
├── workflow.py              # 工作流核心模型
├── workflow_status.py       # 终止状态枚举
├── workflow_runner.py       # 向后兼容入口
├── plan.py                  # 工作流计划生成
├── evaluate.py              # 节点结果评估
├── error_handling.py        # 错误处理
├── cancellation_registry.py # 协作取消注册表
└── runner/                  # 工作流执行子模块
    ├── __init__.py
    ├── workflow_executor.py      # 同步/流式执行引擎
    ├── workflow_lifecycle.py     # 生命周期管理
    ├── workflow_navigator.py     # 节点跳转导航
    └── workflow_termination.py   # 终止与错误处理
```

## 核心组件

### 1. Agent 类

**位置**: `datus/agent/agent.py`

Agent 是整个系统的核心协调器，负责工作流执行、知识库管理、基准测试等功能的统一调度。

**核心属性**:
```python
class Agent:
    args: argparse.Namespace           # 命令行参数
    global_config: AgentConfig         # 全局配置
    db_manager: DBManager              # 数据库管理器
    tools: Dict[str, Any]              # 工具注册表
    storage_modules: Dict[str, bool]   # 已初始化的存储模块
    metadata_store: Optional[SchemaWithValueRAG]   # 元数据存储
    metrics_store: Optional[SemanticMetricsRAG]    # 指标存储
```

**核心方法**:
| 方法 | 功能 | 返回 |
|------|------|------|
| `run(sql_task, check_storage, check_db, run_id)` | 同步执行工作流 | dict |
| `run_stream(sql_task, ...)` | 流式执行（AsyncGenerator） | AsyncGenerator |
| `run_stream_with_metadata(sql_task, metadata)` | 带元数据的流式执行 | AsyncGenerator |
| `create_workflow_runner(check_db, run_id, metadata)` | 创建工作流执行器 | WorkflowRunner |
| `check_db()` | 验证数据库连接 | Dict[str, Any] |
| `probe_llm()` | 测试 LLM 模型连通性 | Dict[str, Any] |
| `bootstrap_kb()` | 初始化知识库组件 | Dict[str, Any] |
| `benchmark()` | 运行基准测试 | Dict[str, Any] |
| `evaluation(log_summary)` | 评估基准测试结果 | Dict[str, Any] |
| `generate_dataset()` | 从轨迹生成数据集 | Dict[str, Any] |

### 2. KnowledgeBaseBootstrapper 类

**位置**: `datus/agent/bootstrap/knowledge_base_bootstrapper.py`

管理知识库存储组件的初始化和引导。

**核心方法**:
| 方法 | 功能 |
|------|------|
| `bootstrap(selected_components)` | 引导指定组件 |
| `_bootstrap_metadata(...)` | 初始化元数据存储 |
| `_bootstrap_metrics(...)` | 初始化指标存储 |
| `_bootstrap_ext_knowledge(...)` | 初始化外部知识存储 |
| `_bootstrap_reference_sql(...)` | 初始化参考 SQL 存储 |
| `check_db()` | 验证数据库连接 |

**支持的组件**:
- `metadata`: Schema 元数据存储
- `metrics`: 语义指标存储
- `document`: 文档存储
- `ext_knowledge`: 外部知识存储
- `reference_sql`: 参考 SQL 存储

### 3. BenchmarkEngine 类

**位置**: `datus/agent/benchmark/benchmark_engine.py`

执行基准测试任务并评估 Agent 性能。

**核心方法**:
| 方法 | 功能 |
|------|------|
| `run()` | 执行基准测试 |
| `run_standard(platform, target_ids, run_id)` | 执行标准基准测试 |
| `run_semantic_layer(path, target_ids, run_id)` | 执行语义层基准测试 |
| `_check_benchmark_file(file_path)` | 验证基准测试文件 |
| `_cleanup_output_paths(path)` | 清理输出路径 |

**支持的基准平台**:
- `spider2`: Spider v2 基准
- `bird_dev`: BIRD 开发集
- `semantic_layer`: 语义层指标基准

### 4. TrajectoryDatasetGenerator 类

**位置**: `datus/agent/dataset/trajectory_dataset_generator.py`

从工作流轨迹文件生成训练数据集。

**核心方法**:
| 方法 | 功能 |
|------|------|
| `generate()` | 生成数据集 |
| `_save_dataset(data, name, format)` | 保存数据集 |
| `_save_parquet(data, name)` | 保存为 Parquet 格式 |

**输出格式**:
- `json`: JSON 格式
- `parquet`: Apache Parquet 格式

---

## 模块交互关系

### 执行流程图

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLI / API                               │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Agent.__init__()                        │
│  - 初始化 args, global_config, db_manager                       │
│  - 检查存储模块状态                                               │
└─────────────────────────────┬───────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
    ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
    │  WorkflowRunner │ │  Bootstrapper   │ │  BenchmarkEngine│
    │  (run/run_stream)│ │  (bootstrap_kb) │ │  (benchmark)    │
    └────────┬────────┘ └────────┬────────┘ └────────┬────────┘
             │                   │                   │
             ▼                   ▼                   ▼
    ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
    │  Workflow       │ │  Storage Modules│ │  Agent.run()    │
    │  (nodes,context)│ │  (Schema,Metric,│ │  (SqlTask)      │
    │                 │ │   Doc, etc.)    │ │                 │
    └─────────────────┘ └─────────────────┘ └─────────────────┘
```

### 与外部模块的交互

| 外部模块 | 交互方式 |
|----------|----------|
| `datus/agent/workflow_runner.py` | `create_workflow_runner()` 创建执行器 |
| `datus/agent/runner/*` | WorkflowRunner 内部委托执行 |
| `datus/configuration/agent_config.py` | 通过 `global_config` 访问配置 |
| `datus/schemas/node_models.py` | `SqlTask` 任务定义 |
| `datus/schemas/action_history.py` | `ActionHistory` 流式事件 |
| `datus/storage/*` | 知识库存储初始化与管理 |
| `datus/tools/db_tools/db_manager.py` | `get_db_manager()` 获取连接 |
| `datus/models/base.py` | `LLMBaseModel` LLM 模型 |
| `datus/utils/benchmark_utils.py` | `evaluate_benchmark_and_report()` |

---

## 主要功能

### 1. 工作流执行

Agent 支持同步和流式两种执行模式：

```python
# 同步执行
result = agent.run(sql_task=sql_task, check_storage=True)

# 流式执行（返回 AsyncGenerator）
async for action in agent.run_stream(sql_task=sql_task):
    print(action.status, action.messages)
```

### 2. 知识库初始化

```python
# 初始化所有知识库组件
agent.bootstrap_kb()

# 或指定组件
from datus.agent.bootstrap import KnowledgeBaseBootstrapper
bootstrapper = KnowledgeBaseBootstrapper(config, args)
result = bootstrapper.bootstrap(["metadata", "metrics"])
```

### 3. 基准测试

```python
# 运行基准测试
result = agent.benchmark()

# 评估结果
eval_result = agent.evaluation(log_summary=True)
```

### 4. 数据集生成

```python
# 从轨迹生成训练数据集
result = agent.generate_dataset()
```

---

## 版本更新记录

### v2.2 (2026-01-28)

**架构重构**:
- `agent.py` 从 904 行精简至 ~165 行（-82%）
- 新增 `bootstrap/` 子模块：知识库初始化逻辑
- 新增 `benchmark/` 子模块：基准测试逻辑
- 新增 `dataset/` 子模块：数据集生成逻辑

**设计改进**:
- 遵循单一职责原则（SRP）
- 子模块可独立测试和复用
- 保持 API 向后兼容

### v2.0 (2026-01-23)

- 完整重写，基于最新代码架构
- 新增流式执行支持（AsyncGenerator）
- 新增协作取消机制

---

## 相关文档

本文档与其他文档存在以下引用和依赖关系：

### 依赖文档（本文档引用的文档）

| 文档 | 路径 | 引用内容 |
|------|------|----------|
| Workflow 模块 | [Datus Agent Workflow 模块介绍.md](./Datus%20Agent%20Workflow%20模块介绍.md) | 工作流执行引擎、节点执行 |
| Configuration 模块 | [Datus Configuration 模块介绍.md](./Datus%20Configuration%20模块介绍.md) | AgentConfig 配置 |
| Schemas 模块 | [Datus Schemas模块介绍.md](./Datus%20Schemas模块介绍.md) | SqlTask, ActionHistory |
| Storage 模块 | [Datus Storage模块介绍.md](./Datus%20Storage模块介绍.md) | 知识库存储组件 |
| CLI 模块 | [Dauts CLI 模块介绍.md](./Dauts%20CLI%20模块介绍.md) | 命令行入口 |

### 被依赖文档（引用本文档的文档）

| 文档 | 路径 | 引用内容 |
|------|------|----------|
| API 模块 | [Datus API模块介绍.md](./Datus%20API模块介绍.md) | Agent 类与 API 端点 |
| CLI 执行命令 | [docs/cli/execution_command.md](../cli/execution_command.md) | `datus run` 命令 |
| CLI 聊天命令 | [docs/cli/chat_command.md](../cli/chat_command.md) | `datus chat` 命令 |
| 快速开始 | [docs/getting_started/Quickstart.md](../getting_started/Quickstart.md) | Agent 使用示例 |

### 文档层级关系

```
docs/
├── modules/
│   ├── Datus Agent 模块介绍.md        ← 本文
│   ├── Datus Agent Workflow 模块介绍.md  ← 工作流执行
│   ├── Datus Agent Node模块介绍.md       ← 节点定义
│   ├── Datus Configuration 模块介绍.md   ← 配置管理
│   ├── Datus Storage模块介绍.md          ← 存储组件
│   ├── Datus Schemas模块介绍.md          ← 数据模型
│   ├── Datus Models模块介绍.md           ← LLM 模型
│   └── ...
│
├── cli/
│   ├── execution_command.md              ← CLI 入口
│   └── chat_command.md                   ← 聊天入口
│
└── getting_started/
    └── Quickstart.md                      ← 快速开始
```
