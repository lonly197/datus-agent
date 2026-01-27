# Datus Prompts 模块介绍

> **文档版本**: v2.0
> **更新日期**: 2026-01-23
> **相关模块**: `datus/prompts/`
> **代码仓库**: [Datus Agent](https://github.com/Datus-ai/Datus-agent)

---

## 模块概述

### 整体设计架构理念

**"版本化提示工程架构"** - 通过 Jinja2 模板 + 文件版本管理实现结构化的 LLM 提示管理

Datus Prompts 模块采用了**模板驱动的提示工程架构**，核心设计理念是：

1. **版本化管理**：每个提示模板都有独立的版本控制
2. **模板化渲染**：使用 Jinja2 实现动态提示生成
3. **分层抽象**：将提示逻辑与业务逻辑分离
4. **可扩展定制**：支持用户自定义提示模板

### 架构层次结构

```
┌─────────────────────────────────────────┐
│           应用层 (Agent/Node)            │
├─────────────────────────────────────────┤
│        提示组装层 (Prompt Functions)     │
├─────────────────────────────────────────┤
│        模板管理层 (PromptManager)        │
├─────────────────────────────────────────┤
│     模板存储层 (Jinja2 Templates)        │
└─────────────────────────────────────────┘
```

---

## 核心组件

### 1. PromptManager

**位置**: `datus/prompts/prompt_manager.py`

文件-based 版本化提示模板管理器，支持 Jinja2 渲染。

**核心属性**:
```python
class PromptManager:
    default_templates_dir: Path  # 内置模板目录
    user_templates_dir: Path     # 用户模板目录 (动态获取)
    _env: Environment            # Jinja2 环境
```

**核心方法**:
| 方法 | 功能 |
|------|------|
| `load_template(name, version)` | 加载模板 |
| `render_template(name, **kwargs)` | 渲染模板 |
| `get_raw_template(name, version)` | 获取原始模板内容 |
| `list_templates()` | 列出所有模板名称 |
| `list_template_versions(name)` | 列出模板所有版本 |
| `get_latest_version(name)` | 获取最新版本 |
| `create_template_version(...)` | 创建新版本 |
| `template_exists(name, version)` | 检查模板是否存在 |
| `get_template_info(name)` | 获取模板信息 |
| `copy_to(src, target, version)` | 复制模板 |

**双层搜索路径**:
```python
# 优先用户自定义目录，然后回退到内置模板
search_paths = [str(self.user_templates_dir), str(self.default_templates_dir)]
```

### 2. Prompt Functions

提示组装函数，位于 `datus/prompts/` 目录下。

| 函数 | 文件 | 功能 |
|------|------|------|
| `get_sql_prompt()` | `gen_sql.py` | SQL 生成提示 |
| `fix_sql_prompt()` | `fix_sql.py` | SQL 修复提示 |
| `get_evaluation_prompt()` | `reflection.py` | 执行结果评估提示 |
| `compare_sql_prompt()` | `compare_sql.py` | SQL 对比提示 |
| `gen_prompt()` | `schema_lineage.py` | Schema 血缘分析提示 |
| `gen_summary_prompt()` | `schema_lineage.py` | 血缘摘要提示 |
| `create_selection_prompt()` | `selection.py` | 候选选择提示 |
| `gen_prompt()` | `output_checking.py` | 输出检查提示 |

---

## 模板文件规范

### 文件命名规范

```
{template_name}_{version}.j2
```

示例:
```
gen_sql_system_1.0.j2
gen_sql_user_1.0.j2
fix_sql_system_1.0.j2
fix_sql_user_1.0.j2
```

### 模板文件列表

**SQL 生成类**:
| 模板文件 | 功能 | 版本 |
|---------|------|------|
| `gen_sql_system_1.0.j2` | SQL 生成系统提示 | 1.0 |
| `gen_sql_user_1.0.j2` | SQL 生成用户提示 | 1.0 |
| `text2sql_system_1.0.j2` | Text2SQL 系统提示 | 1.0 |
| `sql_system_1.0.j2` | 通用 SQL 系统提示 | 1.0 |

**SQL 修复类**:
| 模板文件 | 功能 | 版本 |
|---------|------|------|
| `fix_sql_system_1.0.j2` | SQL 修复系统提示 | 1.0 |
| `fix_sql_user_1.0.j2` | SQL 修复用户提示 | 1.0 |

**对话类**:
| 模板文件 | 功能 | 版本 |
|---------|------|------|
| `chat_system_0.9.j2` | 聊天系统提示 | 0.9, 1.0 |
| `plan_mode_system_1.0.j2` | 规划模式系统提示 | 1.0 |

**评估类**:
| 模板文件 | 功能 | 版本 |
|---------|------|------|
| `evaluation_1.0.j2` | 评估提示 | 1.0 |
| `evaluation_2.0.j2` | 评估提示 | 2.0 |
| `evaluation_2.1.j2` | 评估提示 | 2.1 |

**分析类**:
| 模板文件 | 功能 | 版本 |
|---------|------|------|
| `reasoning_system_1.0.j2` | 推理系统提示 | 1.0 |
| `reasoning_user_1.0.j2` | 推理用户提示 | 1.0 |
| `selection_analysis_1.0.j2` | 候选分析提示 | 1.0 |
| `output_checking_1.0.j2` | 输出检查提示 | 1.0 |

**语义模型类**:
| 模板文件 | 功能 | 版本 |
|---------|------|------|
| `gen_semantic_model_system_1.0.j2` | 语义模型生成系统提示 | 1.0 |
| `gen_metrics_system_1.0.j2` | 指标生成系统提示 | 1.0 |

**其他类**:
| 模板文件 | 功能 | 版本 |
|---------|------|------|
| `date_parser_en_1.0.j2` | 日期解析英文提示 | 1.0 |
| `date_parser_zh_1.0.j2` | 日期解析中文提示 | 1.0 |
| `compare_sql_system_mcp_1.0.j2` | SQL 对比系统提示 | 1.0 |
| `compare_sql_user_1.0.j2` | SQL 对比用户提示 | 1.0 |
| `schema_lineage_system_1.0.j2` | 血缘分析系统提示 | 1.0 |
| `schema_lineage_user_1.0.j2` | 血缘分析用户提示 | 1.0 |
| `schema_lineage_summary_1.0.j2` | 血缘摘要提示 | 1.0 |
| `visualization_system_1.0.j2` | 可视化系统提示 | 1.0 |
| `sql_review_system_1.0.j2` | SQL 审查系统提示 | 1.0 |
| `extract_sql_summary_1.0.j2` | SQL 摘要提取提示 | 1.0 |
| `generate_sql_taxonomy_1.0.j2` | SQL 分类生成提示 | 1.0 |
| `generate_sql_taxonomy_incremental_1.0.j2` | SQL 分类增量提示 | 1.0 |
| `regenerate_sql_name_1.0.j2` | SQL 名称重生成提示 | 1.0 |
| `classify_sql_item_1.0.j2` | SQL 项目分类提示 | 1.0 |
| `gen_sql_summary_system_1.0.j2` | SQL 摘要生成系统提示 | 1.0 |

**LLM 增强提取类**:
| 模板文件 | 功能 | 版本 |
|---------|------|------|
| `enum_extract_system_1.0.j2` | 枚举值提取系统提示 | 1.0 |
| `enum_extract_user_1.0.j2` | 枚举值提取用户提示 | 1.0 |
| `metadata_extract_system_1.0.j2` | 业务元数据提取系统提示 | 1.0 |
| `metadata_extract_user_1.0.j2` | 业务元数据提取用户提示 | 1.0 |
| `design_req_extract_system_1.0.j2` | 设计要件解析系统提示 | 1.0 |
| `design_req_extract_user_1.0.j2` | 设计要件解析用户提示 | 1.0 |

---

## 提示工程设计模式

### 1. SQL 生成提示模式

```python
def get_sql_prompt(
    database_type: str,
    table_schemas: Union[List[TableSchema], str],
    data_details: List[TableValue],
    metrics: List[Metric],
    question: str,
    external_knowledge: str = "",
    prompt_version: str = "1.0",
    # ... 更多参数
) -> List[Dict[str, str]]:
    # 处理 schemas、details、metrics
    # 截断过长内容
    # 渲染模板
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
```

**模板变量**:
- `database_type`: 数据库类型 (snowflake, starrocks, sqlite, etc.)
- `database_notes`: 数据库特定注意事项
- `processed_schemas`: 处理后的表结构信息
- `processed_details`: 处理后的数据详情
- `metrics`: 业务指标
- `knowledge_content`: 外部知识
- `question`: 用户问题
- `current_date`, `date_ranges`: 时间上下文

### 2. SQL 修复提示模式

```python
def fix_sql_prompt(
    sql_task: str,
    prompt_version: str = "1.0",
    sql_context: str = "",
    schemas: list[TableSchema] = None,
    docs: list[str] = None,
) -> List[Dict[str, str]]:
    # 错误上下文 + 修复指令
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
```

### 3. 评估反思提示模式

```python
def get_evaluation_prompt(
    task_description: str,
    sql_generation_result: str,
    sql_execution_result: str,
    prompt_version: str = "2.1",
) -> str:
    # 单一提示用于结果评估
    return prompt_manager.render_template(...)
```

### 4. 候选选择提示模式

```python
def create_selection_prompt(
    candidates: Dict[str, Any],
    prompt_version: str = "1.0",
    max_text_length: int = 500,
) -> str:
    # 截断过长内容避免提示溢出
    # 返回单一用户提示
```

### 5. LLM 增强枚举提取提示模式

```python
def extract_enums_with_llm(
    comment: str,
    llm_model: Any,
    prompt_template: str = DEFAULT_ENUM_EXTRACT_PROMPT,
    cache_ttl: int = 3600,
) -> Tuple[List[Dict[str, str]], bool, float]:
    """使用 LLM 从注释中提取枚举值"""
    # LLM 解析复杂注释格式
    # 返回: (枚举列表, 是否枚举, 置信度)
```

**提示模板示例**:
```jinja2
# enum_extract_user_1.0.j2
分析以下数据库注释，提取所有枚举值对：

Comment: {{ comment }}

任务：
1. 识别枚举格式如 "（0:选项1,1:选项2）"
2. 提取 code-label 对
3. 返回 JSON 结果

输出格式：
{"enums": [{"code": "0", "label": "录入"}], "is_enum": true, "confidence": 0.9}
```

### 6. 业务元数据提取提示模式

```python
def extract_business_metadata_with_llm(
    comment: str,
    column_name: str,
    table_name: str,
    llm_model: Any,
) -> Dict[str, Any]:
    """使用 LLM 提取完整的业务元数据"""
    # 提取字段类型、业务含义、关联概念
    # 返回: {field_type, is_key_field, business_meaning, ...}
```

---

## 模板变量截断机制

为避免提示过长，Prompt Functions 实现了自动截断:

```python
# 默认截断长度
max_table_schemas_length = 4000
max_data_details_length = 2000
max_context_length = 8000
max_value_length = 500
max_text_mark_length = 16
```

**截断处理**:
```python
if len(processed_schemas) > max_table_schemas_length:
    logger.warning("Table schemas is too long, truncating...")
    processed_schemas = processed_schemas[:max_table_schemas_length] + "\n... (truncated)"
```

---

## 数据库特定处理

### Snowflake 方言

```python
if database_type.lower() == DBType.SNOWFLAKE.value.lower():
    database_notes = (
        "\nEnclose all column names in double quotes to comply with "
        "Snowflake syntax requirements and avoid errors. "
        "When referencing table names in Snowflake SQL, you must include "
        "both the database_name and schema_name."
    )
```

### StarRocks 方言

```python
elif database_type.lower() == DBType.STARROCKS.value.lower():
    database_notes = ""  # 无特殊处理
```

---

## 版本化设计优势

1. **渐进式改进**：新版本可以安全部署，不会影响现有功能
2. **回滚支持**：可以快速回退到之前的提示版本
3. **A/B 测试**：可以同时运行多个版本进行对比测试
4. **用户定制**：用户可以覆盖内置提示模板
5. **向后兼容**：始终有内置模板作为 fallback

---

## 使用示例

### 1. 使用默认版本

```python
from datus.prompts.prompt_manager import prompt_manager

# 自动使用最新版本
content = prompt_manager.render_template("gen_sql_user", question="...")
```

### 2. 指定版本

```python
content = prompt_manager.render_template("gen_sql_user", version="1.0", question="...")
```

### 3. 创建新版本

```python
from datus.prompts.prompt_manager import prompt_manager

# 基于现有版本创建新版本
prompt_manager.create_template_version("gen_sql_user", "1.1", base_version="1.0")
```

### 4. 复制模板到用户目录

```python
prompt_manager.copy_to("gen_sql_user", "my_custom_sql", "1.0")
```

---

## 版本更新记录

### v2.1 (2026-01-27)
- 新增 LLM 增强提取类模板（enum_extract, metadata_extract, design_req_extract）
- 新增 LLM 增强枚举提取提示模式（第5节）
- 新增业务元数据提取提示模式（第6节）
- 新增 `EnhancedEnumExtractor` 类相关文档

### v2.0 (2026-01-23)
- 完整重写，基于最新代码架构
- 新增 20+ 模板文件完整清单
- 新增 Prompt Functions 详细文档
- 新增模板变量截断机制说明
- 新增数据库特定处理（Snowflake, StarRocks）
- 新增模板文件分类（SQL 生成、修复、对话、评估、分析等）
- 新增使用示例代码

### v1.0 (2026-01-05)
- 初始版本
- 高层次架构概述
- 版本化提示工程理念

---

## 相关资源

- **项目主页**: [https://datus.ai](https://datus.ai)
- **文档**: [https://docs.datus.ai](https://docs.datus.ai)
- **GitHub**: [https://github.com/Datus-ai/Datus-agent](https://github.com/Datus-ai/Datus-agent)
- **Slack 社区**: [https://join.slack.com/t/datus-ai](https://join.slack.com/t/datus-ai/shared_invite/zt-3g6h4fsdg-iOl5uNoz6A4GOc4xKKWUYg)
