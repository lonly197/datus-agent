# Prompt 模板管理指南

> **文档版本**: v2.0
> **更新日期**: 2026-01-26

---

## 概述

Datus Agent 使用 Jinja2 模板管理 LLM 提示词，支持**双目录加载机制**：

1. **用户自定义模板目录**: `{agent.home}/template/`（优先）
2. **项目内置模板目录**: `datus/prompts/prompt_templates/`（回退）

当请求模板时，系统会优先从用户目录查找，不存在时回退到内置模板。

---

## 目录结构

```
{agent.home}/template/           ← 用户模板（优先）
    └── {name}_{version}.j2

datus/prompts/prompt_templates/  ← 内置模板（回退）
    └── {name}_{version}.j2
```

| 目录 | 环境变量/配置 | 默认值 |
|------|--------------|--------|
| 用户模板 | `agent.home` + `/template/` | `~/.datus/template/` |
| 内置模板 | 固定路径 | `项目目录/datus/prompts/prompt_templates/` |

---

## 模板列表

以下是目前项目内置的所有模板文件（按功能分类）：

### SQL 生成类

| 模板文件 | 功能 |
|---------|------|
| `gen_sql_system_*.j2` | SQL 生成系统提示 |
| `gen_sql_user_*.j2` | SQL 生成用户提示 |
| `text2sql_system_*.j2` | Text2SQL 系统提示 |
| `sql_system_*.j2` | 通用 SQL 系统提示 |

### SQL 修复类

| 模板文件 | 功能 |
|---------|------|
| `fix_sql_system_*.j2` | SQL 修复系统提示 |
| `fix_sql_user_*.j2` | SQL 修复用户提示 |

### 推理分析类

| 模板文件 | 功能 |
|---------|------|
| `reasoning_system_*.j2` | 推理系统提示 |
| `reasoning_user_*.j2` | 推理用户提示 |
| `selection_analysis_*.j2` | 候选分析提示 |
| `output_checking_*.j2` | 输出检查提示 |

### 评估类

| 模板文件 | 功能 |
|---------|------|
| `evaluation_*.j2` | 评估提示 |

### 知识库管理类

| 模板文件 | 功能 |
|---------|------|
| `gen_semantic_model_*.j2` | 语义模型生成 |
| `gen_metrics_*.j2` | 指标生成 |
| `generate_sql_taxonomy_*.j2` | SQL 分类法生成 |
| `generate_sql_taxonomy_incremental_*.j2` | SQL 分类法增量生成 |

### 血缘与架构类

| 模板文件 | 功能 |
|---------|------|
| `schema_lineage_*.j2` | 血缘分析 |
| `schema_lineage_summary_*.j2` | 血缘摘要 |

### 对比与分类类

| 模板文件 | 功能 |
|---------|------|
| `compare_sql_*.j2` | SQL 对比 |
| `compare_sql_system_mcp_*.j2` | SQL 对比（MCP） |
| `classify_sql_item_*.j2` | SQL 项目分类 |

### 日期解析类

| 模板文件 | 功能 |
|---------|------|
| `date_parser_zh_*.j2` | 中文日期解析 |
| `date_parser_en_*.j2` | 英文日期解析 |

### 其他功能类

| 模板文件 | 功能 |
|---------|------|
| `chat_system_*.j2` | 聊天系统提示 |
| `plan_mode_system_*.j2` | 规划模式系统提示 |
| `sql_review_*.j2` | SQL 审查提示 |
| `visualization_*.j2` | 可视化提示 |
| `gen_sql_summary_*.j2` | SQL 摘要生成 |
| `extract_sql_summary_*.j2` | SQL 摘要提取 |
| `regenerate_sql_name_*.j2` | SQL 名称重生成 |

---

## 版本管理

### 文件命名规范

```
{name}_{version}.j2
```

**示例**：
```
gen_sql_user_1.0.j2
gen_sql_user_1.1.j2
fix_sql_system_1.0.j2
```

### 版本号规则

- **语义化版本**: `主版本.次版本`（如 `1.0`, `1.1`, `2.0`）
- **次版本号递增**: 小幅改进或优化
- **主版本号递增**: 重大变更

### 版本选择逻辑

```python
from datus.prompts.prompt_manager import prompt_manager

# 使用最新版本
content = prompt_manager.render_template("gen_sql_user", question="...")

# 使用指定版本
content = prompt_manager.render_template("gen_sql_user", version="1.0", question="...")
```

---

## 模板管理脚本

### 基本用法

```bash
# 进入项目目录
cd /path/to/datus-agent

# 复制缺失的模板到用户目录（默认行为，不覆盖已有文件）
python scripts/update_prompt_templates.py --config ~/.datus/conf/agent.yml

# 强制覆盖所有模板（包括用户修改的版本）
python scripts/update_prompt_templates.py --config ~/.datus/conf/agent.yml --force

# 预览将要执行的更改（不会实际复制）
python scripts/update_prompt_templates.py --config ~/.datus/conf/agent.yml --dry-run

# 列出所有模板及其状态
python scripts/update_prompt_templates.py --config ~/.datus/conf/agent.yml --list

# 同步特定模板
python scripts/update_prompt_templates.py --config ~/.datus/conf/agent.yml --template gen_sql_user

# 同步多个特定模板
python scripts/update_prompt_templates.py --config ~/.datus/conf/agent.yml \
  --template gen_sql_user \
  --template fix_sql_user
```

### 参数说明

| 参数 | 说明 |
|------|------|
| `--config` | **必填**。agent.yml 配置文件路径 |
| `--force` | 可选。强制覆盖已存在的模板 |
| `--dry-run` | 可选。预览模式，不实际执行更改 |
| `--list` | 可选。列出所有模板及其状态 |
| `--template` | 可选。仅同步指定的模板，可多次使用 |
| `--verbose` | 可选。启用详细日志输出 |

### 模板状态

运行 `--list` 命令时显示的状态：

| 状态 | 含义 |
|------|------|
| `Up to date` | 用户模板已是最新版本 |
| `Update available` | 项目有新版本可用 |
| `Only in project` | 模板仅存在于项目目录 |
| `Only in user` | 模板仅存在于用户目录（自定义） |
| `User has newer` | 用户模板版本比项目新 |

---

## 使用场景

### 场景 1：自定义提示模板

1. **复制模板到用户目录**
   ```bash
   python scripts/update_prompt_templates.py --config ~/.datus/conf/agent.yml
   ```

2. **修改模板**
   ```bash
   vim ~/.datus/template/gen_sql_user_1.0.j2
   ```

3. **重启服务使修改生效**

### 场景 2：同步内置模板更新

```bash
# 同步所有模板（强制覆盖）
python scripts/update_prompt_templates.py --config ~/.datus/conf/agent.yml --force

# 或仅同步特定模板
python scripts/update_prompt_templates.py --config ~/.datus/conf/agent.yml --template gen_sql_user
```

### 场景 3：回滚到内置版本

```bash
# 方式 1：删除用户模板
rm ~/.datus/template/gen_sql_user_1.0.j2

# 方式 2：使用 --force 重新同步
python scripts/update_prompt_templates.py --config ~/.datus/conf/agent.yml --force
```

---

## 常见问题

### Q1：修改用户模板后需要重启服务吗？

是的。API Server 在启动时会加载模板，修改后需要重启服务使修改生效。

### Q2：如何查看所有可用模板？

```bash
python scripts/update_prompt_templates.py --config ~/.datus/conf/agent.yml --list
```

### Q3：模板修改后没有生效？

1. 确认模板路径正确（`~/.datus/template/`）
2. 检查文件名格式正确（`{name}_{version}.j2`）
3. 检查文件权限
4. 重启 API Server

### Q4：如何创建新版本的模板？

使用 `prompt_manager.create_template_version()` 方法：

```python
from datus.prompts.prompt_manager import prompt_manager

# 基于最新版本创建 1.1 版
prompt_manager.create_template_version("gen_sql_user", "1.1")

# 基于指定版本创建
prompt_manager.create_template_version("gen_sql_user", "2.0", base_version="1.1")
```

### Q5：如何获取模板信息？

```python
from datus.prompts.prompt_manager import prompt_manager

info = prompt_manager.get_template_info("gen_sql_user")
# {
#   "name": "gen_sql_user",
#   "available_versions": ["1.0", "1.1"],
#   "latest_version": "1.1",
#   "total_versions": 2
# }
```

---

## 相关资源

- **Prompt 模块源码**: [datus/prompts/prompt_manager.py](../../datus/prompts/prompt_manager.py)
- **模板目录**: [datus/prompts/prompt_templates/](../../datus/prompts/prompt_templates/)
- **Prompts 模块介绍**: [Datus Prompts 模块介绍](./modules/Datus%20Prompts模块介绍.md)
- **项目主页**: [https://datus.ai](https://datus.ai)
- **文档**: [https://docs.datus.ai](https://docs.datus.ai)
- **GitHub**: [https://github.com/Datus-ai/Datus-agent](https://github.com/Datus-ai/Datus-agent)
- **Slack 社区**: [https://join.slack.com/t/datus-ai](https://join.slack.com/t/datus-ai/shared_invite/zt-3g6h4fsdg-iOl5uNoz6A4GOc4xKKWUYg)
