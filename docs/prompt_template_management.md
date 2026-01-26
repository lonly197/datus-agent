# Prompt 模板管理指南

> **文档版本**: v1.0
> **更新日期**: 2026-01-26
> **相关脚本**: `scripts/update_prompt_templates.py`

---

## 概述

Datus Agent 使用 Jinja2 模板管理 LLM 提示词，支持**双目录加载机制**：
- **项目内置模板**: `datus/prompts/prompt_templates/`
- **用户自定义模板**: `{agent.home}/template/`

当请求模板时，系统会**优先查找用户目录**，如果不存在则回退到内置模板。

---

## 模板加载优先级

```
{agent.home}/template/  ← 优先（用户自定义）
        ↓
datus/prompts/prompt_templates/  ← 回退（内置默认）
```

### 目录说明

| 目录 | 路径 | 用途 |
|------|------|------|
| 用户模板 | `{agent.home}/template/` | 用户自定义的提示模板 |
| 内置模板 | `项目目录/datus/prompts/prompt_templates/` | 系统内置的默认模板 |

`{agent.home}` 默认为 `~/.datus`，可通过 `agent.yml` 中的 `agent.home` 配置自定义路径。

---

## 使用场景

### 场景 1: 自定义提示模板

当需要修改某个提示模板以适配特定业务场景时：

1. **复制模板到用户目录**
   ```bash
   # 列出所有可用模板
   python -m scripts.update_prompt_templates --config ~/.datus/conf/agent.yml --list

   # 将模板复制到用户目录（不会覆盖已有文件）
   python -m scripts.update_prompt_templates --config ~/.datus/conf/agent.yml
   ```

2. **修改模板**
   ```bash
   # 编辑用户目录中的模板
   vim ~/.datus/template/gen_sql_user_1.0.j2
   ```

3. **验证修改**
   ```bash
   # 重启 API Server 使修改生效
   # ...
   ```

### 场景 2: 更新内置模板

当项目发布新版本提示模板时：

```bash
# 同步所有模板（强制覆盖）
python -m scripts.update_prompt_templates --config ~/.datus/conf/agent.yml --force

# 或仅同步特定模板
python -m scripts.update_prompt_templates --config ~/.datus/conf/agent.yml --template gen_sql_user
```

### 场景 3: 预览更改

```bash
# 预览将要执行的更改（不会实际复制）
python -m scripts.update_prompt_templates --config ~/.datus/conf/agent.yml --dry-run
```

---

## 命令参考

### 基础命令

```bash
# 复制缺失的模板（默认行为，不覆盖已有文件）
python -m scripts.update_prompt_templates --config <config_path>

# 强制覆盖所有模板（包括降级用户修改的版本）
python -m scripts.update_prompt_templates --config <config_path> --force

# 预览将要执行的更改
python -m scripts.update_prompt_templates --config <config_path> --dry-run

# 列出所有模板及其状态
python -m scripts.update_prompt_templates --config <config_path> --list

# 同步特定模板
python -m scripts.update_prompt_templates --config <config_path> --template gen_sql_user

# 同步多个特定模板
python -m scripts.update_prompt_templates --config <config_path> \
  --template gen_sql_user \
  --template fix_sql_user \
  --template gen_sql_system
```

### 完整参数说明

| 参数 | 说明 |
|------|------|
| `--config` | **必填**。agent.yml 配置文件路径 |
| `--force` | 可选。强制覆盖已存在的模板 |
| `--dry-run` | 可选。预览模式，不实际执行更改 |
| `--list` | 可选。列出所有模板及其状态 |
| `--template` | 可选。仅同步指定的模板，可多次使用 |
| `--verbose` | 可选。启用详细日志输出 |

---

## 模板状态说明

运行 `--list` 命令时，会显示以下状态：

| 状态 | 含义 |
|------|------|
| **Up to date** | 用户模板已是最新版本 |
| **Update available** | 项目有新版本可用 |
| **Only in project** | 模板仅存在于项目目录 |
| **Only in user** | 模板仅存在于用户目录（自定义） |
| **User has newer** | 用户模板版本比项目新（可能已自定义） |

---

## 模板版本管理

### 文件命名规范

```
{template_name}_{version}.j2
```

示例：
```
gen_sql_user_1.0.j2
gen_sql_user_1.1.j2
fix_sql_system_1.0.j2
```

### 版本号规则

- 使用语义化版本：`主版本.次版本`（如 `1.0`, `1.1`, `2.0`）
- 次版本号递增表示小幅改进
- 主版本号递增表示重大变更

### 版本选择逻辑

1. **未指定版本**: 自动选择最新版本
2. **指定版本**: 使用指定版本（如 `version="1.0"`）

```python
from datus.prompts.prompt_manager import prompt_manager

# 使用最新版本
content = prompt_manager.render_template("gen_sql_user", question="...")

# 使用指定版本
content = prompt_manager.render_template("gen_sql_user", version="1.0", question="...")
```

---

## 模板列表

### SQL 生成类

| 模板文件 | 功能 | 最新版本 |
|---------|------|---------|
| `gen_sql_system_*.j2` | SQL 生成系统提示 | 1.1 |
| `gen_sql_user_*.j2` | SQL 生成用户提示 | 1.1 |
| `text2sql_system_*.j2` | Text2SQL 系统提示 | 1.0 |
| `sql_system_*.j2` | 通用 SQL 系统提示 | 1.0 |

### SQL 修复类

| 模板文件 | 功能 | 最新版本 |
|---------|------|---------|
| `fix_sql_system_*.j2` | SQL 修复系统提示 | 1.1 |
| `fix_sql_user_*.j2` | SQL 修复用户提示 | 1.1 |

### 推理分析类

| 模板文件 | 功能 | 最新版本 |
|---------|------|---------|
| `reasoning_system_*.j2` | 推理系统提示 | 1.0 |
| `reasoning_user_*.j2` | 推理用户提示 | 1.1 |
| `selection_analysis_*.j2` | 候选分析提示 | 1.0 |
| `output_checking_*.j2` | 输出检查提示 | 1.1 |

### 评估类

| 模板文件 | 功能 | 最新版本 |
|---------|------|---------|
| `evaluation_*.j2` | 评估提示 | 2.1 |

### 其他类

| 模板文件 | 功能 | 最新版本 |
|---------|------|---------|
| `chat_system_*.j2` | 聊天系统提示 | 1.0 |
| `plan_mode_system_*.j2` | 规划模式系统提示 | 1.0 |
| `date_parser_*.j2` | 日期解析提示 | 1.0 |
| `compare_sql_*.j2` | SQL 对比提示 | 1.0 |
| `schema_lineage_*.j2` | 血缘分析提示 | 1.0 |
| `sql_review_*.j2` | SQL 审查提示 | 1.0 |
| `gen_semantic_model_*.j2` | 语义模型生成提示 | 1.0 |
| `gen_metrics_*.j2` | 指标生成提示 | 1.0 |
| `visualization_*.j2` | 可视化提示 | 1.0 |

---

## 常见问题

### Q1: 修改用户模板后需要重启服务吗？

是的。API Server 在启动时会加载模板，修改后需要重启服务使修改生效。

### Q2: 如何回滚到内置模板？

有两种方式：

**方式 1**: 删除用户目录中的模板文件
```bash
rm ~/.datus/template/gen_sql_user_1.0.j2
```

**方式 2**: 使用 `--force` 重新同步
```bash
python -m scripts.update_prompt_templates --config ~/.datus/conf/agent.yml --force
```

### Q3: 如何自定义模板版本？

项目采用文件版本管理，不支持在配置中指定版本号。如需使用特定版本：

1. 复制模板到用户目录：
   ```bash
   python -m scripts.update_prompt_templates --config ~/.datus/conf/agent.yml
   ```

2. 在代码中指定版本：
   ```python
   prompt_manager.render_template("gen_sql_user", version="1.0", question="...")
   ```

### Q4: 模板修改后没有生效？

1. 检查模板路径是否正确：
   ```bash
   python -m scripts.update_prompt_templates --config ~/.datus/conf/agent.yml --list
   ```

2. 确认模板文件名格式正确（`{name}_{version}.j2`）

3. 检查文件权限：
   ```bash
   ls -la ~/.datus/template/
   ```

4. 重启 API Server

### Q5: `{agent.home}` 的默认值是什么？

默认值为 `~/.datus`（用户主目录下的 .datus 文件夹）。可通过 `agent.yml` 中的 `agent.home` 配置自定义路径：

```yaml
agent:
  home: /path/to/custom/datus_home
```

---

## 最佳实践

1. **定期同步**: 当项目更新后，使用 `--list` 检查是否有新版本可用
2. **使用 `--dry-run`**: 在执行同步前，先预览将要执行的更改
3. **备份自定义模板**: 在同步前备份已自定义的模板
4. **版本记录**: 修改模板时记录变更内容和原因
5. **测试验证**: 修改模板后进行功能测试，确保 LLM 输出质量

---

## 相关资源

- **Prompt 模块文档**: [Datus Prompts 模块介绍](./modules/Datus%20Prompts模块介绍.md)
- **项目主页**: [https://datus.ai](https://datus.ai)
- **文档**: [https://docs.datus.ai](https://docs.datus.ai)
- **GitHub**: [https://github.com/Datus-ai/Datus-agent](https://github.com/Datus-ai/Datus-agent)
- **Slack 社区**: [https://join.slack.com/t/datus-ai](https://join.slack.com/t/datus-ai/shared_invite/zt-3g6h4fsdg-iOl5uNoz6A4GOc4xKKWUYg)
