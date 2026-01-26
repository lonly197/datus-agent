# GLM（智谱 AI）模型配置

> **文档版本**: v3.0
> **更新日期**: 2026-01-26

Datus Agent 支持智谱 AI 的 GLM 系列大语言模型，采用 OpenAI 兼容的 API 格式。

## 简介

智谱 AI 是国产大模型厂商，提供 GLM 系列文本模型。API 格式兼容 OpenAI，可通过 `type: glm` 直接配置使用。

**官方文档**: [https://docs.bigmodel.cn/cn/guide/start/model-overview](https://docs.bigmodel.cn/cn/guide/start/model-overview)

## 基本配置

在 `agent.yml` 文件的 `models` 部分添加配置：

```yaml
agent:
  target: GLM-4.7  # 指定默认使用的模型

  models:
    # 旗舰模型
    GLM-4.7:
      type: glm
      api_key: ${GLM_API_KEY}
      model: GLM-4.7

    # 轻量高速模型
    GLM-4.7-FlashX:
      type: glm
      api_key: ${GLM_API_KEY}
      model: GLM-4.7-FlashX

    # 超强性能模型
    GLM-4.6:
      type: glm
      api_key: ${GLM_API_KEY}
      model: GLM-4.6

    # 高性价比模型
    GLM-4.5-Air:
      type: glm
      api_key: ${GLM_API_KEY}
      model: GLM-4.5-Air

    # 免费模型
    GLM-4.7-Flash:
      type: glm
      api_key: ${GLM_API_KEY}
      model: GLM-4.7-Flash
```

## 参数说明

| 参数 | 类型 | 必填 | 说明 | 默认值 |
|------|------|------|------|--------|
| `type` | string | 是 | 模型类型，固定为 `glm` | - |
| `api_key` | string | 是 | 智谱 AI API Key | - |
| `model` | string | 是 | 模型名称 | - |
| `base_url` | string | 否 | API 基础 URL | `https://open.bigmodel.cn/api/paas/v4/` |
| `max_retry` | int | 否 | API 调用最大重试次数 | 3 |
| `retry_interval` | float | 否 | 重试间隔（秒） | 2.0 |

## 环境变量配置

```bash
# 方式 1：使用 GLM_API_KEY（推荐）
export GLM_API_KEY="your-zhipuai-api-key"

# 方式 2：使用 ZHIPUAI_API_KEY（兼容旧配置）
export ZHIPUAI_API_KEY="your-zhipuai-api-key"

# 自定义 API 端点（可选）
export GLM_API_BASE="https://your-custom-endpoint.com/api/paas/v4/"
```

## 支持的模型

基于智谱 AI 官方文档（2025年1月）：

### 文本模型

| 模型名称 | 定位 | 特点 | 上下文长度 | 最大输出 |
|---------|------|------|-----------|---------|
| **GLM-4.7** | 高智能旗舰 | 通用对话、推理与智能体能力全面升级；编程更强、更稳、审美更好 | 200K | 128K |
| **GLM-4.7-FlashX** | 轻量高速 | 小尺寸强能力；适用于中文写作、翻译、长文本、情感/角色扮演等通用场景 | 200K | 128K |
| **GLM-4.6** | 超强性能 | 上下文提升至 200K；高级编码能力、强大推理以及工具调用能力 | 200K | 128K |
| **GLM-4.5-Air** | 高性价比 | 在推理、编码和智能体任务上表现强劲 | 128K | 96K |
| **GLM-4.5-AirX** | 高性价比-极速版 | 推理速度快，价格适中；适用于时效性有较强要求的场景 | 128K | 96K |
| **GLM-4-Long** | 超长输入 | 支持高达 1M 上下文长度；专为处理超长文本和记忆型任务设计 | 1M | 4K |
| **GLM-4-FlashX-250414** | 高速低价 | Flash 增强版本；超快推理速度，更高并发保障 | 128K | 16K |
| **GLM-4.7-Flash** | 免费模型 | 最新基座模型的普惠版本 | 200K | 128K |
| **GLM-4-Flash-250414** | 免费模型 | 超长上下文处理能力；多语言支持；支持外部工具调用 | 128K | 16K |

### 即将弃用模型

| 模型 | 弃用时间 |
|------|---------|
| GLM-Z1 系列 | 2025年11月15日 |
| GLM-4-0520 | 2025年12月30日 |

## 使用示例

### 节点配置

```yaml
nodes:
  generate_sql:
    model: GLM-4.7
    prompt_version: "1.0"

  reflect:
    model: GLM-4.7-Flash
    prompt_version: "2.1"
```

### 代码中使用

```python
from datus.models.glm_model import GlmModel
from datus.configuration.agent_config import ModelConfig

config = ModelConfig(
    type="glm",
    api_key="${GLM_API_KEY}",
    model="GLM-4.7"
)

model = GlmModel(config)
response = model.generate("请生成一个查询用户活跃度的 SQL")
```

## 高级功能

### 深度思考模式

部分 GLM 模型支持深度思考模式（Thinking Mode）：

```python
response = model.generate(
    prompt="分析这道数学题",
    enable_thinking=True,  # 启用深度思考模式
    temperature=0.7,
    max_tokens=2000
)
```

### 获取模型规格

```python
from datus.models.glm_model import GlmModel

model = GlmModel(ModelConfig(type="glm", api_key="...", model="GLM-4.7"))

# 获取上下文长度
print(f"上下文长度: {model.context_tokens()}")  # 200000

# 获取最大输出长度
print(f"最大输出: {model.max_output_tokens()}")  # 128000
```

## 注意事项

### 速率限制

请参考 [智谱 AI 官方速率限制文档](https://docs.bigmodel.cn/cn/api/api-code)，根据您的 API Key 等级了解具体限制。

### Token 限制

| 模型系列 | 上下文长度 | 输出限制 |
|---------|-----------|---------|
| GLM-4.7 / GLM-4.6 | 200K | 128K |
| GLM-4.5-Air 系列 | 128K | 96K |
| GLM-4-Long | 1M | 4K |
| GLM-4-Flash 系列 | 128K | 16K |

### 免费模型配额

GLM-4.7-Flash 和 GLM-4-Flash-250414 为免费模型，有每日调用配额限制。

## 错误码参考

| HTTP 状态码 | 说明 | 解决方法 |
|------------|------|---------|
| 200 | 成功 | - |
| 400 | 参数错误 | 检查接口参数 |
| 401 | 鉴权失败 | 确认 API Key 正确 |
| 429 | 并发/频率超额 | 降低请求频率或联系商务 |
| 500 | 服务器错误 | 稍后重试 |

常见业务错误码：

| 错误码 | 说明 |
|-------|------|
| 1002 | Authorization Token 非法 |
| 1211 | 模型不存在 |
| 1301 | 内容安全拦截 |
| 1302 | 并发数过高 |
| 1303 | 请求频率过高 |

## 故障排除

### API Key 未设置

```python
ValueError: GLM API key must be provided in config or set as GLM_API_KEY/ZHIPUAI_API_KEY environment variable
```

**解决**：确保已正确设置 `GLM_API_KEY` 环境变量。

### 速率限制错误 (429)

```bash
# 检查账户余额
# 降低请求频率
# 或联系智谱 AI 商务扩大并发数
```

### 模型不支持

确保使用正确的模型名称（区分大小写）：
- `GLM-4.7` ✓
- `glm-4.7` ✗

## 相关资源

- [智谱 AI 开放平台](https://bigmodel.cn/)
- [模型概览文档](https://docs.bigmodel.cn/cn/guide/start/model-overview)
- [API 错误码文档](https://docs.bigmodel.cn/cn/api/api-code)
- [OpenAI 兼容接口](https://docs.bigmodel.cn/cn/guide/develop/openai/introduction)
- [模型定价](https://bigmodel.cn/pricing)
