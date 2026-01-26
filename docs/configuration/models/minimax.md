# MiniMax 模型配置

> **文档版本**: v1.0
> **更新日期**: 2026-01-26

Datus Agent 支持 MiniMax 的文本模型系列，包括 MiniMax-M2.1、MiniMax-M2.1-lightning、MiniMax-M2 等。

## 简介

MiniMax 是国产大模型厂商，提供 OpenAI 兼容的 API 接口。Datus 通过 `MiniMaxModel` 类集成支持。

## 基本配置

在 `agent.yml` 文件的 `models` 部分添加以下配置：

```yaml
agent:
  target: MiniMax-M2.1  # 指定默认使用的模型

  models:
    MiniMax-M2.1:
      type: minimax
      api_key: ${MINIMAX_API_KEY}
      model: MiniMax-M2.1

    MiniMax-M2.1-lightning:
      type: minimax
      api_key: ${MINIMAX_API_KEY}
      model: MiniMax-M2.1-lightning

    MiniMax-M2:
      type: minimax
      api_key: ${MINIMAX_API_KEY}
      model: MiniMax-M2
```

## 参数说明

| 参数 | 类型 | 必填 | 说明 | 默认值 |
|------|------|------|------|--------|
| `type` | string | 是 | 模型类型，固定为 `minimax` | - |
| `api_key` | string | 是 | MiniMax API Key | - |
| `model` | string | 是 | 模型名称 | - |
| `base_url` | string | 否 | API 基础 URL | `https://api.minimax.chat/v1/text/chatcompletion_v2` |
| `max_retry` | int | 否 | API 调用最大重试次数 | 3 |
| `retry_interval` | float | 否 | 重试间隔（秒） | 2.0 |

## 环境变量配置

```bash
# 设置 API Key
export MINIMAX_API_KEY="your-minimax-api-key"
```

## 支持的模型

根据 MiniMax 官方文档支持的文本模型：

| 模型名称 | 说明 | 上下文长度 | 输出 Token |
|---------|------|-----------|-----------|
| **MiniMax-M2.1** | 旗舰模型，多语言编程 SOTA，为真实世界复杂任务而生 | 1M (1048576) | 65536 |
| **MiniMax-M2.1-lightning** | 与 M2.1 同等效果，速度大幅提升 | 1M (1048576) | 65536 |
| **MiniMax-M2** | 专为高效编码与 Agent 工作流而生 | 1M (1048576) | 65536 |
| **MiniMax-M2-her** | 文本对话模型，专为角色扮演、多轮对话等场景设计 | 1M (1048576) | 65536 |
| **MiniMax-Text-01** | 早期文本模型（向后兼容） | 1M (1048576) | 65536 |

## 使用示例

### 节点配置

```yaml
nodes:
  generate_sql:
    model: MiniMax-M2.1
    prompt_version: "1.0"

  reflect:
    model: MiniMax-M2.1-lightning
    prompt_version: "2.1"
```

### 代码中使用

```python
from datus.models.minimax_model import MiniMaxModel
from datus.configuration.agent_config import ModelConfig

config = ModelConfig(
    type="minimax",
    api_key="${MINIMAX_API_KEY}",
    model="MiniMax-M2.1"
)

model = MiniMaxModel(config)
response = model.generate("请生成一个查询用户活跃度的 SQL")
```

## API 特性

### OpenAI 兼容

MiniMax 提供 OpenAI 兼容的 API，可以直接使用 OpenAI 客户端调用：

```python
from openai import OpenAI

client = OpenAI(
    api_key="your-minimax-api-key",
    base_url="https://api.minimax.chat/v1/text/chatcompletion_v2"
)

response = client.chat.completions.create(
    model="MiniMax-M2.1",
    messages=[
        {"role": "system", "content": "你是一个 SQL 专家"},
        {"role": "user", "content": "生成一个查询所有用户的 SQL"}
    ]
)
```

### 支持的参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `temperature` | float | 0.7 | 温度参数 |
| `max_tokens` | int | 1000 | 最大生成 token 数 |
| `top_p` | float | 1.0 | Top-p 采样参数 |
| `stream` | bool | false | 是否启用流式输出 |

## 注意事项

### 速率限制

请参考 [MiniMax 官方文档](https://platform.minimaxi.com/docs/guides/rate-limits) 了解速率限制。

### Token 限制

- MiniMax-M2 系列支持最高 1M 上下文长度
- 输出 token 上限为 65536
- 建议根据实际需求设置 `max_tokens` 参数

### 代理设置

如需通过代理访问：

```bash
export HTTPS_PROXY="http://your-proxy:port"
```

## 故障排除

### API Key 未设置

```
ValueError: MiniMax API key not found in config or MINIMAX_API_KEY env var
```

**解决**：确保已设置 `MINIMAX_API_KEY` 环境变量。

### 连接失败

检查网络连接和代理设置：

```bash
# 测试 API 连通性
curl -H "Authorization: Bearer $MINIMAX_API_KEY" \
  https://api.minimax.chat/v1/text/chatcompletion_v2 \
  -d '{"model":"MiniMax-M2.1","messages":[{"role":"user","content":"hi"}]}'
```

### 模型名称错误

确保使用正确的模型名称（区分大小写）：
- `MiniMax-M2.1` ✓
- `minimax-m2.1` ✗

## 相关资源

- [MiniMax 开放平台](https://platform.minimaxi.com/)
- [MiniMax API 文档](https://platform.minimaxi.com/docs/guides/models-intro)
- [MiniMax 模型定价](https://platform.minimaxi.com/docs/guides/pricing)
- [Datus GitHub](https://github.com/Datus-ai/Datus-agent)
