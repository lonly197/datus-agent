# GLM (ZhipuAI) Model Configuration

Datus Agent supports ZhipuAI's GLM (General Language Model) series models, including GLM-4, GLM-4-Plus, GLM-4-Flash, and GLM-4-Air.

## Configuration

To use GLM models, you need to configure the `models` section in your `agent.yml` file.

### Basic Configuration

```yaml
agent:
  models:
    glm-4:
      type: glm
      api_key: ${GLM_API_KEY}
      model: glm-4
    
    glm-4-flash:
      type: glm
      api_key: ${GLM_API_KEY}
      model: glm-4-flash
```

### Parameters

| Parameter | Type | Required | Description | Default |
|-----------|------|----------|-------------|---------|
| `type` | string | Yes | Must be set to `glm` | - |
| `api_key` | string | Yes | Your ZhipuAI API Key. Can be loaded from environment variable `GLM_API_KEY` | - |
| `model` | string | Yes | The specific model name (e.g., `glm-4`, `glm-4-flash`) | - |
| `base_url` | string | No | API Base URL | `https://open.bigmodel.cn/api/paas/v4/` |
| `max_retry` | int | No | Maximum number of retries for API calls | 3 |
| `retry_interval` | float | No | Interval between retries in seconds | 2.0 |

## Environment Variables

You can set the API key using an environment variable to avoid hardcoding it in the configuration file:

```bash
export GLM_API_KEY="your-zhipuai-api-key"
```

## Supported Models

Datus Agent supports the following GLM models:

- **glm-4**: The latest flagship model with strong reasoning and coding capabilities. (128k context)
- **glm-4-plus**: Enhanced version of GLM-4. (128k context)
- **glm-4-flash**: A faster, more cost-effective model suitable for high-volume tasks. (128k context)
- **glm-4-air**: A balanced model for general tasks. (128k context)
- **glm-4-long**: Model optimized for ultra-long context windows (up to 1M tokens).

## Usage Example

Once configured, you can specify the model in your node configurations:

```yaml
nodes:
  generate_sql:
    model: glm-4
    prompt_version: "1.0"
```

## Limitations

- **Rate Limits**: Please refer to ZhipuAI's official documentation for rate limits associated with your API key tier.
- **Token Limits**: While GLM-4 supports 128k context, the output token limit is typically 4096. Ensure your tasks do not require generated responses exceeding this limit.
