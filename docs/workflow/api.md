# Workflow API

## Introduction

The Datus Agent Workflow API Service is a RESTful API that exposes the power of Datus Agent's natural language to SQL capabilities through HTTP endpoints. This service enables applications to integrate intelligent SQL generation, execution, and workflow management into their systems.

## Quick Start

### Starting the Service

```bash
# Start the API service
python -m datus.api.server --host 0.0.0.0 --port 8000

# Start with multiple workers
python -m datus.api.server --workers 4 --port 8000

# Start in daemon mode (background)
python -m datus.api.server --daemon --port 8000
```

## Authentication

### OAuth2 Client Credentials Flow

The API uses OAuth2 client credentials authentication:

#### 1. Get Authentication Token

```bash
curl -X POST "http://localhost:8000/auth/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "client_id=your_client_id&client_secret=your_client_secret&grant_type=client_credentials"
```

#### 2. Use Token in Requests

```bash
curl -X POST "http://localhost:8000/workflows/run" \
  -H "Authorization: Bearer your_jwt_token" \
  -H "Content-Type: application/json" \
  -d '{"workflow": "fixed", "namespace": "your_db", "task": "Show me users"}'
```

### Configuration

Create `auth_clients.yml` to configure clients:

```yaml
clients:
  your_client_id: your_client_secret
  another_client: another_secret

jwt:
  secret_key: your-jwt-secret-key-change-in-production
  algorithm: HS256
  expiration_hours: 2
```

## API Endpoints

### Authentication

#### POST /auth/token

Obtain JWT access token.

**Request:**
```http
POST /auth/token
Content-Type: application/x-www-form-urlencoded

client_id=your_client_id&client_secret=your_client_secret&grant_type=client_credentials
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "Bearer",
  "expires_in": 7200
}
```

### Workflow Execution

#### POST /workflows/run

Execute a workflow to convert natural language to SQL.

**Common Request Parameters:**

| Parameter | Type     | Required | Description |
|-----------|----------|----------|-------------|
| `workflow` | string   | ✅ | Workflow name (text2sql, reflection, fixed, metric_to_sql) |
| `namespace` | string   | ✅ | Database namespace |
| `task` | string   | ✅ | Natural language task description |
| `mode` | string   | ✅ | Execution mode (sync or async) |
| `task_id` | string   | ❌ | Custom task ID for idempotency |
| `catalog_name` | string   | ❌ | Database catalog |
| `database_name` | string   | ❌ | Database name |
| `schema_name` | string   | ❌ | Schema name |
| `current_date` | string   | ❌ | Reference date for time expressions |
| `subject_path` | string[] | ❌ | Business domain |
| `ext_knowledge` | string   | ❌ | Additional business context |

#### Synchronous Mode (mode: "sync")

**Request Headers:**
```
Authorization: Bearer your_jwt_token
Content-Type: application/json
```

**Request Body:**
```json
{
  "workflow": "text2sql",
  "namespace": "your_database_namespace",
  "task": "Show me monthly revenue by product category",
  "mode": "sync",
  "catalog_name": "your_catalog",
  "database_name": "your_database"
}
```

**Response:**
```json
{
  "task_id": "client_20240115143000",
  "status": "completed",
  "workflow": "text2sql",
  "sql": "SELECT DATE_TRUNC('month', order_date) as month, product_category, SUM(amount) as revenue FROM orders WHERE order_date >= '2023-01-01' GROUP BY month, product_category ORDER BY month, revenue DESC",
  "result": [
    {
      "month": "2023-01-01",
      "product_category": "Electronics",
      "revenue": 150000.00
    },
    {
      "month": "2023-01-01",
      "product_category": "Clothing",
      "revenue": 85000.00
    }
  ],
  "metadata": {
    "execution_time": 12.5,
    "nodes_executed": 5,
    "reflection_rounds": 0
  },
  "error": null,
  "execution_time": 12.5
}
```

#### Asynchronous Mode (mode: "async")

**Request Headers:**
```
Authorization: Bearer your_jwt_token
Content-Type: application/json
Accept: text/event-stream
Cache-Control: no-cache
```

**Request Body:**
```json
{
  "workflow": "text2sql",
  "namespace": "your_database_namespace",
  "task": "Show me monthly revenue by product category",
  "mode": "async",
  "catalog_name": "your_catalog",
  "database_name": "your_database"
}
```

**Response (Server-Sent Events stream):**
```
Content-Type: text/event-stream

event: started
data: {"task_id": "client_20240115143000", "workflow": "text2sql"}

event: progress
data: {"message": "Initializing workflow", "progress": 10}

event: node_progress
data: {"node": "schema_linking", "status": "processing", "progress": 25}

event: node_detail
data: {"node": "schema_linking", "description": "Analyzing user query and finding relevant tables", "details": {"tables_found": ["orders", "products"]}}

event: sql_generated
data: {"sql": "SELECT DATE_TRUNC('month', order_date) as month, product_category, SUM(amount) as revenue FROM orders GROUP BY month, product_category"}

event: execution_complete
data: {"status": "success", "rows_affected": 24, "execution_time": 2.1}

event: output_ready
data: {"result": [...], "metadata": {...}}

event: done
data: {"task_id": "client_20240115143000", "status": "completed", "total_time": 15.2}
```

#### POST /workflows/chat_research

Execute a chat research workflow with structured DeepResearchEvent streaming for chatbot applications.

This endpoint provides a specialized interface for deep research tasks with real-time event streaming that matches chatbot UI expectations. It uses the `chat_agentic` workflow internally with plan-mode enabled by default.

**Request Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `namespace` | string | ✅ | Database namespace |
| `task` | string | ✅ | Natural language task description |
| `catalog_name` | string | ❌ | Database catalog |
| `database_name` | string | ❌ | Database name |
| `schema_name` | string | ❌ | Schema name |
| `current_date` | string | ❌ | Reference date for time expressions |
| `domain` | string | ❌ | Business domain |
| `layer1` | string | ❌ | Business layer 1 |
| `layer2` | string | ❌ | Business layer 2 |
| `ext_knowledge` | string | ❌ | Additional business context |
| `plan_mode` | boolean | ❌ | Enable structured plan execution (default: false) |
| `prompt` | string | ❌ | Role definition and task capability prompt to guide the AI agent |
| `prompt_mode` | string | ❌ | How to merge prompt with system prompt: 'replace' or 'append' (default: 'append') |

**Request Headers:**
```
Authorization: Bearer your_jwt_token
Content-Type: application/json
Accept: text/event-stream
Cache-Control: no-cache
```

**Request Body:**
```json
{
  "namespace": "your_database_namespace",
  "task": "从 ODS 试驾表和线索表关联，统计每个月'首次试驾'到'下定'的平均转化周期（天数）。",
  "catalog_name": "your_catalog",
  "database_name": "your_database",
  "plan_mode": true,
  "prompt": "你是【数仓开发助手】。请根据业务逻辑生成高质量的 StarRocks SQL。",
  "prompt_mode": "append"
}
```

**Example with prompt replacement:**
```json
{
  "namespace": "your_database_namespace",
  "task": "分析用户转化周期数据",
  "catalog_name": "your_catalog",
  "database_name": "your_database",
  "prompt": "你是专门负责电商数据分析的AI助手。你需要提供精确的SQL查询和详细的业务分析。请始终使用最优的查询性能优化方案。",
  "prompt_mode": "replace"
}
```

**Response (Server-Sent Events stream):**
```
Content-Type: text/event-stream

data: {"id":"evt_1","planId":"plan_abc123","timestamp":1703123456789,"event":"chat","content":"开始分析您的数据查询需求..."}

data: {"id":"evt_2","planId":"plan_abc123","timestamp":1703123456790,"event":"plan_update","todos":[{"id":"task_1","content":"分析试驾表和线索表结构","status":"in_progress"},{"id":"task_2","content":"生成关联查询SQL","status":"pending"}]}

data: {"id":"evt_3","planId":"plan_abc123","timestamp":1703123456791,"event":"tool_call","toolCallId":"call_456","toolName":"schema_linking","input":{"table":"trial_drive"}}

data: {"id":"evt_4","planId":"plan_abc123","timestamp":1703123456792,"event":"tool_call_result","toolCallId":"call_456","data":{"columns":[{"name":"user_id","type":"string"},{"name":"trial_date","type":"date"}]},"error":false}

data: {"id":"evt_5","planId":"plan_abc123","timestamp":1703123456793,"event":"chat","content":"正在生成SQL查询..."}

data: {"id":"evt_6","planId":"plan_abc123","timestamp":1703123456794,"event":"tool_call","toolCallId":"call_789","toolName":"sql_generation","input":{"query":"转化周期分析"}}

data: {"id":"evt_7","planId":"plan_abc123","timestamp":1703123456795,"event":"tool_call_result","toolCallId":"call_789","data":{"sql":"SELECT DATE_FORMAT(t.trial_date, '%Y-%m') as month, AVG(DATEDIFF(c.order_date, t.trial_date)) as avg_conversion_days FROM trial_drive t JOIN leads c ON t.user_id = c.user_id WHERE t.trial_type = '首次试驾' AND c.status = '下定' GROUP BY month ORDER BY month"},"error":false}

data: {"id":"evt_8","planId":"plan_abc123","timestamp":1703123456796,"event":"complete","content":"SQL生成完成"}

data: {"id":"evt_9","planId":"plan_abc123","timestamp":1703123456797,"event":"report","url":"http://localhost:8000/reports/plan_abc123","data":"<html><body><h1>转化周期分析报告</h1>...</body></html>"}
```

**DeepResearchEvent Types:**

| Event Type | Description | Key Fields |
|------------|-------------|------------|
| `chat` | Chat messages and assistant responses | `content` (string) |
| `plan_update` | Plan execution status updates | `todos` (TodoItem[]) |
| `tool_call` | Tool/function execution start | `toolCallId`, `toolName`, `input` |
| `tool_call_result` | Tool execution completion | `toolCallId`, `data`, `error` |
| `complete` | Task completion | `content` (optional) |
| `report` | Generated report | `url`, `data` (HTML content) |
| `error` | Error occurrence | `error` (error message) |

**TodoItem Structure:**
```json
{
  "id": "task_1",
  "content": "Analyze schema structure",
  "status": "pending|in_progress|completed"
}
```

**Client Implementation Example (JavaScript):**

```javascript
// Connect to SSE stream
const eventSource = new EventSource('/workflows/chat_research', {
  headers: {
    'Authorization': 'Bearer your_jwt_token',
    'Content-Type': 'application/json'
  }
});

// Handle different event types
eventSource.onmessage = (event) => {
  const data = JSON.parse(event.data);

  switch(data.event) {
    case 'chat':
      displayChatMessage(data.content);
      break;

    case 'plan_update':
      updatePlanDisplay(data.todos);
      break;

    case 'tool_call':
      showToolExecution(data.toolName, data.toolCallId);
      break;

    case 'tool_call_result':
      updateToolResult(data.toolCallId, data.data, data.error);
      break;

    case 'complete':
      markTaskComplete(data.content);
      break;

    case 'report':
      displayReport(data.url, data.data);
      break;

    case 'error':
      showError(data.error);
      break;
  }
};

// Send research request with custom prompt
fetch('/workflows/chat_research', {
  method: 'POST',
  headers: {
    'Authorization': 'Bearer your_jwt_token',
    'Content-Type': 'application/json',
    'Accept': 'text/event-stream'
  },
  body: JSON.stringify({
    namespace: 'your_namespace',
    task: 'Generate SQL for monthly conversion analysis',
    plan_mode: true,
    prompt: 'You are a senior data warehouse developer specializing in StarRocks SQL optimization.',
    prompt_mode: 'append'
  })
});
```

### Feedback Submission

#### POST /workflows/feedback

Submit feedback on workflow execution quality.

**Request:**
```json
{
  "task_id": "client_20240115143000",
  "status": "success"
}
```

**Response:**
```json
{
  "task_id": "client_20240115143000",
  "acknowledged": true,
  "recorded_at": "2024-01-15T14:30:15Z"
}
```

## Workflow Types

### reflection
**Intelligent, self-improving SQL generation:**
- Includes reflection for error correction
- Can adapt and retry queries
- Best for complex or uncertain queries

### fixed
**Deterministic SQL generation:**
- Predictable execution path
- No adaptive behavior
- Best for well-understood queries

### metric_to_sql
**Generate SQL from business metrics:**
- Leverages predefined business metrics
- Includes date parsing for temporal queries
- Best for standardized business intelligence

### chat_agentic
**Interactive chat-based SQL generation with tool support:**
- Uses conversational AI with database and filesystem tools
- Supports plan-mode for structured multi-step execution
- Includes real-time streaming of thoughts and tool calls
- Best for complex research tasks and chatbot integrations
- Accessible via `/workflows/chat_research` endpoint for structured event streaming

## Configuration

### Server Configuration

```bash
python -m datus.api.server \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4 \
  --reload \
  --debug
```

### Server Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--host` | Server host address | `127.0.0.1` |
| `--port` | Server port | `8000` |
| `--workers` | Number of worker processes | `1` |
| `--reload` | Auto-reload on code changes | `False` |
| `--debug` | Enable debug mode | `False` |
| `--daemon` | Run in background | `False` |

## Best Practices

### Security
- Use strong, unique JWT secret keys in production
- Rotate client credentials regularly
- Implement rate limiting for production deployments
- Use HTTPS in production environments
- **Prompt Parameter Security**: When using the `prompt` parameter, avoid injecting sensitive information or instructions that could compromise security. The prompt is merged with system instructions and may be logged for debugging purposes.

### Performance
- Use async mode for long-running queries
- Configure appropriate worker count based on expected load
- Monitor memory usage with multiple workers
- Implement client-side timeouts for sync requests

### Error Handling
- Always check response status codes
- Implement retry logic for transient failures
- Handle streaming disconnections in async mode
- Log detailed error information for debugging

## Conclusion

The Datus Agent Workflow API Service provides a powerful, flexible interface for integrating natural language to SQL capabilities into your applications. With support for multiple execution modes, real-time progress streaming, and comprehensive authentication, it enables developers to build intelligent data analysis applications.