# Datus Agent API Server

A FastAPI-based HTTP server that provides REST API access to the Datus Agent functionality.

## Quick Start

### Foreground Mode
```bash
python datus/api/server.py
```

### Daemon Mode
```bash
# Start the server in background
python datus/api/server.py --daemon

# Check server status
python datus/api/server.py --action status

# Stop the server
python datus/api/server.py --action stop

# Restart the server
python datus/api/server.py --action restart
```

## Configuration

### Basic Options
```bash
python datus/api/server.py \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4 \
  --log-level info
```

### Agent Configuration
```bash
python datus/api/server.py \
  --namespace your_namespace \
  --config conf/agent.yml \
  --max_steps 20 \
  --workflow fixed
```

### Daemon Options
```bash
python datus/api/server.py \
  --daemon \
  --pid-file /custom/path/server.pid \
  --daemon-log-file /custom/path/server.log
```

## Default Paths

- **API Endpoint**: `http://localhost:8000`
- **PID File**: `~/.datus/run/datus-agent-api.pid`
- **Log File**: `logs/datus-agent-api.log`

## API Endpoints

Once the server is running, you can access:

- **Interactive Docs**: `http://localhost:8000/docs`
- **Health Check**: `GET /health`
- **Authentication**: `POST /auth/token`
- **Workflow Execution**: `POST /workflows/run`
- **Chat Research**: `POST /workflows/chat_research`
- **Task Management**:
  - `GET /workflows/tasks` - List running and recent tasks
  - `GET /workflows/tasks/{task_id}` - Get task details
  - `DELETE /workflows/tasks/{task_id}` - Cancel a running task
- **Feedback Recording**: `POST /workflows/feedback`

## Plan Executor configuration

You can tune how the server-side plan executor maps todo text to tools and whether a fallback search is attempted. Add an optional `plan_executor` section in your `agent.yml`:

```yaml
plan_executor:
  keyword_tool_map:
    search_table: ["search", "搜索", "表结构", "table", "columns", "列", "字段", "schema"]
    execute_sql: ["execute sql", "执行sql", "执行 sql", "run sql", "执行"]
    report: ["generate report", "生成报告", "write report", "write"]
  enable_fallback: true
```

Notes:
- `keyword_tool_map` maps tool identifiers to a list of keywords. The executor performs a simple substring match (case-insensitive) against the todo content to pick which tool to call.
- `enable_fallback` controls whether the executor will issue a safe `search_table` call when no mapping matches and a DB tool is available.

## SQL Review Tasks

Datus Agent supports specialized SQL review tasks that automatically perform comprehensive code quality analysis. When a task contains review-related keywords (审查, review, 检查, check, 审核, audit), it is automatically identified as an SQL review task and triggers enhanced processing.

### SQL Review Features

- **Automatic Tool Execution**: Pre-executes essential tools (describe_table, search_external_knowledge, read_query, get_table_ddl) before analysis
- **Comprehensive Analysis**: Checks StarRocks SQL compliance, performance optimization, and data consistency
- **Structured Reports**: Generates markdown-formatted review reports with specific sections
- **Fail-Safe Operation**: Continues analysis even if some tools fail, with appropriate warnings

### SQL Review Configuration

Add SQL review specific configuration to your `agent.yml`:

```yaml
plan_executor:
  sql_review_preflight:
    # Enable preflight tool execution (default: true)
    enabled: true
    # Default tool sequence for reviews
    default_tool_sequence:
      - describe_table           # Table structure analysis
      - search_external_knowledge # StarRocks rules retrieval
      - read_query              # SQL validation execution
      - get_table_ddl           # DDL definition retrieval
    # Cache settings
    cache_enabled: true
    cache_ttl_seconds: 300
    # Tool execution timeout
    tool_timeout_seconds: 30
    # Continue on partial failures
    continue_on_failure: true
```

### SQL Review API Usage

```bash
curl --location --request POST 'http://localhost:8000/workflows/chat_research' \
--header 'Accept: text/event-stream' \
--header 'Content-Type: application/json' \
--data-raw '{
    "namespace": "test",
    "catalog_name": "default_catalog",
    "database_name": "test",
    "task": "审查SQL：SELECT * FROM users WHERE created_at > '\''2024-01-01'\''",
    "ext_knowledge": "使用StarRocks 3.3规范进行检查"
  }'
```

### SQL Review Report Format

The system generates structured markdown reports with the following sections:

- **📋 审查概览**: Summary with pass/fail status
- **🔍 审查规则**: Applied review standards
- **📊 审查分析**: Detailed analysis by category
- **⚠️ 发现问题**: Issues found with severity levels
- **💡 优化建议**: Specific improvement recommendations
- **🛠️ 优化后的SQL**: Optimized SQL code when applicable
- **📈 预期效果**: Expected performance improvements

### Monitoring SQL Review Execution

SQL review tasks include comprehensive monitoring:

- **Preflight Tool Tracking**: Records execution time and success/failure for each tool
- **Cache Statistics**: Shows cache hit rates for repeated operations
- **Fail-Safe Annotations**: Warns when tools fail and analysis is based on incomplete data
- **Performance Metrics**: Tool execution times and overall review duration

## Usage Examples

### Health Check
```bash
curl http://localhost:8000/health
```

### Get Access Token
```bash
curl -X POST "http://localhost:8000/auth/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "grant_type=client_credentials&client_id=your_id&client_secret=your_secret"
```

### Run Workflow (Synchronous)
```bash
curl -X POST "http://localhost:8000/workflows/run" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Find all users from database",
    "namespace": "your_namespace",
    "workflow": "workflow_name",
    "mode": "sync"
  }'
```

### Run Workflow (Streaming)
```bash
curl -X POST "http://localhost:8000/workflows/run" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
    "task": "Find all users from database",
    "namespace": "your_namespace",
    "workflow": "workflow_name",
    "mode": "async"
  }'
```

### Record Feedback
```bash
curl -X POST "http://localhost:8000/workflows/feedback" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "your_task_id",
    "status": "success"
  }'
```

### List Tasks
```bash
curl -X GET "http://localhost:8000/workflows/tasks" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### Get Task Details
```bash
curl -X GET "http://localhost:8000/workflows/tasks/your_task_id" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### Cancel Task
```bash
curl -X DELETE "http://localhost:8000/workflows/tasks/your_task_id" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

## Development Mode

For development with auto-reload:
```bash
python datus/api/server.py --reload
```

Note: `--daemon` and `--reload` are mutually exclusive.

## Troubleshooting

### Check if server is running
```bash
python datus/api/server.py --action status
```

### View logs
```bash
tail -f logs/datus-agent-api.log
```

### Force stop
```bash
pkill -f "datus/api/server.py"
```