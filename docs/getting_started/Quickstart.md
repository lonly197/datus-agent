# Quickstart Guide

> **文档版本**: v2.0
> **更新日期**: 2026-01-26

This guide introduces three supported usage **modes** that allow you to adopt Datus based on your needs and environment.

---

## 1. Datus CLI (Chat Mode)

Use Datus like a chatbot: type natural language questions, get SQL or summaries back. Ideal for **ad hoc queries** and **fast metric exploration**.

### Installation

```bash
# Install from PyPI
pip install datus-agent

# Or install development version
pip install git+https://github.com/Datus-ai/Datus-agent.git
```

### Quick Initialization

Use the built-in initialization command to set up the configuration:

```bash
# Interactive initialization (recommended)
datus init

# Or manually copy the template configuration
mkdir -p ~/.datus/conf
cp /path/to/datus-agent/conf/agent.yml.qs ~/.datus/conf/agent.yml
```

### Set Environment Variables

```bash
# Default model is DeepSeek-v3
export DEEPSEEK_API_KEY="your-api-key-here"

# Optional: For other models
export KIMI_API_KEY="your-kimi-api-key"
```

### Launch the CLI

```bash
# Using installed script
datus-cli --namespace local_duckdb

# Or using python module
python -m datus.cli.main --namespace local_duckdb
```

### Start Chatting

```bash
Datus> /list all tables
```

**Example output:**

```
Datus> /list all tables
Processing chat request...
# All Tables in the Database

Here is the complete list of all tables available in the main schema:

## Tables List

| Table Name | Type |
|------------|------|
| bank_failures | table |
| boxplot | table |
| calendar | table |
...

Total Tables: 17

The database contains a diverse set of tables covering various topics including:
- Financial data (bank_failures, gold_vs_bitcoin)
- Demographic data (japan_population, niger_population)
...

Would you like me to explore any specific table in more detail?
Datus>
```

For more command references, see the [CLI documentation](../cli/introduction.md).

---

## 2. Datus Benchmark (Docker Mode)

Run benchmark tests in a pre-configured Docker image to evaluate Datus using standard benchmark datasets: Bird and Spider-snow.

### Pull Image

```bash
docker pull luochen2025/datus-agent
```

### Start Container

```bash
docker run --name datus \
  --env DEEPSEEK_API_KEY=<your_api_key> \
  --env SNOWFLAKE_ACCOUNT=<your_snowflake_account> \
  --env SNOWFLAKE_USERNAME=<your_snowflake_username> \
  --env SNOWFLAKE_PASSWORD=<your_snowflake_password> \
  -d luochen2025/datus-agent
```

### Running Benchmarks

**Note:** Before running benchmarks, you need to download and set up the benchmark databases. Refer to `conf/agent.yml.example` for configuration details.

**Bird Benchmark:**

```bash
# Run specific task by ID
docker exec -it datus python -m datus.main benchmark \
  --namespace bird_sqlite \
  --benchmark bird_dev \
  --benchmark_task_ids 14

# Run all tasks
docker exec -it datus python -m datus.main benchmark \
  --namespace bird_sqlite \
  --benchmark bird_dev
```

**Spider-snow Benchmark:**

```bash
# Run specific task by ID
docker exec -it datus python -m datus.main benchmark \
  --namespace snowflake \
  --benchmark spider2 \
  --benchmark_task_ids sf_bq104

# Run all tasks
docker exec -it datus python -m datus.main benchmark \
  --namespace snowflake \
  --benchmark spider2
```

For more detailed information about Datus benchmarking, see [Benchmark Manual](../benchmark/benchmark_manual.md).

---

## 3. Datus Metric (MetricFlow Integration)

Connect Datus to **MetricFlow** and a data warehouse (e.g., StarRocks) to enable **semantic understanding of metrics** — with support for model-based reasoning, date interpretation, and domain code mapping.

### Prerequisites

- **Datus Agent** already installed
- **MetricFlow CLI**: Requires Python 3.9 (separate venv via Poetry)

### Install MCP Server

```bash
# Find Datus installation path
pip show datus-agent

cd <datus_install_path>/mcp/mcp-metricflow-server
uv sync
```

### Set Up MetricFlow

Create a dedicated directory for your MetricFlow project:

```bash
mkdir -p ~/mf
cd ~/mf
```

> **Note:** You'll reference this directory later as `MF_PROJECT_DIR`.

Clone and set up MetricFlow:

```bash
git clone https://github.com/Datus-ai/metricflow.git
cd metricflow
poetry lock
poetry config virtualenvs.in-project true
poetry install
source .venv/bin/activate
```

Verify installation:

```bash
mf setup
mf tutorial
mf validate-configs
```

### Configure MetricFlow for Datus Demo

Edit `~/.metricflow/config.yml`:

```yaml
model_path: ~/mf/metricflow/semantic_models
email: ''  # Optional
dwh_schema: demo
dwh_dialect: duckdb
dwh_database: ~/.datus/demo/demo.duckdb
```

### Configure Datus

1. Install the filesystem MCP server:

```bash
npm install -g @modelcontextprotocol/server-filesystem
```

2. Add metrics configuration to `~/.datus/conf/agent.yml`:

```yaml
metrics:
  demo:
    subject_path: economic/bank/bank_failures
```

3. Set environment variables:

```bash
# MetricFlow MCP server path + CLI
export METRICFLOW_MCP_DIR=~/mf/metricflow
export MF_PATH=~/mf/metricflow/.venv/bin/mf
export MF_PROJECT_DIR=~/mf/metricflow
export MF_VERBOSE=true

# Filesystem MCP server root for semantic models
mkdir -p ~/mf/metricflow/semantic_models
export FILESYSTEM_MCP_DIRECTORY=~/mf/metricflow/semantic_models
```

### Generate Metrics

1. Start the Datus CLI:

```bash
datus-cli --namespace local_duckdb
```

2. Ask a natural language question:

```bash
Datus> /which state has the highest total asset value of failure bank?
```

3. Run the generated SQL:

```bash
Datus> SELECT
    State,
    SUM("Assets ($mil.)") as Total_Assets_Millions,
    COUNT(*) as Number_of_Failures
FROM demo.main.bank_failures
GROUP BY State
ORDER BY Total_Assets_Millions DESC
LIMIT 1
```

4. Generate metrics:

```bash
Datus> !gen_metrics
```

5. View generated metric definitions:

```bash
cd ~/mf/metricflow/semantic_models
less bank_failures.yaml
```

**Generated YAML example:**

```yaml
data_source:
  name: bank_failures
  description: Bank failures data with state and asset information
  sql_table: demo.main.bank_failures

  measures:
    - name: total_assets_millions
      agg: SUM
      expr: "Assets ($mil.)"
      create_metric: true

    - name: number_of_failures
      agg: COUNT
      expr: "1"
      create_metric: true

  dimensions:
    - name: state
      type: CATEGORICAL
      expr: State

  identifiers:
    - name: bank_failure
      type: PRIMARY
      expr: "CONCAT(State, '-', "Bank Name")"

  mutability:
    type: APPEND_ONLY

---
metric:
  name: state_failure_count_highest_assets
  description: Number of failures in the state with highest total assets
  type: measure_proxy
  type_params:
    measure: number_of_failures
  constraint: "{{ Dimension('state__state') }} = (
    SELECT State
    FROM demo.main.bank_failures
    GROUP BY State
    ORDER BY SUM("Assets ($mil.)") DESC
    LIMIT 1
  )"
  locked_metadata:
    display_name: "Failure Count in Highest Asset State"
    value_format: ",.0f"
    unit: "failures"
    tags:
      - "Banking"
      - "Risk Analysis"
```

---

## Quick Reference

| Mode | Command | Use Case |
|------|---------|----------|
| CLI Chat | `datus-cli --namespace <name>` | Interactive SQL exploration |
| API Server | `datus-agent --host 0.0.0.0 --port 8000` | REST API with SSE streaming |
| Benchmark | `python -m datus.main benchmark ...` | Model evaluation |

### Common Commands

```bash
# CLI with web interface
datus-cli --namespace local_duckdb --web --port 8501

# With custom configuration
datus-cli --namespace local_duckdb --config /path/to/agent.yml

# Enable debug logging
datus-cli --namespace local_duckdb --debug
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `DEEPSEEK_API_KEY` | DeepSeek API key |
| `KIMI_API_KEY` | Kimi (Moonshot) API key |
| `OPENAI_API_KEY` | OpenAI API key |
| `DATUS_CONFIG` | Path to agent.yml configuration |

---

## Next Steps

- [CLI Documentation](../cli/introduction.md)
- [Configuration Guide](../configuration/introduction.md)
- [Benchmark Manual](../benchmark/benchmark_manual.md)
- [GitHub Repository](https://github.com/Datus-ai/Datus-agent)
