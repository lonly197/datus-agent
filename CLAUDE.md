## Overview

**Datus** is an open-source data engineering agent that builds evolvable context for your data system.

- **Datus-CLI**: AI-powered CLI for data engineers
- **Datus-Chat**: Web chatbot with feedback for analysts
- **Datus-API**: APIs for stable data services

### Recent Updates

Text2SQL pipeline hardening with:
- Evidence-Driven Generation (preflight orchestration)
- Robust Error Handling
- Intelligent Caching (QueryCache integration)
- Configuration Validation
- Enhanced Observability

---

## Development Setup

### Installation

```bash
# Development mode (recommended)
make setup-dev
uv pip install -e ".[dev]"

# Build from source
make build && make install-dist

# Install from PyPI
pip install datus-agent
```

### Project Structure

```
/project/
├── datus/              # Core Python package
├── tests/              # Test suite (pytest)
├── conf/               # Configuration templates
├── sample_data/        # Demo datasets
├── docs/               # Documentation
└── .github/workflows/  # CI/CD pipelines
```

### Coding Standards

- **Python**: >= 3.11 (tested on 3.12)
- **Line Length**: 120 characters (Black)
- **Indentation**: 4 spaces
- **Naming**: snake_case (funcs), PascalCase (classes), UPPER_CASE (constants)
- **File Size**: Keep under 500 lines

### Type Safety Pattern

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datus.schemas.action_history import ActionHistory

try:
    from datus.schemas.action_history import ActionHistory
except ImportError:
    ActionHistory = Any  # Runtime fallback
```

---

## Architecture

### Core Components

```
datus/
├── agent/       # Workflow orchestration, 25+ node types
├── api/         # FastAPI REST server
├── cli/         # Command-line interface
├── configuration/ # Config management
├── schemas/     # Data models
├── storage/     # LanceDB, Tantivy
├── utils/       # Utilities
├── prompts/     # LLM templates
└── models/      # ML embeddings
```

### Key Patterns

1. **Node-Based Workflows**: Discrete nodes process data sequentially
2. **Two-Stage Intent Processing**: Heuristic detection + LLM clarification
3. **Schema-First Approach**: Discovery before SQL generation
4. **Preflight Orchestration**: Mandatory preflight tools
5. **Streaming Architecture**: Async/await with SSE

### Database Connectors

See [Datus-adapters](https://github.com/Datus-ai/Datus-adapters):
- Snowflake, StarRocks, MySQL, DuckDB, SQLite

---

## SPARC Methodology

Uses SPARC (Specification, Pseudocode, Architecture, Refinement, Completion) with Claude-Flow.

| Command | Description |
|---------|-------------|
| `npx claude-flow sparc modes` | List available modes |
| `npx claude-flow sparc run <mode> "<task>"` | Execute specific mode |
| `npx claude-flow sparc tdd "<task>"` | TDD workflow |
| `npx claude-flow sparc batch <modes> "<task>"` | Batch parallel |
| `npx claude-flow sparc pipeline "<task>"` | Full pipeline |

**Phases**: Specification → Pseudocode → Architecture → Refinement → Completion

---

## Workflows

### Text2SQL Flow

```yaml
text2sql:
  - intent_analysis       # Fast heuristic detection
  - intent_clarification  # LLM-based clarification
  - schema_discovery      # Schema retrieval
  - schema_validation     # Schema validation
  - generate_sql          # SQL generation
  - execute_sql          # SQL execution
  - result_validation    # Result validation
  - reflect              # Reflection/retry
  - output               # Output generation
```

### Preflight Tools (Mandatory)

- `search_table` - Find relevant tables
- `describe_table` - Get table schemas
- `search_reference_sql` - Find similar queries
- `parse_temporal_expressions` - Parse time expressions

---

## Critical Rules

### Concurrent Execution

1. **ALL operations MUST be concurrent/parallel** in a single message
2. **NEVER save files to root folder** - organize in subdirectories
3. **USE CLAUDE CODE'S TASK TOOL** for spawning agents

**Mandatory Patterns:**
- **TodoWrite**: Batch ALL todos in ONE call (5-10+ minimum)
- **Task tool**: Spawn ALL agents in ONE message
- **File operations**: Batch ALL reads/writes/edits
- **Bash**: Batch ALL terminal operations

### Claude Code vs MCP Tools

| Handles | Claude Code | MCP Tools |
|---------|-------------|-----------|
| Execution | Task tool, file ops, code, bash | Swarm init |
| Strategy | Implementation work | Agent definitions |
| Navigation | Project analysis | Memory, neural |

**Key**: MCP coordinates strategy, Claude Code executes.

---

## Agents (54 Total)

### Core Development
`coder`, `reviewer`, `tester`, `planner`, `researcher`

### Specialized
`backend-dev`, `mobile-dev`, `ml-developer`, `cicd-engineer`, `api-docs`, `system-architect`, `code-analyzer`, `base-template-generator`

### SPARC
`sparc-coord`, `sparc-coder`, `specification`, `pseudocode`, `architecture`, `refinement`

### Testing
`tdd-london-swarm`, `production-validator`

---

## Implementation Patterns

### LLM JSON Parsing

```python
from datus.utils.json_utils import llm_result2json

result = llm_result2json(response_text, expected_type=dict)
if result is None:
    return default_value
```

**DO NOT** use manual string replacement for markdown/JSON cleaning.

### Workflow Termination

- **PROCEED_TO_OUTPUT**: Recovery exhausted, generate report
- **TERMINATE_WITH_ERROR**: Hard failure
- Output node acts as "finally" block - always executes

### Streaming & Task Cancellation

For SSE with `asyncio.create_task()`:

1. DELETE endpoint sets `running_task.meta["cancelled"] = True`
2. Generator checks `running_task.meta.get("cancelled")`
3. When detected, raises `asyncio.CancelledError`

### Bounds & Validation

```python
max_search_range = min(len(self.workflow.node_order), current_idx + 100)
for i in range(current_idx, max_search_range):
    node_id = self.workflow.node_order[i]
    node = self.workflow.nodes.get(node_id)
    if not node:
        logger.warning(f"Node {node_id} not in nodes dict, skipping")
        continue
```

### Markdown Code Block Parsing

```python
if "```json" in response_text:
    parts = response_text.split("```json")
    if len(parts) > 1 and "```" in parts[1]:
        response_text = parts[1].split("```")[0].strip()
    else:
        logger.warning("Malformed markdown code block")
```

---

## Schema & DDL

### StarRocks Fixes

- `SHOW TABLES FROM {db}` → `USE {db}` + `SHOW TABLES`
- `SHOW CREATE TABLE {db}.{table}` → `SHOW CREATE TABLE {table}`

### Storage API

| Method | Function |
|--------|----------|
| `get_schema()` | Single table schema |
| `search_similar()` | Semantic search |
| `search_all()` | All schemas |
| `get_table_schemas()` | Multiple tables |
| `update_table_schema()` | Metadata repair |

When SchemaStorage is empty, implement DDL fallback.

---

## Configuration

### Key Files

- `agent.yml` - Main runtime config
- `conf/*.yml.*` - Templates
- `pyproject.toml` - Package config
- `pytest.ini` - Test config

### Settings

```yaml
# SQL Query Timeout (default: 60s)
default_query_timeout_seconds: 60

# Plan Executor
plan_executor:
  keyword_tool_map: {custom mappings}
  enable_fallback: true

# Models
models:
  providers: [OpenAI, Anthropic, Google]

# Databases
databases:
  - snowflake: {...}
  - starrocks: {...}
  - duckdb: {...}
```

### Environment Setup

```bash
cp conf/agent.yml.template conf/agent.yml
export DATUS_CONFIG=/path/to/agent.yml
export OPENAI_API_KEY=your_key
```

---

## Docker

### Build

```bash
# Download model (required before build)
./download_model.sh
# or
HF_HUB_ENABLE_HF_TRANSFER=1 hf download --resume-download \
  qdrant/all-MiniLM-L6-v2-onnx \
  --local-dir docker/huggingface/fastembed/qdrant--all-MiniLM-L6-v2-onnx \
  --local-dir-use-symlinks False

# Build & run
docker build -t datus-agent:latest .
docker run -p 8080:8080 \
  -v /path/to/agent.yml:/root/.datus/conf/agent.yml \
  datus-agent:latest
```

**Note**: Model files not in repo - download locally before building.

---

## Testing & Quality

### Test Execution

```bash
pytest tests/                       # All tests
pytest -s -vv --tb=short            # Verbose output
pytest -m acceptance                # Acceptance tests
pytest --cov=datus tests/           # With coverage
pytest tests/unit_tests/            # Unit only
pytest tests/test_schema_discovery_node.py  # Specific file
```

### Code Quality

```bash
black datus/ tests/ --line-length=120
isort datus/ tests/ --profile=black
flake8 datus/ tests/ --max-line-length=120
./formatter.sh    # Run all formatters
pre-commit install
```

### Best Practices

- Name tests `test_*.py`
- Keep tests deterministic
- Use small, focused fixtures
- Test success and error paths
- Mock external dependencies

### Test Organization

```
tests/
├── unit_tests/    # Unit tests
├── integration/   # Integration tests
├── api/           # API tests
├── conftest.py    # Pytest fixtures
└── test_*.py      # Test files
```

---

## Git & CI/CD

### Commit Messages

```
feat(scope): add new feature
fix(scope): bug fix
docs(scope): documentation
test(scope): add tests
```

### GitHub Actions

| Workflow | Purpose |
|----------|---------|
| `run-ut.yml` | Unit tests |
| `run-integration.yml` | Integration tests |
| `code-quality.yml` | Code quality checks |
| `python-format-check.yml` | Format validation |

### Pre-commit Hooks

```bash
pre-commit install
```

Configured: Black, Flake8, isort

---

## Build & Release

```bash
# Package management
make build          # Build package
make clean          # Clean artifacts
make check          # Pre-upload check
make test           # Test installation

# Publishing
make upload-test    # Test PyPI
make upload         # PyPI
make publish        # Full workflow

# Quick commands
make quick-build    # Clean + build
make quick-test     # Build + test
make quick-publish  # Clean + build + check + upload
make help           # Show all commands
```

---

## Common Tasks

### Adding a New Node Type

1. Define type in `datus/configuration/node_type.py`
2. Implement in `datus/agent/node/new_node.py`
3. Export in `datus/agent/node/__init__.py`
4. Import in `datus/agent/node/node.py` factory
5. Add handler in `Node.new_instance()`

### Debugging

```bash
export DATUS_LOG_LEVEL=DEBUG
pytest tests/test_workflow.py -s -vv --tb=long
pytest tests/test_sse_compliance.py -v
```

### Diagnostic Commands

```bash
datus-agent --config-check          # Check configuration
datus-agent --test-db-connection    # Test DB connection
datus-agent --validate-workflow     # Validate workflow
datus-agent --version               # Check version
```

---

## Performance & Monitoring

- **Query Caching**: QueryCache with 1-hour TTL for intent clarification
- **Monitoring**: Structured logging, performance metrics, workflow observability
- **Optimization**: Lazy embeddings, async/await, streaming, connection pooling

---

## Documentation Standards

### Before Modifying Docs

1. **API Docs**: Verify against `datus/api/models.py` and `service.py`
2. **Templates**: Use `ls datus/prompts/prompt_templates/*.j2`
3. **Tools**: Verify in actual implementation, not docs

### Common Discrepancies to Catch

- JSON format mismatches
- Non-existent API endpoints
- Incorrect parameter names
- Hardcoded version numbers

### When to Delete vs Update

| Delete When | Update When |
|-------------|-------------|
| Feature no longer exists | Minor fixes needed |
| >3 critical errors | Content is unique |
| Fully duplicated | Template list expansion |

### Review Checklist

- [ ] API endpoint exists in `service.py`
- [ ] Parameters match `models.py`
- [ ] Template lists match actual files
- [ ] Remove hardcoded versions (use `*.j2`)
- [ ] cURL examples have correct `--config` path
- [ ] Check for duplication

### Version Headers

```markdown
> **文档版本**: v1.0
> **更新日期**: YYYY-MM-DD
```

---

## Node Registration

When adding new node types, complete all 5 steps:

1. Define node type in `datus/configuration/node_type.py`
2. Implement node class in `datus/agent/node/new_node.py`
3. Export in `datus/agent/node/__init__.py`
4. Import in `datus/agent/node/node.py` factory
5. Add handler case in `Node.new_instance()`

---

Remember: **Claude Flow coordinates, Claude Code creates!**
