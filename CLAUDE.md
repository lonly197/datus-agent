# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Datus Agent - AI-Powered SQL Data Engineering Agent

## Project Overview

**Datus** is an open-source data engineering agent that builds evolvable context for your data system. It provides three main interfaces:

- **Datus-CLI**: An AI-powered command-line interface for data engineers—think "Claude Code for data engineers." Write SQL, build subagents, and construct context interactively.
- **Datus-Chat**: A web chatbot providing multi-turn conversations with built-in feedback mechanisms for data analysts.
- **Datus-API**: APIs for other agents or applications that need stable, accurate data services.

### Recent Text2SQL Pipeline Hardening

The project has recently undergone comprehensive system hardening for Text2SQL workflows with fixes including:
- Evidence-Driven Generation with preflight orchestration
- Robust Error Handling with standardized error results
- Intelligent Caching with QueryCache integration
- Configuration Validation with runtime validation
- Enhanced Observability with structured logging

## Development Environment Setup

### Installation Commands

```bash
# Development mode (recommended)
make setup-dev
# or
uv pip install -e ".[dev]"

# Install build tools only
make dev-install

# Build and install from source
make build
make install-dist

# Install from PyPI
pip install datus-agent
```

### Build & Release Commands

```bash
# Package management
make build          # Build the package
make clean          # Clean build artifacts
make check          # Check package before upload
make test           # Test the installation

# Publishing
make upload-test    # Upload to Test PyPI
make upload         # Upload to PyPI
make publish        # Full publish workflow (clean + build + check + upload)

# Quick commands
make quick-build    # Clean + build
make quick-test     # Build + test
make quick-publish  # Clean + build + check + upload

# Show all available commands
make help
```

### Testing Commands

```bash
# Run all tests
pytest tests/

# Run with verbose output (configured in pytest.ini)
pytest -s -vv --tb=short --showlocals

# Run acceptance tests
pytest -m acceptance

# Run specific test file
pytest tests/test_schema_discovery_node.py

# Run with coverage
pytest --cov=datus tests/

# Run unit tests only
pytest tests/unit_tests/
```

### Code Quality Commands

```bash
# Format code (Black + isort)
black datus/ tests/ --line-length=120
isort datus/ tests/ --profile=black

# Lint code
flake8 datus/ tests/ --max-line-length=120

# Run all formatters
./formatter.sh

# Install pre-commit hooks
pre-commit install
```

## High-Level Architecture

### Core Components

```
datus/
├── agent/              # Core agent logic and workflow orchestration
│   ├── node/           # Individual workflow nodes (25+ node types)
│   │   ├── intent_analysis_node.py      # Two-stage intent processing
│   │   ├── intent_clarification_node.py # Business intent clarification
│   │   ├── schema_discovery_node.py     # Schema retrieval and linking
│   │   ├── generate_sql_node.py         # SQL generation
│   │   ├── execute_sql_node.py          # SQL execution
│   │   ├── preflight_orchestrator.py   # Preflight tool orchestration
│   │   └── ... (20+ more nodes)
│   ├── workflow.py      # Workflow definition and management
│   ├── plan.py          # Execution plan generation
│   └── agent.py         # Main agent orchestration
├── api/                # FastAPI REST API server
├── cli/                # Command-line interface (Textual/TUI)
├── configuration/      # Configuration management
├── schemas/            # Data models and schemas
├── storage/            # Data storage (LanceDB, Tantivy)
├── utils/              # Utility functions
├── prompts/            # LLM prompt templates
└── models/             # ML models and embeddings
```

### Key Architectural Patterns

1. **Node-Based Workflows**: Each workflow is composed of discrete nodes that process data sequentially
2. **Two-Stage Intent Processing**: Fast heuristic detection + LLM-based clarification
3. **Schema-First Approach**: Schema discovery and validation before SQL generation
4. **Preflight Orchestration**: Evidence-driven generation with mandatory preflight tools
5. **Streaming Architecture**: Async/await with SSE for real-time updates

### Database Connectors

Database connectors are in separate packages (see https://github.com/Datus-ai/Datus-adapters):
- Snowflake, StarRocks, MySQL, DuckDB, SQLite support
- Custom connectors via adapter pattern

## Code Organization & Best Practices

### Project Structure

```
/Users/lonlyhuang/workspace/git/Datus-agent/
├── datus/              # Core Python package
├── tests/              # Test suite (pytest)
├── conf/               # Configuration templates
├── sample_data/        # Demo datasets
├── build_scripts/      # Build automation
├── docs/               # Documentation
└── .github/workflows/  # CI/CD pipelines
```

### Coding Standards

- **Python Version**: Requires Python >= 3.11 (tested on 3.12)
- **Line Length**: 120 characters (Black formatter)
- **Indentation**: 4 spaces
- **Naming**: snake_case (functions), PascalCase (classes), UPPER_CASE (constants)
- **File Size**: Keep files under 500 lines
- **Testing**: Test-first approach with deterministic tests

### Type Safety Pattern

When using types that may cause circular imports:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datus.schemas.action_history import ActionHistory

try:
    from datus.schemas.action_history import ActionHistory
except ImportError:
    ActionHistory = Any  # Runtime fallback
```

### Node Factory Registration

When adding new node types, complete all 5 steps:

1. Define node type in `datus/configuration/node_type.py`
2. Implement node class in `datus/agent/node/new_node.py`
3. Export in `datus/agent/node/__init__.py`
4. Import in `datus/agent/node/node.py` factory
5. Add handler case in `Node.new_instance()`

## SPARC Development Methodology

This project uses SPARC (Specification, Pseudocode, Architecture, Refinement, Completion) with Claude-Flow orchestration.

### SPARC Commands

```bash
# List available modes
npx claude-flow sparc modes

# Execute specific mode
npx claude-flow sparc run <mode> "<task>"

# Run complete TDD workflow
npx claude-flow sparc tdd "<feature>"

# Batch parallel execution
npx claude-flow sparc batch <modes> "<task>"

# Full pipeline processing
npx claude-flow sparc pipeline "<task>"
```

### Workflow Phases

1. **Specification** - Requirements analysis
2. **Pseudocode** - Algorithm design
3. **Architecture** - System design
4. **Refinement** - TDD implementation
5. **Completion** - Integration

## 🚨 CRITICAL: Concurrent Execution & File Management

### Absolute Rules

1. **ALL operations MUST be concurrent/parallel in a single message**
2. **NEVER save working files, text/mds and tests to the root folder**
3. **ALWAYS organize files in appropriate subdirectories**
4. **USE CLAUDE CODE'S TASK TOOL** for spawning agents concurrently

### Golden Rule: "1 MESSAGE = ALL RELATED OPERATIONS"

**MANDATORY PATTERNS:**
- **TodoWrite**: ALWAYS batch ALL todos in ONE call (5-10+ todos minimum)
- **Task tool (Claude Code)**: ALWAYS spawn ALL agents in ONE message with full instructions
- **File operations**: ALWAYS batch ALL reads/writes/edits in ONE message
- **Bash commands**: ALWAYS batch ALL terminal operations in ONE message

### Claude Code vs MCP Tools

**Claude Code Handles ALL EXECUTION:**
- Task tool for spawning and running agents
- File operations (Read, Write, Edit, MultiEdit, Glob, Grep)
- Code generation and programming
- Bash commands and system operations
- Implementation work
- Project navigation and analysis
- TodoWrite and task management
- Git operations

**MCP Tools ONLY COORDINATE:**
- Swarm initialization
- Agent type definitions
- Task orchestration
- Memory management
- Neural features

**KEY**: MCP coordinates strategy, Claude Code's Task tool executes with real agents.

## 🚀 Available Agents (54 Total)

### Core Development
`coder`, `reviewer`, `tester`, `planner`, `researcher`

### Specialized Development
`backend-dev`, `mobile-dev`, `ml-developer`, `cicd-engineer`, `api-docs`, `system-architect`, `code-analyzer`, `base-template-generator`

### SPARC Methodology
`sparc-coord`, `sparc-coder`, `specification`, `pseudocode`, `architecture`, `refinement`

### Testing & Validation
`tdd-london-swarm`, `production-validator`

## Text2SQL Workflow Architecture

### Two-Stage Intent Processing

1. **IntentAnalysisNode** (`datus/agent/node/intent_analysis_node.py`):
   - Fast heuristic-based task type detection
   - Sets `workflow.metadata["detected_intent"]`
   - Skips when execution_mode is pre-specified

2. **IntentClarificationNode** (`datus/agent/node/intent_clarification_node.py`):
   - LLM-based business intent clarification
   - Corrects typos, clarifies ambiguities
   - Sets `workflow.metadata["clarified_task"]`
   - Cached with 1-hour TTL

### Workflow Execution Flow

```yaml
text2sql:
  - intent_analysis       # Step 1: Fast heuristic detection
  - intent_clarification  # Step 2: LLM-based clarification
  - schema_discovery      # Step 3: Schema retrieval
  - schema_validation     # Step 4: Schema validation
  - generate_sql          # Step 5: SQL generation
  - execute_sql          # Step 6: SQL execution
  - result_validation    # Step 7: Result validation
  - reflect              # Step 8: Reflection/retry
  - output               # Step 9: Output generation
```

### Preflight Orchestration

Evidence-driven generation with 4 mandatory tools:
- `search_table` - Find relevant tables
- `describe_table` - Get table schemas
- `search_reference_sql` - Find similar queries
- `parse_temporal_expressions` - Parse time expressions

## Key Implementation Patterns

### LLM JSON Parsing

Use standardized `llm_result2json()` utility:

```python
from datus.utils.json_utils import llm_result2json

result = llm_result2json(response_text, expected_type=dict)
if result is None:
    return default_value
```

**DO NOT** use manual string replacement for markdown/JSON cleaning.

### Workflow Termination Pattern

Ensure output node always executes:

- **PROCEED_TO_OUTPUT** - Recovery exhausted, generate final report
- **TERMINATE_WITH_ERROR** - Hard failure, no recovery possible
- Output node acts as "finally" block - always executes

### FastAPI Streaming & Task Cancellation

For SSE with `asyncio.create_task()`:

1. DELETE endpoint sets `running_task.meta["cancelled"] = True`
2. Generator checks `running_task.meta.get("cancelled")` during execution
3. When detected, raises `asyncio.CancelledError`

Problem: Task and generator are independent execution units. Solution: Cooperative cancellation via meta flag.

### Bounds & Validation

Always add max iteration limits and validate dictionary lookups:

```python
max_search_range = min(len(self.workflow.node_order), current_idx + 100)
for i in range(current_idx, max_search_range):
    node_id = self.workflow.node_order[i]
    node = self.workflow.nodes.get(node_id)
    if not node:
        logger.warning(f"Node {node_id} in node_order but not in nodes dict, skipping")
        continue
```

Validate `split()` results when parsing LLM responses:

```python
if "```json" in response_text:
    parts = response_text.split("```json")
    if len(parts) > 1 and "```" in parts[1]:
        response_text = parts[1].split("```")[0].strip()
    else:
        logger.warning("Malformed markdown code block")
```

## Schema Discovery & DDL Retrieval

### StarRocks Connector Fixes

Fix SQL syntax errors by removing duplicate database name:
- `SHOW TABLES FROM {db}` → `USE {db}` + `SHOW TABLES`
- `SHOW CREATE TABLE {db}.{table}` → `SHOW CREATE TABLE {table}`

### Schema Storage API

- `get_schema()` - Single table schema
- `search_similar()` - Semantic search
- `search_all()` - All schemas
- `get_table_schemas()` - Multiple tables
- `update_table_schema()` - Metadata repair

When SchemaStorage is empty, implement DDL fallback to retrieve schemas from database connector.

## Configuration

### Key Configuration Files

- `agent.yml` - Main runtime configuration
- `conf/*.yml.*` - Configuration templates
- `pyproject.toml` - Python package configuration
- `pytest.ini` - Test configuration

### Important Settings

```yaml
# SQL Query Timeout (default: 60s)
default_query_timeout_seconds: 60

# Plan Executor Configuration
plan_executor:
  keyword_tool_map: {custom mappings}
  enable_fallback: true

# Models Configuration
models:
  providers: [OpenAI, Anthropic, Google]

# Database Connections
databases:
  - snowflake: {...}
  - starrocks: {...}
  - duckdb: {...}
```

### Environment Setup

```bash
# Set up config files
cp conf/agent.yml.template conf/agent.yml

# Set environment variables
export DATUS_CONFIG=/path/to/agent.yml
export OPENAI_API_KEY=your_key
```

## Docker Support

### Build Process

```bash
# Download required model (required before build)
./download_model.sh
# or
HF_HUB_ENABLE_HF_TRANSFER=1 hf download --resume-download qdrant/all-MiniLM-L6-v2-onnx --local-dir docker/huggingface/fastembed/qdrant--all-MiniLM-L6-v2-onnx --local-dir-use-symlinks False

# Build image
docker build -t datus-agent:latest .

# Run container
docker run -p 8080:8080 -v /path/to/agent.yml:/root/.datus/conf/agent.yml datus-agent:latest
```

Model files are not included in Git repository - must be downloaded locally before building.

## Testing Guidelines

### Test Organization

```
tests/
├── unit_tests/          # Unit tests
├── integration/         # Integration tests
├── api/                # API tests
├── conftest.py         # Pytest fixtures
└── test_*.py          # Test files
```

### Test Configuration

Configured in `pytest.ini`:
- Verbose output with `-s -vv`
- Test discovery in `tests/` directory
- Logging enabled at INFO level
- Acceptance tests marked with `@pytest.mark.acceptance`

### Best Practices

- Name tests `test_*.py`
- Keep tests deterministic
- Use small, focused fixtures
- Test both success and error paths
- Mock external dependencies

## Git Workflow & CI/CD

### Commit Messages

Use conventional commits:
- `feat(scope): add new feature`
- `fix(scope): bug fix`
- `docs(scope): documentation`
- `test(scope): add tests`

### GitHub Actions

CI/CD workflows in `.github/workflows/`:
- `run-ut.yml` - Unit tests on self-hosted runner
- `run-integration.yml` - Integration tests
- `code-quality.yml` - Code quality checks
- `python-format-check.yml` - Format validation

### Pre-commit Hooks

Configured in `.pre-commit-config.yaml`:
- Black (formatter)
- Flake8 (linter)
- isort (import sorter)

Install with: `pre-commit install`

## Common Development Tasks

### Adding a New Node Type

1. Create node class in `datus/agent/node/new_node.py`
2. Add type constant to `datus/configuration/node_type.py`
3. Register in node factory (`datus/agent/node/node.py`)
4. Add workflow definition in `datus/agent/workflow.yml`
5. Write unit tests in `tests/`

### Running Specific Tests

```bash
# Test schema discovery
pytest tests/test_schema_discovery_node.py -v

# Test intent analysis
pytest tests/test_intent_analysis_node.py -v

# Test error handling
pytest tests/test_error_handling.py -v

# Run all unit tests
pytest tests/unit_tests/ -v

# Run with coverage
pytest --cov=datus --cov-report=html tests/
```

### Debugging Workflows

```bash
# Enable debug logging
export DATUS_LOG_LEVEL=DEBUG

# Run with verbose pytest
pytest tests/test_workflow.py -s -vv --tb=long

# Check SSE compliance
pytest tests/test_sse_compliance.py -v
```

## Performance & Monitoring

### Query Caching

- QueryCache integration for improved performance
- 1-hour TTL for LLM intent clarification
- Fallback mechanisms for cache failures

### Monitoring

- Structured logging with `structlog`
- Performance metrics collection
- Health checks for preflight tools
- Workflow execution observability

### Optimization

- Lazy loading of embeddings
- Async/await throughout
- Streaming responses for long operations
- Connection pooling for databases

## Troubleshooting

### Common Issues

1. **Build fails**: Run `make clean && make build`
2. **Tests timeout**: Check `default_query_timeout_seconds` in config
3. **Schema discovery fails**: Verify database connector is installed
4. **LLM errors**: Check API keys in environment variables

### Debug Commands

```bash
# Check configuration
datus-agent --config-check

# Test database connection
datus-agent --test-db-connection

# Validate workflow
datus-agent --validate-workflow

# Check version
datus-agent --version
```

## Documentation Standards

### Document Validation Before Changes

When modifying documentation, always verify against source code:

1. **API Documentation**: Check `datus/api/models.py` and `datus/api/service.py` for exact request/response formats
2. **Template Lists**: Use `ls datus/prompts/prompt_templates/*.j2` to get current templates
3. **Tool Parameters**: Verify in the actual tool class implementation, not documentation

**Common discrepancies to catch:**
- JSON return format mismatches (e.g., `check_table_conflicts` missing `target_table` field in docs)
- Non-existent API endpoints (e.g., `/metrics` endpoint doesn't exist)
- Incorrect parameter names (e.g., `tool_timeout_seconds` is not a valid API parameter)
- Hardcoded version numbers that don't match actual files

### When to Delete vs Update Documentation

**Delete documentation when:**
- Document describes features that no longer exist in code
- Document has >3 critical errors that would mislead users
- Document is fully duplicated in another file (e.g., `curl_examples.md` vs `workflow/api.md`)
- Document describes an API endpoint that doesn't exist

**Update documentation when:**
- Minor errors can be fixed with small edits
- Content is unique and valuable
- Template lists need expansion (not correction)

### Document Version Headers

Add version headers to new documentation files:

```markdown
> **文档版本**: v1.0
> **更新日期**: YYYY-MM-DD
```

### Documentation Review Checklist

Before submitting documentation changes:

- [ ] Verify API endpoint exists in `service.py`
- [ ] Check request parameters match `models.py` definitions
- [ ] Confirm template lists match actual files in `prompt_templates/`
- [ ] Remove hardcoded version numbers (use `*.j2` pattern instead)
- [ ] Verify cURL examples use correct `--config` path format
- [ ] Check for duplication with existing docs

## Resources

### Documentation
- [Official Website](https://datus.ai)
- [Documentation](https://docs.datus.ai/)
- [Quickstart Guide](https://docs.datus.ai/getting_started/Quickstart/)
- [Release Notes](https://docs.datus.ai/release_notes/)

### Community
- [Slack](https://join.slack.com/t/datus-ai/shared_invite/zt-3g6h4fsdg-iOl5uNoz6A4GOc4xKKWUYg)
- [GitHub Issues](https://github.com/datus-ai/datus-agent/issues)

### Related Projects
- [Datus Adapters](https://github.com/Datus-ai/Datus-adapters) - Database connectors
- [Claude Flow](https://github.com/ruvnet/claude-flow) - SPARC orchestration

---

Remember: **Claude Flow coordinates, Claude Code creates!**
