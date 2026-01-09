# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Datus is an open-source data engineering agent that builds evolvable context for your data system. It consists of three main components:

- **Datus-CLI**: AI-powered command-line interface for data engineers (Claude Code for data engineers)
- **Datus-Chat**: Web chatbot providing multi-turn conversations with feedback mechanisms
- **Datus-API**: APIs for other agents or applications needing stable, accurate data services

The system automatically builds a living semantic map of company data, combining metadata, metrics, reference SQL, and external knowledge.

## Development Commands

### Package Management and Environment
- **Activate virtual environment**: `source .venv/bin/activate` (required before running tests or starting services)
- **Install dependencies**: `uv pip install -e ".[dev]"` (uses uv with Aliyun mirror)
- **Setup development environment**: `make setup-dev` or `make dev-install`
- **Sync dev dependencies**: `uv sync --dev`

> **IMPORTANT**: Always activate venv before executing test scripts or starting services. Use the command format: `source .venv/bin/activate && <command>`

### Building and Distribution
- **Build package**: `make build` (runs `python build_scripts/build_pypi_package.py build`)
- **Clean build artifacts**: `make clean`
- **Install locally (editable)**: `make install`
- **Test installation**: `make test`
- **Full CI cycle**: `make all` (clean, build, check, test)
- **Publish to PyPI**: `make publish`

### Code Quality and Formatting
- **Format code**: `./formatter.sh` (runs autoflake, black, isort, flake8)
- **Black formatting**: `black --line-length=120 --extend-exclude="/(mcp)/" datus/ tests/`
- **Flake8 linting**: `flake8 --max-line-length=120 --extend-ignore=E203,W503 datus/ tests/`
- **Import sorting**: `isort --profile=black --line-length=120 datus/ tests/`
- **Pre-commit hooks**: Install with `pre-commit install`, run with `pre-commit run --all-files`

### Testing
- **Run all tests**: `pytest` (configured in pytest.ini with `-s -vv --tb=short --showlocals --ignore=tests/test_full_gen_sql.py`)
- **Run integration tests**: See `.github/workflows/run-integration.yml`
- **Test specific file**: `pytest tests/path/to/test.py -v`
- **Parallel testing**: Use `pytest -n auto` (requires pytest-xdist)

## Architecture

### Core Components
- **`datus/agent/`**: Main agent class (`Agent`) and workflow runner (`WorkflowRunner`)
- **`datus/cli/`**: CLI interface (`DatusCLI`) with REPL and web interface (Streamlit)
- **`datus/api/`**: FastAPI-based REST API endpoints
- **`datus/configuration/`**: Agent configuration loading (`agent.yml`)
- **`datus/schemas/`**: Pydantic models for tasks, nodes, actions, and results
- **`datus/models/`**: LLM integration and base model classes
- **`datus/storage/`**: Knowledge base components (metadata, metrics, external knowledge, reference SQL)
- **`datus/tools/`**: Tool implementations (database, filesystem, context search, date parsing, LLM tools)
- **`datus/utils/`**: Utility modules (async, benchmark, constants, exceptions, logging, time)

### Key Architectural Patterns
1. **Workflow-based execution**: Tasks flow through configurable nodes (schema linking, SQL generation, execution)
2. **Semantic knowledge base**: LanceDB stores metadata, metrics, and reference SQL for context-aware queries
3. **Model Context Protocol (MCP)**: Extensible tool integration via MCP servers (see `mcp/mcp-metricflow-server/`)
4. **Subagent system**: Domain-specific chatbots with scoped context and tools
5. **Feedback loop**: User corrections and success stories improve model accuracy over time

### Entry Points
- **`datus-agent`**: Main agent CLI (`datus.main:main`) - `init`, `namespace`, `run`, `benchmark`, `eval`, etc.
- **`datus-cli`**: Interactive CLI (`datus.cli.main:main`) - REPL with `--web` option for Streamlit interface
- **Configuration**: `agent.yml` defines LLM providers, databases, workflows, and node configurations

### Database Support
- Primary: SQLite, DuckDB, Snowflake (with adapters in separate packages)
- Connection management via `DBManager` in `datus.tools.db_tools.db_manager`
- Schema linking supports tables, views, and materialized views

## Configuration

Agent configuration is loaded from `agent.yml` (default locations: `./conf/agent.yml` → `~/.datus/conf/agent.yml`). Key sections:

- **`models`**: LLM provider configurations (OpenAI, Anthropic, Google, etc.)
- **`databases`**: Database connections (Snowflake, StarRocks, SQLite, DuckDB, etc.)
- **`workflows`**: Custom execution plans and node configurations
- **`default_query_timeout_seconds`**: SQL execution timeout (default: 60s)

Namespaces allow multiple database environments within a single configuration.

## Testing and Quality

- **Unit tests**: `tests/unit/` - isolated component tests
- **Integration tests**: `tests/integration/` - require database connections
- **Benchmark tests**: `benchmark/` - performance and accuracy evaluation
- **Code coverage**: Not explicitly configured but can be added with `pytest-cov`
- **Pre-commit**: Enforces black, flake8, isort before commits

## Notes for Contributors

- Python ≥3.12 required (see `pyproject.toml`)
- Uses **uv** for package management with Aliyun mirror default
- Follow existing patterns for new tools, schemas, and storage components
- MCP servers should be placed in `mcp/` directory
- Reference SQL files go in `sample_data/california_schools/reference_sql/` for demos
- SQL history and feedback are stored in LanceDB tables in `{agent.home}/data/`