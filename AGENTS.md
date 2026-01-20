# Repository Guidelines

## Project Structure & Module Organization
- `datus/` is the core Python package (CLI, workflows, prompts, and runtime logic).
- `tests/` holds pytest suites and fixtures; test files follow `test_*.py`.
- `conf/` contains default configuration templates (`*.yml.*`).
- `sample_data/` includes demo datasets and reference SQL used in examples/tests.
- `build_scripts/` and `Makefile` drive packaging, release, and install workflows.
- `docs/` and `docs/assets/` hold documentation and diagrams.

## Build, Test, and Development Commands
- `make setup-dev` installs build tools and development dependencies.
- `uv pip install -e ".[dev]"` installs the package in editable mode.
- `make build` creates the distribution via `build_scripts/build_pypi_package.py`.
- `make install-dist` installs from the built artifacts in `dist/`.
- `pytest` runs the test suite using settings in `pytest.ini`.
- `./formatter.sh` runs autoflake, Black, isort, and flake8 in sequence.

## Coding Style & Naming Conventions
- Python is the primary language; use 4-space indentation.
- Format with Black (line length 120), then isort; flake8 is enforced via pre-commit.
- Use `snake_case` for functions/modules, `PascalCase` for classes, and `UPPER_CASE` for constants.
- The `mcp/` tree is excluded from some formatting and linting rules.

## Testing Guidelines
- Pytest is configured in `pytest.ini` with `tests/` as the root.
- Name tests `test_*.py`; mark acceptance tests with `@pytest.mark.acceptance` and run via `pytest -m acceptance`.
- Keep tests deterministic and prefer small, focused fixtures.

## Commit & Pull Request Guidelines
- Recent history uses conventional commits like `feat(scope): ...` and `fix(scope): ...`.
- PRs should include: a short summary, testing notes (commands + results), and links to related issues.
- For user-facing changes (CLI/docs), include before/after examples or screenshots when relevant.

## Configuration & Security Tips
- Configuration templates live in `conf/`; runtime config is typically `agent.yml`.
- Avoid committing secrets; use environment variables or local config overrides.
