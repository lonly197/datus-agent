# Claude Code Configuration - SPARC Development Environment

## 🚨 CRITICAL: CONCURRENT EXECUTION & FILE MANAGEMENT

**ABSOLUTE RULES**:
1. ALL operations MUST be concurrent/parallel in a single message
2. **NEVER save working files, text/mds and tests to the root folder**
3. ALWAYS organize files in appropriate subdirectories
4. **USE CLAUDE CODE'S TASK TOOL** for spawning agents concurrently, not just MCP

### ⚡ GOLDEN RULE: "1 MESSAGE = ALL RELATED OPERATIONS"

**MANDATORY PATTERNS:**
- **TodoWrite**: ALWAYS batch ALL todos in ONE call (5-10+ todos minimum)
- **Task tool (Claude Code)**: ALWAYS spawn ALL agents in ONE message with full instructions
- **File operations**: ALWAYS batch ALL reads/writes/edits in ONE message
- **Bash commands**: ALWAYS batch ALL terminal operations in ONE message
- **Memory operations**: ALWAYS batch ALL memory store/retrieve in ONE message

### 🎯 CRITICAL: Claude Code Task Tool for Agent Execution

**Claude Code's Task tool is the PRIMARY way to spawn agents:**
```javascript
// ✅ CORRECT: Use Claude Code's Task tool for parallel agent execution
[Single Message]:
  Task("Research agent", "Analyze requirements and patterns...", "researcher")
  Task("Coder agent", "Implement core features...", "coder")
  Task("Tester agent", "Create comprehensive tests...", "tester")
  Task("Reviewer agent", "Review code quality...", "reviewer")
  Task("Architect agent", "Design system architecture...", "system-architect")
```

**MCP tools are ONLY for coordination setup:**
- `mcp__claude-flow__swarm_init` - Initialize coordination topology
- `mcp__claude-flow__agent_spawn` - Define agent types for coordination
- `mcp__claude-flow__task_orchestrate` - Orchestrate high-level workflows

### 📁 File Organization Rules

**NEVER save to root folder. Use these directories:**
- `/src` - Source code files
- `/tests` - Test files
- `/docs` - Documentation and markdown files
- `/config` - Configuration files
- `/scripts` - Utility scripts
- `/examples` - Example code

## Project Overview

This project uses SPARC (Specification, Pseudocode, Architecture, Refinement, Completion) methodology with Claude-Flow orchestration for systematic Test-Driven Development.

## SPARC Commands

### Core Commands
- `npx claude-flow sparc modes` - List available modes
- `npx claude-flow sparc run <mode> "<task>"` - Execute specific mode
- `npx claude-flow sparc tdd "<feature>"` - Run complete TDD workflow
- `npx claude-flow sparc info <mode>` - Get mode details

### Batchtools Commands
- `npx claude-flow sparc batch <modes> "<task>"` - Parallel execution
- `npx claude-flow sparc pipeline "<task>"` - Full pipeline processing
- `npx claude-flow sparc concurrent <mode> "<tasks-file>"` - Multi-task processing

### Build Commands
- `npm run build` - Build project
- `npm run test` - Run tests
- `npm run lint` - Linting
- `npm run typecheck` - Type checking

## SPARC Workflow Phases

1. **Specification** - Requirements analysis (`sparc run spec-pseudocode`)
2. **Pseudocode** - Algorithm design (`sparc run spec-pseudocode`)
3. **Architecture** - System design (`sparc run architect`)
4. **Refinement** - TDD implementation (`sparc tdd`)
5. **Completion** - Integration (`sparc run integration`)

## Code Style & Best Practices

- **Modular Design**: Files under 500 lines
- **Environment Safety**: Never hardcode secrets
- **Test-First**: Write tests before implementation
- **Clean Architecture**: Separate concerns
- **Documentation**: Keep updated

## 🚀 Available Agents (54 Total)

### Core Development
`coder`, `reviewer`, `tester`, `planner`, `researcher`

### Swarm Coordination
`hierarchical-coordinator`, `mesh-coordinator`, `adaptive-coordinator`, `collective-intelligence-coordinator`, `swarm-memory-manager`

### Consensus & Distributed
`byzantine-coordinator`, `raft-manager`, `gossip-coordinator`, `consensus-builder`, `crdt-synchronizer`, `quorum-manager`, `security-manager`

### Performance & Optimization
`perf-analyzer`, `performance-benchmarker`, `task-orchestrator`, `memory-coordinator`, `smart-agent`

### GitHub & Repository
`github-modes`, `pr-manager`, `code-review-swarm`, `issue-tracker`, `release-manager`, `workflow-automation`, `project-board-sync`, `repo-architect`, `multi-repo-swarm`

### SPARC Methodology
`sparc-coord`, `sparc-coder`, `specification`, `pseudocode`, `architecture`, `refinement`

### Specialized Development
`backend-dev`, `mobile-dev`, `ml-developer`, `cicd-engineer`, `api-docs`, `system-architect`, `code-analyzer`, `base-template-generator`

### Testing & Validation
`tdd-london-swarm`, `production-validator`

### Migration & Planning
`migration-planner`, `swarm-init`

## 🎯 Claude Code vs MCP Tools

### Claude Code Handles ALL EXECUTION:
- **Task tool**: Spawn and run agents concurrently for actual work
- File operations (Read, Write, Edit, MultiEdit, Glob, Grep)
- Code generation and programming
- Bash commands and system operations
- Implementation work
- Project navigation and analysis
- TodoWrite and task management
- Git operations
- Package management
- Testing and debugging

### MCP Tools ONLY COORDINATE:
- Swarm initialization (topology setup)
- Agent type definitions (coordination patterns)
- Task orchestration (high-level planning)
- Memory management
- Neural features
- Performance tracking
- GitHub integration

**KEY**: MCP coordinates the strategy, Claude Code's Task tool executes with real agents.

## 🚀 Quick Setup

```bash
# Add MCP servers (Claude Flow required, others optional)
claude mcp add claude-flow npx claude-flow@alpha mcp start
claude mcp add ruv-swarm npx ruv-swarm mcp start  # Optional: Enhanced coordination
claude mcp add flow-nexus npx flow-nexus@latest mcp start  # Optional: Cloud features
```

## MCP Tool Categories

### Coordination
`swarm_init`, `agent_spawn`, `task_orchestrate`

### Monitoring
`swarm_status`, `agent_list`, `agent_metrics`, `task_status`, `task_results`

### Memory & Neural
`memory_usage`, `neural_status`, `neural_train`, `neural_patterns`

### GitHub Integration
`github_swarm`, `repo_analyze`, `pr_enhance`, `issue_triage`, `code_review`

### System
`benchmark_run`, `features_detect`, `swarm_monitor`

### Flow-Nexus MCP Tools (Optional Advanced Features)
Flow-Nexus extends MCP capabilities with 70+ cloud-based orchestration tools:

**Key MCP Tool Categories:**
- **Swarm & Agents**: `swarm_init`, `swarm_scale`, `agent_spawn`, `task_orchestrate`
- **Sandboxes**: `sandbox_create`, `sandbox_execute`, `sandbox_upload` (cloud execution)
- **Templates**: `template_list`, `template_deploy` (pre-built project templates)
- **Neural AI**: `neural_train`, `neural_patterns`, `seraphina_chat` (AI assistant)
- **GitHub**: `github_repo_analyze`, `github_pr_manage` (repository management)
- **Real-time**: `execution_stream_subscribe`, `realtime_subscribe` (live monitoring)
- **Storage**: `storage_upload`, `storage_list` (cloud file management)

**Authentication Required:**
- Register: `mcp__flow-nexus__user_register` or `npx flow-nexus@latest register`
- Login: `mcp__flow-nexus__user_login` or `npx flow-nexus@latest login`
- Access 70+ specialized MCP tools for advanced orchestration

## 🚀 Agent Execution Flow with Claude Code

### The Correct Pattern:

1. **Optional**: Use MCP tools to set up coordination topology
2. **REQUIRED**: Use Claude Code's Task tool to spawn agents that do actual work
3. **REQUIRED**: Each agent runs hooks for coordination
4. **REQUIRED**: Batch all operations in single messages

### Example Full-Stack Development:

```javascript
// Single message with all agent spawning via Claude Code's Task tool
[Parallel Agent Execution]:
  Task("Backend Developer", "Build REST API with Express. Use hooks for coordination.", "backend-dev")
  Task("Frontend Developer", "Create React UI. Coordinate with backend via memory.", "coder")
  Task("Database Architect", "Design PostgreSQL schema. Store schema in memory.", "code-analyzer")
  Task("Test Engineer", "Write Jest tests. Check memory for API contracts.", "tester")
  Task("DevOps Engineer", "Setup Docker and CI/CD. Document in memory.", "cicd-engineer")
  Task("Security Auditor", "Review authentication. Report findings via hooks.", "reviewer")
  
  // All todos batched together
  TodoWrite { todos: [...8-10 todos...] }
  
  // All file operations together
  Write "backend/server.js"
  Write "frontend/App.jsx"
  Write "database/schema.sql"
```

## 📋 Agent Coordination Protocol

### Every Agent Spawned via Task Tool MUST:

**1️⃣ BEFORE Work:**
```bash
npx claude-flow@alpha hooks pre-task --description "[task]"
npx claude-flow@alpha hooks session-restore --session-id "swarm-[id]"
```

**2️⃣ DURING Work:**
```bash
npx claude-flow@alpha hooks post-edit --file "[file]" --memory-key "swarm/[agent]/[step]"
npx claude-flow@alpha hooks notify --message "[what was done]"
```

**3️⃣ AFTER Work:**
```bash
npx claude-flow@alpha hooks post-task --task-id "[task]"
npx claude-flow@alpha hooks session-end --export-metrics true
```

## 🎯 Concurrent Execution Examples

### ✅ CORRECT WORKFLOW: MCP Coordinates, Claude Code Executes

```javascript
// Step 1: MCP tools set up coordination (optional, for complex tasks)
[Single Message - Coordination Setup]:
  mcp__claude-flow__swarm_init { topology: "mesh", maxAgents: 6 }
  mcp__claude-flow__agent_spawn { type: "researcher" }
  mcp__claude-flow__agent_spawn { type: "coder" }
  mcp__claude-flow__agent_spawn { type: "tester" }

// Step 2: Claude Code Task tool spawns ACTUAL agents that do the work
[Single Message - Parallel Agent Execution]:
  // Claude Code's Task tool spawns real agents concurrently
  Task("Research agent", "Analyze API requirements and best practices. Check memory for prior decisions.", "researcher")
  Task("Coder agent", "Implement REST endpoints with authentication. Coordinate via hooks.", "coder")
  Task("Database agent", "Design and implement database schema. Store decisions in memory.", "code-analyzer")
  Task("Tester agent", "Create comprehensive test suite with 90% coverage.", "tester")
  Task("Reviewer agent", "Review code quality and security. Document findings.", "reviewer")
  
  // Batch ALL todos in ONE call
  TodoWrite { todos: [
    {id: "1", content: "Research API patterns", status: "in_progress", priority: "high"},
    {id: "2", content: "Design database schema", status: "in_progress", priority: "high"},
    {id: "3", content: "Implement authentication", status: "pending", priority: "high"},
    {id: "4", content: "Build REST endpoints", status: "pending", priority: "high"},
    {id: "5", content: "Write unit tests", status: "pending", priority: "medium"},
    {id: "6", content: "Integration tests", status: "pending", priority: "medium"},
    {id: "7", content: "API documentation", status: "pending", priority: "low"},
    {id: "8", content: "Performance optimization", status: "pending", priority: "low"}
  ]}
  
  // Parallel file operations
  Bash "mkdir -p app/{src,tests,docs,config}"
  Write "app/package.json"
  Write "app/src/server.js"
  Write "app/tests/server.test.js"
  Write "app/docs/API.md"
```

### ❌ WRONG (Multiple Messages):
```javascript
Message 1: mcp__claude-flow__swarm_init
Message 2: Task("agent 1")
Message 3: TodoWrite { todos: [single todo] }
Message 4: Write "file.js"
// This breaks parallel coordination!
```

## Performance Benefits

- **84.8% SWE-Bench solve rate**
- **32.3% token reduction**
- **2.8-4.4x speed improvement**
- **27+ neural models**

## Hooks Integration

### Pre-Operation
- Auto-assign agents by file type
- Validate commands for safety
- Prepare resources automatically
- Optimize topology by complexity
- Cache searches

### Post-Operation
- Auto-format code
- Train neural patterns
- Update memory
- Analyze performance
- Track token usage

### Session Management
- Generate summaries
- Persist state
- Track metrics
- Restore context
- Export workflows

## Advanced Features (v2.0.0)

- 🚀 Automatic Topology Selection
- ⚡ Parallel Execution (2.8-4.4x speed)
- 🧠 Neural Training
- 📊 Bottleneck Analysis
- 🤖 Smart Auto-Spawning
- 🛡️ Self-Healing Workflows
- 💾 Cross-Session Memory
- 🔗 GitHub Integration

## Integration Tips

1. Start with basic swarm init
2. Scale agents gradually
3. Use memory for context
4. Monitor progress regularly
5. Train patterns from success
6. Enable hooks automation
7. Use GitHub tools first

## Support

- Documentation: https://github.com/ruvnet/claude-flow
- Issues: https://github.com/ruvnet/claude-flow/issues
- Flow-Nexus Platform: https://flow-nexus.ruv.io (registration required for cloud features)

---

Remember: **Claude Flow coordinates, Claude Code creates!**

# Datus-Specific Learnings

## Schema Storage

- SchemaStorage has get_schema() for single table, search_similar() for semantic search, search_all() for all schemas, and get_table_schemas() for multiple tables
- SchemaStorage.update_table_schema() is available for metadata repair (updates or inserts table definitions)
- When SchemaStorage is empty, implement DDL fallback to retrieve schemas from database connector

## Intent Analysis Architecture

### Two-Stage Intent Processing

The text2sql workflow uses a **two-stage intent analysis** approach:

1. **IntentAnalysisNode** (`datus/agent/node/intent_analysis_node.py`):
   - **Purpose**: Task type recognition (text2sql vs sql_review vs data_analysis)
   - **Method**: Fast heuristic-based detection using SQL keywords and patterns
   - **Output**: Sets `workflow.metadata["detected_intent"]`, `["intent_confidence"]`, `["intent_metadata"]`
   - **Skip Logic**: Skips when `execution_mode` is pre-specified (e.g., "text2sql")
   - **LLM Fallback**: Optional LLM classification when heuristic confidence < 0.7

2. **IntentClarificationNode** (`datus/agent/node/intent_clarification_node.py`):
   - **Purpose**: Business intent clarification (理清用户真实分析意图)
   - **Method**: LLM-based clarification with structured JSON output
   - **Capabilities**:
     - Corrects typos (e.g., "华山" → "华南")
     - Clarifies ambiguities (e.g., "最近的销售" → "最近30天的销售数据")
     - Extracts entities (business_terms, time_range, dimensions, metrics)
     - Normalizes queries for better schema discovery
   - **Output**: Sets `workflow.metadata["clarified_task"]`, `["intent_clarification"]`, `["original_task"]`
   - **LLM Caching**: Uses `llm_call_with_retry()` with 1-hour TTL cache

### Workflow Integration

**text2sql workflow** (`datus/agent/workflow.yml`):
```yaml
text2sql:
  - intent_analysis       # Step 1: Confirm task type (fast, heuristic)
  - intent_clarification  # Step 2: Clarify business intent (LLM-based)
  - schema_discovery      # Step 3: Use clarified intent for schema discovery
  - schema_validation
  - generate_sql
  - execute_sql
  - result_validation
  - reflect
  - output
```

### Schema Discovery Enhancement

**SchemaDiscoveryNode** uses the clarified intent:
- Checks `workflow.metadata.get("clarified_task")` first
- Falls back to `workflow.task.task` if no clarification
- Logs clarification metadata (business_terms, typos_fixed) for debugging
- Uses clarified task for semantic search, keyword matching, and LLM inference

### Key Implementation Details

**IntentAnalysisNode** should skip when execution_mode is set:
```python
def should_skip(self, workflow: Workflow) -> bool:
    execution_mode = workflow.metadata.get("execution_mode")
    return execution_mode is not None and execution_mode != ""
```

**IntentClarificationNode** stores results in workflow.metadata (not context.metadata):
```python
self.workflow.metadata["clarified_task"] = clarification_result["clarified_task"]
self.workflow.metadata["intent_clarification"] = clarification_result
self.workflow.metadata["original_task"] = original_task
```

**SchemaDiscoveryNode** uses clarified task:
```python
clarified_task = self.workflow.metadata.get("clarified_task")
if clarified_task and clarified_task != task.task:
    logger.info(f"Using clarified task: '{task.task[:50]}...' → '{clarified_task[:50]}...'")
    task.task = clarified_task
```

### Performance Considerations

- **IntentAnalysis**: Fast heuristic-based (no LLM call in most cases)
- **IntentClarification**: LLM-based (cached with 1-hour TTL)
- **Total Impact**: +1 LLM call per unique query (mitigated by caching)
- **Benefit**: Improved schema discovery accuracy through clarified intent

## Code Quality & Security

### TYPE_CHECKING Imports Pattern

When using types from modules that may create circular imports or are only needed for type hints, use the `TYPE_CHECKING` pattern:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datus.schemas.action_history import ActionHistory, ActionHistoryManager

try:
    from datus.schemas.action_history import ActionHistory, ActionHistoryManager
except ImportError:
    ActionHistory = Any  # Runtime fallback for type hints
    ActionHistoryManager = Any
```

This separates type-time imports from runtime imports, preventing circular dependency issues while maintaining type safety.

### Bounds & Validation for Loops

Always add max iteration limits to prevent unbounded loops and validate dictionary lookups:

```python
# Bad: Unbounded loop
for i in range(current_idx, len(self.workflow.node_order)):
    node = self.workflow.nodes.get(node_id)
    if node:
        break

# Good: Bounded loop with validation
max_search_range = min(len(self.workflow.node_order), current_idx + 100)
for i in range(current_idx, max_search_range):
    node_id = self.workflow.node_order[i]
    node = self.workflow.nodes.get(node_id)
    if not node:
        logger.warning(f"Node {node_id} in node_order but not in nodes dict, skipping")
        continue
    # Process node...
```

Additionally, validate `split()` results when parsing LLM responses:

```python
# Bad: Could fail with malformed markdown
if "```json" in response_text:
    response_text = response_text.split("```json")[1].split("```")[0].strip()

# Good: Bounds checking with error handling
if "```json" in response_text:
    parts = response_text.split("```json")
    if len(parts) > 1 and "```" in parts[1]:
        response_text = parts[1].split("```")[0].strip()
    else:
        logger.warning("Malformed markdown code block")
```

### Node Factory Registration Pattern

When adding new node types to the workflow, all five steps must be completed:

1. **Define node type constant** in `datus/configuration/node_type.py`:
   ```python
   TYPE_NEW_NODE = "new_node"  # Add to ACTION_TYPES list
   ```

2. **Implement the node class** in `datus/agent/node/new_node.py`:
   ```python
   class NewNode(Node):
       def execute(self) -> BaseResult: ...
       async def execute_stream(...) -> AsyncGenerator[ActionHistory, None]: ...
   ```

3. **Export in `__init__.py`** (`datus/agent/node/__init__.py`):
   ```python
   from .new_node import NewNode
   __all__ = [..., "NewNode"]
   ```

4. **Import in factory method** (`datus/agent/node/node.py`):
   ```python
   from datus.agent.node import (
       ...,
       NewNode,  # Add to imports
   )
   ```

5. **Add handler case** in `Node.new_instance()`:
   ```python
   elif node_type == NodeType.TYPE_NEW_NODE:
       return NewNode(node_id, description, node_type, input_data, agent_config)
   ```

**Common Error**: Skipping step 4 or 5 causes `ValueError: Invalid node type: new_node` when the workflow tries to instantiate the node.

### LLM JSON Parsing Standardization

When nodes parse JSON responses from LLMs, use the standardized `llm_result2json()` utility from `datus/utils/json_utils.py`:

```python
from datus.utils.json_utils import llm_result2json

# Parse LLM JSON response (handles markdown, truncation, format errors)
result = llm_result2json(response_text, expected_type=dict)
if result is None:
    # Fallback handling
    return default_value
```

**DO NOT** use manual string replacement for markdown/JSON cleaning:
```python
# Bad: Fragile string replacement
sql_query = sql_query.strip().replace("```json\n", "").replace("\n```", "")
sql_query_dict = json.loads(cleaned_sql)

# Bad: Manual markdown handling without bounds checking
if "```json" in response_text:
    response_text = response_text.split("```json")[1].split("```")[0].strip()
```

**Benefits** of `llm_result2json()`:
- Handles markdown code blocks (```json, ```) automatically
- Repairs truncated JSON (missing closing brackets)
- Repairs malformed JSON (json-repair integration)
- Validates expected types (dict/list)
- Consistent error handling across all nodes

**Nodes updated**: IntentClarificationNode, IntentDetection, GenerateSQLNode, SemanticAgenticNode, SQLSummaryAgenticNode

### Workflow Termination Pattern

When designing workflows with reflection/retry mechanisms, ensure the output node always executes to generate final reports:

- **PROCEED_TO_OUTPUT** status allows workflow to continue to output node when strategies exhausted
- Output node acts as a "finally" block - always executes for report generation regardless of success/failure
- Reflection nodes should use `PROCEED_TO_OUTPUT` instead of `TERMINATE_WITH_ERROR` when max iterations/rounds reached
- This ensures users receive comprehensive reports with error details, SQL queries, and actionable suggestions

**Termination Status Hierarchy** (`datus/agent/workflow_runner.py`):
- `CONTINUE` - Normal execution flow
- `SKIP_TO_REFLECT` - Soft failure, jump to reflection for recovery
- `PROCEED_TO_OUTPUT` - Recovery exhausted, generate final report
- `TERMINATE_WITH_ERROR` - Hard failure, no recovery possible

## FastAPI Streaming & Task Cancellation

### Cooperative Cancellation Pattern for SSE

When implementing FastAPI `StreamingResponse` with `asyncio.create_task()`, the generator and task are independent execution units. Cancelling the task doesn't stop the generator.

**Problem pattern:**
```python
# chat_research endpoint creates two independent units:
stream_task = asyncio.create_task(_run_chat_research_task(...))  # Background wait task
return StreamingResponse(_generate_chat_research_stream(...))   # Generator with actual work
# DELETE endpoint cancels stream_task, but generator continues running!
```

**Solution - Cooperative cancellation via meta flag:**
1. DELETE endpoint sets `running_task.meta["cancelled"] = True` before calling `task.cancel()`
2. Generator checks `running_task.meta.get("cancelled")` at start and during execution loop
3. When detected, generator raises `asyncio.CancelledError` to stop workflow gracefully

**Implementation locations:**
- DELETE endpoint: `datus/api/service.py` (~1685-1688)
- Stream generator: `_generate_chat_research_stream` (~227-231) checks at start
- Workflow loop: `run_chat_research_stream` (~953-958) checks during iteration

**Key insight:** `asyncio.create_task()` creates independent Tasks that don't control generators consumed by framework components like `StreamingResponse`. Use cooperative cancellation patterns when these need to coordinate.

## Text2SQL Schema Discovery Fixes

- Use `get_tables()` not `get_all_table_names()` for database connector API
- LLM should select from actual database tables, not hallucinate names
- Event.planId must match TodoItem.id for proper frontend binding
- Use unified plan ID strategy for preflight tools and error events

# important-instruction-reminders
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.
Never save working files, text/mds and tests to the root folder.
