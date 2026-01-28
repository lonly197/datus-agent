# Plan Hooks Module Split

## Overview

This directory contains the modularized components of the original `plan_hooks.py` file (5916 lines).

## Module Structure

### Completed Modules

1. **types.py** (150 lines)
   - `TaskType` - Task type classifications (TOOL_EXECUTION, LLM_ANALYSIS, HYBRID)
   - `ErrorType` - Error type classifications
   - `PlanningPhaseException` - Exception for planning phase errors
   - `UserCancelledException` - Exception for user cancellation

2. **router.py** (300 lines)
   - `SmartExecutionRouter` - Intelligent task execution router
   - `TaskTypeClassifier` - Smart task type classifier with Chinese/English support

3. **error_handling.py** (400 lines)
   - `ErrorRecoveryStrategy` - Retry and fallback strategies
   - `ErrorHandler` - Comprehensive error handling with auto-fix

4. **monitoring.py** (729 lines)
   - `ExecutionMonitor` - Complete monitoring system with metrics tracking

5. **optimization.py** (533 lines)
   - `QueryCache` - Intelligent caching with TTL
   - `ToolBatchProcessor` - Batch optimization for tool calls

### Remaining Modules (To be extracted from PlanModeHooks)

The original `PlanModeHooks` class (1840-5916, ~4000 lines) needs to be split into:

6. **matching.py** (~800 lines)
   - Tool matching logic
   - LLM reasoning fallback
   - Chinese semantic matching
   - Context-aware matching

7. **core.py** (~1200 lines)
   - PlanModeHooks initialization
   - Lifecycle hooks (on_start, on_end, on_tool_start, etc.)
   - State management
   - Configuration loading

8. **execution.py** (~1200 lines)
   - Execution orchestration
   - Server executor
   - User confirmation handling
   - Replanning logic
   - Step-by-step execution

9. **events.py** (~300 lines)
   - Event emission
   - Plan update events
   - Tool error events
   - Status message events

10. **__init__.py**
    - Re-export all public APIs
    - Maintain backward compatibility

## Key Design Decisions

### 1. Module Dependencies
```
types.py (no dependencies)
  ↓
router.py, error_handling.py (depend on types)
  ↓
monitoring.py, optimization.py (depend on types)
  ↓
matching.py (depends on router, types)
  ↓
core.py, execution.py, events.py (depend on all above)
```

### 2. Circular Dependency Avoidance
- Use `TYPE_CHECKING` for type hints
- Dependency injection for complex interactions
- Keep PlanModeHooks as orchestrator, not container

### 3. Backward Compatibility
- All original APIs preserved
- `__init__.py` re-exports PlanModeHooks
- Existing imports continue to work

## Implementation Status

### ✅ Completed
- types.py - 100%
- router.py - 100%
- error_handling.py - 100%
- monitoring.py - 100%
- optimization.py - 100%

### 🚧 In Progress
- matching.py - Needs extraction from PlanModeHooks (lines ~2042-2433)
- core.py - Needs extraction from PlanModeHooks (lines ~1840-2569)
- execution.py - Needs extraction from PlanModeHooks (lines ~2698-4500)
- events.py - Needs extraction from PlanModeHooks (lines ~1931-2100)

### 📋 TODO
1. Extract matching.py methods:
   - _match_tool_for_todo
   - _match_exact_keywords
   - _execute_llm_reasoning
   - _llm_reasoning_fallback
   - _intelligent_inference
   - Related helpers

2. Extract core.py methods:
   - __init__
   - on_start, on_end, on_tool_start, on_tool_end, on_llm_end
   - _transition_state
   - _load_keyword_map
   - set_execution_event_manager
   - cleanup

3. Extract execution.py methods:
   - _on_plan_generated
   - _get_user_confirmation
   - _handle_replan
   - _handle_execution_step
   - _run_server_executor
   - _execute_tool_with_error_handling
   - Related execution helpers

4. Extract events.py methods:
   - _emit_action
   - _emit_plan_update_event
   - _emit_tool_error_event
   - _emit_status_message
   - Related event helpers

5. Create __init__.py with re-exports

6. Update imports in all modules

7. Verify compilation and tests

## File Sizes

Original: plan_hooks.py - 5916 lines

Modularized:
- types.py: ~150 lines (2.5%)
- router.py: ~300 lines (5.1%)
- error_handling.py: ~400 lines (6.8%)
- monitoring.py: ~729 lines (12.3%)
- optimization.py: ~533 lines (9.0%)
- matching.py: ~800 lines (13.5%) - TODO
- core.py: ~1200 lines (20.3%) - TODO
- execution.py: ~1200 lines (20.3%) - TODO
- events.py: ~300 lines (5.1%) - TODO

Total: ~5,612 lines (94.8%) + documentation

## Testing

After extraction, verify:

```bash
# Check compilation
python -m py_compile datus/cli/plan_hooks/*.py

# Run tests
pytest tests/unit_tests/cli/ -v -k plan_hooks

# Check imports
python -c "from datus.cli.plan_hooks import PlanModeHooks"
```

## Notes

- Original file backed up as `plan_hooks.py.backup`
- All docstrings preserved
- Type hints maintained
- Logging statements preserved
- No functional changes during split
