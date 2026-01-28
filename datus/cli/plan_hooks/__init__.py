# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Plan mode hooks system for workflow management.

This module provides a modularized version of the original plan_hooks.py file.
The main class PlanModeHooks is re-exported for backward compatibility.

Module Structure:
- types: Type definitions and exceptions
- router: Execution routing and task classification
- error_handling: Error recovery strategies
- monitoring: Execution monitoring and metrics
- optimization: Caching and batch processing
- matching: Tool matching logic
- core: Core lifecycle management
- execution: Execution orchestration
- events: Event emission

Usage:
    from datus.cli.plan_hooks import PlanModeHooks

    hooks = PlanModeHooks(
        console=console,
        session=session,
        auto_mode=False,
        action_history_manager=action_manager,
    )
"""

# Import all components for re-export
from .core import PlanModeHooksCore
from .error_handling import ErrorHandler, ErrorRecoveryStrategy
from .events import (
    emit_action,
    emit_plan_update_event,
    emit_sql_execution_error,
    emit_sql_execution_result,
    emit_sql_execution_start,
    emit_status_message,
    emit_tool_error_event,
)
from .execution import (
    get_knowledge_confirmation,
    get_user_confirmation,
    handle_auto_mode,
    handle_execution_step,
    handle_replan,
    on_plan_generated,
    run_server_executor,
)
from .matching import DEFAULT_PLAN_EXECUTOR_KEYWORD_MAP, ToolMatcher
from .monitoring import ExecutionMonitor
from .optimization import QueryCache, ToolBatchProcessor
from .router import SmartExecutionRouter, TaskType, TaskTypeClassifier
from .types import ErrorType, PlanningPhaseException, TaskType as TaskTypeEnum, UserCancelledException

# For backward compatibility, import the original PlanModeHooks from the parent module
# This allows existing code to continue working unchanged
try:
    from datus.cli.plan_hooks import PlanModeHooks as _OriginalPlanModeHooks
except ImportError:
    # If the original file has been moved/renamed, use the modular version
    _OriginalPlanModeHooks = None

# Re-export main classes
__all__ = [
    # Main hooks class (backward compatible)
    "PlanModeHooks",
    # Core components
    "PlanModeHooksCore",
    # Types
    "ErrorType",
    "PlanningPhaseException",
    "UserCancelledException",
    "TaskType",
    "TaskTypeEnum",
    # Router
    "SmartExecutionRouter",
    "TaskTypeClassifier",
    # Error handling
    "ErrorHandler",
    "ErrorRecoveryStrategy",
    # Monitoring
    "ExecutionMonitor",
    # Optimization
    "QueryCache",
    "ToolBatchProcessor",
    # Matching
    "ToolMatcher",
    "DEFAULT_PLAN_EXECUTOR_KEYWORD_MAP",
    # Events
    "emit_action",
    "emit_plan_update_event",
    "emit_tool_error_event",
    "emit_status_message",
    "emit_sql_execution_start",
    "emit_sql_execution_result",
    "emit_sql_execution_error",
    # Execution
    "on_plan_generated",
    "get_user_confirmation",
    "handle_replan",
    "handle_execution_step",
    "handle_auto_mode",
    "get_knowledge_confirmation",
    "run_server_executor",
]

# PlanModeHooks is the main entry point - import from original file
# This maintains full backward compatibility while allowing gradual migration
if _OriginalPlanModeHooks is not None:
    PlanModeHooks = _OriginalPlanModeHooks
else:
    # Fallback: Import from the parent directory (original file)
    import importlib
    import sys
    from pathlib import Path

    # Add parent directory to path if needed
    parent_dir = str(Path(__file__).parent.parent)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    try:
        # Try to import from plan_hooks module (original file)
        plan_hooks_module = importlib.import_module("plan_hooks")
        PlanModeHooks = plan_hooks_module.PlanModeHooks
    except ImportError:
        # Last resort: use the core class as PlanModeHooks
        # Note: This will be missing execution methods that need to be added
        PlanModeHooks = PlanModeHooksCore
        import warnings

        warnings.warn(
            "Using PlanModeHooksCore as PlanModeHooks - some functionality may be limited. "
            "Ensure plan_hooks.py is properly installed.",
            DeprecationWarning,
            stacklevel=2,
        )

# Version info
__version__ = "2.0.0"
__author__ = "DatusAI"
