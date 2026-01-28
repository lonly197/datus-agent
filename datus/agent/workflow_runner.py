# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details

"""
Workflow Runner - Backward Compatibility Module.

This module has been refactored into sub-modules in datus/agent/runner/.
The WorkflowRunner class is now imported from the runner module for
backward compatibility.

New module structure:
- datus/agent/runner/workflow_lifecycle.py
- datus/agent/runner/workflow_navigator.py
- datus/agent/runner/workflow_termination.py
- datus/agent/runner/workflow_executor.py
"""

# Backward compatibility: re-export from new location
from datus.agent.runner import WorkflowExecutor

__all__ = ["WorkflowExecutor"]
