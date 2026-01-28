# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details

"""
SchemaValidationNode implementation - Backward compatibility wrapper.

This file has been refactored into separate modules for better maintainability.
All classes are re-exported from the new modules for backward compatibility.

REFACTORED STRUCTURE:
    datus/agent/node/schema_validation/
    ├── __init__.py       # Public API exports
    ├── core.py           # SchemaValidationNode main class
    ├── validators.py     # Schema coverage validation
    ├── diagnostics.py    # Diagnostic reporting
    └── term_utils.py     # Term processing utilities

MIGRATION GUIDE:
    Old imports (still work):
        from datus.agent.node.schema_validation_node import SchemaValidationNode

    New imports (recommended):
        from datus.agent.node.schema_validation import SchemaValidationNode
        from datus.agent.node.schema_validation.core import SchemaValidationNode
        from datus.agent.node.schema_validation import CoverageValidator, DiagnosticReporter, TermProcessor

For detailed refactoring information, see the module docstrings in each submodule.
"""

# Re-export everything from the new modules for backward compatibility
from datus.agent.node.schema_validation import (
    SchemaValidationNode,
    CoverageValidator,
    DiagnosticReporter,
    TermProcessor,
)

__all__ = [
    "SchemaValidationNode",
    "CoverageValidator",
    "DiagnosticReporter",
    "TermProcessor",
]
