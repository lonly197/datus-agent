# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details

"""
Schema validation node modules.

This package contains the refactored components of the original schema_validation_node.py,
split into focused modules for better maintainability.

Modules:
    - core: SchemaValidationNode class with main execution logic
    - validators: Schema coverage and sufficiency validation
    - diagnostics: Diagnostic reporting and error handling
    - term_utils: Term extraction and processing utilities
"""

# Public API - re-export all classes
from datus.agent.node.schema_validation.core import SchemaValidationNode
from datus.agent.node.schema_validation.validators import CoverageValidator
from datus.agent.node.schema_validation.diagnostics import DiagnosticReporter
from datus.agent.node.schema_validation.term_utils import TermProcessor

__all__ = [
    "SchemaValidationNode",
    "CoverageValidator",
    "DiagnosticReporter",
    "TermProcessor",
]
