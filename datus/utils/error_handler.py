# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Backward compatibility shim for error handling utilities.

The unified error handling implementation lives in datus.utils.error_handling.
This module re-exports the public API for legacy imports.
"""

from datus.utils.error_handling import (  # noqa: F401
    ErrorHandlerMixin,
    LLMMixin,
    LLMRateLimitError,
    LLMTimeoutError,
    NodeErrorResult,
    NodeExecutionResult,
    NodeStatus,
    RetryStrategy,
    check_reflect_node_reachable,
    unified_error_handler,
    with_error_recovery,
)

__all__ = [
    "ErrorHandlerMixin",
    "LLMMixin",
    "LLMRateLimitError",
    "LLMTimeoutError",
    "NodeErrorResult",
    "NodeExecutionResult",
    "NodeStatus",
    "RetryStrategy",
    "check_reflect_node_reachable",
    "unified_error_handler",
    "with_error_recovery",
]
