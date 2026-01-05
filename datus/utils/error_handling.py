# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unified Error Handling for Datus-agent Nodes.

This module provides unified error handling mechanisms for all node types,
ensuring consistent error reporting, logging, and recovery across the system.
"""

import json
import traceback
from functools import wraps
from typing import Any, Dict, List, Optional

from datus.schemas.base import BaseResult
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class NodeErrorResult(BaseResult):
    """Unified node error result with enhanced context information."""

    def __init__(
        self,
        success: bool = False,
        error_code: str = "",
        error_message: str = "",
        error_details: Optional[Dict[str, Any]] = None,
        node_context: Optional[Dict[str, Any]] = None,
        retryable: bool = False,
        recovery_suggestions: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Initialize unified error result.

        Args:
            success: Whether the operation was successful
            error_code: Standardized error code
            error_message: Human-readable error message
            error_details: Detailed error information
            node_context: Context information about the node execution
            retryable: Whether this error can be retried
            recovery_suggestions: List of suggested recovery actions
        """
        super().__init__(success=success, error=error_message, **kwargs)

        self.error_code = error_code
        self.error_details = error_details or {}
        self.node_context = node_context or {}
        self.retryable = retryable
        self.recovery_suggestions = recovery_suggestions or []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        base_dict = super().to_dict() if hasattr(super(), "to_dict") else {"success": self.success, "error": self.error}

        return {
            **base_dict,
            "error_code": self.error_code,
            "error_details": self.error_details,
            "node_context": self.node_context,
            "retryable": self.retryable,
            "recovery_suggestions": self.recovery_suggestions,
        }


def unified_error_handler(node_type: str, operation: str):
    """
    Unified error handling decorator for node methods.

    This decorator provides consistent error handling across all node types,
    ensuring proper logging, error classification, and context preservation.

    Args:
        node_type: The type of node (e.g., 'ExecuteSQLNode')
        operation: The operation being performed (e.g., 'sql_execution')

    Returns:
        Decorated function that handles errors consistently
    """

    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except DatusException as e:
                # Re-raise DatusException as-is since it's already properly formatted
                logger.error(
                    f"{node_type}.{operation} failed with DatusException: {e}",
                    extra={
                        "error_code": e.code.code,
                        "node_type": node_type,
                        "operation": operation,
                        "node_id": getattr(self, "id", "unknown"),
                    },
                )
                raise
            except json.JSONDecodeError as e:
                # Handle JSON parsing errors specifically
                error_details = {
                    "json_error": str(e),
                    "doc": e.doc[:500] + "..." if len(e.doc) > 500 else e.doc,
                    "pos": e.pos,
                }
                error_result = _create_node_error_result(
                    self,
                    ErrorCode.COMMON_JSON_PARSE_ERROR,
                    f"JSON parsing failed in {operation}: {str(e)}",
                    operation,
                    error_details,
                )
                self.result = error_result
                return error_result
            except ConnectionError as e:
                # Handle connection errors
                error_details = {"connection_error": str(e), "operation": operation}
                error_result = _create_node_error_result(
                    self,
                    ErrorCode.DB_CONNECTION_FAILED,
                    f"Connection failed during {operation}: {str(e)}",
                    operation,
                    error_details,
                    retryable=True,
                )
                self.result = error_result
                return error_result
            except TimeoutError as e:
                # Handle timeout errors
                error_details = {"timeout_error": str(e), "operation": operation}
                error_result = _create_node_error_result(
                    self,
                    ErrorCode.DB_EXECUTION_TIMEOUT,
                    f"Operation timed out during {operation}: {str(e)}",
                    operation,
                    error_details,
                    retryable=True,
                )
                self.result = error_result
                return error_result
            except Exception as e:
                # Handle all other exceptions
                logger.error(
                    f"{node_type}.{operation} failed with unexpected error: {str(e)}",
                    exc_info=True,
                    extra={
                        "node_type": node_type,
                        "operation": operation,
                        "node_id": getattr(self, "id", "unknown"),
                        "stack_trace": traceback.format_exc(),
                    },
                )

                error_result = _create_node_error_result(
                    self,
                    ErrorCode.NODE_EXECUTION_FAILED,
                    f"{operation} failed: {str(e)}",
                    operation,
                    {"unexpected_error": str(e), "stack_trace": traceback.format_exc()},
                    retryable=_is_retryable_error(e),
                )
                self.result = error_result
                return error_result

        return wrapper

    return decorator


def _create_node_error_result(
    node_instance,
    error_code: ErrorCode,
    error_message: str,
    operation: str,
    error_details: Optional[Dict[str, Any]] = None,
    retryable: bool = False,
) -> NodeErrorResult:
    """
    Create a standardized node error result.

    Args:
        node_instance: The node instance that encountered the error
        error_code: The error code
        error_message: Human-readable error message
        operation: The operation that failed
        error_details: Additional error details
        retryable: Whether the error is retryable

    Returns:
        NodeErrorResult with comprehensive error information
    """
    # Gather node context information
    node_context = {
        "node_id": getattr(node_instance, "id", "unknown"),
        "node_type": getattr(node_instance, "type", "unknown"),
        "operation": operation,
        "start_time": getattr(node_instance, "start_time", None),
        "input_summary": _summarize_node_input(node_instance),
    }

    # Determine recovery suggestions based on error type
    recovery_suggestions = _generate_recovery_suggestions(error_code, operation)

    return NodeErrorResult(
        success=False,
        error_code=error_code.code,
        error_message=error_message,
        error_details={**(error_details or {}), "stack_trace": traceback.format_exc()},
        node_context=node_context,
        retryable=retryable,
        recovery_suggestions=recovery_suggestions,
    )


def _summarize_node_input(node_instance) -> Dict[str, Any]:
    """
    Generate a summary of node input for error context.

    Args:
        node_instance: The node instance

    Returns:
        Dictionary with input summary
    """
    if not hasattr(node_instance, "input") or not node_instance.input:
        return {}

    input_obj = node_instance.input
    summary = {}

    # Extract common fields that are useful for debugging
    common_fields = ["sql_query", "task", "database_name", "table_schemas", "task_id"]
    for field in common_fields:
        if hasattr(input_obj, field):
            value = getattr(input_obj, field)
            if isinstance(value, str) and len(value) > 100:
                summary[field] = value[:100] + "..."
            elif isinstance(value, list) and len(value) > 3:
                summary[field] = f"{len(value)} items"
            else:
                summary[field] = value

    return summary


def _is_retryable_error(error: Exception) -> bool:
    """
    Determine if an error is retryable based on its type.

    Args:
        error: The exception that occurred

    Returns:
        True if the error is likely retryable
    """
    retryable_error_types = (
        ConnectionError,
        TimeoutError,
        OSError,  # Includes network-related OS errors
    )

    # Check for specific error messages that indicate retryable conditions
    error_message = str(error).lower()
    retryable_messages = ["connection", "timeout", "network", "temporary", "unavailable", "overload"]

    return isinstance(error, retryable_error_types) or any(msg in error_message for msg in retryable_messages)


def _generate_recovery_suggestions(error_code: ErrorCode, operation: str) -> List[str]:
    """
    Generate recovery suggestions based on error code and operation.

    Args:
        error_code: The error code
        operation: The operation that failed

    Returns:
        List of recovery suggestions
    """
    suggestions = []

    # Error-code-specific suggestions
    if error_code == ErrorCode.DB_CONNECTION_FAILED:
        suggestions.extend(
            ["Check database connection settings", "Verify database server is running", "Check network connectivity"]
        )
    elif error_code == ErrorCode.DB_EXECUTION_TIMEOUT:
        suggestions.extend(["Increase query timeout settings", "Optimize the SQL query", "Check database performance"])
    elif error_code == ErrorCode.COMMON_JSON_PARSE_ERROR:
        suggestions.extend(
            ["Validate JSON format", "Check for special characters in the input", "Review the prompt template"]
        )
    elif error_code == ErrorCode.MODEL_REQUEST_FAILED:
        suggestions.extend(
            ["Check API key and authentication", "Verify model service availability", "Check rate limits and quotas"]
        )
    elif error_code == ErrorCode.NODE_EXECUTION_FAILED:
        suggestions.extend(
            ["Review input parameters", "Check system resources", "Enable debug logging for more details"]
        )

    # Operation-specific suggestions
    if operation == "sql_execution":
        suggestions.extend(["Validate SQL syntax", "Check table and column names", "Verify database permissions"])
    elif operation == "schema_linking":
        suggestions.extend(
            ["Check database schema access", "Verify table metadata availability", "Try with different matching rates"]
        )
    elif operation == "model_generation":
        suggestions.extend(
            ["Check model configuration", "Validate prompt templates", "Try with different model parameters"]
        )

    return suggestions[:5]  # Limit to top 5 suggestions
