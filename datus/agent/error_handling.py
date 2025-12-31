# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unified Error Handling for Datus-agent Node System.

This module provides standardized error handling mechanisms for all nodes,
including unified error result formats, error handling decorators, and
consistent error logging and reporting.
"""

import traceback
from functools import wraps
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

from pydantic import ValidationError

from datus.schemas.base import BaseResult
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

F = TypeVar('F')


class NodeErrorResult(BaseResult):
    """Unified error result for all nodes with enhanced error information."""

    error_code: str = ""
    error_details: Dict[str, Any] = {}
    node_context: Dict[str, Any] = {}
    retryable: bool = False
    recovery_suggestions: List[str] = []

    def __init__(
        self,
        success: bool = False,
        error_code: str = "",
        error_message: str = "",
        error_details: Optional[Dict[str, Any]] = None,
        node_context: Optional[Dict[str, Any]] = None,
        retryable: bool = False,
        recovery_suggestions: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__(
            success=success,
            error=error_message if error_message else "",
            **kwargs
        )
        self.error_code = error_code
        self.error_details = error_details or {}
        self.node_context = node_context or {}
        self.retryable = retryable
        self.recovery_suggestions = recovery_suggestions or []


def unified_error_handler(node_type: str, operation: str, log_errors: bool = True):
    """
    Unified error handling decorator for node methods.

    This decorator provides standardized error handling for node methods,
    converting various exception types to unified error results.

    Args:
        node_type: The type/name of the node (e.g., "ExecuteSQLNode")
        operation: The operation being performed (e.g., "sql_execution")
        log_errors: Whether to log errors automatically

    Returns:
        Decorated function that returns NodeErrorResult on errors
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                result = func(self, *args, **kwargs)
                # If the function returns a result with success=True, return it as-is
                if hasattr(result, 'success') and result.success:
                    return result
                # If no success attribute, assume it's successful and return as-is
                return result
            except DatusException as e:
                # Already a standardized DatusException, return as-is
                if log_errors:
                    logger.error(
                        f"{node_type}.{operation} failed with DatusException: {e}",
                        extra={
                            "error_code": e.code.code,
                            "node_type": node_type,
                            "operation": operation
                        }
                    )
                return _create_node_error_result(
                    self, e.code, str(e), operation, error_details=e.message_args, retryable=retryable
                )
            except ValidationError as e:
                error_details = {
                    "validation_errors": [str(err) for err in e.errors()],
                    "model_name": getattr(e, '__class__', {}).get('__name__', 'Unknown')
                }
                if log_errors:
                    logger.error(
                        f"{node_type}.{operation} failed with validation error: {str(e)}",
                        extra={
                            "node_type": node_type,
                            "operation": operation,
                            "error_type": "validation_error"
                        }
                    )
                return _create_node_error_result(
                    self, ErrorCode.COMMON_VALIDATION_FAILED, f"Validation failed: {str(e)}",
                    operation, error_details=error_details, retryable=retryable
                )
            except (TimeoutError, OSError) as e:
                # Network/connection related errors
                error_code = ErrorCode.DB_CONNECTION_TIMEOUT if "timeout" in str(e).lower() else ErrorCode.DB_CONNECTION_FAILED
                if log_errors:
                    logger.error(
                        f"{node_type}.{operation} failed with connection error: {str(e)}",
                        extra={
                            "node_type": node_type,
                            "operation": operation,
                            "error_type": "connection_error"
                        }
                    )
                return _create_node_error_result(
                    self, error_code, f"Connection error: {str(e)}",
                    operation, error_details={"connection_error": str(e)}, retryable=True
                )
            except Exception as e:
                # Generic error handling
                if log_errors:
                    logger.error(
                        f"{node_type}.{operation} failed: {str(e)}",
                        exc_info=True,
                        extra={
                            "node_type": node_type,
                            "operation": operation,
                            "error_type": type(e).__name__
                        }
                    )
                return _create_node_error_result(
                    self, ErrorCode.NODE_EXECUTION_FAILED, str(e), operation, retryable=retryable
                )
        return wrapper
    return decorator


def _create_node_error_result(
    node_instance,
    error_code: ErrorCode,
    error_message: str,
    operation: str,
    error_details: Optional[Dict[str, Any]] = None,
    retryable: bool = False
) -> NodeErrorResult:
    """
    Create a standardized NodeErrorResult with node context.

    Args:
        node_instance: The node instance that encountered the error
        error_code: The error code
        error_message: Human-readable error message
        operation: The operation that failed
        error_details: Additional error details
        retryable: Whether this error is retryable

    Returns:
        NodeErrorResult with complete context
    """
    # Extract node context
    node_context = {
        "node_id": getattr(node_instance, 'id', 'unknown'),
        "node_type": getattr(node_instance, 'type', 'unknown'),
        "operation": operation,
        "start_time": getattr(node_instance, 'start_time', None),
    }

    # Add input summary if available
    if hasattr(node_instance, '_summarize_input'):
        try:
            node_context["input_summary"] = node_instance._summarize_input()
        except Exception:
            node_context["input_summary"] = {"error": "Failed to summarize input"}

    # Enhanced error details
    enhanced_details = error_details or {}
    enhanced_details.update({
        "stack_trace": traceback.format_exc(),
        "operation": operation,
        "timestamp": __import__('time').time()
    })

    # Generate recovery suggestions and retryable flag based on error type
    recovery_suggestions = []
    retryable = False

    if error_code == ErrorCode.DB_CONNECTION_FAILED:
        recovery_suggestions = [
            "Check database connection string and credentials",
            "Verify database server is running and accessible",
            "Check network connectivity and firewall settings"
        ]
        retryable = True
    elif error_code == ErrorCode.DB_EXECUTION_TIMEOUT:
        recovery_suggestions = [
            "Consider optimizing the SQL query",
            "Check database performance and indexes",
            "Increase timeout limit if appropriate"
        ]
        retryable = True
    elif error_code == ErrorCode.COMMON_VALIDATION_FAILED:
        recovery_suggestions = [
            "Verify input data format and required fields",
            "Check data types and constraints",
            "Review input validation rules"
        ]
    elif error_code == ErrorCode.MODEL_REQUEST_FAILED:
        recovery_suggestions = [
            "Check LLM service availability and API keys",
            "Verify model configuration and parameters",
            "Retry the request or use a different model"
        ]
        retryable = True

    return NodeErrorResult(
        success=False,
        error_code=error_code.code,
        error_message=error_message,
        error_details=enhanced_details,
        node_context=node_context,
        retryable=retryable,
        recovery_suggestions=recovery_suggestions
    )


class ErrorHandlerMixin:
    """
    Mixin class providing error handling utilities for nodes.

    This mixin provides common error handling patterns that nodes can use
    to ensure consistent error reporting and logging.
    """

    def create_error_result(
        self,
        error_code: ErrorCode,
        error_message: str,
        operation: str,
        error_details: Optional[Dict[str, Any]] = None,
        retryable: bool = False
    ) -> NodeErrorResult:
        """
        Create a standardized error result for this node.

        Args:
            error_code: The error code
            error_message: Human-readable error message
            operation: The operation that failed
            error_details: Additional error details
            retryable: Whether this error is retryable

        Returns:
            NodeErrorResult with complete node context
        """
        return _create_node_error_result(
            self, error_code, error_message, operation, error_details, retryable
        )

    def log_node_error(
        self,
        error: Exception,
        operation: str,
        additional_context: Optional[Dict[str, Any]] = None
    ):
        """
        Log a node error with consistent formatting.

        Args:
            error: The exception that occurred
            operation: The operation that failed
            additional_context: Additional context to include in logs
        """
        log_context = {
            "node_id": getattr(self, 'id', 'unknown'),
            "node_type": getattr(self, 'type', 'unknown'),
            "operation": operation,
            "error_type": type(error).__name__
        }

        if additional_context:
            log_context.update(additional_context)

        logger.error(
            f"Node {getattr(self, 'type', 'unknown')} operation '{operation}' failed: {str(error)}",
            exc_info=True,
            extra=log_context
        )

    def summarize_input(self) -> Dict[str, Any]:
        """
        Generate a summary of the node's input for error context.

        Returns:
            Dictionary containing input summary
        """
        summary = {}
        if not hasattr(self, 'input') or not self.input:
            return summary

        try:
            # Handle different input types
            if hasattr(self.input, '__dict__'):
                for key, value in self.input.__dict__.items():
                    if key in ['sql_query', 'task', 'database_name', 'table_schemas', 'input_text']:
                        if isinstance(value, str) and len(value) > 100:
                            summary[key] = value[:100] + "..."
                        else:
                            summary[key] = str(value)
            elif isinstance(self.input, dict):
                for key, value in self.input.items():
                    if key in ['sql_query', 'task', 'database_name', 'input_text']:
                        if isinstance(value, str) and len(value) > 100:
                            summary[key] = value[:100] + "..."
                        else:
                            summary[key] = value
        except Exception as e:
            summary["summary_error"] = f"Failed to summarize input: {str(e)}"

        return summary


def with_error_recovery(
    max_retries: int = 3,
    backoff_factor: float = 1.0,
    retryable_errors: Optional[List[Type[Exception]]] = None
):
    """
    Decorator that adds retry logic with exponential backoff for retryable errors.

    Args:
        max_retries: Maximum number of retry attempts
        backoff_factor: Backoff multiplier for exponential backoff
        retryable_errors: List of exception types that should be retried

    Returns:
        Decorated function with retry logic
    """
    if retryable_errors is None:
        retryable_errors = [ConnectionError, TimeoutError, OSError]

    def decorator(func: F) -> F:
        @wraps(func)
        async def async_wrapper(self, *args, **kwargs):
            import asyncio

            for attempt in range(max_retries + 1):
                try:
                    return await func(self, *args, **kwargs)
                except Exception as e:
                    if attempt == max_retries or not any(isinstance(e, err_type) for err_type in retryable_errors):
                        raise

                    wait_time = backoff_factor * (2 ** attempt)
                    logger.warning(
                        f"Attempt {attempt + 1} failed for {func.__name__}, retrying in {wait_time}s: {str(e)}"
                    )
                    await asyncio.sleep(wait_time)

        @wraps(func)
        def sync_wrapper(self, *args, **kwargs):
            import time

            for attempt in range(max_retries + 1):
                try:
                    return func(self, *args, **kwargs)
                except Exception as e:
                    if attempt == max_retries or not any(isinstance(e, err_type) for err_type in retryable_errors):
                        raise

                    wait_time = backoff_factor * (2 ** attempt)
                    logger.warning(
                        f"Attempt {attempt + 1} failed for {func.__name__}, retrying in {wait_time}s: {str(e)}"
                    )
                    time.sleep(wait_time)

        # Return appropriate wrapper based on whether the function is async
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator
