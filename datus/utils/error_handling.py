# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unified Error Handling for Datus-agent Node System.

This module provides comprehensive error handling mechanisms for all nodes,
including:
- Unified error result formats (NodeErrorResult, NodeExecutionResult)
- Error handling decorators (unified_error_handler, with_error_recovery)
- LLM retry logic with caching (LLMMixin)
- Consistent error logging and reporting
- Workflow utilities (check_reflect_node_reachable)
"""

import asyncio
import json
import time
import traceback
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, cast

from pydantic import ValidationError
from tenacity import (
    after_log,
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from datus.schemas.base import BaseResult
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


# =============================================================================
# Result Classes
# =============================================================================


class NodeStatus(Enum):
    """Unified node execution status."""

    SUCCESS = "success"
    SOFT_FAILED = "soft_failed"  # Recoverable failure
    FAILED = "failed"  # Hard failure, terminate workflow
    PROCESSING = "processing"


class RetryStrategy(Enum):
    """Retry strategy for failed operations."""

    NONE = "none"  # No retry
    EXPONENTIAL_BACKOFF = "exponential_backoff"  # Exponential backoff
    FIXED_DELAY = "fixed_delay"  # Fixed delay between retries
    IMMEDIATE = "immediate"  # Retry immediately


class NodeErrorResult(BaseResult):
    """
    Unified node error result with enhanced context information.

    This class provides a standardized format for error reporting across all nodes,
    including error details, context, retry information, and recovery suggestions.
    """

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
        super().__init__(success=success, error=error_message if error_message else "", **kwargs)
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


@dataclass
class NodeExecutionResult:
    """
    Unified result structure for node execution.

    This provides a consistent interface for node execution outcomes,
    enabling proper error propagation and retry logic.

    Used primarily by LLMMixin for LLM operation results.
    """

    status: NodeStatus
    error_code: Optional[ErrorCode] = None
    error_message: Optional[str] = None
    retry_strategy: RetryStrategy = RetryStrategy.NONE
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Ensure metadata is initialized."""
        if self.metadata is None:
            self.metadata = {}

    @classmethod
    def success(cls, metadata: Optional[Dict[str, Any]] = None) -> "NodeExecutionResult":
        """Create a successful result."""
        return cls(status=NodeStatus.SUCCESS, metadata=metadata or {})

    @classmethod
    def soft_failure(
        cls,
        error_message: str,
        error_code: ErrorCode = ErrorCode.NODE_EXECUTION_FAILED,
        metadata: Optional[Dict[str, Any]] = None,
        retry_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF,
    ) -> "NodeExecutionResult":
        """Create a soft failure result (recoverable)."""
        return cls(
            status=NodeStatus.SOFT_FAILED,
            error_code=error_code,
            error_message=error_message,
            metadata=metadata or {},
            retry_strategy=retry_strategy,
        )

    @classmethod
    def hard_failure(
        cls,
        error_message: str,
        error_code: ErrorCode = ErrorCode.NODE_EXECUTION_FAILED,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "NodeExecutionResult":
        """Create a hard failure result (terminate workflow)."""
        return cls(
            status=NodeStatus.FAILED,
            error_code=error_code,
            error_message=error_message,
            metadata=metadata or {},
            retry_strategy=RetryStrategy.NONE,
        )

    @property
    def is_success(self) -> bool:
        """Check if execution was successful."""
        return self.status == NodeStatus.SUCCESS

    @property
    def is_soft_failure(self) -> bool:
        """Check if this is a recoverable failure."""
        return self.status == NodeStatus.SOFT_FAILED

    @property
    def is_hard_failure(self) -> bool:
        """Check if this is a terminal failure."""
        return self.status == NodeStatus.FAILED

    @property
    def can_retry(self) -> bool:
        """Check if this execution can be retried."""
        return self.retry_strategy != RetryStrategy.NONE


# =============================================================================
# LLM Error Classes
# =============================================================================


class RetryableLLMError(Exception):
    """Base class for retryable LLM errors."""


class LLMTimeoutError(RetryableLLMError):
    """LLM request timeout."""


class LLMRateLimitError(RetryableLLMError):
    """LLM rate limit exceeded."""


class LLMInvalidResponseError(Exception):
    """LLM returned invalid response (non-retryable)."""


# =============================================================================
# Decorators
# =============================================================================


def unified_error_handler(node_type: str, operation: str, log_errors: bool = True):
    """
    Unified error handling decorator for node methods.

    This decorator provides consistent error handling across all node types,
    ensuring proper logging, error classification, and context preservation.

    Handles:
    - DatusException (already formatted)
    - ValidationError (Pydantic validation errors)
    - JSON parsing errors
    - Connection/Timeout errors (retryable)
    - Generic exceptions

    Args:
        node_type: The type of node (e.g., 'ExecuteSQLNode')
        operation: The operation being performed (e.g., 'sql_execution')
        log_errors: Whether to log errors automatically

    Returns:
        Decorated function that handles errors consistently
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                result = func(self, *args, **kwargs)
                # If the function returns a result with success=True, return it as-is
                if hasattr(result, "success") and result.success:
                    return result
                # If no success attribute, assume it's successful and return as-is
                return result
            except DatusException as e:
                # Already a standardized DatusException
                if log_errors:
                    logger.error(
                        f"{node_type}.{operation} failed with DatusException: {e}",
                        extra={
                            "error_code": e.code.code,
                            "node_type": node_type,
                            "operation": operation,
                            "node_id": getattr(self, "id", "unknown"),
                        },
                    )
                error_result = _create_node_error_result(
                    self, e.code, str(e), operation, error_details=e.message_args
                )
                self.result = error_result
                return error_result
            except ValidationError as e:
                # Pydantic validation errors
                error_details = {
                    "validation_errors": [str(err) for err in e.errors()],
                    "model_name": getattr(e, "__class__", {}).get("__name__", "Unknown"),
                }
                if log_errors:
                    logger.error(
                        f"{node_type}.{operation} failed with validation error: {str(e)}",
                        extra={
                            "node_type": node_type,
                            "operation": operation,
                            "error_type": "validation_error",
                        },
                    )
                error_result = _create_node_error_result(
                    self,
                    ErrorCode.COMMON_VALIDATION_FAILED,
                    f"Validation failed: {str(e)}",
                    operation,
                    error_details=error_details,
                )
                self.result = error_result
                return error_result
            except json.JSONDecodeError as e:
                # JSON parsing errors specifically
                if log_errors:
                    logger.debug(
                        f"JSON parsing error details: {e}",
                        extra={
                            "json_error": str(e),
                            "doc_preview": (e.doc[:500] + "...") if len(e.doc) > 500 else e.doc,
                            "pos": e.pos,
                        },
                    )
                error_details = {"json_error": str(e), "pos": e.pos, "doc_preview": e.doc[:500] if len(e.doc) > 500 else e.doc}
                error_result = _create_node_error_result(
                    self,
                    ErrorCode.COMMON_JSON_PARSE_ERROR,
                    f"JSON parsing failed in {operation}: {str(e)}",
                    operation,
                    error_details,
                )
                self.result = error_result
                return error_result
            except (TimeoutError, ConnectionError, OSError) as e:
                # Network/connection related errors
                if isinstance(e, TimeoutError):
                    error_code = ErrorCode.DB_EXECUTION_TIMEOUT
                    retryable = True
                elif isinstance(e, ConnectionError):
                    error_code = ErrorCode.DB_CONNECTION_FAILED
                    retryable = True
                else:
                    # OSError but not TimeoutError or ConnectionError
                    error_code = ErrorCode.DB_CONNECTION_FAILED
                    retryable = True

                if log_errors:
                    logger.error(
                        f"{node_type}.{operation} failed with connection error: {str(e)}",
                        extra={
                            "node_type": node_type,
                            "operation": operation,
                            "error_type": type(e).__name__,
                        },
                    )
                error_result = _create_node_error_result(
                    self,
                    error_code,
                    f"Connection error: {str(e)}",
                    operation,
                    error_details={"connection_error": str(e)},
                    retryable=retryable,
                )
                self.result = error_result
                return error_result
            except Exception as e:
                # Generic error handling
                if log_errors:
                    logger.error(
                        f"{node_type}.{operation} failed: {str(e)}",
                        exc_info=True,
                        extra={
                            "node_type": node_type,
                            "operation": operation,
                            "node_id": getattr(self, "id", "unknown"),
                            "error_type": type(e).__name__,
                            "stack_trace": traceback.format_exc(),
                        },
                    )
                error_result = _create_node_error_result(
                    self,
                    ErrorCode.NODE_EXECUTION_FAILED,
                    str(e),
                    operation,
                    error_details={"unexpected_error": str(e), "stack_trace": traceback.format_exc()},
                    retryable=_is_retryable_error(e),
                )
                self.result = error_result
                return error_result

        return wrapper

    return decorator


def with_error_recovery(
    max_retries: int = 3, backoff_factor: float = 1.0, retryable_errors: Optional[List[Type[Exception]]] = None
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
            for attempt in range(max_retries + 1):
                try:
                    return await func(self, *args, **kwargs)
                except Exception as e:
                    if attempt == max_retries or not any(isinstance(e, err_type) for err_type in retryable_errors):
                        raise

                    wait_time = backoff_factor * (2**attempt)
                    logger.warning(
                        f"Attempt {attempt + 1} failed for {func.__name__}, retrying in {wait_time}s: {str(e)}"
                    )
                    await asyncio.sleep(wait_time)

        @wraps(func)
        def sync_wrapper(self, *args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return func(self, *args, **kwargs)
                except Exception as e:
                    if attempt == max_retries or not any(isinstance(e, err_type) for err_type in retryable_errors):
                        raise

                    wait_time = backoff_factor * (2**attempt)
                    logger.warning(
                        f"Attempt {attempt + 1} failed for {func.__name__}, retrying in {wait_time}s: {str(e)}"
                    )
                    time.sleep(wait_time)

        # Return appropriate wrapper based on whether the function is async
        if asyncio.iscoroutinefunction(func):
            return cast(F, async_wrapper)
        else:
            return cast(F, sync_wrapper)

    return decorator


# =============================================================================
# Mixin Classes
# =============================================================================


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
        retryable: bool = False,
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
        return _create_node_error_result(self, error_code, error_message, operation, error_details, retryable)

    def log_node_error(self, error: Exception, operation: str, additional_context: Optional[Dict[str, Any]] = None):
        """
        Log a node error with consistent formatting.

        Args:
            error: The exception that occurred
            operation: The operation that failed
            additional_context: Additional context to include in logs
        """
        log_context = {
            "node_id": getattr(self, "id", "unknown"),
            "node_type": getattr(self, "type", "unknown"),
            "operation": operation,
            "error_type": type(error).__name__,
        }

        if additional_context:
            log_context.update(additional_context)

        logger.error(
            f"Node {getattr(self, 'type', 'unknown')} operation '{operation}' failed: {str(error)}",
            exc_info=True,
            extra=log_context,
        )

    def summarize_input(self) -> Dict[str, Any]:
        """
        Generate a summary of the node's input for error context.

        Returns:
            Dictionary containing input summary
        """
        summary: Dict[str, Any] = {}

        if not hasattr(self, "input") or not self.input:
            return summary

        try:
            # Handle different input types
            if hasattr(self.input, "__dict__"):
                for key, value in self.input.__dict__.items():
                    if key in ["sql_query", "task", "database_name", "table_schemas", "input_text"]:
                        if isinstance(value, str) and len(value) > 100:
                            summary[key] = value[:100] + "..."
                        else:
                            summary[key] = str(value)
            elif isinstance(self.input, dict):
                for key, value in self.input.items():
                    if key in ["sql_query", "task", "database_name", "input_text"]:
                        if isinstance(value, str) and len(value) > 100:
                            summary[key] = value[:100] + "..."
                        else:
                            summary[key] = value
        except Exception as e:
            summary["summary_error"] = f"Failed to summarize input: {str(e)}"

        return summary


class LLMMixin:
    """
    Mixin class providing robust LLM calling with retry logic.

    Features:
    - Automatic retry with exponential backoff for retryable errors
    - Response caching to reduce redundant API calls
    - Intelligent error detection (rate limits, timeouts, invalid responses)
    - Detailed logging and metrics

    Usage:
        class MyNode(Node, LLMMixin):
            async def run(self):
                response = await self.llm_call_with_retry(
                    prompt="...",
                    operation_name="table_discovery"
                )
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._llm_call_cache = {}  # {cache_key: (expires_at, value)}

    async def llm_call_with_retry(
        self,
        prompt: str,
        operation_name: str = "llm_call",
        max_retries: int = 3,
        cache_key: Optional[str] = None,
        cache_ttl_seconds: Optional[int] = None,
        **llm_kwargs,
    ) -> Dict[str, Any]:
        """
        Call LLM with automatic retry and caching.

        Args:
            prompt: The prompt to send to LLM
            operation_name: Name for logging/metrics
            max_retries: Maximum number of retry attempts
            cache_key: Optional cache key to reuse results
            cache_ttl_seconds: Cache TTL in seconds (None = no expiration)
            **llm_kwargs: Additional arguments to pass to LLM

        Returns:
            Dict containing LLM response

        Raises:
            NodeExecutionResult: If all retries exhausted
        """
        # Check cache first
        if cache_key and cache_key in self._llm_call_cache:
            expires_at, cached_value = self._llm_call_cache[cache_key]
            if expires_at is None or expires_at > time.time():
                logger.debug(f"LLM cache hit for operation: {operation_name}")
                return cached_value
            del self._llm_call_cache[cache_key]

        # Get model from instance
        if not hasattr(self, "model"):
            raise RuntimeError("LLMMixin requires 'model' attribute on the instance")

        model = self.model

        # Configure retry based on operation type
        if operation_name in ["table_discovery", "term_extraction"]:
            # Discovery operations need more retries
            max_retries = max(max_retries, 3)
            multiplier = 1
            min_wait = 2
        else:
            multiplier = 1
            min_wait = 4

        @retry(
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential(multiplier=multiplier, min=min_wait, max=60),
            retry=retry_if_exception_type(RetryableLLMError),
            before_sleep=before_sleep_log(logger, logger.level) if hasattr(logger, "level") else None,
            after=after_log(logger, logger.level) if hasattr(logger, "level") else None,
            reraise=True,
        )
        def _llm_call_sync():
            """Synchronous LLM call with retry."""
            try:
                if hasattr(model, "generate_with_json_output"):
                    return model.generate_with_json_output(prompt, **llm_kwargs)
                elif hasattr(model, "generate"):
                    return {"result": model.generate(prompt, **llm_kwargs)}
                else:
                    raise ValueError(f"Model {type(model)} has no compatible generate method")
            except asyncio.TimeoutError as e:
                logger.warning(f"LLM timeout for {operation_name}: {e}")
                raise LLMTimeoutError(f"LLM request timeout: {operation_name}") from e
            except Exception as e:
                error_msg = str(e).lower()
                # Detect rate limiting
                if "rate limit" in error_msg or "429" in error_msg or "quota" in error_msg:
                    logger.warning(f"LLM rate limit for {operation_name}: {e}")
                    raise LLMRateLimitError(f"Rate limit exceeded: {operation_name}") from e
                # Detect invalid JSON/response
                elif "json" in error_msg or "parse" in error_msg or "invalid" in error_msg:
                    logger.warning(f"LLM invalid response for {operation_name}: {e}")
                    raise LLMInvalidResponseError(f"Invalid LLM response: {operation_name}") from e
                # Other errors are not retryable
                else:
                    logger.error(f"LLM non-retryable error for {operation_name}: {e}")
                    raise

        try:
            start_time = time.time()
            result = _llm_call_sync()
            elapsed = time.time() - start_time

            # Get retry statistics if available
            retry_stats = getattr(_llm_call_sync, "retry", None)
            if retry_stats and hasattr(retry_stats, "statistics"):
                attempt_number = retry_stats.statistics.get("attempt_number", 1)
            else:
                attempt_number = 1

            logger.info(f"LLM call succeeded: {operation_name} " f"(took {elapsed:.2f}s, attempt: {attempt_number})")

            # Cache successful results
            if cache_key:
                ttl = cache_ttl_seconds if cache_ttl_seconds and cache_ttl_seconds > 0 else None
                expires_at = time.time() + ttl if ttl else None
                self._llm_call_cache[cache_key] = (expires_at, result)

            return result

        except Exception as e:
            logger.error(f"LLM call failed after {max_retries} retries: {operation_name} - {e}")

            # Return structured error instead of raising
            raise NodeExecutionResult.soft_failure(
                error_message=f"LLM call failed for {operation_name}: {str(e)}",
                error_code=ErrorCode.NODE_EXECUTION_FAILED,
                metadata={"operation": operation_name, "retries": max_retries},
            )


# =============================================================================
# Helper Functions
# =============================================================================


def _create_node_error_result(
    node_instance: Any,
    error_code: ErrorCode,
    error_message: str,
    operation: str,
    error_details: Optional[Dict[str, Any]] = None,
    retryable: bool = False,
) -> NodeErrorResult:
    """
    Create a standardized NodeErrorResult with node context.

    Args:
        node_instance: The node instance that encountered the error
        error_code: The error code
        error_message: Human-readable error message
        operation: The operation that failed
        error_details: Additional error details
        retryable: Whether this error is retryable (note: may be overridden by error type logic)

    Returns:
        NodeErrorResult with complete context
    """
    # Extract node context
    node_context = {
        "node_id": getattr(node_instance, "id", "unknown"),
        "node_type": getattr(node_instance, "type", "unknown"),
        "operation": operation,
        "start_time": getattr(node_instance, "start_time", None),
    }

    # Add input summary if available
    if hasattr(node_instance, "summarize_input"):
        try:
            node_context["input_summary"] = node_instance.summarize_input()
        except Exception as e:
            logger.error(f"Failed to summarize input: {e}", exc_info=True)
            node_context["input_summary"] = {"error": "Failed to summarize input"}
    elif hasattr(node_instance, "_summarize_input"):
        try:
            node_context["input_summary"] = node_instance._summarize_input()
        except Exception as e:
            logger.error(f"Failed to summarize input: {e}", exc_info=True)
            node_context["input_summary"] = {"error": "Failed to summarize input"}

    # Enhanced error details
    enhanced_details = error_details or {}
    try:
        enhanced_details.update(
            {"stack_trace": traceback.format_exc(), "operation": operation, "timestamp": time.time()}
        )
    except Exception as e:
        logger.warning(f"Failed to update error details: {e}")
        enhanced_details["update_error"] = str(e)

    # Generate recovery suggestions based on error type
    # Note: retryable flag is set based on error type below; caller-provided value is ignored
    recovery_suggestions = _generate_recovery_suggestions(error_code, operation)
    retryable = _is_retryable_by_code(error_code) or retryable

    return NodeErrorResult(
        success=False,
        error_code=error_code.code,
        error_message=error_message,
        error_details=enhanced_details,
        node_context=node_context,
        retryable=retryable,
        recovery_suggestions=recovery_suggestions,
    )


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


def _is_retryable_by_code(error_code: ErrorCode) -> bool:
    """
    Determine if an error is retryable based on its error code.

    Args:
        error_code: The error code

    Returns:
        True if the error is retryable by default
    """
    retryable_codes = [
        ErrorCode.DB_CONNECTION_FAILED,
        ErrorCode.DB_EXECUTION_TIMEOUT,
        ErrorCode.MODEL_REQUEST_FAILED,
    ]
    return error_code in retryable_codes


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
            [
                "Check database connection string and credentials",
                "Verify database server is running and accessible",
                "Check network connectivity and firewall settings",
            ]
        )
    elif error_code == ErrorCode.DB_EXECUTION_TIMEOUT:
        suggestions.extend(
            ["Consider optimizing the SQL query", "Check database performance and indexes", "Increase timeout limit if appropriate"]
        )
    elif error_code == ErrorCode.COMMON_VALIDATION_FAILED:
        suggestions.extend(["Verify input data format and required fields", "Check data types and constraints", "Review input validation rules"])
    elif error_code == ErrorCode.COMMON_JSON_PARSE_ERROR:
        suggestions.extend(["Validate JSON format", "Check for special characters in the input", "Review the prompt template"])
    elif error_code == ErrorCode.MODEL_REQUEST_FAILED:
        suggestions.extend(
            ["Check LLM service availability and API keys", "Verify model configuration and parameters", "Retry the request or use a different model"]
        )
    elif error_code == ErrorCode.NODE_EXECUTION_FAILED:
        suggestions.extend(["Review input parameters", "Check system resources", "Enable debug logging for more details"])

    # Operation-specific suggestions
    if operation == "sql_execution":
        suggestions.extend(["Validate SQL syntax", "Check table and column names", "Verify database permissions"])
    elif operation == "schema_linking":
        suggestions.extend(["Check database schema access", "Verify table metadata availability", "Try with different matching rates"])
    elif operation == "model_generation":
        suggestions.extend(["Check model configuration", "Validate prompt templates", "Try with different model parameters"])

    return suggestions[:5]  # Limit to top 5 suggestions


def check_reflect_node_reachable(workflow) -> bool:
    """
    Check if a reflect node is reachable from the current node.

    This is more accurate than just checking if a reflect node exists,
    as it verifies the reflect node is in the execution path.

    Args:
        workflow: The workflow instance

    Returns:
        True if reflect node can be reached, False otherwise
    """
    if not workflow or not workflow.nodes:
        return False

    # Check if any reflect node exists
    reflect_nodes = [
        (node_id, node)
        for node_id, node in workflow.nodes.items()
        if node.type == "reflect" and node.status not in ["completed", "skipped"]
    ]

    if not reflect_nodes:
        return False

    # Get current node index
    current_idx = workflow.current_node_index
    if current_idx is None:
        return False

    # Check if any reflect node is after current position
    for node_id, node in reflect_nodes:
        # Get node index from workflow
        node_idx = None
        for idx, (n_id, _) in enumerate(workflow.nodes.items()):
            if n_id == node_id:
                node_idx = idx
                break

        if node_idx is not None and node_idx > current_idx:
            logger.debug(f"Reflect node {node_id} is reachable (index {node_idx} > {current_idx})")
            return True

    logger.debug("No reachable reflect nodes found")
    return False
