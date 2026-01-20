# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unified error handling and retry mechanisms for node execution.
"""

import asyncio
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

from tenacity import (
    after_log,
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from datus.utils.exceptions import ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


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


@dataclass
class NodeExecutionResult:
    """
    Unified result structure for node execution.

    This provides a consistent interface for node execution outcomes,
    enabling proper error propagation and retry logic.
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


class RetryableLLMError(Exception):
    """Base class for retryable LLM errors."""


class LLMTimeoutError(RetryableLLMError):
    """LLM request timeout."""


class LLMRateLimitError(RetryableLLMError):
    """LLM rate limit exceeded."""


class LLMInvalidResponseError(Exception):
    """LLM returned invalid response (non-retryable)."""


class LLMMixin:
    """
    Mixin class providing robust LLM calling with retry logic.

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
