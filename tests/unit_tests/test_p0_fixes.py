# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for P0 fixes.

Tests for:
1. Unified error handling mechanism
2. Soft Failure judgment logic
3. LLM retry mechanism
4. Business term configuration
5. Context update thread safety
"""

from unittest.mock import Mock, patch

import pytest

from datus.configuration.business_term_config import (
    SCHEMA_VALIDATION_CONFIG,
    get_business_term_mapping,
    get_schema_term_mapping,
    get_table_keyword_pattern,
)
from datus.utils.context_lock import (
    ContextSnapshot,
    ContextUpdateLock,
    atomic_context_merge,
    safe_context_update,
)
from datus.utils.error_handler import (
    LLMRateLimitError,
    LLMTimeoutError,
    NodeExecutionResult,
    RetryStrategy,
    check_reflect_node_reachable,
)


class TestNodeExecutionResult:
    """Test unified error handling mechanism."""

    def test_success_result(self):
        """Test creating a success result."""
        result = NodeExecutionResult.success(metadata={"tables": 5})
        assert result.is_success
        assert not result.is_soft_failure
        assert not result.is_hard_failure
        assert result.metadata["tables"] == 5

    def test_soft_failure_result(self):
        """Test creating a soft failure result."""
        result = NodeExecutionResult.soft_failure(
            error_message="No tables found",
            metadata={"query": "test"},
        )
        assert result.is_soft_failure
        assert not result.is_success
        assert not result.is_hard_failure
        assert result.can_retry

    def test_hard_failure_result(self):
        """Test creating a hard failure result."""
        result = NodeExecutionResult.hard_failure(
            error_message="Database connection failed",
        )
        assert result.is_hard_failure
        assert not result.is_success
        assert not result.is_soft_failure
        assert not result.can_retry


class TestCheckReflectNodeReachable:
    """Test Soft Failure judgment logic fix."""

    def test_no_workflow(self):
        """Test with no workflow."""
        assert not check_reflect_node_reachable(None)

    def test_no_reflect_node(self):
        """Test workflow without reflect node."""
        workflow = Mock()
        workflow.nodes = {"node1": Mock(type="text2sql"), "node2": Mock(type="validation")}
        workflow.current_node_index = 0
        assert not check_reflect_node_reachable(workflow)

    def test_reflect_node_reachable(self):
        """Test reflect node is reachable."""
        workflow = Mock()
        node1 = Mock(type="text2sql")
        node2 = Mock(type="validation")
        node3 = Mock(type="reflect", status="pending")

        # Create a mock dict with items() method
        workflow.nodes = {"node1": node1, "node2": node2, "node3": node3}
        workflow.current_node_index = 0

        assert check_reflect_node_reachable(workflow)

    def test_reflect_node_not_reachable_already_executed(self):
        """Test reflect node already executed."""
        workflow = Mock()
        node1 = Mock(type="text2sql")
        node2 = Mock(type="reflect", status="completed")

        workflow.nodes = {"node1": node1, "node2": node2}
        workflow.current_node_index = 2  # After reflect node

        assert not check_reflect_node_reachable(workflow)


class TestBusinessTermConfig:
    """Test business term configuration."""

    def test_get_business_term_mapping(self):
        """Test getting business term mappings."""
        mappings = get_business_term_mapping("试驾")
        assert isinstance(mappings, list)
        assert len(mappings) > 0
        assert "test_drive" in mappings or "dwd_assign_dlr_clue_fact_di" in mappings

    def test_get_schema_term_mapping(self):
        """Test getting schema term mappings."""
        mappings = get_schema_term_mapping("订单")
        assert isinstance(mappings, list)
        assert len(mappings) > 0
        assert "order" in mappings or "orders" in mappings

    def test_get_table_keyword_pattern(self):
        """Test getting table keyword patterns."""
        table_name = get_table_keyword_pattern("用户")
        assert table_name == "users"

        # Test non-existent term
        table_name = get_table_keyword_pattern("nonexistent")
        assert table_name == ""

    def test_coverage_thresholds_config(self):
        """Test coverage thresholds configuration."""
        thresholds = SCHEMA_VALIDATION_CONFIG["coverage_thresholds"]
        assert "simple" in thresholds
        assert "medium" in thresholds
        assert "complex" in thresholds

        assert thresholds["simple"]["threshold"] == 0.5
        assert thresholds["medium"]["threshold"] == 0.3
        assert thresholds["complex"]["threshold"] == 0.2


class TestContextUpdateLock:
    """Test thread-safe context updates."""

    def test_context_lock_singleton(self):
        """Test that ContextUpdateLock is a singleton."""
        lock1 = ContextUpdateLock()
        lock2 = ContextUpdateLock()
        assert lock1 is lock2

    def test_safe_context_update(self):
        """Test safe context update."""
        context = Mock()
        context.value = 0

        def update_func():
            context.value = 42
            return {"updated": True}

        result = safe_context_update(context, update_func, "test_update")
        assert result["success"]
        assert context.value == 42

    def test_safe_context_update_with_error(self):
        """Test safe context update with error."""
        context = Mock()

        def update_func():
            raise ValueError("Test error")

        result = safe_context_update(context, update_func, "test_update")
        assert not result["success"]
        assert "error" in result

    def test_atomic_context_merge(self):
        """Test atomic context merge."""
        context = Mock()
        context.list_attr = [1, 2, 3]

        updates = {"list_attr": [4, 5]}

        # Test extend strategy
        result = atomic_context_merge(context, updates, merge_strategy="extend")
        assert result["success"]
        assert context.list_attr == [1, 2, 3, 4, 5]

    def test_context_snapshot(self):
        """Test context snapshot and restore."""
        context = Mock()
        context.value1 = "test"
        context.value2 = [1, 2, 3]

        snapshot = ContextSnapshot(context)
        snapshot.capture()

        # Modify context
        context.value1 = "modified"
        context.value2.append(4)

        # Restore
        success = snapshot.restore()
        assert success
        assert context.value1 == "test"
        assert context.value2 == [1, 2, 3]


class TestLLMRetryMechanism:
    """Test LLM retry mechanism (integration test)."""

    @patch("datus.utils.error_handler.logger")
    def test_llm_timeout_classification(self, mock_logger):
        """Test that timeout errors are properly classified."""
        # The error handler should classify asyncio.TimeoutError as LLMTimeoutError
        # This is tested via the actual retry mechanism in integration tests
        assert LLMTimeoutError.__name__ == "LLMTimeoutError"
        assert LLMRateLimitError.__name__ == "LLMRateLimitError"

    def test_retry_strategy_enum(self):
        """Test retry strategy enum."""
        assert RetryStrategy.NONE.value == "none"
        assert RetryStrategy.EXPONENTIAL_BACKOFF.value == "exponential_backoff"
        assert RetryStrategy.IMMEDIATE.value == "immediate"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
