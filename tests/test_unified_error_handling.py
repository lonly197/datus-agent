# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Tests for unified error handling in Datus-agent node system.
"""

import pytest

from datus.agent.error_handling import ErrorHandlerMixin, NodeErrorResult
from datus.utils.exceptions import ErrorCode


class MockNode(ErrorHandlerMixin):
    """Mock node for testing error handling."""

    def __init__(self, node_id="test_node", node_type="test"):
        self.id = node_id
        self.type = node_type
        self.input = None


class TestNodeErrorResult:
    """Test NodeErrorResult class."""

    def test_node_error_result_creation(self):
        """Test creating NodeErrorResult with basic parameters."""
        result = NodeErrorResult(
            success=False,
            error_code="TEST001",
            error_message="Test error",
            error_details={"test": "detail"},
            retryable=True,
            recovery_suggestions=["Try again"]
        )

        assert result.success is False
        assert result.error_code == "TEST001"
        assert result.error == "Test error"
        assert result.error_details == {"test": "detail"}
        assert result.retryable is True
        assert result.recovery_suggestions == ["Try again"]

    def test_node_error_result_defaults(self):
        """Test NodeErrorResult with default values."""
        result = NodeErrorResult()

        assert result.success is False
        assert result.error_code == ""
        assert result.error == ""
        assert result.error_details == {}
        assert result.node_context == {}
        assert result.retryable is False
        assert result.recovery_suggestions == []


class TestErrorHandlerMixin:
    """Test ErrorHandlerMixin functionality."""

    def test_create_error_result(self):
        """Test creating error result through mixin."""
        node = MockNode("test_node", "test_type")

        result = node.create_error_result(
            ErrorCode.NODE_EXECUTION_FAILED,
            "Test execution failed",
            "test_operation",
            error_details={"extra": "info"}
        )

        assert isinstance(result, NodeErrorResult)
        assert result.success is False
        assert result.error_code == ErrorCode.NODE_EXECUTION_FAILED.code
        assert result.error == "Test execution failed"
        assert result.error_details["extra"] == "info"
        assert result.retryable is False  # NODE_EXECUTION_FAILED is not retryable by default
        assert result.node_context["node_id"] == "test_node"
        assert result.node_context["node_type"] == "test_type"
        assert result.node_context["operation"] == "test_operation"

    def test_summarize_input_no_input(self):
        """Test input summarization when no input is available."""
        node = MockNode()
        node.input = None

        summary = node.summarize_input()
        assert summary == {}

    def test_summarize_input_with_dict(self):
        """Test input summarization with dictionary input."""
        node = MockNode()
        node.input = {
            "sql_query": "SELECT * FROM test_table",
            "database_name": "test_db",
            "other_field": "ignored"
        }

        summary = node.summarize_input()
        assert summary["sql_query"] == "SELECT * FROM test_table"
        assert summary["database_name"] == "test_db"
        assert "other_field" not in summary