# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for error classification functionality."""

import pytest

from datus.agent.node.chat_agentic_node import ChatAgenticNode


class TestErrorClassification:
    """Test error classification methods in ChatAgenticNode."""

    @pytest.fixture
    def node(self, mock_agent_config, mock_db_func_tool):
        """Create a ChatAgenticNode instance for testing."""
        return ChatAgenticNode(
            agent_id="test_agent",
            agent_config=mock_agent_config,
            db_func_tool=mock_db_func_tool,
        )

    def test_syntax_error_classification(self, node):
        """Test classification of syntax errors."""
        result = node._classify_error_type("syntax error near 'SELECT'", "validate_sql")
        assert result == "syntax_error"

    def test_parse_error_classification(self, node):
        """Test classification of parse errors."""
        result = node._classify_error_type("unable to parse SQL", "read_query")
        assert result == "syntax_error"

    def test_table_not_found_classification(self, node):
        """Test classification of table not found errors."""
        result = node._classify_error_type("Table 'users' not found", "describe_table")
        assert result == "table_not_found"

    def test_table_not_found_in_chinese(self, node):
        """Test classification of table not found errors in Chinese."""
        result = node._classify_error_type("表 'users' 不存在", "get_table_ddl")
        assert result == "table_not_found"

    def test_permission_error_classification(self, node):
        """Test classification of permission errors."""
        result = node._classify_error_type("permission denied for table", "read_query")
        assert result == "permission_error"

    def test_access_denied_classification(self, node):
        """Test classification of access denied errors."""
        result = node._classify_error_type("access denied for user", "describe_table")
        assert result == "permission_error"

    def test_timeout_error_classification(self, node):
        """Test classification of timeout errors."""
        result = node._classify_error_type("query timed out after 30s", "read_query")
        assert result == "timeout_error"

    def test_connection_error_classification(self, node):
        """Test classification of connection errors."""
        result = node._classify_error_type("could not connect to database", "read_query")
        assert result == "connection_error"

    def test_unknown_error_classification(self, node):
        """Test classification of unknown errors."""
        result = node._classify_error_type("some random error message", "describe_table")
        assert result == "unknown_error"

    def test_recovery_suggestions_syntax_error(self, node):
        """Test recovery suggestions for syntax errors."""
        suggestions = node._get_recovery_suggestions("syntax_error")
        assert len(suggestions) > 0
        assert any("SELECT" in s or "FROM" in s for s in suggestions)
        assert any("括号" in s or "引号" in s for s in suggestions)

    def test_recovery_suggestions_table_not_found(self, node):
        """Test recovery suggestions for table not found errors."""
        suggestions = node._get_recovery_suggestions("table_not_found")
        assert len(suggestions) > 0
        assert any("表名" in s or "拼写" in s for s in suggestions)

    def test_recovery_suggestions_permission_error(self, node):
        """Test recovery suggestions for permission errors."""
        suggestions = node._get_recovery_suggestions("permission_error")
        assert len(suggestions) > 0
        assert any("权限" in s for s in suggestions)

    def test_recovery_suggestions_timeout_error(self, node):
        """Test recovery suggestions for timeout errors."""
        suggestions = node._get_recovery_suggestions("timeout_error")
        assert len(suggestions) > 0
        assert any("简化" in s or "WHERE" in s for s in suggestions)

    def test_recovery_suggestions_connection_error(self, node):
        """Test recovery suggestions for connection errors."""
        suggestions = node._get_recovery_suggestions("connection_error")
        assert len(suggestions) > 0
        assert any("连接" in s or "网络" in s for s in suggestions)

    def test_recovery_suggestions_unknown_error(self, node):
        """Test recovery suggestions for unknown errors."""
        suggestions = node._get_recovery_suggestions("unknown_error")
        assert len(suggestions) > 0
        assert any("日志" in s or "技术支持" in s for s in suggestions)


class TestDynamicToolAdjustment:
    """Test dynamic tool sequence adjustment based on table availability."""

    @pytest.fixture
    def node(self, mock_agent_config, mock_db_func_tool):
        """Create a ChatAgenticNode instance for testing."""
        return ChatAgenticNode(
            agent_id="test_agent",
            agent_config=mock_agent_config,
            db_func_tool=mock_db_func_tool,
        )

    def test_no_adjustment_when_all_tables_exist(self, node):
        """Test that tools are not adjusted when all tables exist."""
        required_tools = ["describe_table", "read_query", "get_table_ddl"]
        table_names = ["users", "orders"]
        table_existence = {
            "users": {"result": {"table_exists": True}},
            "orders": {"result": {"table_exists": True}},
        }

        adjusted = node._adjust_tool_sequence_for_missing_tables(required_tools, table_names, table_existence)

        assert adjusted == required_tools

    def test_skip_schema_tools_when_tables_missing(self, node):
        """Test that schema-dependent tools are skipped when tables are missing."""
        required_tools = ["describe_table", "read_query", "get_table_ddl"]
        table_names = ["users"]
        table_existence = {"users": {"result": {"table_exists": False}}}

        adjusted = node._adjust_tool_sequence_for_missing_tables(required_tools, table_names, table_existence)

        assert "describe_table" not in adjusted
        assert "get_table_ddl" not in adjusted
        assert "read_query" in adjusted

    def test_no_adjustment_for_empty_table_names(self, node):
        """Test that tools are not adjusted when table_names is empty."""
        required_tools = ["describe_table", "read_query", "get_table_ddl"]
        table_names = []
        table_existence = {}

        adjusted = node._adjust_tool_sequence_for_missing_tables(required_tools, table_names, table_existence)

        assert adjusted == required_tools

    def test_partial_table_existence(self, node):
        """Test behavior when some tables exist and some don't."""
        required_tools = ["describe_table", "read_query", "get_table_ddl"]
        table_names = ["users", "nonexistent_table"]
        table_existence = {
            "users": {"result": {"table_exists": True}},
            "nonexistent_table": {"result": {"table_exists": False}},
        }

        adjusted = node._adjust_tool_sequence_for_missing_tables(required_tools, table_names, table_existence)

        # Should skip schema tools because at least one table is missing
        assert "describe_table" not in adjusted
        assert "get_table_ddl" not in adjusted
        assert "read_query" in adjusted


# Mock fixtures
@pytest.fixture
def mock_agent_config():
    """Create a mock agent configuration."""
    from unittest.mock import MagicMock

    config = MagicMock()
    config.db_type = "starrocks"
    config.plan_executor_config = {}
    return config


@pytest.fixture
def mock_db_func_tool():
    """Create a mock database function tool."""
    from unittest.mock import MagicMock

    tool = MagicMock()
    tool.check_table_exists = MagicMock()
    tool.describe_table = MagicMock()
    tool.get_table_ddl = MagicMock()
    tool.read_query = MagicMock()
    return tool
