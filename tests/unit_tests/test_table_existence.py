# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for table existence checking functionality."""

from unittest.mock import MagicMock, Mock

import pytest

from datus.tools.func_tool.database import DBFuncTool
from datus.tools.schemas import FuncToolResult


class TestTableExistenceChecking:
    """Test table existence checking in DBFuncTool."""

    @pytest.fixture
    def mock_connector(self):
        """Create a mock database connector."""
        connector = MagicMock()
        connector.get_tables = MagicMock()
        return connector

    @pytest.fixture
    def db_func_tool(self, mock_connector, mock_agent_config):
        """Create a DBFuncTool instance for testing."""
        tool = DBFuncTool(
            connector=mock_connector,
            agent_config=mock_agent_config,
        )
        tool.logger = MagicMock()  # Mock logger
        return tool

    def test_table_exists_returns_true(self, db_func_tool, mock_connector):
        """Test that check_table_exists returns True for existing table."""
        mock_connector.get_tables.return_value = [
            {"name": "users"},
            {"name": "orders"},
            {"name": "products"},
        ]

        result = db_func_tool.check_table_exists("users")

        assert result.success == 1
        assert result.result["table_exists"] is True
        assert "suggestions" in result.result
        assert result.result["suggestions"] == []

    def test_table_exists_case_insensitive(self, db_func_tool, mock_connector):
        """Test that check_table_exists is case-insensitive."""
        mock_connector.get_tables.return_value = [
            {"name": "Users"},
            {"name": "Orders"},
        ]

        result = db_func_tool.check_table_exists("users")

        assert result.success == 1
        assert result.result["table_exists"] is True

    def test_table_not_found_returns_false(self, db_func_tool, mock_connector):
        """Test that check_table_exists returns False for non-existing table."""
        mock_connector.get_tables.return_value = [
            {"name": "users"},
            {"name": "orders"},
        ]

        result = db_func_tool.check_table_exists("products")

        assert result.success == 1
        assert result.result["table_exists"] is False

    def test_table_not_found_with_suggestions(self, db_func_tool, mock_connector):
        """Test that check_table_exists provides suggestions for similar table names."""
        mock_connector.get_tables.return_value = [
            {"name": "users"},
            {"name": "user_profiles"},
            {"name": "orders"},
        ]

        result = db_func_tool.check_table_exists("usr")

        assert result.success == 1
        assert result.result["table_exists"] is False
        assert "users" in result.result["suggestions"]

    def test_table_not_found_no_suggestions(self, db_func_tool, mock_connector):
        """Test that check_table_exists returns empty suggestions when no similar tables."""
        mock_connector.get_tables.return_value = [
            {"name": "products"},
            {"name": "orders"},
        ]

        result = db_func_tool.check_table_exists("xyzabc")

        assert result.success == 1
        assert result.result["table_exists"] is False
        assert len(result.result["suggestions"]) == 0

    def test_table_existence_with_catalog_database_schema(self, db_func_tool, mock_connector):
        """Test that check_table_exists passes catalog, database, and schema parameters."""
        mock_connector.get_tables.return_value = [{"name": "users"}]

        result = db_func_tool.check_table_exists(
            table_name="users",
            catalog="my_catalog",
            database="my_database",
            schema_name="my_schema",
        )

        mock_connector.get_tables.assert_called_once_with(
            catalog_name="my_catalog",
            database_name="my_database",
            schema_name="my_schema",
        )
        assert result.success == 1

    def test_table_existence_with_empty_tables_list(self, db_func_tool, mock_connector):
        """Test that check_table_exists handles empty table list."""
        mock_connector.get_tables.return_value = []

        result = db_func_tool.check_table_exists("users")

        assert result.success == 1
        assert result.result["table_exists"] is False
        assert result.result["available_tables"] == []

    def test_table_existence_error_handling(self, db_func_tool, mock_connector):
        """Test that check_table_exists handles exceptions gracefully."""
        mock_connector.get_tables.side_effect = Exception("Connection error")

        result = db_func_tool.check_table_exists("users")

        assert result.success == 0
        assert "error" in result.__dict__

    def test_available_tables_limited_to_20(self, db_func_tool, mock_connector):
        """Test that available_tables list is limited to 20 entries."""
        # Create 25 tables
        tables = [{"name": f"table_{i}"} for i in range(25)]
        mock_connector.get_tables.return_value = tables

        result = db_func_tool.check_table_exists("users")

        assert result.success == 1
        assert len(result.result["available_tables"]) == 20


class TestTableExistenceIntegration:
    """Integration tests for table existence checking with ChatAgenticNode."""

    @pytest.fixture
    def node(self, mock_agent_config, mock_db_func_tool):
        """Create a ChatAgenticNode instance for testing."""
        from datus.agent.node.chat_agentic_node import ChatAgenticNode

        return ChatAgenticNode(
            agent_id="test_agent",
            agent_config=mock_agent_config,
            db_func_tool=mock_db_func_tool,
        )

    def test_execute_preflight_tool_check_table_exists(self, node):
        """Test that check_table_exists tool is executed correctly."""
        import asyncio

        async def run_test():
            # Mock the check_table_exists result
            node.db_func_tool.check_table_exists.return_value = FuncToolResult(
                result={"table_exists": True, "suggestions": [], "available_tables": ["users", "orders"]}
            )

            result = await node._execute_preflight_tool(
                tool_name="check_table_exists",
                sql_query="SELECT * FROM users",
                table_names=["users"],
                catalog="",
                database="",
                schema="",
            )

            assert result["success"] == 1
            assert result["result"]["table_exists"] is True
            node.db_func_tool.check_table_exists.assert_called_once()

        asyncio.run(run_test())


# Mock fixtures
@pytest.fixture
def mock_agent_config():
    """Create a mock agent configuration."""
    config = MagicMock()
    config.db_type = "starrocks"
    config.plan_executor_config = {}
    return config


@pytest.fixture
def mock_db_func_tool():
    """Create a mock database function tool."""
    tool = MagicMock()
    tool.check_table_exists = MagicMock()
    tool.describe_table = MagicMock()
    tool.get_table_ddl = MagicMock()
    tool.read_query = MagicMock()
    return tool
