# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Tests for ExecuteSQLNode unified error handling.
"""

from unittest.mock import Mock, patch

import pytest

from datus.agent.error_handling import NodeErrorResult
from datus.agent.node.execute_sql_node import ExecuteSQLNode
from datus.configuration.agent_config import AgentConfig
from datus.schemas.node_models import ExecuteSQLInput
from datus.utils.exceptions import ErrorCode


class TestExecuteSQLErrorHandling:
    """Test ExecuteSQLNode error handling."""

    @pytest.fixture
    def agent_config(self):
        """Create a mock agent config."""
        config = Mock(spec=AgentConfig)
        config.namespaces = {"default": {"test_db": {}}}
        config.current_namespace = "default"
        config.nodes = {}  # Add nodes attribute for initialization
        # Mock model configuration to avoid initialization issues
        config.models = {}
        config.active_model = Mock()
        config.active_model.type = "openai"
        config.active_model.model = "gpt-3.5-turbo"
        return config

    @pytest.fixture
    def execute_sql_node(self, agent_config):
        """Create ExecuteSQLNode instance."""
        node = ExecuteSQLNode(
            node_id="test_execute_sql",
            description="Test ExecuteSQL Node",
            node_type="execute_sql",
            input_data=ExecuteSQLInput(sql_query="SELECT * FROM test_table", database_name="test_db"),
            agent_config=agent_config,
        )
        return node

    def test_successful_execution(self, execute_sql_node):
        """Test successful SQL execution."""
        mock_result = Mock()
        mock_result.success = True
        mock_result.sql_return = "test data"
        mock_result.row_count = 5

        with patch.object(execute_sql_node, "_sql_connector") as mock_connector:
            mock_db = Mock()
            mock_db.execute.return_value = mock_result
            mock_connector.return_value = mock_db

            result = execute_sql_node.run()

            assert result.success is True
            assert result.sql_return == "test data"
            assert result.row_count == 5

    def test_database_connection_failed(self, execute_sql_node):
        """Test database connection failure."""
        with patch.object(execute_sql_node, "_sql_connector", return_value=None):
            result = execute_sql_node.run()

            assert isinstance(result, NodeErrorResult)
            assert result.success is False
            assert result.error_code == ErrorCode.DB_CONNECTION_FAILED.code
            assert "Database connection not initialized" in result.error
            assert result.node_context["node_id"] == "test_execute_sql"
            assert result.node_context["node_type"] == "execute_sql"

    def test_sql_execution_timeout(self, execute_sql_node):
        """Test SQL execution timeout."""
        from concurrent.futures import TimeoutError as FuturesTimeoutError

        execute_sql_node.input.query_timeout_seconds = 5

        with patch.object(execute_sql_node, "_sql_connector") as mock_connector:
            mock_db = Mock()
            # Simulate timeout
            with patch("datus.agent.node.execute_sql_node.ThreadPoolExecutor") as mock_executor:
                mock_future = Mock()
                mock_future.result.side_effect = FuturesTimeoutError()
                mock_executor.return_value.__enter__.return_value.submit.return_value = mock_future

                mock_connector.return_value = mock_db

                result = execute_sql_node.run()

                assert isinstance(result, NodeErrorResult)
                assert result.success is False
                assert result.error_code == ErrorCode.DB_EXECUTION_TIMEOUT.code
                assert "timed out after 5 seconds" in result.error
                assert result.retryable is True

    def test_generic_execution_error(self, execute_sql_node):
        """Test generic SQL execution error."""
        with patch.object(execute_sql_node, "_sql_connector") as mock_connector:
            mock_db = Mock()
            mock_db.execute.side_effect = Exception("Database error")

            mock_connector.return_value = mock_db

            result = execute_sql_node.run()

            assert isinstance(result, NodeErrorResult)
            assert result.success is False
            assert result.error_code == ErrorCode.NODE_EXECUTION_FAILED.code
            assert "Database error" in result.error
            assert "stack_trace" in result.error_details
