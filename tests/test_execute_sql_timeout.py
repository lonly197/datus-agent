# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import time
import unittest
from unittest.mock import MagicMock, patch

from datus.agent.node.execute_sql_node import ExecuteSQLNode
from datus.configuration.agent_config import AgentConfig
from datus.schemas.node_models import ExecuteSQLInput, ExecuteSQLResult


class TestExecuteSQLTimeout(unittest.TestCase):
    """Test timeout functionality in ExecuteSQLNode."""

    def setUp(self):
        """Set up test fixtures."""
        self.agent_config = AgentConfig(
            target="test",
            models={"test": MagicMock()},
            nodes={"test": MagicMock()},
            rag_base_path="/tmp",
        )
        self.agent_config.default_query_timeout_seconds = 2  # 2 second timeout for testing

        self.node = ExecuteSQLNode(
            node_id="test_node",
            description="Test Execute SQL Node",
            node_type="execute_sql",
            input_data=None,
            agent_config=self.agent_config,
        )

    def test_timeout_from_config(self):
        """Test that timeout is applied from agent config when not specified in input."""
        # Mock a slow connector that sleeps longer than timeout
        mock_connector = MagicMock()
        mock_result = ExecuteSQLResult(
            success=True, sql_query="SELECT * FROM test", sql_return="test data", row_count=1
        )

        def slow_execute(input_data):
            time.sleep(3)  # Sleep longer than 2 second timeout
            return mock_result

        mock_connector.execute.side_effect = slow_execute

        # Mock the _sql_connector method
        with patch.object(self.node, "_sql_connector", return_value=mock_connector):
            # Set up input without explicit timeout
            self.node.input = ExecuteSQLInput(
                database_name="test",
                sql_query="SELECT * FROM test",
            )

            # Execute and check timeout
            result = self.node._execute_sql()

            # Verify timeout occurred
            self.assertFalse(result.success)
            self.assertIn("timed out after 2 seconds", result.error)
            self.assertEqual(result.sql_query, "SELECT * FROM test")

            # Verify connector.close was called for cleanup
            mock_connector.close.assert_called_once()

    def test_timeout_from_input(self):
        """Test that timeout is applied from input when specified."""
        # Mock a slow connector
        mock_connector = MagicMock()
        mock_result = ExecuteSQLResult(
            success=True, sql_query="SELECT * FROM test", sql_return="test data", row_count=1
        )

        def slow_execute(input_data):
            time.sleep(5)  # Sleep longer than 1 second timeout
            return mock_result

        mock_connector.execute.side_effect = slow_execute

        # Mock the _sql_connector method
        with patch.object(self.node, "_sql_connector", return_value=mock_connector):
            # Set up input with explicit timeout (1 second)
            self.node.input = ExecuteSQLInput(
                database_name="test",
                sql_query="SELECT * FROM test",
                query_timeout_seconds=1,
            )

            # Execute and check timeout
            result = self.node._execute_sql()

            # Verify timeout occurred with input timeout
            self.assertFalse(result.success)
            self.assertIn("timed out after 1 seconds", result.error)

            # Verify connector.close was called for cleanup
            mock_connector.close.assert_called_once()

    def test_no_timeout_when_disabled(self):
        """Test that no timeout is applied when timeout is 0 or None."""
        # Mock a connector
        mock_connector = MagicMock()
        mock_result = ExecuteSQLResult(
            success=True, sql_query="SELECT * FROM test", sql_return="test data", row_count=1
        )

        def fast_execute(input_data):
            return mock_result

        mock_connector.execute.side_effect = fast_execute

        # Mock the _sql_connector method
        with patch.object(self.node, "_sql_connector", return_value=mock_connector):
            # Set up input with timeout disabled
            self.node.input = ExecuteSQLInput(
                database_name="test",
                sql_query="SELECT * FROM test",
                query_timeout_seconds=0,  # Disable timeout
            )

            # Execute and check normal execution
            result = self.node._execute_sql()

            # Verify normal execution occurred
            self.assertTrue(result.success)
            self.assertEqual(result.sql_return, "test data")

            # Verify close was not called during timeout cleanup
            mock_connector.close.assert_not_called()

    def test_normal_execution_without_timeout(self):
        """Test normal execution when no timeout is configured."""
        # Create config without default timeout
        agent_config = AgentConfig(
            target="test",
            models={"test": MagicMock()},
            nodes={"test": MagicMock()},
            rag_base_path="/tmp",
        )
        # Don't set default_query_timeout_seconds

        node = ExecuteSQLNode(
            node_id="test_node",
            description="Test Execute SQL Node",
            node_type="execute_sql",
            input_data=None,
            agent_config=agent_config,
        )

        # Mock a connector
        mock_connector = MagicMock()
        mock_result = ExecuteSQLResult(
            success=True, sql_query="SELECT * FROM test", sql_return="test data", row_count=1
        )

        mock_connector.execute.return_value = mock_result

        # Mock the _sql_connector method
        with patch.object(node, "_sql_connector", return_value=mock_connector):
            # Set up input without timeout
            node.input = ExecuteSQLInput(
                database_name="test",
                sql_query="SELECT * FROM test",
            )

            # Execute and check normal execution
            result = node._execute_sql()

            # Verify normal execution occurred
            self.assertTrue(result.success)
            self.assertEqual(result.sql_return, "test data")

            # Verify close was not called
            mock_connector.close.assert_not_called()


if __name__ == "__main__":
    unittest.main()
