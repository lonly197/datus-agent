# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import unittest
from unittest.mock import MagicMock, patch

from datus.agent.workflow import Workflow
from datus.agent.workflow_runner import WorkflowRunner
from datus.schemas.node_models import SqlTask


class TestWorkflowStepCount(unittest.TestCase):
    """Test workflow finalization step counting functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.sql_task = SqlTask(
            id="test_task",
            database_type="sqlite",
            task="SELECT * FROM test_table",
        )

        # Create mock agent config
        self.agent_config = MagicMock()
        self.agent_config.get_trajectory_run_dir.return_value = "/tmp/test_trajectory"

        # Create workflow runner
        self.runner = WorkflowRunner(
            args=MagicMock(),
            agent_config=self.agent_config,
            run_id="test_run_123",
        )

        # Create mock workflow
        self.mock_workflow = MagicMock()
        self.runner.workflow = self.mock_workflow

    def test_finalize_workflow_completes_node_count(self):
        """Test that _finalize_workflow computes completed_nodes correctly."""
        # Mock workflow nodes with different statuses
        mock_node1 = MagicMock()
        mock_node1.status = "completed"

        mock_node2 = MagicMock()
        mock_node2.status = "completed"

        mock_node3 = MagicMock()
        mock_node3.status = "failed"

        mock_node4 = MagicMock()
        mock_node4.status = "pending"

        self.mock_workflow.nodes = {
            "node1": mock_node1,
            "node2": mock_node2,
            "node3": mock_node3,
            "node4": mock_node4,
        }

        self.mock_workflow.task = self.sql_task
        self.mock_workflow.get_final_result.return_value = {"test": "result"}

        # Mock workflow methods
        self.mock_workflow.display = MagicMock()
        self.mock_workflow.save = MagicMock()

        # Mock the file operations and time
        with patch('datus.agent.workflow_runner.os.makedirs'), \
             patch('datus.agent.workflow_runner.time.time', return_value=1234567890), \
             patch('datus.agent.workflow_runner.logger') as mock_logger:

            # Call _finalize_workflow
            result = self.runner._finalize_workflow(step_count=5)

            # Verify completed_nodes calculation (2 completed nodes)
            self.assertEqual(result["completed_nodes"], 2)
            self.assertEqual(result["steps"], 5)
            self.assertEqual(result["run_id"], "test_run_123")
            self.assertIn("final_result", result)

            # Verify logging includes both step count and completed nodes
            mock_logger.info.assert_any_call("Workflow execution completed. StepsAttempted:5 CompletedNodes:2")

    def test_finalize_workflow_zero_completed_nodes(self):
        """Test that _finalize_workflow handles zero completed nodes."""
        # Mock workflow nodes all failed or pending
        mock_node1 = MagicMock()
        mock_node1.status = "failed"

        mock_node2 = MagicMock()
        mock_node2.status = "pending"

        mock_node3 = MagicMock()
        mock_node3.status = "running"

        self.mock_workflow.nodes = {
            "node1": mock_node1,
            "node2": mock_node2,
            "node3": mock_node3,
        }

        self.mock_workflow.task = self.sql_task
        self.mock_workflow.get_final_result.return_value = {"test": "result"}

        # Mock workflow methods
        self.mock_workflow.display = MagicMock()
        self.mock_workflow.save = MagicMock()

        # Mock the file operations and time
        with patch('datus.agent.workflow_runner.os.makedirs'), \
             patch('datus.agent.workflow_runner.time.time', return_value=1234567890), \
             patch('datus.agent.workflow_runner.logger') as mock_logger:

            # Call _finalize_workflow
            result = self.runner._finalize_workflow(step_count=3)

            # Verify completed_nodes is 0
            self.assertEqual(result["completed_nodes"], 0)
            self.assertEqual(result["steps"], 3)

            # Verify logging
            mock_logger.info.assert_any_call("Workflow execution completed. StepsAttempted:3 CompletedNodes:0")

    def test_finalize_workflow_all_completed_nodes(self):
        """Test that _finalize_workflow handles all nodes completed."""
        # Mock workflow nodes all completed
        mock_node1 = MagicMock()
        mock_node1.status = "completed"

        mock_node2 = MagicMock()
        mock_node2.status = "completed"

        self.mock_workflow.nodes = {
            "node1": mock_node1,
            "node2": mock_node2,
        }

        self.mock_workflow.task = self.sql_task
        self.mock_workflow.get_final_result.return_value = {"test": "result"}

        # Mock workflow methods
        self.mock_workflow.display = MagicMock()
        self.mock_workflow.save = MagicMock()

        # Mock the file operations and time
        with patch('datus.agent.workflow_runner.os.makedirs'), \
             patch('datus.agent.workflow_runner.time.time', return_value=1234567890), \
             patch('datus.agent.workflow_runner.logger') as mock_logger:

            # Call _finalize_workflow
            result = self.runner._finalize_workflow(step_count=2)

            # Verify completed_nodes equals total nodes
            self.assertEqual(result["completed_nodes"], 2)
            self.assertEqual(result["steps"], 2)

            # Verify logging
            mock_logger.info.assert_any_call("Workflow execution completed. StepsAttempted:2 CompletedNodes:2")

    def test_finalize_workflow_no_workflow(self):
        """Test that _finalize_workflow handles None workflow gracefully."""
        runner = WorkflowRunner(
            args=MagicMock(),
            agent_config=self.agent_config,
            run_id="test_run_123",
        )
        runner.workflow = None

        result = runner._finalize_workflow(step_count=1)

        # Should return empty dict
        self.assertEqual(result, {})


if __name__ == "__main__":
    unittest.main()
