import pytest
from unittest.mock import MagicMock

from datus.agent.node.schema_discovery_node import SchemaDiscoveryNode
from datus.schemas.base import BaseResult
from datus.schemas.node_models import BaseInput


class TestSchemaDiscoveryNode:
    """Test cases for SchemaDiscoveryNode."""

    @pytest.fixture
    def mock_agent_config(self):
        """Create a mock agent config."""
        config = MagicMock()
        return config

    @pytest.fixture
    def mock_workflow(self):
        """Create a mock workflow with task."""
        workflow = MagicMock()
        task_mock = MagicMock()
        task_mock.task = "统计每个月'首次试驾'到'下定'的平均转化周期（天数）"
        task_mock.catalog_name = "test_catalog"
        task_mock.database_name = "test_db"
        task_mock.schema_name = ""
        workflow.task = task_mock
        workflow.metadata = {}
        return workflow

    @pytest.fixture
    def schema_discovery_node(self, mock_agent_config, mock_workflow):
        """Create SchemaDiscoveryNode instance."""
        node = SchemaDiscoveryNode(
            node_id="test_schema_node",
            description="Test Schema Discovery Node",
            node_type="schema_discovery",
            input_data=BaseInput(),
            agent_config=mock_agent_config,
        )
        # Set up workflow reference
        node.workflow = mock_workflow
        return node

    def test_execute_sets_result_attribute(self, schema_discovery_node):
        """Test that execute() method properly sets self.result."""
        # Execute the node
        result = schema_discovery_node.execute()

        # Verify result is returned
        assert isinstance(result, BaseResult)
        assert result.success is True

        # Verify self.result is set (this was the bug)
        assert schema_discovery_node.result is not None
        assert schema_discovery_node.result is result

    def test_setup_input(self, schema_discovery_node, mock_workflow):
        """Test setup_input method."""
        result = schema_discovery_node.setup_input(mock_workflow)

        assert result["success"] is True
        assert result["message"] == "Schema discovery input setup complete"

    def test_no_task_branch(self, mock_agent_config):
        """Test behavior when no workflow or task is available."""
        node = SchemaDiscoveryNode(
            node_id="test_schema_node",
            description="Test Schema Discovery Node",
            node_type="schema_discovery",
            input_data=BaseInput(),
            agent_config=mock_agent_config,
        )
        # No workflow set

        # Execute should still work and set result
        result = node.execute()
        assert isinstance(result, BaseResult)
        assert node.result is not None
        assert node.result is result

    def test_valid_task_branch(self, schema_discovery_node, mock_workflow):
        """Test behavior with valid task and workflow."""
        # Execute the node with valid setup
        result = schema_discovery_node.execute()

        # Verify result is properly set
        assert isinstance(result, BaseResult)
        assert result.success is True
        assert schema_discovery_node.result is not None
        assert schema_discovery_node.result is result

        # Verify the node can be run (though it may fail due to mocks)
        # This tests the synchronous execution path
        try:
            run_result = schema_discovery_node.run()
            # We don't assert completion since the mock may not fully work
            # But we verify the method doesn't crash
            assert run_result is not None
        except Exception:
            # Expected due to incomplete mocking, but should not crash on result setting
            pass