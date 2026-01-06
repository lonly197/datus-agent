import pytest
from unittest.mock import MagicMock

from datus.agent.node.intent_analysis_node import IntentAnalysisNode
from datus.schemas.base import BaseResult
from datus.schemas.node_models import BaseInput


class TestIntentAnalysisNode:
    """Test cases for IntentAnalysisNode."""

    @pytest.fixture
    def mock_agent_config(self):
        """Create a mock agent config."""
        config = MagicMock()
        config.intent_detector_llm_fallback = False
        return config

    @pytest.fixture
    def mock_workflow(self):
        """Create a mock workflow with task."""
        workflow = MagicMock()
        task_mock = MagicMock()
        task_mock.task = "统计每个月'首次试驾'到'下定'的平均转化周期（天数）"
        workflow.task = task_mock
        workflow.metadata = {}
        return workflow

    @pytest.fixture
    def intent_analysis_node(self, mock_agent_config, mock_workflow):
        """Create IntentAnalysisNode instance."""
        node = IntentAnalysisNode(
            node_id="test_intent_node",
            description="Test Intent Analysis Node",
            node_type="intent_analysis",
            input_data=BaseInput(),
            agent_config=mock_agent_config,
        )
        # Set up workflow reference
        node.workflow = mock_workflow
        return node

    def test_execute_sets_result_attribute(self, intent_analysis_node):
        """Test that execute() method properly sets self.result."""
        # Execute the node
        result = intent_analysis_node.execute()

        # Verify result is returned
        assert isinstance(result, BaseResult)
        assert result.success is True

        # Verify self.result is set (this was the bug)
        assert intent_analysis_node.result is not None
        assert intent_analysis_node.result is result

    def test_node_run_completes_successfully(self, intent_analysis_node):
        """Test that Node.run() completes successfully and sets workflow metadata."""
        # Run the node
        run_result = intent_analysis_node.run()

        # Verify node completed successfully
        assert intent_analysis_node.status == "completed"
        assert run_result is not None

        # Verify workflow metadata was set
        assert "detected_intent" in intent_analysis_node.workflow.metadata
        assert "intent_confidence" in intent_analysis_node.workflow.metadata
        assert "intent_metadata" in intent_analysis_node.workflow.metadata

        # Verify intent was detected as SQL-related
        detected_intent = intent_analysis_node.workflow.metadata["detected_intent"]
        assert detected_intent in ["sql", "sql_related"]

        # Verify confidence is reasonable
        confidence = intent_analysis_node.workflow.metadata["intent_confidence"]
        assert isinstance(confidence, (int, float))
        assert 0.0 <= confidence <= 1.0

    def test_setup_input(self, intent_analysis_node, mock_workflow):
        """Test setup_input method."""
        result = intent_analysis_node.setup_input(mock_workflow)

        assert result["success"] is True
        assert result["message"] == "Intent analysis input setup complete"

    def test_update_context(self, intent_analysis_node, mock_workflow):
        """Test update_context method."""
        result = intent_analysis_node.update_context(mock_workflow)

        assert result["success"] is True
        assert result["message"] == "Intent analysis context update complete"

    def test_no_task_text(self, mock_agent_config, mock_workflow):
        """Test behavior when no task text is provided."""
        # Create node with empty task
        mock_workflow.task.task = ""
        node = IntentAnalysisNode(
            node_id="test_intent_node",
            description="Test Intent Analysis Node",
            node_type="intent_analysis",
            input_data=BaseInput(),
            agent_config=mock_agent_config,
        )
        node.workflow = mock_workflow

        # Execute should still work (handled gracefully)
        result = node.execute()
        assert isinstance(result, BaseResult)

    def test_chinese_text_intent_detection(self, mock_agent_config, mock_workflow):
        """Test intent detection with Chinese text (the actual failing case)."""
        # This is the exact text from the failing log
        chinese_task = "统计每个月'首次试驾'到'下定'的平均转化周期（天数）"
        mock_workflow.task.task = chinese_task

        node = IntentAnalysisNode(
            node_id="test_intent_node",
            description="Test Intent Analysis Node",
            node_type="intent_analysis",
            input_data=BaseInput(),
            agent_config=mock_agent_config,
        )
        node.workflow = mock_workflow

        # Run the node
        run_result = node.run()

        # Verify it completes successfully
        assert node.status == "completed"

        # Verify metadata contains expected values
        metadata = node.workflow.metadata
        assert "detected_intent" in metadata
        assert metadata["detected_intent"] in ["sql", "sql_related"]
        assert "intent_confidence" in metadata
        assert "intent_metadata" in metadata