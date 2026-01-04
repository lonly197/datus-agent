# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for the new text2sql workflow nodes: IntentAnalysisNode and SchemaDiscoveryNode.
"""

from unittest.mock import MagicMock, patch

import pytest

from datus.agent.node.intent_analysis_node import IntentAnalysisNode
from datus.agent.node.schema_discovery_node import SchemaDiscoveryNode
from datus.schemas.action_history import ActionRole, ActionStatus
from datus.schemas.node_models import BaseInput, SqlTask


class TestIntentAnalysisNode:
    """Test cases for IntentAnalysisNode."""

    @pytest.fixture
    def mock_workflow(self):
        """Create a mock workflow."""
        workflow = MagicMock()
        workflow.task = SqlTask(
            id="test_task",
            task="Show me sales data",
            database_type="sqlite",
            catalog_name="",
            database_name="test",
            schema_name="",
            subject_path=[],
            external_knowledge="",
        )
        workflow.metadata = {}
        return workflow

    @pytest.fixture
    def mock_agent_config(self):
        """Create a mock agent config."""
        config = MagicMock()
        return config

    @pytest.fixture
    def intent_node(self, mock_agent_config):
        """Create IntentAnalysisNode instance."""
        return IntentAnalysisNode(
            node_id="intent_node",
            description="Test intent analysis",
            node_type="intent_analysis",
            input_data=BaseInput(),
            agent_config=mock_agent_config,
        )

    @pytest.mark.asyncio
    async def test_setup_input(self, intent_node, mock_workflow):
        """Test setup_input method."""
        intent_node.workflow = mock_workflow
        result = await intent_node.setup_input(mock_workflow)
        assert result["success"] is True
        assert result["message"] == "Intent analysis input setup complete"

    @pytest.mark.asyncio
    async def test_run_with_valid_task(self, intent_node, mock_workflow):
        """Test successful intent analysis run."""
        intent_node.workflow = mock_workflow

        # Mock intent detection
        with patch("datus.agent.intent_detection.IntentDetector") as mock_detector_class:
            mock_detector = MagicMock()
            mock_detector.detect_intent_heuristic.return_value = MagicMock(
                intent="text2sql", confidence=0.9, metadata={"test": "data"}
            )
            mock_detector_class.return_value = mock_detector

            # Collect actions from run
            actions = []
            async for action in intent_node.run():
                actions.append(action)

            # Verify one action was emitted
            assert len(actions) == 1
            action = actions[0]

            # Verify action properties
            assert action.role == ActionRole.TOOL
            assert action.action_type == "intent_analysis"
            assert action.status == ActionStatus.SUCCESS
            assert action.output["intent"] == "text2sql"
            assert action.output["confidence"] == 0.9

            # Verify workflow metadata was updated
            assert mock_workflow.metadata["detected_intent"] == "text2sql"
            assert mock_workflow.metadata["intent_confidence"] == 0.9

    @pytest.mark.asyncio
    async def test_run_with_empty_task(self, intent_node, mock_workflow):
        """Test intent analysis with empty task text."""
        mock_workflow.task.task = ""
        intent_node.workflow = mock_workflow

        actions = []
        async for action in intent_node.run():
            actions.append(action)

        # Should emit error action
        assert len(actions) == 1
        action = actions[0]
        assert action.status == ActionStatus.FAILED
        assert "No task text provided" in action.messages

    @pytest.mark.asyncio
    async def test_run_with_exception(self, intent_node, mock_workflow):
        """Test intent analysis with exception during detection."""
        intent_node.workflow = mock_workflow

        # Mock intent detection to raise exception
        with patch("datus.agent.intent_detection.IntentDetector") as mock_detector_class:
            mock_detector = MagicMock()
            mock_detector.detect_intent_heuristic.side_effect = Exception("Detection failed")
            mock_detector_class.return_value = mock_detector

            actions = []
            async for action in intent_node.run():
                actions.append(action)

            # Should emit error action
            assert len(actions) == 1
            action = actions[0]
            assert action.status == ActionStatus.FAILED
            assert "Detection failed" in action.output["error"]


class TestSchemaDiscoveryNode:
    """Test cases for SchemaDiscoveryNode."""

    @pytest.fixture
    def mock_workflow(self):
        """Create a mock workflow."""
        workflow = MagicMock()
        workflow.task = SqlTask(
            id="test_task",
            task="Show me sales data",
            database_type="sqlite",
            catalog_name="test_catalog",
            database_name="test_db",
            schema_name="public",
            subject_path=[],
            external_knowledge="",
        )
        workflow.context = MagicMock()
        return workflow

    @pytest.fixture
    def mock_agent_config(self):
        """Create a mock agent config."""
        config = MagicMock()
        config.rag_storage_path.return_value = "/tmp/test_rag"
        return config

    @pytest.fixture
    def schema_node(self, mock_agent_config):
        """Create SchemaDiscoveryNode instance."""
        return SchemaDiscoveryNode(
            node_id="schema_node",
            description="Test schema discovery",
            node_type="schema_discovery",
            input_data=BaseInput(),
            agent_config=mock_agent_config,
        )

    @pytest.mark.asyncio
    async def test_setup_input(self, schema_node, mock_workflow):
        """Test setup_input method."""
        schema_node.workflow = mock_workflow
        result = await schema_node.setup_input(mock_workflow)
        assert result["success"] is True
        assert "Schema discovery input setup complete" in result["message"]

    @pytest.mark.asyncio
    async def test_run_with_successful_discovery(self, schema_node, mock_workflow):
        """Test successful schema discovery."""
        schema_node.workflow = mock_workflow

        # Mock schema RAG
        with patch("datus.storage.schema_metadata.SchemaWithValueRAG") as mock_rag_class:
            mock_rag = MagicMock()
            mock_schemas = [MagicMock(table_name="users"), MagicMock(table_name="orders")]
            mock_values = [MagicMock(), MagicMock()]
            mock_rag.search_tables.return_value = (mock_schemas, mock_values)
            mock_rag_class.return_value = mock_rag

            actions = []
            async for action in schema_node.run():
                actions.append(action)

            # Verify one action was emitted
            assert len(actions) == 1
            action = actions[0]

            # Verify action properties
            assert action.role == ActionRole.TOOL
            assert action.action_type == "schema_discovery"
            assert action.status == ActionStatus.SUCCESS

            # Verify context was updated
            mock_workflow.context.update_schema_and_values.assert_called_once_with(mock_schemas, mock_values)

    @pytest.mark.asyncio
    async def test_run_with_exception(self, schema_node, mock_workflow):
        """Test schema discovery with exception."""
        schema_node.workflow = mock_workflow

        # Mock schema RAG to raise exception
        with patch("datus.storage.schema_metadata.SchemaWithValueRAG") as mock_rag_class:
            mock_rag = MagicMock()
            mock_rag.search_tables.side_effect = Exception("Schema search failed")
            mock_rag_class.return_value = mock_rag

            actions = []
            async for action in schema_node.run():
                actions.append(action)

            # Should emit error action
            assert len(actions) == 1
            action = actions[0]
            assert action.status == ActionStatus.FAILED
            assert "Schema search failed" in action.output["error"]

    @pytest.mark.asyncio
    async def test_run_with_empty_tables(self, schema_node, mock_workflow):
        """Test schema discovery when no tables are found."""
        schema_node.workflow = mock_workflow

        # Mock schema RAG to return empty results
        with patch("datus.storage.schema_metadata.SchemaWithValueRAG") as mock_rag_class:
            mock_rag = MagicMock()
            mock_rag.search_tables.return_value = ([], [])
            mock_rag_class.return_value = mock_rag

            actions = []
            async for action in schema_node.run():
                actions.append(action)

            # Should still emit success action
            assert len(actions) == 1
            action = actions[0]
            assert action.status == ActionStatus.SUCCESS
            assert "No tables found" in action.output["message"]
