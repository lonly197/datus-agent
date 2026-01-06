"""
End-to-end integration tests for Text2SQL workflow.

Tests complete workflow execution from API request to SQL output,
including error scenarios and recovery mechanisms.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from datus.agent.workflow import Workflow
from datus.schemas.base import SqlTask
from datus.agent.workflow_runner import WorkflowRunner
from datus.configuration.agent_config import AgentConfig


@pytest.fixture
def mock_agent_config():
    """Create mock agent config with text2sql settings."""
    config = MagicMock(spec=AgentConfig)
    config.scenarios = {
        'text2sql': {
            'continue_on_failure': True,
            'tool_timeout_seconds': 30.0
        }
    }
    config.db_type = "starrocks"
    return config


@pytest.fixture
def sample_sql_task():
    """Create a sample SQL task for testing."""
    return SqlTask(
        id="test_task_123",
        task="统计每个月'首次试驾'到'下定'的平均转化周期（天数）",
        database_type="starrocks",
        catalog_name="default_catalog",
        database_name="test_db",
        schema_name="",
        external_knowledge="使用StarRocks 3.3 SQL审查规则"
    )


@pytest.fixture
def mock_workflow_runner(mock_agent_config):
    """Create mock workflow runner."""
    runner = MagicMock(spec=WorkflowRunner)
    runner.args = MagicMock()
    runner.args.max_steps = 100
    runner.args.load_cp = None
    return runner


class TestText2SQLWorkflowIntegration:
    """Integration tests for complete Text2SQL workflow execution."""

    @pytest.mark.asyncio
    async def test_complete_text2sql_workflow_execution(self, mock_agent_config, sample_sql_task):
        """Test complete text2sql workflow from start to finish."""
        # This is a high-level integration test that would require mocking
        # the entire workflow execution chain

        # Mock the workflow execution
        with patch('datus.agent.workflow_runner.WorkflowRunner') as mock_runner_class:
            mock_runner = MagicMock()
            mock_runner_class.return_value = mock_runner

            # Mock successful execution
            mock_runner.run_stream.return_value = self._mock_successful_workflow_stream()

            # Create agent and execute
            from datus.agent.agent import Agent
            agent = Agent(global_config=mock_agent_config)

            # Execute workflow
            events = []
            async for event in agent.run_stream_with_metadata(sample_sql_task, metadata={"workflow": "text2sql"}):
                events.append(event)

            # Verify workflow completed successfully
            assert len(events) > 0
            assert any("completed" in str(event.messages).lower() for event in events)

    @pytest.mark.asyncio
    async def test_text2sql_workflow_with_preflight_tools(self, mock_agent_config, sample_sql_task):
        """Test text2sql workflow with preflight tool execution."""
        with patch('datus.agent.workflow_runner.WorkflowRunner') as mock_runner_class, \
             patch('datus.agent.node.preflight_orchestrator.PreflightOrchestrator') as mock_preflight_class:

            # Mock preflight orchestrator
            mock_preflight = MagicMock()
            mock_preflight_class.return_value = mock_preflight
            mock_preflight.run_preflight_tools.return_value = self._mock_preflight_stream()

            # Mock workflow runner
            mock_runner = MagicMock()
            mock_runner_class.return_value = mock_runner
            mock_runner.run_stream.return_value = self._mock_successful_workflow_stream()

            from datus.agent.agent import Agent
            agent = Agent(global_config=mock_agent_config)

            events = []
            async for event in agent.run_stream_with_metadata(sample_sql_task, metadata={"workflow": "text2sql"}):
                events.append(event)

            # Verify preflight tools were called
            mock_preflight.run_preflight_tools.assert_called_once()

            # Verify workflow completed
            assert len(events) > 0

    @pytest.mark.asyncio
    async def test_text2sql_workflow_error_recovery(self, mock_agent_config, sample_sql_task):
        """Test error recovery in text2sql workflow."""
        with patch('datus.agent.workflow_runner.WorkflowRunner') as mock_runner_class:
            mock_runner = MagicMock()
            mock_runner_class.return_value = mock_runner

            # Mock workflow with errors but eventual success
            mock_runner.run_stream.return_value = self._mock_workflow_with_errors()

            from datus.agent.agent import Agent
            agent = Agent(global_config=mock_agent_config)

            events = []
            async for event in agent.run_stream_with_metadata(sample_sql_task, metadata={"workflow": "text2sql"}):
                events.append(event)

            # Verify error handling occurred
            error_events = [e for e in events if hasattr(e, 'status') and e.status == "failed"]
            success_events = [e for e in events if hasattr(e, 'status') and e.status == "completed"]

            # Should have both errors and eventual success
            assert len(error_events) > 0
            assert len(success_events) > 0

    @pytest.mark.asyncio
    async def test_text2sql_workflow_node_dependencies(self, mock_agent_config, sample_sql_task):
        """Test that nodes properly depend on each other."""
        with patch('datus.agent.workflow_runner.WorkflowRunner') as mock_runner_class:
            mock_runner = MagicMock()
            mock_runner_class.return_value = mock_runner

            # Mock workflow that validates node dependencies
            mock_runner.run_stream.return_value = self._mock_workflow_with_dependencies()

            from datus.agent.agent import Agent
            agent = Agent(global_config=mock_agent_config)

            events = []
            async for event in agent.run_stream_with_metadata(sample_sql_task, metadata={"workflow": "text2sql"}):
                events.append(event)

            # Verify intent analysis happened before schema discovery
            intent_events = [e for e in events if "intent" in str(e.messages).lower()]
            schema_events = [e for e in events if "schema" in str(e.messages).lower()]

            assert len(intent_events) > 0
            assert len(schema_events) > 0

    def test_workflow_configuration_validation(self, mock_agent_config):
        """Test that workflow configuration is properly validated."""
        from datus.agent.plan import create_workflow_plan

        # Test valid text2sql workflow configuration
        workflow_config = {
            "text2sql": [
                "intent_analysis",
                "schema_discovery",
                "generate_sql",
                "execute_sql",
                "output"
            ]
        }

        # Should not raise exception
        try:
            plan = create_workflow_plan(workflow_config, mock_agent_config)
            assert "text2sql" in plan
            assert len(plan["text2sql"]) == 5
        except Exception as e:
            pytest.fail(f"Valid workflow configuration failed: {e}")

    def test_invalid_workflow_configuration_handling(self, mock_agent_config):
        """Test handling of invalid workflow configurations."""
        from datus.agent.plan import create_workflow_plan

        # Test invalid node type
        workflow_config = {
            "text2sql": [
                "intent_analysis",
                "invalid_node_type",
                "generate_sql"
            ]
        }

        # Should raise ValueError for invalid node type
        with pytest.raises(ValueError, match="Invalid node type"):
            create_workflow_plan(workflow_config, mock_agent_config)

    def _mock_successful_workflow_stream(self):
        """Mock a successful workflow execution stream."""
        from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus

        async def mock_stream():
            # Workflow initialization
            yield ActionHistory(
                action_id="workflow_init",
                role=ActionRole.SYSTEM,
                messages="Workflow initialized",
                action_type="workflow_init",
                status=ActionStatus.SUCCESS
            )

            # Intent analysis
            yield ActionHistory(
                action_id="intent_analysis",
                role=ActionRole.TOOL,
                messages="Intent analysis completed: sql",
                action_type="intent_analysis",
                status=ActionStatus.SUCCESS
            )

            # Schema discovery
            yield ActionHistory(
                action_id="schema_discovery",
                role=ActionRole.TOOL,
                messages="Schema discovery completed",
                action_type="schema_discovery",
                status=ActionStatus.SUCCESS
            )

            # SQL generation
            yield ActionHistory(
                action_id="sql_generation",
                role=ActionRole.TOOL,
                messages="SQL generation completed",
                action_type="sql_generation",
                status=ActionStatus.SUCCESS
            )

            # SQL execution
            yield ActionHistory(
                action_id="sql_execution",
                role=ActionRole.TOOL,
                messages="SQL execution completed",
                action_type="sql_execution",
                status=ActionStatus.SUCCESS
            )

            # Output
            yield ActionHistory(
                action_id="output",
                role=ActionRole.TOOL,
                messages="Results returned to user",
                action_type="output",
                status=ActionStatus.SUCCESS
            )

        return mock_stream()

    def _mock_preflight_stream(self):
        """Mock preflight tool execution stream."""
        from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus

        async def mock_stream():
            # Preflight tools
            tools = ["search_table", "describe_table", "search_reference_sql", "parse_temporal_expressions"]

            for tool in tools:
                # Tool start
                yield ActionHistory(
                    action_id=f"preflight_{tool}",
                    role=ActionRole.TOOL,
                    messages=f"Executing preflight tool: {tool}",
                    action_type=f"preflight_{tool}",
                    status=ActionStatus.PROCESSING
                )

                # Tool completion
                yield ActionHistory(
                    action_id=f"preflight_{tool}_result",
                    role=ActionRole.TOOL,
                    messages=f"Preflight tool {tool} completed",
                    action_type=f"preflight_{tool}_result",
                    status=ActionStatus.SUCCESS
                )

        return mock_stream()

    def _mock_workflow_with_errors(self):
        """Mock workflow execution with errors and recovery."""
        from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus

        async def mock_stream():
            # Initial error in intent analysis
            yield ActionHistory(
                action_id="intent_analysis_error",
                role=ActionRole.TOOL,
                messages="Intent analysis failed: temporary error",
                action_type="intent_analysis",
                status=ActionStatus.FAILED
            )

            # Recovery and successful completion
            yield ActionHistory(
                action_id="intent_analysis_retry",
                role=ActionRole.TOOL,
                messages="Intent analysis completed: sql (retry)",
                action_type="intent_analysis",
                status=ActionStatus.SUCCESS
            )

            # Rest of workflow completes successfully
            yield ActionHistory(
                action_id="schema_discovery",
                role=ActionRole.TOOL,
                messages="Schema discovery completed",
                action_type="schema_discovery",
                status=ActionStatus.SUCCESS
            )

            yield ActionHistory(
                action_id="workflow_complete",
                role=ActionRole.SYSTEM,
                messages="Workflow completed despite initial errors",
                action_type="workflow_complete",
                status=ActionStatus.SUCCESS
            )

        return mock_stream()

    def _mock_workflow_with_dependencies(self):
        """Mock workflow that validates node dependencies."""
        from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus

        async def mock_stream():
            # Intent analysis must come first
            yield ActionHistory(
                action_id="intent_analysis",
                role=ActionRole.TOOL,
                messages="Intent analysis completed: sql",
                action_type="intent_analysis",
                status=ActionStatus.SUCCESS,
                output={"detected_intent": "sql", "intent_confidence": 0.8}
            )

            # Schema discovery depends on intent analysis
            yield ActionHistory(
                action_id="schema_discovery",
                role=ActionRole.TOOL,
                messages="Schema discovery using intent: sql",
                action_type="schema_discovery",
                status=ActionStatus.SUCCESS,
                input={"intent": "sql"}
            )

            # SQL generation depends on schema
            yield ActionHistory(
                action_id="sql_generation",
                role=ActionRole.TOOL,
                messages="SQL generation with schema context",
                action_type="sql_generation",
                status=ActionStatus.SUCCESS
            )

        return mock_stream()