"""
API tests for chat research endpoint with text2sql scenarios.

Tests the complete API flow from HTTP request to SSE response streaming.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient
from datus.api.service import app
from datus.schemas.api_models import ChatResearchRequest


@pytest.fixture
def test_client():
    """Create test client for API testing."""
    return TestClient(app)


@pytest.fixture
def sample_text2sql_request():
    """Create sample text2sql request."""
    return ChatResearchRequest(
        task="统计每个月'首次试驾'到'下定'的平均转化周期（天数）",
        namespace="test",
        prompt="你是【数仓开发助手】。使用StarRocks数据库，遵循数据仓库开发规范和SQL最佳实践。",
        prompt_mode="append",
        catalog_name="default_catalog",
        database_name="test_db"
    )


class TestChatResearchText2SQLAPI:
    """API tests for text2sql chat research endpoint."""

    def test_chat_research_text2sql_request_validation(self, test_client, sample_text2sql_request):
        """Test that text2sql requests are properly validated."""
        with patch('datus.api.service.run_chat_research_stream') as mock_stream:
            mock_stream.return_value = self._mock_sse_response()

            response = test_client.post(
                "/workflows/chat_research",
                json=sample_text2sql_request.model_dump(),
                headers={"Accept": "text/event-stream"}
            )

            # Should accept the request
            assert response.status_code == 200
            mock_stream.assert_called_once()

    def test_chat_research_sse_headers(self, test_client, sample_text2sql_request):
        """Test SSE response headers."""
        with patch('datus.api.service.run_chat_research_stream') as mock_stream:
            mock_stream.return_value = self._mock_sse_response()

            response = test_client.post(
                "/workflows/chat_research",
                json=sample_text2sql_request.model_dump(),
                headers={"Accept": "text/event-stream"}
            )

            # Check SSE headers
            assert response.status_code == 200
            assert "text/plain" in response.headers.get("content-type", "")
            # Note: FastAPI TestClient doesn't perfectly simulate SSE streaming

    def test_chat_research_missing_sse_header(self, test_client, sample_text2sql_request):
        """Test rejection when SSE header is missing."""
        response = test_client.post(
            "/workflows/chat_research",
            json=sample_text2sql_request.model_dump()
        )

        # Should be rejected without SSE header
        assert response.status_code == 400

    @patch('datus.api.service.get_current_client')
    @patch('datus.api.service.Agent')
    def test_text2sql_workflow_initialization(self, mock_agent_class, mock_get_client, test_client, sample_text2sql_request):
        """Test that text2sql workflow is properly initialized."""
        # Mock client authentication
        mock_get_client.return_value = "test_client"

        # Mock agent
        mock_agent = MagicMock()
        mock_agent.global_config.db_type = "starrocks"
        mock_agent_class.return_value = mock_agent

        # Mock successful workflow execution
        mock_agent.run_stream_with_metadata.return_value = self._mock_successful_workflow_events()

        with patch('datus.api.service.run_chat_research_stream') as mock_stream:
            mock_stream.return_value = self._mock_sse_response()

            response = test_client.post(
                "/workflows/chat_research",
                json=sample_text2sql_request.model_dump(),
                headers={"Accept": "text/event-stream"}
            )

            assert response.status_code == 200

    def test_text2sql_task_type_detection(self, sample_text2sql_request):
        """Test that text2sql tasks are properly categorized."""
        from datus.api.service import ChatResearchAPI

        api = ChatResearchAPI()

        # Test task type detection
        task_type = api._identify_task_type(sample_text2sql_request.task)

        assert task_type == "text2sql"

    def test_text2sql_task_configuration(self, sample_text2sql_request):
        """Test text2sql task configuration generation."""
        from datus.api.service import ChatResearchAPI

        api = ChatResearchAPI()

        # Test configuration generation
        config = api._configure_task_processing("text2sql", sample_text2sql_request)

        assert config["workflow"] == "text2sql"
        assert config["plan_mode"] is False
        assert config["auto_execute_plan"] is False
        assert config["system_prompt"] == "text2sql_system"
        assert "required_tool_sequence" in config
        assert len(config["required_tool_sequence"]) == 4

    def test_text2sql_sql_task_creation(self, sample_text2sql_request):
        """Test SQL task creation for text2sql."""
        from datus.api.service import ChatResearchAPI
        from unittest.mock import patch

        api = ChatResearchAPI()

        with patch('datus.api.service.Agent') as mock_agent_class:
            mock_agent = MagicMock()
            mock_agent.global_config.db_type = "starrocks"
            mock_agent.global_config.output_dir = "/tmp"
            mock_agent_class.return_value = mock_agent

            # Test SQL task creation
            sql_task = api._create_sql_task(sample_text2sql_request, "text2sql", "test_client_123")

            assert sql_task.task == sample_text2sql_request.task
            assert sql_task.database_type == "starrocks"
            assert sql_task.external_knowledge == "使用StarRocks 3.3 SQL审查规则"

    @patch('datus.api.service.get_current_client')
    def test_text2sql_error_handling(self, mock_get_client, test_client):
        """Test error handling in text2sql workflow."""
        mock_get_client.return_value = "test_client"

        # Test with invalid request (missing required fields)
        invalid_request = {
            "task": "",  # Empty task
            "namespace": "test"
        }

        response = test_client.post(
            "/workflows/chat_research",
            json=invalid_request,
            headers={"Accept": "text/event-stream"}
        )

        # Should handle gracefully
        assert response.status_code in [200, 400, 422]  # Various possible error codes

    def test_task_type_detection_edge_cases(self):
        """Test task type detection with various inputs."""
        from datus.api.service import ChatResearchAPI

        api = ChatResearchAPI()

        test_cases = [
            ("统计销售额", "text2sql"),
            ("检查SQL语法", "sql_review"),
            ("分析销售趋势", "data_analysis"),
            ("", "text2sql"),  # Default fallback
            ("SELECT * FROM table", "text2sql"),  # SQL query
        ]

        for task_text, expected_type in test_cases:
            detected_type = api._identify_task_type(task_text)
            assert detected_type == expected_type, f"Failed for task: {task_text}"

    def _mock_sse_response(self):
        """Mock SSE response generator."""
        async def mock_response():
            yield "data: {\"event\": \"workflow_started\"}\n\n"
            yield "data: {\"event\": \"preflight_tools_started\"}\n\n"
            yield "data: {\"event\": \"intent_analysis_completed\"}\n\n"
            yield "data: {\"event\": \"schema_discovery_completed\"}\n\n"
            yield "data: {\"event\": \"sql_generation_completed\"}\n\n"
            yield "data: {\"event\": \"sql_execution_completed\"}\n\n"
            yield "data: {\"event\": \"workflow_completed\", \"sql\": \"SELECT * FROM test\"}\n\n"

        return mock_response()

    def _mock_successful_workflow_events(self):
        """Mock successful workflow execution events."""
        from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus

        async def mock_events():
            # Preflight tools
            yield ActionHistory(
                action_id="preflight_search_table",
                role=ActionRole.TOOL,
                messages="Table search completed",
                action_type="preflight_search_table",
                status=ActionStatus.SUCCESS
            )

            # Workflow nodes
            yield ActionHistory(
                action_id="intent_analysis",
                role=ActionRole.TOOL,
                messages="Intent analysis: sql",
                action_type="intent_analysis",
                status=ActionStatus.SUCCESS
            )

            yield ActionHistory(
                action_id="sql_generation",
                role=ActionRole.TOOL,
                messages="SQL generated successfully",
                action_type="sql_generation",
                status=ActionStatus.SUCCESS,
                output={"sql": "SELECT COUNT(*) FROM orders"}
            )

        return mock_events()