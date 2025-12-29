# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from datus.agent.intent_detection import IntentDetector, IntentResult, detect_sql_intent
from datus.api.models import RunWorkflowRequest


class TestIntentDetection:
    """Test intent detection functionality."""

    def test_keyword_detection_sql_query(self):
        """Test keyword detection for SQL generation queries."""
        detector = IntentDetector()

        # Chinese SQL generation query
        result = detector.detect_sql_intent_by_keyword("从 ODS 试驾表和线索表关联，统计每个月'首次试驾'到'下定'的平均转化周期")
        assert result[0] is True  # Should detect SQL intent
        assert "试驾" in result[1]["keyword_matches"]
        assert "线索" in result[1]["keyword_matches"]

    def test_keyword_detection_sql_review(self):
        """Test keyword detection for SQL review queries."""
        detector = IntentDetector()

        result = detector.detect_sql_intent_by_keyword("审查以下SQL：SELECT * FROM dwd_assign_dlr_clue_fact_di")
        assert result[0] is True
        assert "sql" in result[1]["keyword_matches"]
        assert "审查" in result[1]["keyword_matches"]

    def test_keyword_detection_non_sql(self):
        """Test keyword detection for non-SQL queries."""
        detector = IntentDetector()

        result = detector.detect_sql_intent_by_keyword("What is the weather today?")
        assert result[0] is False

    def test_sql_pattern_detection(self):
        """Test SQL pattern detection."""
        detector = IntentDetector()

        result = detector.detect_sql_intent_by_keyword("SELECT user_id, name FROM users WHERE status = 'active'")
        assert result[0] is True
        assert len(result[1]["pattern_matches"]) > 0

    @pytest.mark.asyncio
    async def test_llm_fallback(self):
        """Test LLM fallback classification."""
        detector = IntentDetector()

        # Mock LLM model
        mock_model = AsyncMock()
        mock_model.generate.return_value = '{"intent": "sql_generation", "confidence": 0.9}'

        result = await detector.detect_sql_intent(
            text="Generate a report on user activity", model=mock_model, use_llm_fallback=True
        )

        assert result.intent == "sql_generation"
        assert result.confidence == 0.9
        mock_model.generate.assert_called_once()

    def test_intent_result_creation(self):
        """Test IntentResult dataclass."""
        result = IntentResult(intent="sql_generation", confidence=0.85, metadata={"source": "keyword"})

        assert result.intent == "sql_generation"
        assert result.confidence == 0.85
        assert result.metadata["source"] == "keyword"


class TestAutoInjectionIntegration:
    """Integration tests for auto-injection functionality."""

    @pytest.mark.asyncio
    async def test_auto_injection_with_sql_intent(self):
        """Test end-to-end auto-injection when SQL intent is detected."""
        from datus.api.service import DatusAPIService
        from datus.schemas.node_models import SqlTask

        # Mock agent and config
        mock_agent = MagicMock()
        mock_config = MagicMock()
        mock_config.db_type = "starrocks"
        mock_config.current_metric_meta.return_value.ext_knowledge = ""
        mock_config.rag_storage_path.return_value = "/tmp/test"
        mock_agent.global_config = mock_config

        # Mock ExtKnowledgeStore
        with patch("datus.api.service.ExtKnowledgeStore") as mock_store_class:
            mock_store = MagicMock()
            # Mock search results
            mock_results = MagicMock()
            mock_results.__iter__.return_value = [
                {"terminology": "partition_pruning_basic", "explanation": "强制分区裁剪"},
                {"terminology": "forbid_select_star", "explanation": "禁止SELECT *"},
            ]
            mock_results.__len__.return_value = 2
            mock_store.search_knowledge.return_value = mock_results
            mock_store.table_size.return_value = 10  # Non-empty store
            mock_store_class.return_value = mock_store

            service = DatusAPIService()

            # Create request with plan_mode but no ext_knowledge
            request = RunWorkflowRequest(
                task="从 ODS 试驾表和线索表关联，统计转化周期",
                workflow="chat_agentic_plan",
                namespace="test",
                plan_mode=True,
                ext_knowledge=None,  # Explicitly no ext_knowledge
            )

            # Call _create_sql_task
            sql_task = await service._create_sql_task(request, "test_task", mock_agent)

            # Verify auto-injection occurred
            assert isinstance(sql_task, SqlTask)
            assert "Auto-detected relevant knowledge:" in sql_task.external_knowledge
            assert "partition_pruning_basic" in sql_task.external_knowledge
            assert "forbid_select_star" in sql_task.external_knowledge

    @pytest.mark.asyncio
    async def test_no_injection_without_plan_mode(self):
        """Test that auto-injection doesn't happen without plan_mode."""
        from datus.api.service import DatusAPIService

        mock_agent = MagicMock()
        mock_config = MagicMock()
        mock_config.db_type = "starrocks"
        mock_config.current_metric_meta.return_value.ext_knowledge = ""
        mock_agent.global_config = mock_config

        service = DatusAPIService()

        # Create request without plan_mode
        request = RunWorkflowRequest(
            task="Generate SQL report",
            workflow="chat_agentic",
            namespace="test",
            plan_mode=False,  # Not plan mode
            ext_knowledge=None,
        )

        sql_task = await service._create_sql_task(request, "test_task", mock_agent)

        # Verify no auto-injection occurred
        assert sql_task.external_knowledge == ""

    @pytest.mark.asyncio
    async def test_no_injection_with_explicit_knowledge(self):
        """Test that auto-injection doesn't override explicit ext_knowledge."""
        from datus.api.service import DatusAPIService

        mock_agent = MagicMock()
        mock_config = MagicMock()
        mock_config.db_type = "starrocks"
        mock_config.current_metric_meta.return_value.ext_knowledge = ""
        mock_agent.global_config = mock_config

        service = DatusAPIService()

        # Create request with explicit ext_knowledge
        explicit_knowledge = "Custom business rules"
        request = RunWorkflowRequest(
            task="Generate SQL report",
            workflow="chat_agentic_plan",
            namespace="test",
            plan_mode=True,
            ext_knowledge=explicit_knowledge,  # Explicit knowledge provided
        )

        sql_task = await service._create_sql_task(request, "test_task", mock_agent)

        # Verify explicit knowledge is preserved (not auto-injected over)
        assert sql_task.external_knowledge == explicit_knowledge

    def test_context_search_tools_integration(self):
        """Test that ContextSearchTools properly exposes external knowledge search."""
        from datus.tools.func_tool.context_search import ContextSearchTools

        mock_config = MagicMock()
        mock_config.rag_storage_path.return_value = "/tmp/test"

        # Mock ExtKnowledgeStore
        with patch("datus.tools.func_tool.context_search.ExtKnowledgeStore") as mock_store_class:
            mock_store = MagicMock()
            mock_store.table_size.return_value = 5  # Has knowledge
            mock_store_class.return_value = mock_store

            tools = ContextSearchTools(mock_config)

            # Verify external knowledge search is available
            available_tools = tools.available_tools()
            tool_names = [tool.__name__ for tool in available_tools]

            assert "search_external_knowledge" in str(available_tools)
            assert tools.has_ext_knowledge is True

    def test_empty_knowledge_store(self):
        """Test behavior when external knowledge store is empty."""
        from datus.tools.func_tool.context_search import ContextSearchTools

        mock_config = MagicMock()
        mock_config.rag_storage_path.return_value = "/tmp/test"

        # Mock empty ExtKnowledgeStore
        with patch("datus.tools.func_tool.context_search.ExtKnowledgeStore") as mock_store_class:
            mock_store = MagicMock()
            mock_store.table_size.return_value = 0  # Empty store
            mock_store_class.return_value = mock_store

            tools = ContextSearchTools(mock_config)

            assert tools.has_ext_knowledge is False

            # External knowledge search should not be available
            available_tools = tools.available_tools()
            tool_names = [tool.__name__ for tool in available_tools]
            assert "search_external_knowledge" not in str(available_tools)


if __name__ == "__main__":
    pytest.main([__file__])
