import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from datus.agent.node.chat_agentic_node import PreflightOrchestrator
from datus.schemas.action_history import ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.node_models import SqlTask


class TestEnhancedPreflightTools:
    """Test enhanced preflight tools functionality."""

    @pytest.fixture
    def mock_workflow(self):
        """Create a mock workflow with enhanced tool sequence."""
        workflow = MagicMock()
        workflow.metadata = {
            "required_tool_sequence": ["analyze_query_plan", "check_table_conflicts", "validate_partitioning"]
        }
        workflow.task = SqlTask(
            task="审查SQL：SELECT * FROM test_table WHERE id = 1",
            database_type="starrocks",
            database_name="test",
            catalog_name="default_catalog",
        )
        workflow.context = MagicMock()
        workflow.context.preflight_results = {}
        return workflow

    @pytest.fixture
    def mock_action_history_manager(self):
        """Create a mock action history manager."""
        return MagicMock(spec=ActionHistoryManager)

    @pytest.fixture
    def chat_node(self):
        """Create a ChatAgenticNode instance with mocked dependencies."""
        with patch("datus.agent.node.chat_agentic_node.GenSQLAgenticNode.__init__", return_value=None):
            from datus.agent.node.chat_agentic_node import ChatAgenticNode

            node = ChatAgenticNode(node_id="test_node", description="Test node", node_type="chat")

            # Mock required attributes
            node.db_func_tool = MagicMock()
            node.plan_hooks = MagicMock()
            node.plan_hooks.enable_query_caching = True
            node.plan_hooks.query_cache = MagicMock()
            node.execution_event_manager = MagicMock()

            return node

    @pytest.fixture
    def orchestrator(self, chat_node):
        """Create a PreflightOrchestrator instance."""
        return PreflightOrchestrator(chat_node)

    @pytest.mark.asyncio
    async def test_analyze_query_plan_tool(self, orchestrator, chat_node):
        """Test analyze_query_plan tool execution."""
        # Mock the db_func_tool response
        mock_result = {
            "success": True,
            "plan_text": "EXPLAIN output...",
            "estimated_rows": 1000,
            "estimated_cost": 150.5,
            "hotspots": [
                {
                    "reason": "full_table_scan",
                    "node": "TableScan(test_table)",
                    "severity": "high",
                    "recommendation": "Add index on id column",
                }
            ],
            "join_analysis": {"join_count": 0, "join_types": [], "join_order_issues": []},
            "index_usage": {
                "indexes_used": [],
                "missing_indexes": ["idx_test_table_id"],
                "index_effectiveness": "poor",
            },
            "warnings": [],
        }
        chat_node.db_func_tool.analyze_query_plan.return_value = mock_result

        # Mock cache miss
        chat_node.plan_hooks.query_cache.get_enhanced.return_value = None

        result = await orchestrator._call_enhanced_tool(
            "analyze_query_plan", "SELECT * FROM test_table WHERE id = 1", ["test_table"], "default_catalog", "test", ""
        )

        assert result["success"] is True
        assert result["estimated_rows"] == 1000
        assert len(result["hotspots"]) == 1
        assert result["index_usage"]["index_effectiveness"] == "poor"

        # Verify cache was checked and set
        chat_node.plan_hooks.query_cache.get_enhanced.assert_called_once()
        chat_node.plan_hooks.query_cache.set_enhanced.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_table_conflicts_tool(self, orchestrator, chat_node):
        """Test check_table_conflicts tool execution."""
        # Mock the db_func_tool response
        mock_result = {
            "success": True,
            "exists_similar": True,
            "target_table": {
                "name": "test_table",
                "columns": ["id", "name", "created_at"],
                "ddl_hash": "abc123",
                "estimated_rows": 10000,
            },
            "matches": [
                {
                    "table_name": "test_table_backup",
                    "similarity_score": 0.85,
                    "conflict_type": "similar_business",
                    "matching_columns": ["id", "name"],
                    "column_similarity": 0.8,
                    "business_conflict": "可能存在用户数据重复",
                    "recommendation": "建议评估是否可以复用现有表",
                }
            ],
            "duplicate_build_risk": "medium",
            "layering_violations": [],
        }
        chat_node.db_func_tool.check_table_conflicts.return_value = mock_result

        # Mock cache miss
        chat_node.plan_hooks.query_cache.get_enhanced.return_value = None

        result = await orchestrator._call_enhanced_tool(
            "check_table_conflicts",
            "SELECT * FROM test_table WHERE id = 1",
            ["test_table"],
            "default_catalog",
            "test",
            "",
        )

        assert result["success"] is True
        assert result["exists_similar"] is True
        assert len(result["matches"]) == 1
        assert result["duplicate_build_risk"] == "medium"

    @pytest.mark.asyncio
    async def test_validate_partitioning_tool(self, orchestrator, chat_node):
        """Test validate_partitioning tool execution."""
        # Mock the db_func_tool response
        mock_result = {
            "success": True,
            "partitioned": True,
            "partition_info": {
                "partition_key": "created_at",
                "partition_type": "time_based",
                "partition_count": 30,
                "partition_expression": "date_trunc('day', created_at)",
            },
            "validation_results": {
                "partition_key_valid": True,
                "granularity_appropriate": True,
                "data_distribution_even": False,
                "pruning_opportunities": True,
            },
            "issues": [
                {
                    "severity": "medium",
                    "issue_type": "uneven_distribution",
                    "description": "某些分区数据量偏大",
                    "recommendation": "考虑调整分区策略",
                }
            ],
            "recommended_partition": {
                "suggested_key": "created_at",
                "suggested_type": "time_based",
                "estimated_partitions": 12,
                "rationale": "建议按月分区以获得更好的数据分布",
            },
            "performance_impact": {
                "query_speed_improvement": "significant",
                "storage_efficiency": "improved",
                "maintenance_overhead": "acceptable",
            },
        }
        chat_node.db_func_tool.validate_partitioning.return_value = mock_result

        result = await orchestrator._call_enhanced_tool(
            "validate_partitioning",
            "SELECT * FROM test_table WHERE created_at >= '2024-01-01'",
            ["test_table"],
            "default_catalog",
            "test",
            "",
        )

        assert result["success"] is True
        assert result["partitioned"] is True
        assert result["partition_info"]["partition_key"] == "created_at"
        assert len(result["issues"]) == 1

    @pytest.mark.asyncio
    async def test_cache_hit_optimization(self, orchestrator, chat_node):
        """Test that cache hits work correctly."""
        cached_result = {
            "success": True,
            "plan_text": "Cached EXPLAIN output...",
            "estimated_rows": 500,
            "cached": True,
        }

        # Mock cache hit
        chat_node.plan_hooks.query_cache.get_enhanced.return_value = cached_result

        result = await orchestrator._call_enhanced_tool(
            "analyze_query_plan", "SELECT * FROM test_table WHERE id = 1", ["test_table"], "default_catalog", "test", ""
        )

        assert result == cached_result
        # Verify cache was checked but not set (since it was a hit)
        chat_node.plan_hooks.query_cache.get_enhanced.assert_called_once()
        chat_node.plan_hooks.query_cache.set_enhanced.assert_not_called()

    @pytest.mark.asyncio
    async def test_error_handling(self, orchestrator, chat_node):
        """Test error handling for failed tool calls."""
        # Mock tool failure
        chat_node.db_func_tool.analyze_query_plan.side_effect = Exception("Database connection failed")

        # Mock cache miss
        chat_node.plan_hooks.query_cache.get_enhanced.return_value = None

        result = await orchestrator._call_enhanced_tool(
            "analyze_query_plan", "SELECT * FROM test_table WHERE id = 1", ["test_table"], "default_catalog", "test", ""
        )

        assert result["success"] is False
        assert "Database connection failed" in result["error"]

    @pytest.mark.asyncio
    async def test_context_injection(self, orchestrator, mock_workflow):
        """Test that tool results are properly injected into workflow context."""
        # Mock successful tool result
        mock_result = {"success": True, "plan_text": "Test plan", "hotspots": []}

        orchestrator._inject_tool_result_into_context(mock_workflow, "analyze_query_plan", mock_result)

        # Verify context was updated
        assert hasattr(mock_workflow.context, "preflight_results")
        assert "query_plan_analysis" in mock_workflow.context.preflight_results
        assert mock_workflow.context.preflight_results["query_plan_analysis"]["plan_text"] == "Test plan"

    @pytest.mark.asyncio
    async def test_batch_processing(self, orchestrator):
        """Test batch processing of table-related tools."""
        batch = [
            (MagicMock(), {"table_name": "table1", "catalog": "cat", "database": "db", "schema": "sch"}),
            (MagicMock(), {"table_name": "table2", "catalog": "cat", "database": "db", "schema": "sch"}),
        ]

        # Mock db_func_tool responses
        orchestrator.db_func_tool.check_table_conflicts.side_effect = [
            {"success": True, "table_name": "table1"},
            {"success": True, "table_name": "table2"},
        ]

        results = await orchestrator._process_tool_batch("check_table_conflicts", batch, MagicMock())

        assert len(results) == 2
        assert results[0]["table_name"] == "table1"
        assert results[1]["table_name"] == "table2"

    def test_error_classification(self, orchestrator):
        """Test error type classification."""
        assert orchestrator._classify_error_type("permission denied", "analyze_query_plan") == "permission_error"
        assert orchestrator._classify_error_type("timeout", "read_query") == "timeout_error"
        assert orchestrator._classify_error_type("table does not exist", "describe_table") == "table_not_found"
        assert orchestrator._classify_error_type("connection failed", "read_query") == "connection_error"
        assert orchestrator._classify_error_type("unknown error", "analyze_query_plan") == "unknown_error"
