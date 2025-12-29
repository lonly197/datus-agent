import pytest

from datus.api.service import DatusAPIService


class TestTaskIdentification:
    """Test task type identification for SQL review tasks."""

    def test_identify_sql_review_task(self):
        """Test that SQL review tasks are correctly identified."""
        service = DatusAPIService()

        # Test various SQL review keywords
        review_tasks = [
            "审查以下SQL：SELECT * FROM table",
            "请评审这个SQL查询",
            "检查SQL代码质量",
            "审核数据库查询",
            "analyze this SQL for review",
            "check SQL quality and performance",
        ]

        for task in review_tasks:
            result = service._identify_task_type(task)
            assert result == "sql_review", f"Failed to identify task: {task}"

    def test_identify_data_analysis_task(self):
        """Test that data analysis tasks are correctly identified."""
        service = DatusAPIService()

        analysis_tasks = [
            "分析用户行为数据",
            "统计销售趋势",
            "对比不同时期的业绩",
            "汇总各部门数据",
            "analyze user behavior data",
            "compare sales performance",
        ]

        for task in analysis_tasks:
            result = service._identify_task_type(task)
            assert result == "data_analysis", f"Failed to identify task: {task}"

    def test_identify_text2sql_task(self):
        """Test that regular text2sql tasks are correctly identified."""
        service = DatusAPIService()

        sql_tasks = [
            "查询用户表中的所有记录",
            "给我一个统计销售额的SQL",
            "创建报表查询",
            "find all users in the database",
            "generate SQL for sales report",
        ]

        for task in sql_tasks:
            result = service._identify_task_type(task)
            assert result == "text2sql", f"Failed to identify task: {task}"

    def test_configure_sql_review_processing(self):
        """Test that sql_review tasks get correct configuration."""
        service = DatusAPIService()

        # Mock request object
        class MockRequest:
            pass

        request = MockRequest()
        config = service._configure_task_processing("sql_review", request)

        assert config["workflow"] == "chat_agentic"
        assert config["plan_mode"] is False
        assert config["auto_execute_plan"] is False
        assert config["system_prompt"] == "sql_review"
        assert config["output_format"] == "markdown"
        assert "required_tool_sequence" in config
        assert isinstance(config["required_tool_sequence"], list)
        assert len(config["required_tool_sequence"]) > 0

    def test_configure_data_analysis_processing(self):
        """Test that data_analysis tasks get correct configuration."""
        service = DatusAPIService()

        class MockRequest:
            pass

        request = MockRequest()
        config = service._configure_task_processing("data_analysis", request)

        assert config["workflow"] == "chat_agentic_plan"
        assert config["plan_mode"] is True
        assert config["auto_execute_plan"] is True
        assert config["system_prompt"] == "plan_mode"
        assert config["output_format"] == "json"

    def test_configure_text2sql_processing(self):
        """Test that text2sql tasks get correct configuration."""
        service = DatusAPIService()

        class MockRequest:
            pass

        request = MockRequest()
        config = service._configure_task_processing("text2sql", request)

        assert config["workflow"] == "chat_agentic"
        assert config["plan_mode"] is False
        assert config["auto_execute_plan"] is False
        assert config["system_prompt"] == "chat_system"
        assert config["output_format"] == "json"
