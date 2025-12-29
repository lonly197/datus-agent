#!/usr/bin/env python3
"""Test script for task type identification and configuration logic."""

class MockChatResearchRequest:
    """Mock request object for testing."""
    def __init__(self, task, namespace="test", catalog_name="default_catalog",
                 database_name="test", ext_knowledge="", plan_mode=True, auto_execute_plan=True):
        self.task = task
        self.namespace = namespace
        self.catalog_name = catalog_name
        self.database_name = database_name
        self.ext_knowledge = ext_knowledge
        self.plan_mode = plan_mode
        self.auto_execute_plan = auto_execute_plan

def identify_task_type(task_text: str) -> str:
    """识别任务类型"""
    task_lower = task_text.lower()

    # SQL审查任务特征
    review_keywords = [
        "审查", "review", "检查", "check", "审核", "audit",
        "质量", "quality", "评估", "evaluate", "分析sql", "analyze sql"
    ]
    if any(keyword in task_lower for keyword in review_keywords):
        return "sql_review"

    # 数据分析任务特征
    analysis_keywords = [
        "分析", "analysis", "对比", "compare", "趋势", "trend",
        "统计", "statistics", "汇总", "summary", "报告", "report"
    ]
    if any(keyword in task_lower for keyword in analysis_keywords):
        return "data_analysis"

    # 默认Text2SQL
    return "text2sql"

def configure_task_processing(task_type: str, request) -> dict:
    """根据任务类型配置处理参数"""

    if task_type == "sql_review":
        # SQL审查：不使用plan模式，使用专门的审查提示词
        return {
            "workflow": "chat_agentic",  # 不使用plan模式
            "plan_mode": False,
            "auto_execute_plan": False,
            "system_prompt": "sql_review_system",  # 专门的SQL审查提示词
            "output_format": "markdown"  # 指定输出格式
        }
    elif task_type == "data_analysis":
        # 数据分析：使用plan模式
        return {
            "workflow": "chat_agentic_plan",
            "plan_mode": True,
            "auto_execute_plan": True,
            "system_prompt": "plan_mode",
            "output_format": "json"
        }
    else:  # text2sql
        # Text2SQL：标准模式
        return {
            "workflow": "chat_agentic",
            "plan_mode": False,
            "auto_execute_plan": False,
            "system_prompt": "chat_system",
            "output_format": "json"
        }

def test_task_identification():
    """Test task type identification logic."""

    # Test cases
    test_cases = [
        # SQL审查任务
        {
            "task": "审查以下SQL：SELECT * FROM table WHERE id = 1",
            "expected_type": "sql_review"
        },
        {
            "task": "请检查这个SQL的性能：SELECT * FROM users",
            "expected_type": "sql_review"
        },
        {
            "task": "分析用户表的数据分布情况",
            "expected_type": "data_analysis"
        },
        {
            "task": "统计各部门的销售额",
            "expected_type": "data_analysis"
        },
        {
            "task": "帮我查询用户总数",
            "expected_type": "text2sql"
        },
        {
            "task": "SELECT * FROM users WHERE age > 18",
            "expected_type": "text2sql"
        }
    ]

    print("Testing task type identification:")
    print("=" * 50)

    for i, test_case in enumerate(test_cases, 1):
        task_text = test_case["task"]
        expected = test_case["expected_type"]

        identified_type = identify_task_type(task_text)
        status = "✓" if identified_type == expected else "✗"

        print("2d")
        print(f"  Task: {task_text}")
        print(f"  Expected: {expected}, Got: {identified_type}")
        print()

def test_task_configuration():
    """Test task configuration logic."""

    # Create a mock request
    request = MockChatResearchRequest(
        task="审查SQL：SELECT * FROM table"
    )

    task_types = ["sql_review", "data_analysis", "text2sql"]

    print("Testing task configuration:")
    print("=" * 50)

    for task_type in task_types:
        config = configure_task_processing(task_type, request)

        print(f"Task Type: {task_type}")
        print(f"  Workflow: {config['workflow']}")
        print(f"  Plan Mode: {config['plan_mode']}")
        print(f"  Auto Execute: {config['auto_execute_plan']}")
        print(f"  System Prompt: {config['system_prompt']}")
        print(f"  Output Format: {config['output_format']}")
        print()

if __name__ == "__main__":
    try:
        test_task_identification()
        test_task_configuration()
        print("All tests completed successfully!")
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
