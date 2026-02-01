import asyncio

import pytest

from datus.agent.node.schema_validation_node import SchemaValidationNode
from datus.agent.node.sql_validate_node import SQLValidateNode
from datus.agent.workflow import Workflow
from datus.agent.workflow_status import WorkflowTerminationStatus
from datus.schemas.action_history import ActionStatus
from datus.schemas.node_models import SQLContext, SqlTask, TableSchema


@pytest.fixture(autouse=True)
def _disable_workflow_tools(monkeypatch):
    """Avoid initializing DB tools during unit tests."""
    monkeypatch.setattr(Workflow, "_init_tools", lambda self: None, raising=False)


@pytest.mark.asyncio
async def test_schema_validation_hard_block_sets_termination(monkeypatch):
    """Coverage极低时应直接跳转反射，避免继续生成SQL。"""

    async def fake_validate(self, task, context):
        return {
            "is_sufficient": False,
            "table_count": 1,
            "missing_definitions": [],
            "query_terms": [],
            "coverage_score": 0.0,
            "coverage_threshold": 0.2,
            "covered_terms": [],
            "uncovered_terms": [],
            "critical_terms": ["首次"],
            "critical_terms_covered": [],
            "critical_terms_uncovered": ["首次"],
            "critical_coverage_score": 0.0,
            "critical_coverage_threshold": 0.5,
            "term_evidence": {},
            "table_coverage": {},
            "invalid_definitions": [],
        }

    monkeypatch.setattr(SchemaValidationNode, "_validate_schema_coverage", fake_validate, raising=False)

    task = SqlTask(
        id="t1",
        database_type="starrocks",
        task="统计每个月‘首次试驾’到‘下定’的平均转化周期（天数)",
        catalog_name="default_catalog",
        database_name="test",
        schema_name="",
        output_dir="/tmp",
        external_knowledge="",
        schema_linking_type="table",
        date_ranges="",
    )
    wf = Workflow("test", task=task)
    wf.context.table_schemas = [
        TableSchema(
            identifier="default_catalog.test..dummy.table",
            catalog_name="default_catalog",
            table_name="dummy",
            database_name="test",
            schema_name="",
            definition="CREATE TABLE dummy(id INT)",
        )
    ]

    node = SchemaValidationNode("sv", "schema validation", "schema_validation")
    node.workflow = wf

    node.setup_input(wf)
    async for _ in node.run():
        pass

    assert node.last_action_status == ActionStatus.FAILED
    assert wf.metadata.get("termination_status") in (
        WorkflowTerminationStatus.RETRY_SQL,
        WorkflowTerminationStatus.SKIP_TO_REFLECT,
    )
    assert wf.metadata.get("termination_reason") == "schema_insufficient_coverage"


@pytest.mark.asyncio
async def test_sql_validate_flags_missing_columns(monkeypatch):
    """SQLValidateNode应基于DDL识别缺失列并触发反射跳转。"""

    task = SqlTask(
        id="t2",
        database_type="starrocks",
        task="select clue_status",
        catalog_name="default_catalog",
        database_name="test",
        schema_name="",
        output_dir="/tmp",
        external_knowledge="",
        schema_linking_type="table",
        date_ranges="",
    )
    wf = Workflow("test", task=task)
    wf.context.table_schemas = [
        TableSchema(
            identifier="default_catalog.test..dwd_assign_dlr_clue_fact_di.table",
            catalog_name="default_catalog",
            table_name="dwd_assign_dlr_clue_fact_di",
            database_name="test",
            schema_name="",
            definition="CREATE TABLE dwd_assign_dlr_clue_fact_di(dealer_code string, status int)",
        )
    ]
    wf.context.sql_contexts.append(
        SQLContext(
            sql_query="select clue_status from dwd_assign_dlr_clue_fact_di",
            explanation="",
            sql_return="",
            sql_error="",
            row_count=0,
            reflection_strategy="",
            reflection_explanation="",
        )
    )

    node = SQLValidateNode("sv", "sql validate", "sql_validate")
    node.workflow = wf

    node.setup_input(wf)
    async for _ in node.run():
        pass

    node.update_context(wf)

    assert node.result.data["columns_exist"] is False
    assert any("clue_status" in err for err in node.result.data["errors"])
    assert wf.metadata.get("termination_status") in (
        WorkflowTerminationStatus.RETRY_SQL,
        WorkflowTerminationStatus.SKIP_TO_REFLECT,
    )


@pytest.mark.asyncio
async def test_sql_validate_allows_cte_and_alias_columns():
    """SQLValidateNode不应将CTE/别名列当作物理列校验。"""

    task = SqlTask(
        id="t3",
        database_type="starrocks",
        task="按月统计转化漏斗",
        catalog_name="default_catalog",
        database_name="test",
        schema_name="",
        output_dir="/tmp",
        external_knowledge="",
        schema_linking_type="table",
        date_ranges="",
    )
    wf = Workflow("test", task=task)
    wf.context.table_schemas = [
        TableSchema(
            identifier="default_catalog.test..dws_clue_dlr_clue_testdrive_order_2h_di_test.table",
            catalog_name="default_catalog",
            table_name="dws_clue_dlr_clue_testdrive_order_2h_di_test",
            database_name="test",
            schema_name="",
            definition=(
                "CREATE TABLE dws_clue_dlr_clue_testdrive_order_2h_di_test("
                "dealer_clue_code string, clue_create_time string, is_valid_clue int)"
            ),
        )
    ]
    wf.context.sql_contexts.append(
        SQLContext(
            sql_query=(
                "WITH monthly_data AS ("
                "SELECT DATE_FORMAT(clue_create_time, '%Y-%m') AS month, "
                "COUNT(DISTINCT t.dealer_clue_code) AS total_clues, "
                "COUNT(DISTINCT CASE WHEN t.is_valid_clue = 1 THEN t.dealer_clue_code END) AS valid_clues "
                "FROM dws_clue_dlr_clue_testdrive_order_2h_di_test t "
                "GROUP BY DATE_FORMAT(clue_create_time, '%Y-%m')) "
                "SELECT month, total_clues, valid_clues FROM monthly_data"
            ),
            explanation="",
            sql_return="",
            sql_error="",
            row_count=0,
            reflection_strategy="",
            reflection_explanation="",
        )
    )

    node = SQLValidateNode("sv", "sql validate", "sql_validate")
    node.workflow = wf

    node.setup_input(wf)
    async for _ in node.run():
        pass

    assert node.result.data["columns_exist"] is True
    assert node.result.data["tables_exist"] is True


@pytest.mark.asyncio
async def test_sql_validate_cte_column_list_and_nested_subquery():
    """CTE显式列清单与子查询别名列不应触发列缺失。"""

    task = SqlTask(
        id="t4",
        database_type="starrocks",
        task="复杂CTE与子查询",
        catalog_name="default_catalog",
        database_name="test",
        schema_name="",
        output_dir="/tmp",
        external_knowledge="",
        schema_linking_type="table",
        date_ranges="",
    )
    wf = Workflow("test", task=task)
    wf.context.table_schemas = [
        TableSchema(
            identifier="default_catalog.test..t_orders.table",
            catalog_name="default_catalog",
            table_name="t_orders",
            database_name="test",
            schema_name="",
            definition="CREATE TABLE t_orders(order_id string, order_time string, amount int)",
        )
    ]
    wf.context.sql_contexts.append(
        SQLContext(
            sql_query=(
                "WITH base(month, cnt) AS ("
                "SELECT DATE_FORMAT(order_time, '%Y-%m') AS month, COUNT(*) AS cnt "
                "FROM t_orders GROUP BY DATE_FORMAT(order_time, '%Y-%m')) "
                "SELECT b.month, b.cnt, x.cnt2 "
                "FROM base b "
                "JOIN (SELECT month, cnt AS cnt2 FROM base) x ON x.month = b.month"
            ),
            explanation="",
            sql_return="",
            sql_error="",
            row_count=0,
            reflection_strategy="",
            reflection_explanation="",
        )
    )

    node = SQLValidateNode("sv", "sql validate", "sql_validate")
    node.workflow = wf

    node.setup_input(wf)
    async for _ in node.run():
        pass

    assert node.result.data["columns_exist"] is True
    assert node.result.data["tables_exist"] is True


@pytest.mark.asyncio
async def test_sql_validate_cte_window_function_alias():
    """窗口函数别名列在CTE中应被识别为虚拟列。"""

    task = SqlTask(
        id="t5",
        database_type="starrocks",
        task="窗口函数别名",
        catalog_name="default_catalog",
        database_name="test",
        schema_name="",
        output_dir="/tmp",
        external_knowledge="",
        schema_linking_type="table",
        date_ranges="",
    )
    wf = Workflow("test", task=task)
    wf.context.table_schemas = [
        TableSchema(
            identifier="default_catalog.test..t_clue.table",
            catalog_name="default_catalog",
            table_name="t_clue",
            database_name="test",
            schema_name="",
            definition="CREATE TABLE t_clue(dealer_clue_code string, clue_create_time string)",
        )
    ]
    wf.context.sql_contexts.append(
        SQLContext(
            sql_query=(
                "WITH ranked AS ("
                "SELECT dealer_clue_code, "
                "ROW_NUMBER() OVER (PARTITION BY dealer_clue_code ORDER BY clue_create_time) AS rn "
                "FROM t_clue) "
                "SELECT dealer_clue_code FROM ranked WHERE rn = 1"
            ),
            explanation="",
            sql_return="",
            sql_error="",
            row_count=0,
            reflection_strategy="",
            reflection_explanation="",
        )
    )

    node = SQLValidateNode("sv", "sql validate", "sql_validate")
    node.workflow = wf

    node.setup_input(wf)
    async for _ in node.run():
        pass

    assert node.result.data["columns_exist"] is True
    assert node.result.data["tables_exist"] is True


@pytest.mark.asyncio
async def test_sql_validate_union_subquery_alias_columns():
    """UNION子查询输出列应被识别为虚拟列。"""

    task = SqlTask(
        id="t6",
        database_type="starrocks",
        task="union子查询",
        catalog_name="default_catalog",
        database_name="test",
        schema_name="",
        output_dir="/tmp",
        external_knowledge="",
        schema_linking_type="table",
        date_ranges="",
    )
    wf = Workflow("test", task=task)
    wf.context.table_schemas = [
        TableSchema(
            identifier="default_catalog.test..t_orders.table",
            catalog_name="default_catalog",
            table_name="t_orders",
            database_name="test",
            schema_name="",
            definition="CREATE TABLE t_orders(order_time string, amount int)",
        )
    ]
    wf.context.sql_contexts.append(
        SQLContext(
            sql_query=(
                "SELECT u.month, u.cnt FROM ("
                "SELECT DATE_FORMAT(order_time, '%Y-%m') AS month, COUNT(*) AS cnt "
                "FROM t_orders GROUP BY DATE_FORMAT(order_time, '%Y-%m') "
                "UNION ALL "
                "SELECT DATE_FORMAT(order_time, '%Y-%m') AS month, COUNT(*) AS cnt "
                "FROM t_orders GROUP BY DATE_FORMAT(order_time, '%Y-%m')) u "
                "WHERE u.cnt > 0"
            ),
            explanation="",
            sql_return="",
            sql_error="",
            row_count=0,
            reflection_strategy="",
            reflection_explanation="",
        )
    )

    node = SQLValidateNode("sv", "sql validate", "sql_validate")
    node.workflow = wf

    node.setup_input(wf)
    async for _ in node.run():
        pass

    assert node.result.data["columns_exist"] is True
    assert node.result.data["tables_exist"] is True
