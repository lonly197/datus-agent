import asyncio

import pytest

from datus.agent.node.schema_validation_node import SchemaValidationNode
from datus.agent.node.sql_validate_node import SQLValidateNode
from datus.agent.workflow import Workflow
from datus.agent.workflow_status import WorkflowTerminationStatus
from datus.schemas.action_history import ActionStatus
from datus.schemas.node_models import SQLContext, SqlTask, TableSchema


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
    assert wf.metadata.get("termination_status") == WorkflowTerminationStatus.SKIP_TO_REFLECT
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
    assert wf.metadata.get("termination_status") == WorkflowTerminationStatus.SKIP_TO_REFLECT
