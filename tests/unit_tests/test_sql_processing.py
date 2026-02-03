# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from datus.api.event_converter.sql_processing import generate_execution_report


def test_execution_report_marks_execute_failure():
    metadata = {
        "failure_stage": "execute_sql",
        "termination_reason": "Column not found",
        "sql_validation": {"syntax_valid": True},
    }
    report = generate_execution_report(0, metadata)
    assert "执行状态" in report
    assert "SQL执行失败" in report
    assert "Column not found" in report
    assert "SQL适合生产使用" in report and "否" in report


def test_execution_report_marks_validation_failure():
    metadata = {
        "failure_stage": "result_validation",
        "sql_validation": {
            "syntax_valid": True,
            "tables_exist": True,
            "columns_exist": False,
            "has_dangerous_ops": False,
        },
        "result_validation": {"is_valid": False, "reason": "missing key metric"},
    }
    report = generate_execution_report(12, metadata)
    assert "验证失败类型" in report
    assert "SQL验证" in report
    assert "结果验证" in report
    assert "missing key metric" in report
