# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details

from enum import Enum


class WorkflowTerminationStatus(str, Enum):
    """工作流终止状态"""

    CONTINUE = "continue"  # 继续执行
    SKIP_TO_REFLECT = "skip_to_reflect"  # 跳转到反思节点
    PROCEED_TO_OUTPUT = "proceed_to_output"  # 继续执行到输出节点（生成报告）
    TERMINATE_WITH_ERROR = "terminate_with_error"  # 终止并报错
    TERMINATE_SUCCESS = "terminate_success"  # 成功终止
