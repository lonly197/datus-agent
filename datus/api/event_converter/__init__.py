# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Event converter package for mapping ActionHistory to DeepResearchEvent format.

This package has been refactored from the original monolithic event_converter.py
(2390 lines) into focused modules for better maintainability:

Modules:
- core: Main DeepResearchEventConverter class
- virtual_steps: Virtual step management for Text2SQL workflows
- normalization: Identifier and status normalization utilities
- sql_processing: SQL report generation and DDL parsing
- event_validation: Event flow validation and helper methods
- streaming: SSE stream conversion utilities
"""

# Re-export all event models
from .models import (
    ChatEvent,
    CompleteEvent,
    DeepResearchEvent,
    ErrorEvent,
    PlanUpdateEvent,
    ReportEvent,
    TodoItem,
    TodoStatus,
    ToolCallEvent,
    ToolCallResultEvent,
)

# Re-export main converter class
from .core import DeepResearchEventConverter

# Re-export streaming utilities
from .streaming import convert_stream_to_events

# Re-export virtual steps for external use
from .virtual_steps import VIRTUAL_STEPS

# Re-export normalization functions for external use
from .normalization import (
    normalize_node_type,
    normalize_tool_name,
    normalize_todo_status,
    hash_text,
)

# Re-export validation functions for external use
from .event_validation import (
    extract_plan_from_output,
    extract_callid_from_output,
    extract_todo_id_from_action,
    get_unified_plan_id,
    find_tool_call_id,
    is_internal_todo_update,
    extract_node_type_from_action,
    validate_event_flow,
    generate_event_id,
)

# Re-export SQL processing functions for external use
from .sql_processing import (
    generate_sql_summary,
    format_diagnostic_report,
    parse_ddl_comments,
    extract_table_info,
    analyze_relationships,
    parse_sql_structure,
    infer_field_usage,
    get_field_comment,
    escape_markdown_table_cell,
)

__all__ = [
    # Main converter
    "DeepResearchEventConverter",

    # Event models
    "ChatEvent",
    "CompleteEvent",
    "DeepResearchEvent",
    "ErrorEvent",
    "PlanUpdateEvent",
    "ReportEvent",
    "TodoItem",
    "TodoStatus",
    "ToolCallEvent",
    "ToolCallResultEvent",

    # Streaming
    "convert_stream_to_events",

    # Virtual steps
    "VIRTUAL_STEPS",

    # Normalization
    "normalize_node_type",
    "normalize_tool_name",
    "normalize_todo_status",
    "hash_text",

    # Validation
    "extract_plan_from_output",
    "extract_callid_from_output",
    "extract_todo_id_from_action",
    "get_unified_plan_id",
    "find_tool_call_id",
    "is_internal_todo_update",
    "extract_node_type_from_action",
    "validate_event_flow",
    "generate_event_id",

    # SQL processing
    "generate_sql_summary",
    "format_diagnostic_report",
    "parse_ddl_comments",
    "extract_table_info",
    "analyze_relationships",
    "parse_sql_structure",
    "infer_field_usage",
    "get_field_comment",
    "escape_markdown_table_cell",
]
