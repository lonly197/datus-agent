# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
API models for the Datus Agent FastAPI service.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class Mode(str, Enum):
    """Execution mode enum."""

    SYNC = "sync"
    ASYNC = "async"


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service health status")
    version: str = Field(..., description="Service version")
    database_status: Dict[str, str] = Field(..., description="Database connection status")
    llm_status: str = Field(..., description="LLM service status")


class RunWorkflowRequest(BaseModel):
    """Request model for workflow execution."""

    workflow: str = Field(..., description="Workflow name, e.g., nl2sql")
    namespace: str = Field(..., description="Database namespace")
    task: str = Field(..., description="Natural language task description")
    mode: Mode = Field(Mode.SYNC, description="Execution mode: sync or async")
    task_id: Optional[str] = Field(None, description="Custom task ID for idempotency")
    catalog_name: Optional[str] = Field(None, description="Catalog name")
    database_name: Optional[str] = Field(None, description="Database name")
    schema_name: Optional[str] = Field(None, description="Schema name")
    current_date: Optional[str] = Field(None, description="Current date reference for relative time expressions")
    subject_path: Optional[List[str]] = Field(None, description="Subject path for the task")
    ext_knowledge: Optional[str] = Field(None, description="External knowledge for the task")
    plan_mode: Optional[bool] = Field(False, description="Enable plan mode for structured execution")


class RunWorkflowResponse(BaseModel):
    """Response model for workflow execution."""

    task_id: str = Field(..., description="Unique task identifier")
    status: str = Field(..., description="Workflow execution status")
    workflow: str = Field(..., description="Workflow name")
    sql: Optional[str] = Field(None, description="Generated SQL query")
    result: Optional[List[Dict[str, Any]]] = Field(None, description="Workflow execution results")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    error: Optional[str] = Field(None, description="Error message if any")
    execution_time: Optional[float] = Field(None, description="Execution time in seconds")


class TokenResponse(BaseModel):
    """Token response model."""

    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(..., description="Token type, always 'Bearer'")
    expires_in: int = Field(..., description="Token expiration time in seconds")


class TaskStartResponse(BaseModel):
    """Response model for task start confirmation."""

    task_id: str = Field(..., description="Confirmed task ID")
    status: str = Field("accepted", description="Task acceptance status")
    message: str = Field(..., description="Confirmation message")
    estimated_start_time: Optional[str] = Field(None, description="Estimated start time")


class FeedbackStatus(str, Enum):
    """Feedback status enum."""

    SUCCESS = "success"
    FAILED = "failed"


class FeedbackRequest(BaseModel):
    """Request model for user feedback."""

    task_id: str = Field(..., description="Target task ID")
    status: FeedbackStatus = Field(..., description="Task execution status feedback")


class FeedbackResponse(BaseModel):
    """Response model for feedback submission."""

    task_id: str = Field(..., description="Task ID that feedback was recorded for")
    acknowledged: bool = Field(..., description="Whether feedback was successfully recorded")
    recorded_at: str = Field(..., description="ISO timestamp when feedback was recorded")


class ChatResearchRequest(BaseModel):
    """Request model for chat research execution."""

    namespace: str = Field(..., description="Database namespace")
    task: str = Field(..., description="Natural language task description")
    task_id: Optional[str] = Field(None, description="Custom task ID for idempotency (frontend messageId)")
    catalog_name: Optional[str] = Field(None, description="Catalog name")
    database_name: Optional[str] = Field(None, description="Database name")
    schema_name: Optional[str] = Field(None, description="Schema name")
    current_date: Optional[str] = Field(None, description="Current date reference")
    domain: Optional[str] = Field(None, description="Domain for the task")
    layer1: Optional[str] = Field(None, description="Layer1 for the task")
    layer2: Optional[str] = Field(None, description="Layer2 for the task")
    ext_knowledge: Optional[str] = Field(None, description="External knowledge")
    plan_mode: Optional[bool] = Field(False, description="Enable plan mode for structured execution")
    auto_execute_plan: Optional[bool] = Field(False, description="Auto execute plan without user confirmation")
    prompt: Optional[str] = Field(None, description="Role definition and task capability prompt to guide the AI agent")
    prompt_mode: Optional[str] = Field(
        "append", description="How to merge prompt with system prompt: 'replace' or 'append' (default)"
    )
    execution_mode: Optional[str] = Field(
        None,
        description="Execution mode override. Supports predefined scenarios ('text2sql', 'sql_review', 'data_analysis', 'deep_analysis') or direct workflow names ('chat_agentic_plan', 'metric_to_sql', etc.). Falls back to auto-detection if invalid or unspecified.",
    )


class DeepResearchEventType(str, Enum):
    """Event types for deep research streaming."""

    CHAT = "chat"
    PLAN_UPDATE = "plan_update"
    TOOL_CALL = "tool_call"
    TOOL_CALL_RESULT = "tool_call_result"
    SQL_EXECUTION_START = "sql_execution_start"
    SQL_EXECUTION_PROGRESS = "sql_execution_progress"
    SQL_EXECUTION_RESULT = "sql_execution_result"
    SQL_EXECUTION_ERROR = "sql_execution_error"
    COMPLETE = "complete"
    REPORT = "report"
    ERROR = "error"


class TodoStatus(str, Enum):
    """Plan item execution status."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class TodoItem(BaseModel):
    """Plan execution item."""

    id: str = Field(..., description="Plan item ID")
    content: str = Field(..., description="Plan item content")
    status: TodoStatus = Field(..., description="Execution status")


class BaseEvent(BaseModel):
    """Base event structure."""

    id: str = Field(..., description="Unique event ID")
    planId: Optional[str] = Field(None, description="Plan ID if applicable")
    timestamp: int = Field(..., description="Event timestamp in milliseconds")
    event: DeepResearchEventType = Field(..., description="Event type")


class ChatEvent(BaseEvent):
    """Chat message event."""

    event: DeepResearchEventType = DeepResearchEventType.CHAT
    content: str = Field(..., description="Chat content (markdown/text/sql/html)")


class PlanUpdateEvent(BaseEvent):
    """Plan update event."""

    event: DeepResearchEventType = DeepResearchEventType.PLAN_UPDATE
    todos: List[TodoItem] = Field(..., description="List of plan items")


class ToolCallEvent(BaseEvent):
    """Tool call event."""

    event: DeepResearchEventType = DeepResearchEventType.TOOL_CALL
    toolCallId: str = Field(..., description="Tool call ID")
    toolName: str = Field(..., description="Tool name")
    input: Dict[str, Any] = Field(..., description="Tool input parameters")


class ToolCallResultEvent(BaseEvent):
    """Tool call result event."""

    event: DeepResearchEventType = DeepResearchEventType.TOOL_CALL_RESULT
    toolCallId: str = Field(..., description="Corresponding tool call ID")
    data: Any = Field(..., description="Tool execution result")
    error: bool = Field(..., description="Whether execution failed")


class CompleteEvent(BaseEvent):
    """Task completion event."""

    event: DeepResearchEventType = DeepResearchEventType.COMPLETE
    content: Optional[str] = Field(None, description="Final result content")


class ReportEvent(BaseEvent):
    """Report generation event."""

    event: DeepResearchEventType = DeepResearchEventType.REPORT
    url: Optional[str] = Field("", description="Report URL (optional)")
    data: Optional[str] = Field("", description="Report HTML content (optional)")


class ErrorEvent(BaseEvent):
    """Error event."""

    event: DeepResearchEventType = DeepResearchEventType.ERROR
    error: str = Field(..., description="Error message")


class SqlExecutionStartEvent(BaseEvent):
    """SQL execution start event."""

    event: DeepResearchEventType = DeepResearchEventType.SQL_EXECUTION_START
    sqlQuery: str = Field(..., description="SQL query being executed")
    databaseName: Optional[str] = Field(None, description="Target database name")
    estimatedRows: Optional[int] = Field(None, description="Estimated number of rows (if available)")


class SqlExecutionProgressEvent(BaseEvent):
    """SQL execution progress event."""

    event: DeepResearchEventType = DeepResearchEventType.SQL_EXECUTION_PROGRESS
    sqlQuery: str = Field(..., description="SQL query being executed")
    progress: float = Field(..., description="Execution progress (0.0 to 1.0)")
    currentStep: str = Field(..., description="Current execution step description")
    elapsedTime: Optional[int] = Field(None, description="Elapsed time in milliseconds")


class SqlExecutionResultEvent(BaseEvent):
    """SQL execution result event."""

    event: DeepResearchEventType = DeepResearchEventType.SQL_EXECUTION_RESULT
    sqlQuery: str = Field(..., description="Executed SQL query")
    rowCount: int = Field(..., description="Number of rows returned/affected")
    executionTime: int = Field(..., description="Execution time in milliseconds")
    data: Optional[Any] = Field(None, description="Query result data (first N rows or summary)")
    hasMoreData: bool = Field(False, description="Whether there are more results available")
    dataPreview: Optional[str] = Field(None, description="Text preview of results")


class SqlExecutionErrorEvent(BaseEvent):
    """SQL execution error event."""

    event: DeepResearchEventType = DeepResearchEventType.SQL_EXECUTION_ERROR
    sqlQuery: str = Field(..., description="Failed SQL query")
    error: str = Field(..., description="Error message")
    errorType: str = Field(..., description="Error type (syntax, permission, timeout, etc.)")
    suggestions: List[str] = Field(default_factory=list, description="Suggested fixes or recovery actions")
    canRetry: bool = Field(False, description="Whether the query can be retried")


# Union type for all events

DeepResearchEvent = Union[
    ChatEvent,
    PlanUpdateEvent,
    ToolCallEvent,
    ToolCallResultEvent,
    SqlExecutionStartEvent,
    SqlExecutionProgressEvent,
    SqlExecutionResultEvent,
    SqlExecutionErrorEvent,
    CompleteEvent,
    ReportEvent,
    ErrorEvent,
]
