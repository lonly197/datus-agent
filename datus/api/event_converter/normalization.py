# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Normalization utilities for event conversion.

This module provides functions to normalize various identifiers and statuses
to ensure consistent event generation across different workflow types.
"""

import hashlib
from typing import Optional


def normalize_node_type(node_type: Optional[str]) -> str:
    """Normalize action/node type for virtual step matching.

    - Strips common prefixes added by execution wrappers (tool_, llm_, workflow_, execution_)
    - Strips common suffixes used for result/error markers (_result, _error, _failed, _success)

    Args:
        node_type: The raw node type from action

    Returns:
        Normalized node type string
    """
    if not node_type:
        return ""

    normalized = str(node_type)

    # Remove known prefixes
    for prefix in ("tool_", "llm_", "workflow_", "execution_"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]
            break

    # Remove known suffixes
    for suffix in ("_result", "_error", "_failed", "_success"):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
            break

    return normalized


def normalize_tool_name(action_type: str) -> str:
    """Normalize tool name for ToolCallEvent/toolCallResultEvent.

    Args:
        action_type: The raw action type

    Returns:
        Normalized tool name
    """
    normalized = normalize_node_type(action_type)
    if normalized.startswith("preflight_"):
        normalized = normalized.replace("preflight_", "", 1)
    return normalized


def normalize_todo_status(status: Optional[str]) -> str:
    """Normalize TodoStatus strings across legacy and frontend enums.

    Args:
        status: The raw status string

    Returns:
        Normalized status string
    """
    if not status:
        return "pending"
    normalized = str(status).lower()
    if normalized == "failed":
        return "error"
    return normalized


def hash_text(s: str) -> str:
    """Generate hash for text deduplication.

    Args:
        s: Text string to hash

    Returns:
        Hexadecimal hash string
    """
    try:
        return hashlib.sha1(s.strip().encode("utf-8")).hexdigest()
    except Exception:
        return ""
