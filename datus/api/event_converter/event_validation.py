# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Event validation and helper utilities for event conversion.

This module provides functions to validate event flows, extract IDs from actions,
and perform helper operations for event generation.
"""

import json
import re
import uuid
from typing import Any, Dict, List, Optional

from datus.schemas.action_history import ActionHistory
from datus.utils.loggings import get_logger

from .models import DeepResearchEvent


def extract_plan_from_output(output: Any) -> Dict[str, Any]:
    """Extract plan information from action output.

    Args:
        output: Action output (dict, string, or other)

    Returns:
        Extracted plan information dict
    """
    if isinstance(output, dict):
        return output

    if isinstance(output, str):
        # Try parsing as JSON
        try:
            return json.loads(output)
        except json.JSONDecodeError:
            pass

        # Try extracting JSON from markdown code block
        match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', output, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

    return {}


def extract_callid_from_output(output: Any) -> Optional[str]:
    """Extract call ID from action output.

    Args:
        output: Action output (dict, string, or other)

    Returns:
        Call ID string or None
    """
    if isinstance(output, dict):
        return output.get("call_id") or output.get("callid")

    if isinstance(output, str):
        # Try parsing as JSON
        try:
            data = json.loads(output)
            return data.get("call_id") or data.get("callid")
        except json.JSONDecodeError:
            pass

        # Try extracting JSON from markdown code block
        match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', output, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1))
                return data.get("call_id") or data.get("callid")
            except json.JSONDecodeError:
                pass

    return None


def try_parse_json_like(obj: Any) -> Optional[Dict[str, Any]]:
    """Try to parse object as JSON-like dict.

    Args:
        obj: Object to parse (dict, string, or other)

    Returns:
        Parsed dict or None
    """
    if isinstance(obj, dict):
        return obj

    if isinstance(obj, str):
        obj = obj.strip()
        if not obj:
            return None

        # Try direct JSON parse
        try:
            return json.loads(obj)
        except json.JSONDecodeError:
            pass

        # Try extracting JSON from markdown code block
        match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', obj, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Try extracting JSON object with regex
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', obj, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

    return None


def extract_todo_id_from_action(action: ActionHistory) -> Optional[str]:
    """Extract todo ID from action metadata.

    Args:
        action: ActionHistory object

    Returns:
        Todo ID string or None
    """
    # Check metadata first
    if action.metadata and isinstance(action.metadata, dict):
        todo_id = action.metadata.get("todo_id")
        if todo_id:
            return str(todo_id)

    # Check input for todo_id field
    if action.input and isinstance(action.input, dict):
        todo_id = action.input.get("todo_id")
        if todo_id:
            return str(todo_id)

    # Check output for todo_id field
    if action.output and isinstance(action.output, dict):
        todo_id = action.output.get("todo_id")
        if todo_id:
            return str(todo_id)

    return None


def get_unified_plan_id(
    action: ActionHistory,
    force_associate: bool = False,
    virtual_plan_id: Optional[str] = None
) -> Optional[str]:
    """Get unified plan ID for action events.

    Args:
        action: ActionHistory object
        force_associate: Force association with virtual plan
        virtual_plan_id: Virtual plan ID to use if available

    Returns:
        Plan ID string or None
    """
    # First, try to extract from action metadata
    if action.metadata and isinstance(action.metadata, dict):
        plan_id = action.metadata.get("plan_id")
        if plan_id:
            return str(plan_id)

    # Check input for plan_id
    if action.input and isinstance(action.input, dict):
        plan_id = action.input.get("plan_id")
        if plan_id:
            return str(plan_id)

    # Check output for plan_id
    if action.output and isinstance(action.output, dict):
        plan_id = action.output.get("plan_id")
        if plan_id:
            return str(plan_id)

    # Force associate with virtual plan for certain action types
    if force_associate and virtual_plan_id:
        return virtual_plan_id

    return None


def find_tool_call_id(action: ActionHistory, tool_call_map: Dict[str, str]) -> Optional[str]:
    """Find tool call ID for action.

    Args:
        action: ActionHistory object
        tool_call_map: Mapping of action_id to tool_call_id

    Returns:
        Tool call ID string or None
    """
    # Check if we already have a mapping
    if action.action_id in tool_call_map:
        return tool_call_map[action.action_id]

    # Try to extract from action metadata
    if action.metadata and isinstance(action.metadata, dict):
        call_id = action.metadata.get("tool_call_id")
        if call_id:
            return str(call_id)

    # Try to extract from input
    if action.input and isinstance(action.input, dict):
        call_id = action.input.get("tool_call_id")
        if call_id:
            return str(call_id)

    return None


def is_internal_todo_update(action: ActionHistory) -> bool:
    """Check if action is an internal todo update.

    Args:
        action: ActionHistory object

    Returns:
        True if internal todo update, False otherwise
    """
    # Check action type
    if action.action_type in ["todo_update", "update_todo"]:
        return True

    # Check metadata
    if action.metadata and isinstance(action.metadata, dict):
        if action.metadata.get("internal_todo"):
            return True

    return False


def extract_node_type_from_action(action: ActionHistory) -> Optional[str]:
    """Extract node type from action.

    Args:
        action: ActionHistory object

    Returns:
        Node type string or None
    """
    # Direct action_type
    if action.action_type:
        return action.action_type

    # Check metadata
    if action.metadata and isinstance(action.metadata, dict):
        node_type = action.metadata.get("node_type")
        if node_type:
            return str(node_type)

    # Check input
    if action.input and isinstance(action.input, dict):
        node_type = action.input.get("node_type")
        if node_type:
            return str(node_type)

    return None


def validate_event_flow(
    action_type: str,
    events: List[DeepResearchEvent],
    logger=None
) -> bool:
    """Validate event flow for critical actions.

    Args:
        action_type: Type of action being validated
        events: List of generated events
        logger: Logger instance

    Returns:
        True if validation passes, False otherwise
    """
    if not logger:
        logger = get_logger(__name__)

    # Skip validation for non-critical actions
    critical_actions = [
        "generate_sql",
        "execute_sql",
        "schema_discovery",
        "output_generation"
    ]
    if action_type not in critical_actions:
        return True

    # Check that at least one event was generated
    if not events:
        logger.warning(f"No events generated for critical action: {action_type}")
        return False

    # Check that all events have required fields
    for event in events:
        if not event.id:
            logger.warning(f"Event missing ID for action: {action_type}")
            return False

        if not event.timestamp:
            logger.warning(f"Event missing timestamp for action: {action_type}")
            return False

    return True


def generate_event_id(prefix: str = "event") -> str:
    """Generate unique event ID.

    Args:
        prefix: ID prefix

    Returns:
        Unique event ID string
    """
    return f"{prefix}_{uuid.uuid4().hex[:16]}"
