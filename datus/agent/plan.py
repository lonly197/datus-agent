# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import os
from pathlib import Path
from typing import List, Optional

import yaml
from agents import Tool

from datus.agent.node import Node
from datus.agent.node.intent_analysis_node import IntentAnalysisNode
from datus.agent.node.schema_discovery_node import SchemaDiscoveryNode
from datus.agent.workflow import Workflow
from datus.configuration.agent_config import AgentConfig
from datus.configuration.node_type import NodeType
from datus.schemas.node_models import SqlTask
from datus.schemas.schema_linking_node_models import SchemaLinkingInput
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

# Node type alias mapping for config normalization
NODE_TYPE_ALIASES = {
    # Reasoning aliases
    "reason_sql": NodeType.TYPE_REASONING,
    "reasoning_sql": NodeType.TYPE_REASONING,
    "reason": NodeType.TYPE_REASONING,
    # Reflection aliases
    "reflection": NodeType.TYPE_REFLECT,
    "reflect": NodeType.TYPE_REFLECT,
    # Execution aliases
    "execute": NodeType.TYPE_EXECUTE_SQL,
    # Chat aliases
    "chat": NodeType.TYPE_CHAT,
    "chat_agentic": NodeType.TYPE_CHAT,
    # SQL generation aliases
    "chatbot": NodeType.TYPE_GENSQL,  # Deprecated, use sql_chatbot
    "sql_chatbot": NodeType.TYPE_GENSQL,
    "sql_generation": NodeType.TYPE_GENSQL,
    # Syntax/execution preview aliases
    "syntax_validation": NodeType.TYPE_CHAT,
    "execution_preview": NodeType.TYPE_CHAT,
}


def load_builtin_workflow_config() -> dict:
    current_dir = Path(__file__).parent
    config_path = current_dir / "workflow.yml"

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Workflow configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Validate workflow configuration
    validation_errors = validate_workflow_config(config)
    if validation_errors:
        error_msg = "Workflow configuration validation failed:\n" + "\n".join(
            f"  - {error}" for error in validation_errors
        )
        raise ValueError(error_msg)

    logger.debug(f"Workflow configuration loaded and validated: {config_path}")

    return config


def validate_workflow_config(workflow_config: dict) -> List[str]:
    """
    Validate workflow configuration for correctness.

    Args:
        workflow_config: Workflow configuration dictionary

    Returns:
        List of validation error messages
    """
    errors = []

    if not isinstance(workflow_config, dict):
        errors.append("Workflow config must be a dictionary")
        return errors

    if "workflow" not in workflow_config:
        errors.append("Missing 'workflow' section in configuration")
        return errors

    workflows = workflow_config["workflow"]
    if not isinstance(workflows, dict):
        errors.append("'workflow' section must be a dictionary")
        return errors

    # Validate all workflows with specific validators
    for workflow_name, workflow_nodes in workflows.items():
        if workflow_name == "text2sql":
            # Text2SQL workflow with comprehensive validation
            text2sql_errors = _validate_text2sql_workflow(workflow_nodes)
            errors.extend(text2sql_errors)
        elif workflow_name in ["reflection", "fixed", "dynamic"]:
            # Standard SQL workflows require core nodes
            node_errors = _validate_standard_sql_workflow(workflow_name, workflow_nodes)
            errors.extend(node_errors)
        elif workflow_name == "metric_to_sql":
            # Metric-based workflow requires specific nodes
            node_errors = _validate_metric_to_sql_workflow(workflow_nodes)
            errors.extend(node_errors)
        elif workflow_name in ["chat_agentic", "chat_agentic_plan", "gensql_agentic"]:
            # Agentic workflows require chat/gensql node and output
            node_errors = _validate_agentic_workflow(workflow_name, workflow_nodes)
            errors.extend(node_errors)
        else:
            # Generic validation for unknown workflow types
            node_errors = _validate_workflow_nodes(workflow_name, workflow_nodes)
            errors.extend(node_errors)

    return errors


def _validate_text2sql_workflow(workflow_nodes: List[str]) -> List[str]:
    """Validate text2sql workflow configuration."""
    errors = []

    if not isinstance(workflow_nodes, list):
        errors.append("text2sql workflow must be a list of node names")
        return errors

    required_nodes = ["intent_analysis", "schema_discovery", "generate_sql", "execute_sql", "output"]
    for required_node in required_nodes:
        if required_node not in workflow_nodes:
            errors.append(f"text2sql workflow missing required node: {required_node}")

    # Validate node order (intent_analysis and schema_discovery should come first)
    if "intent_analysis" in workflow_nodes and "schema_discovery" in workflow_nodes:
        intent_idx = workflow_nodes.index("intent_analysis")
        schema_idx = workflow_nodes.index("schema_discovery")
        if intent_idx > schema_idx:
            errors.append("intent_analysis should come before schema_discovery in text2sql workflow")

    return errors


def _validate_workflow_nodes(workflow_name: str, workflow_nodes: List[str]) -> List[str]:
    """Validate workflow node definitions."""
    errors = []

    if not isinstance(workflow_nodes, list):
        errors.append(f"Workflow '{workflow_name}' must be a list of node names")
        return errors

    # Basic validation - ensure all nodes are strings
    for node in workflow_nodes:
        if not isinstance(node, str):
            errors.append(f"Workflow '{workflow_name}': node '{node}' must be a string")

    return errors


def _validate_standard_sql_workflow(workflow_name: str, workflow_nodes: List[str]) -> List[str]:
    """Validate standard SQL workflow has required nodes.

    Standard workflows include: reflection, fixed, dynamic
    These workflows should have core SQL generation nodes.
    """
    errors = []

    if not isinstance(workflow_nodes, list):
        errors.append(f"{workflow_name} workflow must be a list of node names")
        return errors

    # Required nodes for standard SQL workflows
    required_nodes = ["schema_linking", "generate_sql", "execute_sql", "output"]
    for required_node in required_nodes:
        if required_node not in workflow_nodes:
            errors.append(
                f"{workflow_name} workflow missing required node: {required_node}"
            )

    return errors


def _validate_metric_to_sql_workflow(workflow_nodes: List[str]) -> List[str]:
    """Validate metric_to_sql workflow has required nodes."""
    errors = []

    if not isinstance(workflow_nodes, list):
        errors.append("metric_to_sql workflow must be a list of node names")
        return errors

    # Metric workflow requires specific nodes for metric handling
    required_nodes = ["schema_linking", "search_metrics", "generate_sql", "execute_sql", "output"]
    for required_node in required_nodes:
        if required_node not in workflow_nodes:
            errors.append(
                f"metric_to_sql workflow missing required node: {required_node}"
            )

    return errors


def _validate_agentic_workflow(workflow_name: str, workflow_nodes: List[str]) -> List[str]:
    """Validate agentic workflow has required nodes.

    Agentic workflows include: chat_agentic, chat_agentic_plan, gensql_agentic
    These workflows focus on conversational AI interactions.
    """
    errors = []

    if not isinstance(workflow_nodes, list):
        errors.append(f"{workflow_name} workflow must be a list of node names")
        return errors

    # All agentic workflows require output node
    if "output" not in workflow_nodes:
        errors.append(
            f"{workflow_name} workflow missing required node: output"
        )

    # Agentic workflows should have a chat/gensql node
    chat_patterns = ["chat_agentic", "sql_chatbot", "chat"]
    has_chat_node = any(
        any(pattern in node for pattern in chat_patterns)
        for node in workflow_nodes
    )

    if not has_chat_node:
        errors.append(
            f"{workflow_name} workflow should have a chat or sql_chatbot node"
        )

    # Plan workflows should not have execute_sql (plan-only mode)
    if workflow_name.endswith("_plan"):
        if "execute_sql" in workflow_nodes:
            errors.append(
                f"{workflow_name} workflow should not contain execute_sql node (plan-only mode)"
            )

    return errors


def create_nodes_from_config(
    workflow_config: list,
    sql_task: SqlTask,
    agent_config: Optional[AgentConfig] = None,
    tools: Optional[List[Tool]] = None,
) -> List[Node]:
    nodes = []

    start_node = Node.new_instance(
        node_id="node_0",
        description=NodeType.get_description(NodeType.TYPE_BEGIN),
        node_type=NodeType.TYPE_BEGIN,
        input_data=sql_task,
        agent_config=agent_config,
        tools=tools,
    )
    nodes.append(start_node)

    # Process workflow config that may contain nested structures
    processed_nodes = _process_workflow_config(workflow_config, sql_task, agent_config, tools=tools)
    nodes.extend(processed_nodes)

    logger.info(f"Generated workflow with {len(nodes)} nodes")

    return nodes


def _process_workflow_config(
    config: list,
    sql_task: SqlTask,
    agent_config: Optional[AgentConfig] = None,
    start_index: int = 1,
    node_id_prefix: str = "node",
    tools: Optional[List[Tool]] = None,
) -> List[Node]:
    """Process workflow configuration that may contain nested parallel structures"""
    nodes = []
    current_index = start_index

    for item in config:
        if isinstance(item, str):
            # Simple node type
            node_id = f"{node_id_prefix}_{current_index}"
            node = _create_single_node(item, node_id, sql_task, agent_config, tools)
            nodes.append(node)
            current_index += 1

        elif isinstance(item, dict):
            # Handle nested structures like parallel
            for key, value in item.items():
                if key == "parallel" and isinstance(value, list):
                    # Create a parallel node
                    parallel_children = value
                    node_id = f"{node_id_prefix}_{current_index}"

                    from datus.schemas.parallel_node_models import ParallelInput

                    parallel_input = ParallelInput(
                        child_nodes=parallel_children, shared_input=None  # Will be set up during execution
                    )

                    parallel_node = Node.new_instance(
                        node_id=node_id,
                        description=NodeType.get_description(NodeType.TYPE_PARALLEL),
                        node_type=NodeType.TYPE_PARALLEL,
                        input_data=parallel_input,
                        agent_config=agent_config,
                        tools=tools,
                    )
                    nodes.append(parallel_node)
                    current_index += 1

                elif key == "selection":
                    # Create a selection node (if it's specified as dict with criteria)
                    node_id = f"{node_id_prefix}_{current_index}"

                    from datus.schemas.parallel_node_models import SelectionInput

                    selection_criteria = value if isinstance(value, str) else "best_quality"
                    selection_input = SelectionInput(
                        candidate_results={},  # Will be populated during execution
                        selection_criteria=selection_criteria,
                    )

                    selection_node = Node.new_instance(
                        node_id=node_id,
                        description=NodeType.get_description(NodeType.TYPE_SELECTION),
                        node_type=NodeType.TYPE_SELECTION,
                        input_data=selection_input,
                        agent_config=agent_config,
                    )
                    nodes.append(selection_node)
                    current_index += 1
                else:
                    # Handle other dict-based configurations if needed
                    logger.warning(f"Unknown configuration item: {key}")

        elif item == "selection":
            # Simple selection node
            node_id = f"{node_id_prefix}_{current_index}"

            from datus.schemas.parallel_node_models import SelectionInput

            selection_input = SelectionInput(
                candidate_results={}, selection_criteria="best_quality"  # Will be populated during execution
            )

            selection_node = Node.new_instance(
                node_id=node_id,
                description=NodeType.get_description(NodeType.TYPE_SELECTION),
                node_type=NodeType.TYPE_SELECTION,
                input_data=selection_input,
                agent_config=agent_config,
            )
            nodes.append(selection_node)
            current_index += 1

    return nodes


def _create_single_node(
    node_type: str,
    node_id: str,
    sql_task: SqlTask,
    agent_config: Optional[AgentConfig] = None,
    tools: Optional[List[Tool]] = None,
) -> Node:
    # Normalize aliases from config using dictionary lookup
    normalized_type = NODE_TYPE_ALIASES.get(node_type, node_type)

    # Check if node_type is defined in agentic_nodes config - if so, map to gensql
    if (
        agent_config
        and hasattr(agent_config, "agentic_nodes")
        and node_type in agent_config.agentic_nodes
    ):
        normalized_type = NodeType.TYPE_GENSQL

    description = NodeType.get_description(normalized_type)

    input_data = None
    if normalized_type == NodeType.TYPE_SCHEMA_LINKING:
        input_data = SchemaLinkingInput.from_sql_task(
            sql_task=sql_task,
            matching_rate=agent_config.schema_linking_rate if agent_config else "fast",
        )

    # Use standard Node.new_instance for all node types
    node = Node.new_instance(
        node_id=node_id,
        description=description,
        node_type=normalized_type,
        input_data=input_data,
        agent_config=agent_config,
        tools=tools,
        node_name=(
            node_type if normalized_type == NodeType.TYPE_GENSQL else None
        ),  # Pass original name for gensql nodes
    )

    return node


def generate_workflow(
    task: SqlTask,
    plan_type: str = "reflection",
    agent_config: Optional[AgentConfig] = None,
) -> Workflow:
    logger.info(f"Generating workflow for task based on plan type '{plan_type}': {task}")

    if not plan_type and agent_config:
        plan_type = agent_config.workflow_plan
    elif not plan_type:
        plan_type = "reflection"  # fallback to default

    if agent_config and plan_type in agent_config.custom_workflows:
        logger.info(f"Using custom workflow '{plan_type}' from configuration")
        selected_workflow = agent_config.custom_workflows[plan_type]
    else:
        # Check builtin workflows
        config = load_builtin_workflow_config()
        workflows = config.get("workflow", {})

        if plan_type not in workflows:
            if agent_config and agent_config.custom_workflows:
                available_custom = list(agent_config.custom_workflows.keys())
                available_builtin = list(workflows.keys())
                raise ValueError(
                    f"Invalid plan type '{plan_type}'. "
                    f"Available builtin workflows: {available_builtin}, "
                    f"custom workflows: {available_custom}"
                )
            else:
                available_builtin = list(workflows.keys())
                raise ValueError(f"Invalid plan type '{plan_type}'. Available builtin workflows: {available_builtin}")

        selected_workflow = workflows[plan_type]

    # support { steps: [...] } structure for custom workflows
    workflow_steps = selected_workflow
    workflow_config = None
    if isinstance(selected_workflow, dict):
        if "steps" in selected_workflow:
            workflow_steps = selected_workflow["steps"]
            # Extract config if available
            if "config" in selected_workflow:
                workflow_config = selected_workflow["config"]

    workflow = Workflow(
        name=f"SQL Query Workflow ({plan_type})",
        task=task,
        agent_config=agent_config,
    )

    # Store workflow config in the workflow object if available
    if workflow_config:
        workflow.workflow_config = workflow_config

    nodes = create_nodes_from_config(workflow_steps, task, agent_config, workflow.tools)

    for node in nodes:
        workflow.add_node(node)
    if task.tables and agent_config is not None:
        from datus.storage.schema_metadata import SchemaWithValueRAG

        try:
            rag = SchemaWithValueRAG(agent_config=agent_config)
            schemas, values = rag.search_tables(
                task.tables, task.catalog_name, task.database_name, task.schema_name, dialect=task.database_type
            )
            if len(schemas) != len(task.tables):
                schema_table_names = [item.table_name for item in schemas]
                logger.warning(
                    f"The obtained table schema is: {schema_table_names}; "
                    f"The table required for the task is: {schemas}"
                )
            logger.debug(f"Use task tables: {schemas}")
            workflow.context.update_schema_and_values(schemas, values)
        except Exception as e:
            logger.warning(f"Failed to obtain the schema corresponding to {task.tables}: {e}")

    logger.info(f"Generated workflow with {len(nodes)} nodes")
    return workflow
