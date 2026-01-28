# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details

"""
External knowledge enhancement for schema discovery.

This module provides functionality to enhance schema discovery with
external domain knowledge and LLM-based schema matching.
"""

from typing import List, Optional
from datus.tools.func_tool.context_search import ContextSearchTools
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


async def search_and_enhance_external_knowledge(
    user_query: str,
    subject_path: Optional[List[str]],
    agent_config,
    workflow,
) -> str:
    """
    Search for relevant external knowledge based on user query.

    This method enhances the query context with domain knowledge before schema discovery.

    Args:
        user_query: The user's natural language query
        subject_path: Optional subject hierarchy path (e.g., ['Finance', 'Revenue', 'Q1'])
        agent_config: Agent configuration
        workflow: Workflow instance

    Returns:
        Formatted string of relevant knowledge entries
    """
    try:
        if not agent_config:
            return ""

        # Check if external knowledge enhancement is enabled (with backward compatibility)
        if not hasattr(agent_config, "schema_discovery_config"):
            return ""

        if not agent_config.schema_discovery_config.external_knowledge_enabled:
            logger.debug("External knowledge enhancement is disabled")
            return ""

        context_search_tools = ContextSearchTools(agent_config)

        # Check if external knowledge store is available
        if not context_search_tools.has_ext_knowledge:
            logger.debug("External knowledge store is empty, skipping search")
            return ""

        # Convert subject_path to domain/layer1/layer2 format
        domain = subject_path[0] if len(subject_path) > 0 else ""
        layer1 = subject_path[1] if len(subject_path) > 1 else ""
        layer2 = subject_path[2] if len(subject_path) > 2 else ""

        # Get top_n from configuration (with backward compatibility)
        if hasattr(agent_config, "schema_discovery_config"):
            top_n = agent_config.schema_discovery_config.external_knowledge_top_n
        else:
            top_n = 5  # Default value for backward compatibility

        search_result = context_search_tools.search_external_knowledge(
            query_text=user_query,
            domain=domain,
            layer1=layer1,
            layer2=layer2,
            top_n=top_n,
        )

        if search_result.success and search_result.result:
            knowledge_items = []
            for result in search_result.result:
                terminology = result.get("terminology", "")
                explanation = result.get("explanation", "")
                if terminology and explanation:
                    knowledge_items.append(f"- {terminology}: {explanation}")

            if knowledge_items:
                formatted_knowledge = "\n".join(knowledge_items)
                logger.info(f"Found {len(knowledge_items)} relevant external knowledge entries")
                return formatted_knowledge

        return ""

    except Exception as e:
        logger.warning(f"Failed to search external knowledge: {str(e)}")
        return ""


def combine_knowledge(original: str, enhanced: str) -> str:
    """
    Combine original knowledge and searched knowledge.

    Args:
        original: Original knowledge text
        enhanced: Enhanced knowledge from external search

    Returns:
        Combined knowledge string
    """
    parts = []
    if original:
        parts.append(original)
    if enhanced:
        parts.append(f"Relevant Business Knowledge:\n{enhanced}")
    return "\n\n".join(parts)


async def llm_based_schema_matching(
    query: str,
    matching_rate: str,
    agent_config,
    workflow,
    model,
) -> List[str]:
    """
    Use LLM-based schema matching for large datasets.

    This method leverages MatchSchemaTool for intelligent table selection
    when dealing with databases containing many tables.

    Args:
        query: User query text
        matching_rate: Matching rate strategy (should be "from_llm")
        agent_config: Agent configuration
        workflow: Workflow instance
        model: LLM model instance

    Returns:
        List of selected table names
    """
    if not agent_config or matching_rate != "from_llm":
        return []

    # Check if LLM matching is enabled (with backward compatibility)
    if not hasattr(agent_config, "schema_discovery_config"):
        return []

    if not agent_config.schema_discovery_config.llm_matching_enabled:
        logger.debug("LLM-based schema matching is disabled")
        return []

    try:
        from datus.schemas.schema_linking_node_models import SchemaLinkingInput
        from datus.tools.lineage_graph_tools.schema_lineage import SchemaLineageTool

        # Get task from workflow
        task = workflow.task if workflow else None
        if not task:
            return []

        # Create SchemaLinkingInput for MatchSchemaTool
        linking_input = SchemaLinkingInput(
            input_text=query,
            database_type=getattr(task, "database_type", "sqlite"),
            catalog_name=getattr(task, "catalog_name", ""),
            database_name=getattr(task, "database_name", ""),
            schema_name=getattr(task, "schema_name", ""),
            matching_rate=matching_rate,
            table_type=getattr(task, "schema_linking_type", "table"),
        )

        # Use SchemaLineageTool with MatchSchemaTool
        tool = SchemaLineageTool(agent_config=agent_config)
        result = tool.execute(linking_input, model)

        if result.success and result.table_schemas:
            table_names = [schema.table_name for schema in result.table_schemas]
            logger.info(f"LLM-based matching found {len(table_names)} tables")
            return table_names

        return []

    except Exception as e:
        logger.warning(f"LLM-based schema matching failed: {e}")
        return []
