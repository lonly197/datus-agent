# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details

"""
Sub-agent Knowledge Base Refresher.

Handles refreshing scoped knowledge bases for sub-agents
after global bootstrap operations.
"""

from typing import Optional

from datus.configuration.agent_config import AgentConfig
from datus.schemas.agent_models import SubAgentConfig
from datus.storage.sub_agent_kb_bootstrap import SUPPORTED_COMPONENTS as SUB_AGENT_COMPONENTS
from datus.storage.sub_agent_kb_bootstrap import SubAgentBootstrapper
from datus.utils.constants import SYS_SUB_AGENTS
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


def refresh_scoped_agents(
    agent_config: AgentConfig,
    component: str,
    kb_strategy: str,
) -> None:
    """
    Rebuild scoped knowledge bases for sub-agents after global bootstrap.

    Args:
        agent_config: Agent configuration
        component: The component that was bootstrapped (e.g., "metadata", "metrics")
        kb_strategy: Knowledge base update strategy ("overwrite" or "incremental")
    """
    if component not in SUB_AGENT_COMPONENTS:
        return
    if kb_strategy not in {"overwrite", "incremental"}:
        return

    agent_nodes = getattr(agent_config, "agentic_nodes", {}) or {}
    if not agent_nodes:
        return
    current_namespace = agent_config.current_namespace

    for name, raw_config in agent_nodes.items():
        if name in SYS_SUB_AGENTS:
            continue

        from pydantic import ValidationError

        try:
            sub_config = SubAgentConfig.model_validate(raw_config)
        except ValidationError as exc:
            logger.warning(f"Skipping sub-agent '{name}' due to invalid configuration: {exc}")
            continue

        if not sub_config.is_in_namespace(current_namespace):
            logger.debug(
                f"Skipping sub-agent '{name}' for component '{component}' "
                f"because there is no corresponding scope context configured under namespace {current_namespace}"
            )
            continue

        try:
            bootstrapper = SubAgentBootstrapper(
                sub_agent=sub_config,
                agent_config=agent_config,
            )
            logger.info(
                f"Running SubAgentBootstrapper for sub-agent '{name}' (component={component}, "
                f"strategy=overwrite, storage={bootstrapper.storage_path})"
            )
            result = bootstrapper.run([component], "overwrite")
            if not result.should_bootstrap:
                reason = result.reason or "No scoped context provided"
                logger.info(f"SubAgentBootstrapper skipped for sub-agent '{name}': {reason}")
            else:
                component_summaries = []
                for comp_result in result.results:
                    summary = f"{comp_result.component}:{comp_result.status}"
                    if comp_result.message:
                        summary = f"{summary} ({comp_result.message})"
                    component_summaries.append(summary)
                component_summaries_str = (
                    ", ".join(component_summaries) if component_summaries else "no component results"
                )
                logger.info(
                    f"Bootstrap finished for sub-agent '{name}' (storage={result.storage_path}): "
                    f"{component_summaries_str}"
                )
        except Exception as exc:
            logger.warning(f"Failed to refresh scoped KB for sub-agent '{name}': {exc}")
