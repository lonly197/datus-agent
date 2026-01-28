# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details

"""
Table discovery engine orchestrator.

This module provides the main orchestration logic for discovering candidate tables,
coordinating between different search strategies.
"""

import time
from typing import Any, Dict, List, Tuple

from datus.utils.chinese_query_utils import contains_chinese
from datus.utils.loggings import get_logger
from datus.utils.schema_discovery_metrics import (
    ExternalKnowledgeMetrics,
    SchemaDiscoveryMetrics,
    SearchStage,
)

from .llm_discovery import llm_based_table_discovery
from .search_strategies import (
    context_based_discovery,
    keyword_table_discovery,
    semantic_table_discovery,
)

logger = get_logger(__name__)


async def discover_candidate_tables(
    task,
    intent: str,
    workflow,
    agent_config,
    metrics: SchemaDiscoveryMetrics,
    llm_call_func,
    llm_schema_matching_func,
    search_enhance_func,
    combine_knowledge_func,
    apply_progressive_func,
    extract_tables_func,
    rewrite_fts_func,
    rerank_check_func,
) -> Tuple[List[str], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Discover candidate tables based on task and intent.

    Enhanced v2.6: Added comprehensive metrics tracking for all search stages.

    Args:
        task: The SQL task
        intent: Detected intent type
        workflow: Workflow instance
        agent_config: Agent configuration
        metrics: Metrics collector
        llm_call_func: LLM call function with retry
        llm_schema_matching_func: LLM schema matching function
        search_enhance_func: External knowledge search function
        combine_knowledge_func: Knowledge combination function
        apply_progressive_func: Progressive matching function
        extract_tables_func: Table extraction from text function
        rewrite_fts_func: FTS query rewrite function
        rerank_check_func: Rerank resource check function

    Returns:
        Tuple of (candidate table names, candidate details, discovery statistics)
    """
    candidate_details: Dict[str, Dict[str, Any]] = {}
    candidate_order: List[str] = []
    discovery_stats: Dict[str, Any] = {}

    def record_candidate(
        table_name: str,
        source: str,
        score: Optional[float] = None,
        matched_terms: Optional[List[str]] = None,
    ) -> None:
        """Record a candidate table with its discovery source and metadata."""
        if table_name not in candidate_details:
            candidate_details[table_name] = {
                "table_name": table_name,
                "sources": [],
                "scores": {},
                "matched_terms": [],
                "order_index": len(candidate_order),
            }
            candidate_order.append(table_name)
        entry = candidate_details[table_name]
        if source not in entry["sources"]:
            entry["sources"].append(source)
        if score is not None:
            entry["scores"][source] = score
        if matched_terms:
            for term in matched_terms:
                if term not in entry["matched_terms"]:
                    entry["matched_terms"].append(term)

    # Stage 1: Explicit tables (Highest Priority)
    metrics.start_stage(SearchStage.EXPLICIT)
    if hasattr(task, "tables") and task.tables:
        for table_name in task.tables:
            record_candidate(table_name, "explicit")
        logger.info(f"Using explicitly specified tables: {task.tables}")
        discovery_stats["explicit_tables"] = len(task.tables)
    metrics.end_stage(
        SearchStage.EXPLICIT,
        success=bool(hasattr(task, "tables") and task.tables),
        tables_found=len(task.tables) if hasattr(task, "tables") and task.tables else 0,
        tables_returned=len(task.tables) if hasattr(task, "tables") and task.tables else 0,
    )

    # If intent is text2sql or sql, try to discover tables
    if intent in ["text2sql", "sql"] and hasattr(task, "task"):
        task_text = task.task.lower()

        # Apply progressive matching based on reflection round
        if agent_config and hasattr(agent_config, "schema_discovery_config"):
            base_matching_rate = agent_config.schema_discovery_config.base_matching_rate
        else:
            base_matching_rate = "fast"
        final_matching_rate = apply_progressive_func(base_matching_rate)

        # External knowledge enhancement with metrics
        metrics.start_stage(SearchStage.EXTERNAL_KNOWLEDGE)
        ek_start_time = time.time()
        if workflow and hasattr(workflow, "task"):
            try:
                enhanced_knowledge = await search_enhance_func(
                    workflow.task.task,
                    getattr(workflow.task, "subject_path", None),
                )

                if enhanced_knowledge:
                    original_knowledge = getattr(workflow.task, "external_knowledge", "")
                    combined_knowledge = combine_knowledge_func(original_knowledge, enhanced_knowledge)
                    workflow.task.external_knowledge = combined_knowledge
                    logger.info(f"Enhanced external knowledge with {len(enhanced_knowledge.split(chr(10)))} entries")

                    ek_metrics = ExternalKnowledgeMetrics(
                        attempted=True,
                        success=True,
                        entries_found=len(enhanced_knowledge.split(chr(10))),
                        entries_used=len(enhanced_knowledge.split(chr(10))),
                        duration_ms=(time.time() - ek_start_time) * 1000,
                        query_text=task.task,
                    )
                    metrics.record_external_knowledge(ek_metrics)
                    metrics.end_stage(SearchStage.EXTERNAL_KNOWLEDGE, success=True, tables_found=0)
                else:
                    metrics.end_stage(SearchStage.EXTERNAL_KNOWLEDGE, success=True, tables_found=0)
            except Exception as e:
                ek_metrics = ExternalKnowledgeMetrics(
                    attempted=True,
                    success=False,
                    error_type=type(e).__name__,
                    error_message=str(e),
                )
                metrics.record_external_knowledge(ek_metrics)
                metrics.end_stage(SearchStage.EXTERNAL_KNOWLEDGE, success=False, tables_found=0)
                logger.warning(f"External knowledge enhancement failed: {e}")
        else:
            metrics.end_stage(SearchStage.EXTERNAL_KNOWLEDGE, success=False, tables_found=0)

        # Check if LLM matching should be used
        metrics.start_stage(SearchStage.LLM_MATCHING)
        if final_matching_rate == "from_llm":
            logger.info("[LLM Matching] Using LLM-based schema matching for large datasets")
            llm_tables = await llm_schema_matching_func(task.task, final_matching_rate)
            if llm_tables:
                for table_name in llm_tables:
                    record_candidate(table_name, "llm_schema_matching")
                logger.info(f"[LLM Matching] Found {len(llm_tables)} tables via LLM inference")
            discovery_stats["llm_schema_matching_tables"] = len(llm_tables or [])
            metrics.end_stage(
                SearchStage.LLM_MATCHING,
                success=bool(llm_tables),
                tables_found=len(llm_tables or []),
            )
        else:
            metrics.end_stage(SearchStage.LLM_MATCHING, success=True, tables_found=0)

        # Stage 1: Semantic Search
        metrics.start_stage(SearchStage.SEMANTIC)
        top_n = 20
        semantic_tables, semantic_stats = await semantic_table_discovery(
            task_text=task.task,
            top_n=top_n,
            agent_config=agent_config,
            workflow=workflow,
            rerank_check_func=rerank_check_func,
            rewrite_fts_func=rewrite_fts_func,
        )
        if semantic_tables:
            for result in semantic_tables:
                record_candidate(result["table_name"], "semantic", score=result.get("score"))
            logger.info(f"[Stage 1] Found {len(semantic_tables)} tables via semantic search (top_n={top_n})")
        discovery_stats.update(semantic_stats)
        metrics.end_stage(
            SearchStage.SEMANTIC,
            success=bool(semantic_tables),
            tables_found=len(semantic_tables),
            metadata=semantic_stats,
        )

        # Stage 1: Keyword Matching
        metrics.start_stage(SearchStage.KEYWORD)
        keyword_tables = keyword_table_discovery(
            task_text=task_text,
            agent_config=agent_config,
            extract_tables_func=extract_tables_func,
        )
        if keyword_tables:
            for table_name, matched_terms in keyword_tables.items():
                record_candidate(table_name, "keyword", matched_terms=matched_terms)
            logger.info(f"[Stage 1] Found {len(keyword_tables)} tables via keyword matching")
        discovery_stats["keyword_tables"] = len(keyword_tables)
        metrics.end_stage(
            SearchStage.KEYWORD,
            success=bool(keyword_tables),
            tables_found=len(keyword_tables),
        )

        # Stage 1.5: LLM Inference
        llm_tables = await llm_based_table_discovery(
            query=task.task,
            all_tables=None,  # Will be fetched internally
            llm_call_func=llm_call_func,
        )
        if llm_tables:
            for table_name in llm_tables:
                record_candidate(table_name, "llm")
            logger.info(f"[Stage 1.5] Found {len(llm_tables)} tables via LLM inference")
        discovery_stats["llm_tables"] = len(llm_tables or [])

        candidate_tables = candidate_order[:]

        # Stage 2: Context Search
        metrics.start_stage(SearchStage.CONTEXT_SEARCH)
        context_threshold = 10
        if agent_config and hasattr(agent_config, "schema_discovery_config"):
            context_threshold = agent_config.schema_discovery_config.context_search_threshold

        should_context_search = len(candidate_tables) < context_threshold
        if (
            not should_context_search
            and contains_chinese(task_text)
            and len(llm_tables or []) <= 3
        ):
            should_context_search = True
            logger.info(
                "[Stage 2] Chinese query with limited LLM recall; running context search despite %d tables",
                len(candidate_tables),
            )

        if should_context_search:
            context_tables = await context_based_discovery(
                query=task.task,
                agent_config=agent_config,
                workflow=workflow,
            )
            if context_tables:
                for table_name in context_tables:
                    record_candidate(table_name, "context_search")
                logger.info(f"[Stage 2] Found {len(context_tables)} tables via context search")
            discovery_stats["context_search_tables"] = len(context_tables)
            metrics.end_stage(
                SearchStage.CONTEXT_SEARCH,
                success=bool(context_tables),
                tables_found=len(context_tables),
            )
        else:
            metrics.end_stage(SearchStage.CONTEXT_SEARCH, success=True, tables_found=0)

    # Finalize candidates
    max_tables = None
    if agent_config and hasattr(agent_config, "schema_discovery_config"):
        max_tables = agent_config.schema_discovery_config.max_candidate_tables

    candidate_tables, candidate_details_list = finalize_candidates(
        candidate_details=candidate_details,
        candidate_order=candidate_order,
        max_tables=max_tables,
        prefer_keyword=False,
        agent_config=agent_config,
    )

    return candidate_tables, candidate_details_list, discovery_stats


def finalize_candidates(
    candidate_details: Dict[str, Dict[str, Any]],
    candidate_order: List[str],
    max_tables: Optional[int],
    prefer_keyword: bool = False,
    agent_config = None,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Finalize and prioritize candidate tables.

    Args:
        candidate_details: Dictionary of candidate table details
        candidate_order: Order of discovery
        max_tables: Maximum number of tables to return
        prefer_keyword: Whether to prioritize keyword matches
        agent_config: Agent configuration

    Returns:
        Tuple of (finalized table names, detailed list with rankings)
    """
    priority_map = {
        "explicit": 5,
        "semantic": 4,
        "keyword": 3,
        "llm_schema_matching": 3,
        "llm": 2,
        "context_search": 2,
        "fallback": 1,
    }
    if prefer_keyword:
        priority_map["keyword"] = 4
        priority_map["semantic"] = 3
        priority_map["llm_schema_matching"] = 4
        priority_map["llm"] = 4

    details_list: List[Dict[str, Any]] = []
    for table_name in candidate_order:
        entry = candidate_details[table_name]
        sources = entry.get("sources", [])
        priority = max((priority_map.get(source, 0) for source in sources), default=0)
        semantic_score = entry.get("scores", {}).get("semantic", 0.0)
        details_list.append(
            {
                "table_name": table_name,
                "sources": sources,
                "scores": entry.get("scores", {}),
                "matched_terms": entry.get("matched_terms", []),
                "priority": priority,
                "semantic_score": semantic_score,
                "order_index": entry.get("order_index", 0),
            }
        )

    # Apply prefix-based whitelist/blacklist adjustments
    cfg = agent_config.schema_discovery_config if agent_config else None
    whitelist = getattr(cfg, "table_prefix_whitelist", []) if cfg else []
    blacklist = getattr(cfg, "table_prefix_blacklist", []) if cfg else []
    bl_penalty = getattr(cfg, "prefix_blacklist_penalty", 0.0) if cfg else 0.0
    wl_bonus = getattr(cfg, "prefix_whitelist_bonus", 0.0) if cfg else 0.0

    def _match_prefix(name: str, prefixes: List[str]) -> bool:
        return any(name.startswith(p.lower()) for p in prefixes)

    adjusted: List[Dict[str, Any]] = []
    for item in details_list:
        name_l = item["table_name"].lower()
        if whitelist and _match_prefix(name_l, [p.lower() for p in whitelist]):
            item["priority"] += 1
            item["semantic_score"] += wl_bonus
            item.setdefault("scores", {})["prefix_bonus"] = wl_bonus
        if blacklist and _match_prefix(name_l, [p.lower() for p in blacklist]):
            item["priority"] -= 1
            item["semantic_score"] *= max(0.0, 1.0 - bl_penalty)
            item.setdefault("scores", {})["prefix_penalty"] = bl_penalty
        adjusted.append(item)
    details_list = adjusted

    details_list.sort(
        key=lambda item: (
            item["priority"],
            item["semantic_score"],
            len(item["matched_terms"]),
            -item["order_index"],
        ),
        reverse=True,
    )

    if isinstance(max_tables, int) and max_tables > 0 and len(details_list) > max_tables:
        logger.info(f"Limiting candidate tables from {len(details_list)} to {max_tables}")
        limited = details_list[:max_tables]
        limited_names = {item["table_name"] for item in limited}
        llm_candidates = [
            item
            for item in details_list
            if "llm" in item.get("sources", []) and item["table_name"] not in limited_names
        ]
        if llm_candidates:
            for llm_item in llm_candidates:
                if llm_item["table_name"] in limited_names:
                    continue
                for idx in range(len(limited) - 1, -1, -1):
                    if "llm" in limited[idx].get("sources", []):
                        continue
                    limited_names.discard(limited[idx]["table_name"])
                    limited[idx] = llm_item
                    limited_names.add(llm_item["table_name"])
                    break
        details_list = limited

    candidate_tables = [item["table_name"] for item in details_list]
    for idx, item in enumerate(details_list, start=1):
        item["rank"] = idx

    return candidate_tables, details_list
