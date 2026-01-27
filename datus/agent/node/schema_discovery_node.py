# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
SchemaDiscoveryNode implementation for discovering relevant schema and tables.
"""

import os
import sys
import subprocess
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from datus.agent.node.node import Node, execute_with_async_stream
from datus.agent.workflow import Workflow
from datus.configuration.agent_config import AgentConfig
from datus.configuration.business_term_config import (
    LLM_TABLE_DISCOVERY_CONFIG, get_business_term_mapping)
from datus.configuration.node_config import (
    DEFAULT_HYBRID_COMMENT_BONUS,
    DEFAULT_HYBRID_RERANK_MIN_CPU_COUNT,
    DEFAULT_HYBRID_RERANK_MIN_MEMORY_GB,
    DEFAULT_HYBRID_RERANK_MIN_TABLES,
    DEFAULT_HYBRID_RERANK_TOP_N,
    DEFAULT_HYBRID_RERANK_WEIGHT,
    DEFAULT_RERANK_MODEL,
)
from datus.schemas.action_history import (ActionHistory, ActionHistoryManager,
                                          ActionRole, ActionStatus)
from datus.schemas.base import BaseResult
from datus.schemas.node_models import (BaseInput, Metric, ReferenceSql,
                                       TableSchema)
from datus.storage.schema_metadata import SchemaWithValueRAG
from datus.tools.db_tools.db_manager import get_db_manager
from datus.tools.func_tool.context_search import ContextSearchTools
from datus.utils.context_lock import safe_context_update
from datus.utils.error_handler import LLMMixin, NodeExecutionResult
from datus.utils.exceptions import ErrorCode
from datus.utils.loggings import get_logger
from datus.utils.constants import DBType
from datus.utils.sql_utils import (
    ddl_has_missing_commas,
    extract_table_names,
    is_likely_truncated_ddl,
)

logger = get_logger(__name__)


def _contains_chinese(text: str) -> bool:
    """
    Detect if text contains Chinese characters.

    Args:
        text: Input text to check

    Returns:
        True if text contains Chinese characters (CJK Unified Ideographs),
        False otherwise
    """
    if not text:
        return False
    # CJK Unified Ideographs range: U+4E00 to U+9FFF
    # Also includes CJK Extension A: U+3400 to U+4DBF
    for char in text:
        code_point = ord(char)
        if (0x4E00 <= code_point <= 0x9FFF) or (0x3400 <= code_point <= 0x4DBF):
            return True
    return False


class SchemaDiscoveryNode(Node, LLMMixin):
    """
    Node for discovering relevant schema and tables for a query.

    This node analyzes the query intent and discovers candidate tables/columns
    that may be relevant, populating the workflow context for downstream nodes.
    """

    def __init__(
        self,
        node_id: str,
        description: str,
        node_type: str,
        input_data: Optional[BaseInput] = None,
        agent_config: Optional[AgentConfig] = None,
        tools: Optional[list] = None,
    ):
        Node.__init__(
            self,
            node_id=node_id,
            description=description,
            node_type=node_type,
            input_data=input_data,
            agent_config=agent_config,
            tools=tools,
        )
        LLMMixin.__init__(self)
        self._rerank_check_cache: Dict[tuple, Dict[str, Any]] = {}

    def _get_available_cpu_count(self) -> int:
        try:
            return len(os.sched_getaffinity(0))
        except Exception:
            return os.cpu_count() or 0

    def _get_available_memory_bytes(self) -> int:
        try:
            import psutil

            return int(psutil.virtual_memory().available)
        except Exception:
            pass

        if sys.platform.startswith("linux"):
            try:
                with open("/proc/meminfo", "r", encoding="utf-8") as handle:
                    for line in handle:
                        if line.startswith("MemAvailable:"):
                            parts = line.split()
                            if len(parts) >= 2 and parts[1].isdigit():
                                return int(parts[1]) * 1024
            except Exception:
                return 0
            return 0

        if sys.platform == "darwin":
            try:
                result = subprocess.run(["vm_stat"], capture_output=True, text=True, check=False)
                if result.returncode != 0:
                    return 0
                page_size = 4096
                total_pages = 0
                for line in result.stdout.splitlines():
                    if "page size of" in line:
                        digits = "".join(ch for ch in line if ch.isdigit())
                        if digits:
                            page_size = int(digits)
                    if line.startswith("Pages free") or line.startswith("Pages inactive") or line.startswith(
                        "Pages speculative"
                    ):
                        parts = line.replace(".", "").split(":")
                        if len(parts) == 2:
                            count = parts[1].strip()
                            if count.isdigit():
                                total_pages += int(count)
                return total_pages * page_size
            except Exception:
                return 0

        if sys.platform == "win32":
            try:
                import ctypes

                class MemoryStatus(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_uint32),
                        ("dwMemoryLoad", ctypes.c_uint32),
                        ("ullTotalPhys", ctypes.c_uint64),
                        ("ullAvailPhys", ctypes.c_uint64),
                        ("ullTotalPageFile", ctypes.c_uint64),
                        ("ullAvailPageFile", ctypes.c_uint64),
                        ("ullTotalVirtual", ctypes.c_uint64),
                        ("ullAvailVirtual", ctypes.c_uint64),
                        ("ullAvailExtendedVirtual", ctypes.c_uint64),
                    ]

                status = MemoryStatus()
                status.dwLength = ctypes.sizeof(MemoryStatus)
                if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                    return int(status.ullAvailPhys)
            except Exception:
                return 0

        return 0

    def _rewrite_fts_query_with_llm(self, query_text: str) -> str:
        if not query_text or not self.model:
            return query_text
        try:
            prompt = (
                "You are a data analyst. Rewrite the user query into short Chinese keywords for schema search.\n"
                "Return ONLY JSON: {\"query\": \"<keywords>\"}.\n"
                "Constraints:\n"
                "- Use 3-8 concise keywords\n"
                "- Keep product or model names as-is (e.g., 铂智3X)\n"
                "- Remove filler words\n"
                f"User query: {query_text}\n"
            )
            response = self.call_llm(prompt, operation_name="schema_discovery_fts_rewrite")
        except Exception as exc:
            logger.debug(f"LLM rewrite failed: {exc}")
            return query_text
        if isinstance(response, dict):
            rewritten = response.get("query", "")
            if isinstance(rewritten, str) and rewritten.strip():
                return rewritten.strip()
        return query_text

    def _check_rerank_resources(
        self, model_path: str, min_cpu_count: int, min_memory_gb: float
    ) -> Dict[str, Any]:
        available_cpus = self._get_available_cpu_count()
        available_memory_bytes = self._get_available_memory_bytes()
        available_memory_gb = available_memory_bytes / (1024 ** 3) if available_memory_bytes else 0.0
        model_exists = bool(model_path) and os.path.exists(model_path)

        reasons = []
        if not model_exists:
            reasons.append(f"model_not_found:{model_path}")
        if min_cpu_count > 0 and available_cpus < min_cpu_count:
            reasons.append(f"cpu_insufficient:{available_cpus}<{min_cpu_count}")
        if min_memory_gb > 0 and available_memory_gb < min_memory_gb:
            reasons.append(f"memory_insufficient:{available_memory_gb:.2f}<{min_memory_gb}")

        # Check runtime dependencies for reranker
        dependencies_ok = True
        missing_deps = []
        try:
            import sentence_transformers
            import torch
        except ImportError as e:
            dependencies_ok = False
            missing_deps.append(str(e).split("'")[1] if "'" in str(e) else "unknown")
            reasons.append(f"missing_dependencies:{','.join(missing_deps)}")

        return {
            "ok": not reasons,
            "reasons": reasons,
            "available_cpus": available_cpus,
            "available_memory_gb": available_memory_gb,
            "model_exists": model_exists,
            "dependencies_ok": dependencies_ok,
            "missing_dependencies": missing_deps,
        }

    def _get_rerank_check(
        self, model_path: str, min_cpu_count: int, min_memory_gb: float
    ) -> Dict[str, Any]:
        cache_key = (model_path, min_cpu_count, min_memory_gb)
        cached = self._rerank_check_cache.get(cache_key)
        if cached is not None:
            return cached
        result = self._check_rerank_resources(model_path, min_cpu_count, min_memory_gb)
        self._rerank_check_cache[cache_key] = result
        return result

    def setup_input(self, workflow: Workflow) -> Dict[str, Any]:
        """
        Setup schema discovery input from workflow context.

        Args:
            workflow: Workflow instance containing context and task

        Returns:
            Dictionary with success status
        """
        if not self.input:
            self.input = BaseInput()

        return {"success": True, "message": "Schema discovery input setup complete"}

    async def run(self) -> AsyncGenerator[ActionHistory, None]:
        """
        Run schema discovery to find relevant tables and columns.

        Yields:
            ActionHistory: Progress and result actions
        """
        if not self.workflow:
            logger.error("Workflow not initialized in SchemaDiscoveryNode")
            return

        try:
            if not self.workflow.task:
                error_result = self.create_error_result(
                    ErrorCode.NODE_EXECUTION_FAILED,
                    "No workflow or task available for schema discovery",
                    "schema_discovery",
                    {"workflow_id": getattr(self.workflow, "id", "unknown") if self.workflow else "unknown"},
                )
                yield ActionHistory(
                    action_id=f"{self.id}_error",
                    role=ActionRole.TOOL,
                    messages=f"Schema discovery failed: {error_result.error_message}",
                    action_type="schema_discovery",
                    input={},
                    status=ActionStatus.FAILED,
                    output={"error": error_result.error_message, "error_code": error_result.error_code},
                )
                return

            task = self.workflow.task

            # Safely access intent from workflow metadata with comprehensive validation
            intent = "text2sql"  # Default fallback

            if hasattr(self.workflow, "metadata") and self.workflow.metadata:
                detected_intent = self.workflow.metadata.get("detected_intent")
                try:
                    confidence = float(self.workflow.metadata.get("intent_confidence", 0.0))
                except (ValueError, TypeError):
                    confidence = 0.0

                if detected_intent and isinstance(detected_intent, str) and confidence > 0.3:
                    intent = detected_intent
                    logger.debug(f"Using detected intent from workflow metadata: {intent} (confidence: {confidence})")
                else:
                    logger.warning(
                        f"Intent detection unreliable (intent: {detected_intent}, confidence: {confidence}), using default 'text2sql'"
                    )
            else:
                logger.warning("Workflow metadata not available, using default intent 'text2sql'")

            # ✅ Use clarified_task if available (from IntentClarificationNode)
            # Check if intent clarification has occurred and use the clarified version
            if hasattr(self.workflow, "metadata") and self.workflow.metadata:
                clarified_task = self.workflow.metadata.get("clarified_task")

                if clarified_task and clarified_task != task.task:
                    logger.info(
                        f"Using clarified task for schema discovery: "
                        f"'{task.task[:50]}...' → '{clarified_task[:50]}...'"
                    )
                    # ✅ Fix: Temporarily use clarified task for schema discovery with safe restoration
                    # Store original task in instance variable for safe restoration
                    self._original_task_text = task.task
                    task.task = clarified_task

                    # Store clarification metadata for logging
                    clarification_result = self.workflow.metadata.get("intent_clarification", {})
                    if clarification_result:
                        entities = clarification_result.get("entities", {})
                        corrections = clarification_result.get("corrections", {})
                        logger.info(
                            f"Intent clarification metadata: "
                            f"business_terms={entities.get('business_terms', [])}, "
                            f"typos_fixed={corrections.get('typos_fixed', [])}"
                        )

            # Validate task has required attributes
            if not hasattr(task, "task") or not task.task:
                error_result = self.create_error_result(
                    ErrorCode.NODE_EXECUTION_FAILED,
                    "Task has no content to analyze for schema discovery",
                    "schema_discovery",
                    {"task_id": getattr(task, "id", "unknown")},
                )
                yield ActionHistory(
                    action_id=f"{self.id}_error",
                    role=ActionRole.TOOL,
                    messages=f"Schema discovery failed: {error_result.error_message}",
                    action_type="schema_discovery",
                    input={"task_content": getattr(task, "task", "")},
                    status=ActionStatus.FAILED,
                    output={"error": error_result.error_message, "error_code": error_result.error_code},
                )
                return

            # Get candidate tables based on intent and task
            candidate_tables, candidate_details, discovery_stats = await self._discover_candidate_tables(task, intent)

            # If we have candidate tables, load their schemas
            if candidate_tables:
                await self._load_table_schemas(task, candidate_tables)
            else:
                # Fail-Fast: If no tables found after all attempts, do not proceed
                error_msg = f"No candidate tables found for task '{task.task[:50]}...'. Cannot generate SQL without schema context."
                logger.error(error_msg)

                error_result = self.create_error_result(
                    ErrorCode.NODE_EXECUTION_FAILED,
                    error_msg,
                    "schema_discovery",
                    {"task_id": getattr(task, "id", "unknown")},
                )
                yield ActionHistory(
                    action_id=f"{self.id}_error_nofound",
                    role=ActionRole.TOOL,
                    messages=f"Schema discovery failed: {error_msg}\nTry explicitly mentioning table names or checking database connection.",
                    action_type="schema_discovery",
                    input={"intent": intent},
                    status=ActionStatus.FAILED,
                    output={"error": error_msg, "error_code": ErrorCode.NODE_EXECUTION_FAILED},
                )

                # Mark node as failed
                self.result = BaseResult(success=False, error=error_msg)
                return

            # Emit success action with results
            yield ActionHistory(
                action_id=f"{self.id}_schema_discovery",
                role=ActionRole.TOOL,
                messages=f"Schema discovery completed: found {len(candidate_tables)} candidate tables",
                action_type="schema_discovery",
                input={
                    "intent": intent,
                    "catalog": task.catalog_name,
                    "database": task.database_name,
                    "schema": task.schema_name or None,
                    "query_context": self._build_query_context(task),
                },
                status=ActionStatus.SUCCESS,
                output={
                    "candidate_tables": candidate_tables,
                    "table_count": len(candidate_tables),
                    "intent": intent,
                    "table_candidates": candidate_details,
                    "discovery_stats": discovery_stats,
                },
            )

            # ✅ Fix: Set self.result to ensure node success
            self.result = BaseResult(
                success=True,
                data={
                    "candidate_tables": candidate_tables,
                    "table_count": len(candidate_tables),
                    "intent": intent,
                    "table_candidates": candidate_details,
                    "discovery_stats": discovery_stats,
                },
            )

            logger.info(
                f"Schema discovery completed: found {len(candidate_tables)} candidate tables for intent '{intent}'"
            )

        except Exception as e:
            logger.error(f"Schema discovery failed: {e}")
            error_result = self.create_error_result(
                ErrorCode.NODE_EXECUTION_FAILED,
                f"Schema discovery execution failed: {str(e)}",
                "schema_discovery",
                {
                    "task_id": (
                        getattr(self.workflow.task, "id", "unknown")
                        if self.workflow and self.workflow.task
                        else "unknown"
                    )
                },
            )
            yield ActionHistory(
                action_id=f"{self.id}_error",
                role=ActionRole.TOOL,
                messages=f"Schema discovery failed: {error_result.error_message}",
                action_type="schema_discovery",
                input={"intent": intent},
                status=ActionStatus.FAILED,
                output={"error": error_result.error_message, "error_code": error_result.error_code},
            )

        finally:
            # ✅ Fix: Restore original task to prevent parameter pollution
            # Check if we have the original_task_text from the clarified task modification
            original_task_text = getattr(self, "_original_task_text", None)
            if original_task_text is not None and self.workflow and self.workflow.task:
                self.workflow.task.task = original_task_text
                logger.info(f"Restored original task to prevent parameter pollution")
                # Clear the stored original task
                self._original_task_text = None

    async def _discover_candidate_tables(
        self, task, intent: str
    ) -> Tuple[List[str], List[Dict[str, Any]], Dict[str, Any]]:
        """
        Discover candidate tables based on task and intent.

        Args:
            task: The SQL task
            intent: Detected intent type

        Returns:
            List of candidate table names
        """
        # For now, use a simple heuristic approach
        # In production, this could use semantic search or LLM-based table discovery

        candidate_details: Dict[str, Dict[str, Any]] = {}
        candidate_order: List[str] = []
        discovery_stats: Dict[str, Any] = {}

        def record_candidate(
            table_name: str,
            source: str,
            score: Optional[float] = None,
            matched_terms: Optional[List[str]] = None,
        ) -> None:
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

        # 1. If tables are explicitly mentioned in the task, use those (Highest Priority)
        if hasattr(task, "tables") and task.tables:
            for table_name in task.tables:
                record_candidate(table_name, "explicit")
            logger.info(f"Using explicitly specified tables: {task.tables}")
            discovery_stats["explicit_tables"] = len(task.tables)

        # If intent is text2sql or sql (from intent analysis), try to discover tables
        if intent in ["text2sql", "sql"] and hasattr(task, "task"):
            task_text = task.task.lower()

            # ✅ NEW: Apply progressive matching based on reflection round (from SchemaLinkingNode)
            if self.agent_config and hasattr(self.agent_config, "schema_discovery_config"):
                base_matching_rate = self.agent_config.schema_discovery_config.base_matching_rate
            else:
                base_matching_rate = "fast"  # Default for backward compatibility
            final_matching_rate = self._apply_progressive_matching(base_matching_rate)

            # ✅ NEW: External knowledge enhancement (from SchemaLinkingNode)
            if self.workflow and hasattr(self.workflow, "task"):
                enhanced_knowledge = await self._search_and_enhance_external_knowledge(
                    self.workflow.task.task,
                    getattr(self.workflow.task, "subject_path", None),
                )

                if enhanced_knowledge:
                    original_knowledge = getattr(self.workflow.task, "external_knowledge", "")
                    combined_knowledge = self._combine_knowledge(original_knowledge, enhanced_knowledge)
                    self.workflow.task.external_knowledge = combined_knowledge
                    logger.info(f"Enhanced external knowledge with {len(enhanced_knowledge.split(chr(10)))} entries")

            # ✅ NEW: Check if LLM matching should be used (from SchemaLinkingNode)
            if final_matching_rate == "from_llm":
                logger.info(f"[LLM Matching] Using LLM-based schema matching for large datasets")
                llm_tables = await self._llm_based_schema_matching(task.task, final_matching_rate)
                if llm_tables:
                    for table_name in llm_tables:
                        record_candidate(table_name, "llm_schema_matching")
                    logger.info(f"[LLM Matching] Found {len(llm_tables)} tables via LLM inference")
                discovery_stats["llm_schema_matching_tables"] = len(llm_tables or [])

            # --- Stage 1: Fast Cache/Keyword & Semantic (Hybrid Search) ---
            # Align with design: semantic search uses top_n=20 by default.
            top_n = 20

            # 1. Semantic Search (High Priority)
            semantic_tables, semantic_stats = await self._semantic_table_discovery(task.task, top_n=top_n)
            if semantic_tables:
                for result in semantic_tables:
                    record_candidate(result["table_name"], "semantic", score=result.get("score"))
                logger.info(
                    f"[Stage 1] Found {len(semantic_tables)} tables via semantic search (top_n={top_n})"
                )
            discovery_stats.update(semantic_stats)

            # 2. Keyword Matching (Medium Priority)
            # Optimization: Always run keyword search to improve recall (Hybrid Search)
            keyword_tables = self._keyword_table_discovery(task_text)
            if keyword_tables:
                for table_name, matched_terms in keyword_tables.items():
                    record_candidate(table_name, "keyword", matched_terms=matched_terms)
                logger.info(f"[Stage 1] Found {len(keyword_tables)} tables via keyword matching")
            discovery_stats["keyword_tables"] = len(keyword_tables)

            # 3. LLM Inference (Stage 1.5) - Enhance recall for Chinese/Ambiguous queries
            llm_tables = await self._llm_based_table_discovery(task.task)
            if llm_tables:
                for table_name in llm_tables:
                    record_candidate(table_name, "llm")
                logger.info(f"[Stage 1.5] Found {len(llm_tables)} tables via LLM inference")
            discovery_stats["llm_tables"] = len(llm_tables or [])

            candidate_tables = candidate_order[:]

            # --- Stage 2: Deep Metadata Scan (Context Search) ---
            # ✅ Optimized: More aggressive trigger condition (was <3, now <10)
            # This ensures better recall by using context search more frequently
            context_threshold = 10
            if self.agent_config and hasattr(self.agent_config, "schema_discovery_config"):
                context_threshold = self.agent_config.schema_discovery_config.context_search_threshold
            should_context_search = len(candidate_tables) < context_threshold
            if (
                not should_context_search
                and _contains_chinese(task_text)
                and len(llm_tables or []) <= 3
            ):
                should_context_search = True
                logger.info(
                    "[Stage 2] Chinese query with limited LLM recall; running context search despite %d tables",
                    len(candidate_tables),
                )
            if should_context_search:
                if len(candidate_tables) < context_threshold:
                    logger.info(
                        f"[Stage 2] Tables found ({len(candidate_tables)}) below threshold ({context_threshold}), "
                        f"initiating Context Search..."
                    )
                else:
                    logger.info(f"[Stage 2] Initiating Context Search with {len(candidate_tables)} tables")
                context_tables = await self._context_based_discovery(task.task)
                if context_tables:
                    for table_name in context_tables:
                        record_candidate(table_name, "context_search")
                    logger.info(f"[Stage 2] Found {len(context_tables)} tables via context search")
                discovery_stats["context_tables"] = len(context_tables or [])
            else:
                discovery_stats["context_tables"] = 0

            candidate_tables = candidate_order[:]

            # 4. Fallback: Get All Tables (Low Priority)
            # Only if absolutely no tables found from previous stages
            if not candidate_tables:
                logger.info("[Stage 3] No tables found, attempting Fallback (Get All Tables)...")
                fallback_tables = await self._fallback_get_all_tables(task)
                if fallback_tables:
                    for table_name in fallback_tables:
                        record_candidate(table_name, "fallback")
                    logger.warning(f"[Stage 3] Used fallback: {len(fallback_tables)} tables")
                discovery_stats["fallback_tables"] = len(fallback_tables or [])
            else:
                discovery_stats["fallback_tables"] = 0

        max_tables = None
        if self.agent_config and hasattr(self.agent_config, "schema_discovery_config"):
            max_tables = getattr(self.agent_config.schema_discovery_config, "max_candidate_tables", None)

        candidate_tables, candidate_details_list = self._finalize_candidates(
            candidate_details,
            candidate_order,
            max_tables,
            prefer_keyword=_contains_chinese(task_text),
        )
        discovery_stats["pre_limit_count"] = len(candidate_details)
        discovery_stats["max_candidate_tables"] = max_tables
        discovery_stats["final_table_count"] = len(candidate_tables)
        logger.info(
            "Schema discovery summary: "
            f"explicit={discovery_stats.get('explicit_tables', 0)}, "
            f"semantic={discovery_stats.get('semantic_tables', 0)}, "
            f"keyword={discovery_stats.get('keyword_tables', 0)}, "
            f"llm={discovery_stats.get('llm_tables', 0)}, "
            f"context={discovery_stats.get('context_tables', 0)}, "
            f"fallback={discovery_stats.get('fallback_tables', 0)}, "
            f"final={discovery_stats.get('final_table_count', 0)}"
        )
        return candidate_tables, candidate_details_list, discovery_stats

    def _finalize_candidates(
        self,
        candidate_details: Dict[str, Dict[str, Any]],
        candidate_order: List[str],
        max_tables: Optional[int],
        prefer_keyword: bool = False,
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
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
        cfg = self.agent_config.schema_discovery_config if self.agent_config else None
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

    def _build_query_context(self, task) -> Dict[str, Any]:
        context: Dict[str, Any] = {"task": task.task}
        if self.workflow and hasattr(self.workflow, "metadata") and self.workflow.metadata:
            clarified_task = self.workflow.metadata.get("clarified_task")
            if clarified_task:
                context["clarified_task"] = clarified_task
            clarification = self.workflow.metadata.get("intent_clarification", {})
            if isinstance(clarification, dict):
                entities = clarification.get("entities", {})
                if entities:
                    context["business_terms"] = entities.get("business_terms", [])
                    context["dimensions"] = entities.get("dimensions", [])
                    context["metrics"] = entities.get("metrics", [])
                    context["time_range"] = entities.get("time_range")
        return context

    async def _context_based_discovery(self, query: str) -> List[str]:
        """
        Stage 2: Discover tables via Metrics and Reference SQL.

        This method systematically searches for metrics and reference SQL,
        then updates the workflow context with discovered knowledge.
        """
        found_tables = []
        try:
            if not self.agent_config:
                return []

            context_search = ContextSearchTools(self.agent_config)

            # 1. Search Metrics - Systematic search (no conditional check)
            try:
                metric_result = context_search.search_metrics(query_text=query, top_n=5)
                if metric_result.success and metric_result.result:
                    # Convert results to Metric objects and update context
                    metrics = []
                    for item in metric_result.result:
                        if isinstance(item, dict):
                            metric = Metric(
                                name=item.get("name", ""), llm_text=item.get("llm_text", item.get("description", ""))
                            )
                            metrics.append(metric)

                    if metrics and self.workflow:

                        def update_metrics():
                            self.workflow.context.update_metrics(metrics)
                            return {"metrics_count": len(metrics)}

                        result = safe_context_update(
                            self.workflow.context,
                            update_metrics,
                            operation_name="update_metrics",
                        )

                        if result["success"]:
                            logger.info(f"Updated context with {len(metrics)} metrics")
                        else:
                            logger.warning(f"Metrics update failed: {result.get('error')}")
            except Exception as e:
                logger.warning(f"Metrics search failed: {e}")

            # 2. Search Reference SQL - Systematic search (always execute)
            try:
                ref_sql_result = context_search.search_reference_sql(query_text=query, top_n=5)
                if ref_sql_result.success and ref_sql_result.result:
                    # Convert results to ReferenceSql objects and update context
                    reference_sqls = []
                    for item in ref_sql_result.result:
                        if isinstance(item, dict):
                            ref_sql = ReferenceSql(
                                name=item.get("name", item.get("summary", ""))[:100],
                                sql=item.get("sql", ""),
                                comment=item.get("comment", ""),
                                summary=item.get("summary", ""),
                                tags=item.get("tags", ""),
                            )
                            reference_sqls.append(ref_sql)

                    if reference_sqls and self.workflow:

                        def update_reference_sqls():
                            self.workflow.context.update_reference_sqls(reference_sqls)
                            return {"ref_sql_count": len(reference_sqls)}

                        result = safe_context_update(
                            self.workflow.context,
                            update_reference_sqls,
                            operation_name="update_reference_sqls",
                        )

                        if result["success"]:
                            logger.info(f"Updated context with {len(reference_sqls)} reference SQLs")
                            # Extract table names from reference SQLs and populate found_tables
                            for ref_sql in reference_sqls:
                                if ref_sql.sql:
                                    try:
                                        tables = extract_table_names(ref_sql.sql)
                                        found_tables.extend(tables)
                                    except Exception as e:
                                        logger.debug(f"Failed to extract tables from SQL: {e}")
                        else:
                            logger.warning(f"Reference SQLs update failed: {result.get('error')}")
            except Exception as e:
                logger.warning(f"Reference SQL search failed: {e}")

        except Exception as e:
            logger.warning(f"Context-based discovery failed: {e}")

        return found_tables

    async def _update_context(self, schemas: List[TableSchema], task) -> None:
        """
        Update the workflow context with discovered schemas.
        """
        if not schemas:
            return

        # Add to schemas list
        context_schemas = task.schemas or []

        # Merge new schemas, avoiding duplicates
        existing_tables = {s.table_name for s in context_schemas}
        new_schemas = [s for s in schemas if s.table_name not in existing_tables]

        if new_schemas:
            context_schemas.extend(new_schemas)
            task.schemas = context_schemas

            # Also update tables list for simple access
            current_tables = task.tables or []
            new_table_names = [s.table_name for s in new_schemas]
            current_tables.extend(new_table_names)
            task.tables = list(set(current_tables))

            logger.info(f"Updated context with {len(new_schemas)} new schemas")

    async def _semantic_table_discovery(
        self, task_text: str, top_n: int = 20
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Discover tables using semantic vector search with comment enhancement.

        ✅ Enhanced: Concatenate table_comment with definition for embedding (improves precision).
        ✅ Enhanced: Filter by business_tags if query contains domain keywords.
        ✅ Enhanced: Use row_count to prioritize frequently accessed tables.

        Args:
            task_text: User query text
            top_n: Number of results to retrieve (adjusted by progressive matching)

        Returns:
            List of table names
        """
        try:
            from datus.configuration.business_term_config import \
                infer_business_tags
            from datus.utils.device_utils import get_device

            if not self.agent_config:
                return []

            tables: List[Dict[str, Any]] = []
            stats: Dict[str, Any] = {
                "semantic_top_n": top_n,
                "semantic_tables": 0,
                "semantic_total_hits": 0,
            }

            # Get similarity threshold from configuration (with backward compatibility)
            if hasattr(self.agent_config, "schema_discovery_config"):
                base_similarity_threshold = self.agent_config.schema_discovery_config.semantic_similarity_threshold
            else:
                base_similarity_threshold = 0.5  # Default threshold for backward compatibility

            # ✅ Dynamic Similarity Threshold: Use lower threshold for Chinese queries
            # Chinese queries often have lower semantic similarity with English table names
            # due to cross-lingual embedding mismatch. Lower threshold improves recall.
            if _contains_chinese(task_text):
                # Get reduction factor from configuration (with backward compatibility)
                if hasattr(self.agent_config, "schema_discovery_config"):
                    reduction_factor = self.agent_config.schema_discovery_config.chinese_query_threshold_reduction
                else:
                    reduction_factor = 0.6  # Default: apply 40% reduction for Chinese queries

                similarity_threshold = base_similarity_threshold * reduction_factor
                logger.info(
                    f"Chinese query detected, using reduced similarity threshold: "
                    f"{similarity_threshold:.2f} (base: {base_similarity_threshold}, factor: {reduction_factor})"
                )
            else:
                similarity_threshold = base_similarity_threshold
            stats["semantic_similarity_threshold"] = similarity_threshold

            # ✅ NEW: Detect query domain for business tag filtering
            query_tags = infer_business_tags(task_text, [])

            # Build enhanced search text with domain hints
            enhanced_query = task_text
            if query_tags:
                enhanced_query = f"{task_text} domain:{','.join(query_tags)}"

            # 1. Use SchemaWithValueRAG for direct table semantic search
            rag = SchemaWithValueRAG(agent_config=self.agent_config)
            # ✅ Enhanced: Use dynamic top_n from progressive matching
            schema_results, _ = rag.search_similar(query_text=enhanced_query, top_n=top_n)

            candidate_map: Dict[str, Dict[str, Any]] = {}
            vector_total_hits = 0
            fts_total_hits = 0
            rerank_total_hits = 0

            cfg = self.agent_config.schema_discovery_config if self.agent_config else None
            hybrid_enabled = getattr(cfg, "hybrid_search_enabled", True)
            use_fts = getattr(cfg, "hybrid_use_fts", True)
            vector_weight = getattr(cfg, "hybrid_vector_weight", 0.6)
            fts_weight = getattr(cfg, "hybrid_fts_weight", 0.3)
            row_count_weight = getattr(cfg, "hybrid_row_count_weight", 0.2)
            tag_bonus_weight = getattr(cfg, "hybrid_tag_bonus", 0.1)
            comment_bonus_weight = getattr(cfg, "hybrid_comment_bonus", DEFAULT_HYBRID_COMMENT_BONUS)
            rerank_enabled = getattr(cfg, "hybrid_rerank_enabled", False)
            rerank_weight = getattr(cfg, "hybrid_rerank_weight", DEFAULT_HYBRID_RERANK_WEIGHT)
            rerank_min_tables = getattr(cfg, "hybrid_rerank_min_tables", DEFAULT_HYBRID_RERANK_MIN_TABLES)
            rerank_top_n = getattr(cfg, "hybrid_rerank_top_n", DEFAULT_HYBRID_RERANK_TOP_N)
            rerank_model = getattr(cfg, "hybrid_rerank_model", DEFAULT_RERANK_MODEL)
            rerank_column = getattr(cfg, "hybrid_rerank_column", "definition")
            rerank_min_cpu_count = getattr(cfg, "hybrid_rerank_min_cpu_count", DEFAULT_HYBRID_RERANK_MIN_CPU_COUNT)
            rerank_min_memory_gb = getattr(cfg, "hybrid_rerank_min_memory_gb", DEFAULT_HYBRID_RERANK_MIN_MEMORY_GB)
            rerank_device = get_device()
            rerank_check = None
            if rerank_enabled:
                rerank_check = self._get_rerank_check(rerank_model, rerank_min_cpu_count, rerank_min_memory_gb)
                if not rerank_check["ok"]:
                    logger.info(
                        "Rerank disabled due to resource/model constraints: "
                        + ", ".join(rerank_check["reasons"])
                    )
                    rerank_enabled = False
            stats.update(
                {
                    "hybrid_enabled": hybrid_enabled,
                    "hybrid_use_fts": use_fts,
                    "hybrid_vector_weight": vector_weight,
                    "hybrid_fts_weight": fts_weight,
                    "hybrid_row_count_weight": row_count_weight,
                    "hybrid_tag_bonus": tag_bonus_weight,
                    "hybrid_comment_bonus": comment_bonus_weight,
                    "hybrid_rerank_enabled": rerank_enabled,
                    "hybrid_rerank_weight": rerank_weight,
                    "hybrid_rerank_min_tables": rerank_min_tables,
                    "hybrid_rerank_top_n": rerank_top_n,
                    "hybrid_rerank_model": rerank_model,
                    "hybrid_rerank_column": rerank_column,
                    "hybrid_rerank_min_cpu_count": rerank_min_cpu_count,
                    "hybrid_rerank_min_memory_gb": rerank_min_memory_gb,
                    "hybrid_rerank_model_exists": rerank_check["model_exists"] if rerank_check else False,
                    "hybrid_rerank_available_cpus": rerank_check["available_cpus"] if rerank_check else 0,
                    "hybrid_rerank_available_memory_gb": (
                        rerank_check["available_memory_gb"] if rerank_check else 0.0
                    ),
                    "hybrid_rerank_device": rerank_device,
                }
            )

            if schema_results and len(schema_results) > 0:
                try:
                    table_names = schema_results.column("table_name").to_pylist()
                    vector_total_hits = len(table_names)
                    distances = (
                        schema_results.column("_distance").to_pylist()
                        if "_distance" in schema_results.column_names
                        else [None] * len(table_names)
                    )
                    table_comments = (
                        schema_results.column("table_comment").to_pylist()
                        if "table_comment" in schema_results.column_names
                        else [""] * len(table_names)
                    )
                    row_counts = (
                        schema_results.column("row_count").to_pylist()
                        if "row_count" in schema_results.column_names
                        else [0] * len(table_names)
                    )
                    business_tags_list = []
                    if "business_tags" in schema_results.column_names:
                        for tags in schema_results.column("business_tags").to_pylist():
                            business_tags_list.append(tags if isinstance(tags, list) else [])
                    else:
                        business_tags_list = [[]] * len(table_names)

                    for table_name, distance, table_comment, row_count, tags in zip(
                        table_names, distances, table_comments, row_counts, business_tags_list
                    ):
                        similarity = 0.1 if distance is None else 1.0 / (1.0 + distance)
                        if similarity_threshold and similarity < similarity_threshold:
                            continue
                        candidate = candidate_map.setdefault(
                            table_name,
                            {
                                "vector_score": 0.0,
                                "fts_score": 0.0,
                                "rerank_score": 0.0,
                                "table_comment": table_comment or "",
                                "row_count": row_count or 0,
                                "tags": tags or [],
                            },
                        )
                        candidate["vector_score"] = max(candidate["vector_score"], similarity)
                        if table_comment:
                            candidate["table_comment"] = table_comment
                        if row_count:
                            candidate["row_count"] = row_count
                        if tags:
                            candidate["tags"] = tags

                except Exception as filter_error:
                    logger.warning(f"Vector search parsing failed: {filter_error}.")

            fts_results = None
            if hybrid_enabled and use_fts:
                try:
                    task_context = getattr(self.workflow, "task", None) if self.workflow else None
                    fts_catalog = task_context.catalog_name if task_context else ""
                    fts_database = task_context.database_name if task_context else ""
                    fts_schema = task_context.schema_name if task_context else ""
                    fts_table_type = (
                        getattr(task_context, "schema_linking_type", "table") if task_context else "table"
                    )
                    fts_query = enhanced_query
                    cfg = self.agent_config.schema_discovery_config if self.agent_config else None
                    llm_rewrite_enabled = bool(getattr(cfg, "llm_fts_rewrite_enabled", False))
                    min_chars = int(getattr(cfg, "llm_fts_rewrite_min_chars", 6))
                    if llm_rewrite_enabled and len(fts_query or "") >= min_chars and _contains_chinese(fts_query):
                        fts_query = self._rewrite_fts_query_with_llm(fts_query)
                    fts_results = rag.schema_store.search_fts(
                        query_text=fts_query,
                        catalog_name=fts_catalog or "",
                        database_name=fts_database or "",
                        schema_name=fts_schema or "",
                        table_type=fts_table_type,
                        top_n=top_n,
                        select_fields=["table_name", "table_comment", "row_count", "business_tags"],
                    )
                except Exception as fts_error:
                    logger.warning(f"FTS search failed: {fts_error}")

            if fts_results is not None and len(fts_results) > 0:
                fts_table_names = fts_results.column("table_name").to_pylist()
                fts_total_hits = len(fts_table_names)
                fts_scores = (
                    fts_results.column("_score").to_pylist() if "_score" in fts_results.column_names else []
                )
                max_fts_score = max(fts_scores) if fts_scores else 0.0
                fts_table_comments = (
                    fts_results.column("table_comment").to_pylist()
                    if "table_comment" in fts_results.column_names
                    else [""] * len(fts_table_names)
                )
                fts_row_counts = (
                    fts_results.column("row_count").to_pylist()
                    if "row_count" in fts_results.column_names
                    else [0] * len(fts_table_names)
                )
                fts_tags_list = []
                if "business_tags" in fts_results.column_names:
                    for tags in fts_results.column("business_tags").to_pylist():
                        fts_tags_list.append(tags if isinstance(tags, list) else [])
                else:
                    fts_tags_list = [[]] * len(fts_table_names)

                for idx, (table_name, table_comment, row_count, tags) in enumerate(
                    zip(fts_table_names, fts_table_comments, fts_row_counts, fts_tags_list)
                ):
                    raw_score = fts_scores[idx] if idx < len(fts_scores) else 1.0
                    norm_score = (raw_score / max_fts_score) if max_fts_score else 0.5
                    candidate = candidate_map.setdefault(
                        table_name,
                        {
                            "vector_score": 0.0,
                            "fts_score": 0.0,
                            "rerank_score": 0.0,
                            "table_comment": table_comment or "",
                            "row_count": row_count or 0,
                            "tags": tags or [],
                        },
                    )
                    candidate["fts_score"] = max(candidate["fts_score"], norm_score)
                    if table_comment:
                        candidate["table_comment"] = table_comment
                    if row_count:
                        candidate["row_count"] = row_count
                    if tags:
                        candidate["tags"] = tags

            rerank_results = None
            if hybrid_enabled and rerank_enabled:
                candidate_count = max(len(candidate_map), vector_total_hits, fts_total_hits)
                if candidate_count < rerank_min_tables:
                    logger.info(
                        f"Rerank skipped: candidates={candidate_count} below min_tables={rerank_min_tables}"
                    )
                else:
                    try:
                        reranker_key = f"{rerank_model}:{rerank_column}"
                        cached_key = getattr(self, "_schema_reranker_key", "")
                        if cached_key == reranker_key:
                            reranker = getattr(self, "_schema_reranker", None)
                        else:
                            reranker = None

                        if reranker is None:
                            try:
                                from lancedb.rerankers import CrossEncoderReranker
                            except Exception as import_error:
                                logger.warning(f"Rerank disabled: failed to import CrossEncoderReranker: {import_error}")
                                reranker = None
                            else:
                                reranker = CrossEncoderReranker(
                                    model_name=rerank_model,
                                    device=rerank_device,
                                    column=rerank_column,
                                )
                                self._schema_reranker = reranker
                                self._schema_reranker_key = reranker_key

                        if reranker:
                            rag.schema_store.reranker = reranker
                            rerank_results, _ = rag.search_similar(
                                query_text=enhanced_query,
                                top_n=rerank_top_n,
                                use_rerank=True,
                            )
                    except Exception as rerank_error:
                        logger.warning(f"Rerank search failed: {rerank_error}")

            if rerank_results is not None and len(rerank_results) > 0:
                rerank_table_names = rerank_results.column("table_name").to_pylist()
                rerank_total_hits = len(rerank_table_names)
                rerank_scores = []
                score_column = None
                for candidate_column in ("_score", "_relevance", "_distance"):
                    if candidate_column in rerank_results.column_names:
                        score_column = candidate_column
                        break
                if score_column == "_distance":
                    distances = rerank_results.column("_distance").to_pylist()
                    rerank_scores = [0.1 if d is None else 1.0 / (1.0 + d) for d in distances]
                elif score_column:
                    rerank_scores = rerank_results.column(score_column).to_pylist()
                else:
                    rerank_scores = []

                max_rerank_score = max(rerank_scores) if rerank_scores else 0.0
                for idx, table_name in enumerate(rerank_table_names):
                    raw_score = rerank_scores[idx] if idx < len(rerank_scores) else None
                    if raw_score is None:
                        score = 1.0 - (idx / max(len(rerank_table_names), 1))
                    elif score_column == "_distance":
                        score = raw_score
                    else:
                        score = (raw_score / max_rerank_score) if max_rerank_score else 0.5

                    candidate = candidate_map.setdefault(
                        table_name,
                        {
                            "vector_score": 0.0,
                            "fts_score": 0.0,
                            "rerank_score": 0.0,
                            "table_comment": "",
                            "row_count": 0,
                            "tags": [],
                        },
                    )
                    candidate["rerank_score"] = max(candidate["rerank_score"], score)

            if candidate_map:
                scored_tables = []
                for table_name, payload in candidate_map.items():
                    row_count = payload.get("row_count", 0) or 0
                    tags = payload.get("tags", []) or []
                    table_comment = payload.get("table_comment", "") or ""
                    vector_score = payload.get("vector_score", 0.0) or 0.0
                    fts_score = payload.get("fts_score", 0.0) or 0.0
                    rerank_score = payload.get("rerank_score", 0.0) or 0.0

                    row_count_score = min(row_count / 1_000_000, 1.0) if row_count else 0
                    tag_bonus = len(set(tags) & set(query_tags)) * tag_bonus_weight if query_tags else 0
                    comment_bonus = 0.0
                    if table_comment and any(term.lower() in table_comment.lower() for term in task_text.split()):
                        comment_bonus = comment_bonus_weight

                    if not hybrid_enabled:
                        final_score = vector_score
                    else:
                        final_score = (vector_score * vector_weight) + (fts_score * fts_weight)
                        final_score += row_count_score * row_count_weight
                        final_score += rerank_score * rerank_weight
                    final_score += tag_bonus + comment_bonus
                    scored_tables.append((table_name, final_score))

                scored_tables.sort(key=lambda x: x[1], reverse=True)
                tables.extend([{"table_name": name, "score": score} for name, score in scored_tables[:top_n]])
                stats["semantic_tables"] = len(scored_tables[:top_n])
                stats["semantic_total_hits"] = max(vector_total_hits, fts_total_hits, rerank_total_hits)
                stats["semantic_vector_hits"] = vector_total_hits
                stats["semantic_fts_hits"] = fts_total_hits
                stats["semantic_rerank_hits"] = rerank_total_hits

                logger.info(
                    f"Hybrid semantic search produced {len(scored_tables[:top_n])} tables "
                    f"(vector={vector_total_hits}, fts={fts_total_hits}, tags={query_tags})"
                )
            elif vector_total_hits:
                logger.warning(
                    f"No tables passed similarity threshold ({similarity_threshold}). "
                    f"Using all {vector_total_hits} vector results as fallback."
                )
                tables.extend([{"table_name": name, "score": 0.0} for name in schema_results.column("table_name")])
                stats["semantic_tables"] = vector_total_hits
                stats["semantic_total_hits"] = vector_total_hits

            # 2. Use ContextSearchTools for metrics/business logic search (complementary)
            context_search = ContextSearchTools(self.agent_config)
            if context_search.has_metrics:
                result = context_search.search_metrics(query_text=task_text, top_n=5)
                if result.success and result.result:
                    # Logic to extract tables from metrics would go here
                    # For now, we rely primarily on SchemaWithValueRAG
                    pass

            return tables, stats
        except Exception as e:
            logger.warning(f"Semantic table discovery failed: {e}")
            return [], {"semantic_tables": 0, "semantic_total_hits": 0}

    def _keyword_table_discovery(self, task_text: str) -> Dict[str, List[str]]:
        """
        Discover tables using keyword matching and External Knowledge.

        Args:
            task_text: Lowercase user query text

        Returns:
            List of table names
        """
        candidate_tables: Dict[str, List[str]] = {}

        # 1. Check hardcoded configuration
        from datus.configuration.business_term_config import \
            TABLE_KEYWORD_PATTERNS

        for keyword, table_name in TABLE_KEYWORD_PATTERNS.items():
            if keyword in task_text:
                for name in {table_name, table_name[:-1] if table_name.endswith("s") else f"{table_name}s"}:
                    candidate_tables.setdefault(name, [])
                    if keyword not in candidate_tables[name]:
                        candidate_tables[name].append(keyword)

                business_mappings = get_business_term_mapping(keyword)
                if business_mappings:
                    for name in business_mappings:
                        candidate_tables.setdefault(name, [])
                        if keyword not in candidate_tables[name]:
                            candidate_tables[name].append(keyword)

        # 2. Check External Knowledge Store
        try:
            from datus.storage.cache import get_storage_cache_instance

            storage_cache = get_storage_cache_instance(self.agent_config)
            ext_knowledge = storage_cache.ext_knowledge_storage()

            # Search for relevant terms in the knowledge base
            results = ext_knowledge.search_knowledge(query_text=task_text, top_n=5)

            if results:
                for item in results:
                    explanation = item.get("explanation", "")
                    potential_tables = self._extract_potential_tables_from_text(explanation)
                    for name in potential_tables:
                        candidate_tables.setdefault(name, [])
                        if "external_knowledge" not in candidate_tables[name]:
                            candidate_tables[name].append("external_knowledge")

        except (ConnectionError, TimeoutError) as e:
            logger.warning(f"External knowledge search failed (network): {e}")
            # Continue with hardcoded mappings as fallback
        except Exception as e:
            logger.warning(f"External knowledge search failed: {e}")
            # Continue with hardcoded mappings as fallback

        return candidate_tables

    def _extract_potential_tables_from_text(self, text: str) -> List[str]:
        """Helper to extract potential table names (snake_case words) from text."""
        import re

        # Simple regex for potential table names (snake_case, at least one underscore)
        return re.findall(r"\b[a-z]+_[a-z0-9_]+\b", text)

    async def _llm_based_table_discovery(self, query: str, all_tables: List[str] = None) -> List[str]:
        """
        Stage 1.5: Use LLM to select relevant tables from actual database tables.

        ✅ Fixed: LLM selects from real DB tables instead of hallucinating.
        The LLM is given the actual list of database tables and asked to select
        the most relevant ones for the query. This prevents hallucination of
        non-existent table names.

        Args:
            query: User query text
            all_tables: Optional list of actual database tables (fetched if not provided)

        Returns:
            List of table names that exist in the database
        """
        if not query:
            return []

        # Get all actual tables from database if not provided
        if all_tables is None:
            all_tables = await self._get_all_database_tables()

        if not all_tables:
            logger.debug("No database tables available for LLM-based discovery")
            return []

        # ✅ Use configured prompt template with actual table list
        prompt_template = LLM_TABLE_DISCOVERY_CONFIG.get("prompt_template", "")
        # Format table list for prompt (limit to prevent token overflow)
        tables_list = "\n".join(f"- {t}" for t in all_tables[:100])
        prompt = prompt_template.format(tables_list=tables_list, query=query)

        try:
            # ✅ Use LLMMixin with retry and caching
            cache_key = f"llm_table_discovery:{hash(query)}:{hash(tuple(all_tables))}"
            max_retries = LLM_TABLE_DISCOVERY_CONFIG.get("max_retries", 3)
            cache_enabled = LLM_TABLE_DISCOVERY_CONFIG.get("cache_enabled", True)
            cache_ttl_seconds = LLM_TABLE_DISCOVERY_CONFIG.get("cache_ttl_seconds") if cache_enabled else None

            response = await self.llm_call_with_retry(
                prompt=prompt,
                operation_name="table_discovery",
                cache_key=cache_key if cache_enabled else None,
                max_retries=max_retries,
                cache_ttl_seconds=cache_ttl_seconds,
            )

            tables = response.get("tables", [])
            # ✅ Fixed: Validate that returned tables actually exist in database
            all_tables_set = set(all_tables)
            validated_tables = [str(t) for t in tables if isinstance(t, (str, int, float)) and str(t) in all_tables_set]
            unique_tables = list(set(validated_tables))

            if len(tables) != len(validated_tables):
                invalid_tables = set(str(t) for t in tables if isinstance(t, (str, int, float))) - all_tables_set
                logger.warning(
                    f"LLM returned {len(invalid_tables)} invalid table(s) that don't exist: {invalid_tables}"
                )

            logger.info(f"LLM selected {len(unique_tables)} tables from actual database: {unique_tables}")
            return unique_tables
        except NodeExecutionResult as e:
            logger.warning(f"LLM table discovery failed after retries: {e.error_message}")
            return []
        except Exception as e:
            logger.warning(f"LLM table discovery failed with unexpected error: {e}")
            return []

    async def _get_all_database_tables(self) -> List[str]:
        """
        Fetch all table names from the database(s).
        Supports multi-database configuration by iterating through all configured databases.

        Returns:
            List of all table names in the database(s)
        """
        try:
            if not self.workflow or not self.workflow.task:
                return []

            task = self.workflow.task
            db_manager = get_db_manager()

            # Check if multiple databases are configured
            db_configs = self.agent_config.current_db_configs()
            is_multi_db = len(db_configs) > 1

            all_tables = []

            if is_multi_db:
                logger.info(f"Multi-database mode detected: scanning {len(db_configs)} databases...")
                for logic_name, db_config in db_configs.items():
                    try:
                        connector = db_manager.get_conn(self.agent_config.current_namespace, logic_name)
                        if not connector:
                            continue

                        tables = connector.get_tables(
                            catalog_name=db_config.catalog or "",
                            database_name=db_config.database or "",
                            schema_name=db_config.schema or "",
                        )
                        # Prefix with logic_name to distinguish tables from different DBs
                        prefixed_tables = [f"{logic_name}.{t}" for t in tables]
                        all_tables.extend(prefixed_tables)
                    except Exception as e:
                        logger.warning(f"Failed to get tables for database {logic_name}: {e}")
            else:
                # Single database mode (legacy behavior)
                connector = db_manager.get_conn(self.agent_config.current_namespace, task.database_name)

                if not connector:
                    logger.warning(f"Could not get database connector for {task.database_name}")
                    return []

                # Get all tables from database using the correct method name
                # BaseSqlConnector defines get_tables() as the abstract method
                all_tables = connector.get_tables(
                    catalog_name=task.catalog_name or "",
                    database_name=task.database_name or "",
                    schema_name=task.schema_name or "",
                )

            logger.debug(f"Retrieved {len(all_tables)} tables from database(s)")
            return all_tables

        except Exception as e:
            logger.warning(f"Failed to retrieve database tables: {e}")
            return []

    async def _fallback_get_all_tables(self, task) -> List[str]:
        """
        Fallback: Get all tables from the database if no candidates found.

        Args:
            task: The SQL task

        Returns:
            List of all table names (limited to top N)
        """
        try:
            # ✅ Fixed: Use correct function name (get_db_manager not db_manager_instance)
            db_manager = get_db_manager()
            connector = db_manager.get_conn(self.agent_config.current_namespace, task.database_name)

            if not connector:
                return []

            # Get all tables using the correct method name
            # BaseSqlConnector defines get_tables() as the abstract method
            all_tables = connector.get_tables(
                catalog_name=task.catalog_name or "",
                database_name=task.database_name or "",
                schema_name=task.schema_name or "",
            )

            # Limit to reasonable number to avoid context overflow
            max_tables = 50
            if self.agent_config and hasattr(self.agent_config, "schema_discovery_config"):
                max_tables = self.agent_config.schema_discovery_config.fallback_table_limit
            if len(all_tables) > max_tables:
                logger.warning(f"Too many tables ({len(all_tables)}), limiting to first {max_tables} for fallback")
                return all_tables[:max_tables]

            return all_tables

        except Exception as e:
            logger.warning(f"Fallback table discovery failed: {e}")
            return []

    async def _load_table_schemas(self, task, candidate_tables: List[str]) -> None:
        """
        Load schemas for candidate tables and update workflow context.
        Includes automatic metadata repair if definitions are missing.

        ✅ Fixed: Uses thread-safe context update to prevent race conditions.

        Args:
            task: The SQL task
            candidate_tables: List of table names to load schemas for
        """
        if not candidate_tables or not self.workflow:
            return

        try:
            # Check if multiple databases are configured
            if self.agent_config is None:
                db_configs = {}
            else:
                db_configs = self.agent_config.current_db_configs()
            is_multi_db = len(db_configs) > 1

            # --- Repair Logic Start ---
            if self.agent_config:
                try:
                    from datus.storage.cache import get_storage_cache_instance

                    storage_cache = get_storage_cache_instance(self.agent_config)
                    schema_storage = storage_cache.schema_storage()

                    # Check for missing definitions
                    current_schemas = schema_storage.get_table_schemas(candidate_tables)
                    missing_tables = []
                    invalid_tables = []

                    # ✅ Fixed: Convert PyArrow Table to list of dicts for proper iteration
                    # get_table_schemas() returns pa.Table, iterating directly yields ChunkedArray (columns)
                    # to_pylist() converts to list of dicts where each dict represents a row
                    schema_dicts = current_schemas.to_pylist()
                    definition_by_table = {
                        str(schema.get("table_name", "")): schema.get("definition") for schema in schema_dicts
                    }
                    for table_name in candidate_tables:
                        # Handle potential prefix
                        lookup_name = table_name.split(".")[-1] if "." in table_name else table_name
                        definition = definition_by_table.get(lookup_name)

                        if not definition or not str(definition).strip():
                            missing_tables.append(table_name)
                            continue

                        definition_text = str(definition)
                        if ddl_has_missing_commas(definition_text) or is_likely_truncated_ddl(definition_text):
                            invalid_tables.append(table_name)

                    if missing_tables or invalid_tables:
                        logger.info(
                            "Found %d tables with missing metadata and %d with invalid DDL. Attempting repair...",
                            len(missing_tables),
                            len(invalid_tables),
                        )
                        tables_to_repair = sorted(set(missing_tables + invalid_tables))
                        await self._repair_metadata(tables_to_repair, schema_storage, task)
                        logger.info("Retrying DDL fallback for tables with missing or invalid metadata...")
                        await self._ddl_fallback_and_retry(tables_to_repair, task)

                except Exception as e:
                    logger.warning(f"Metadata repair pre-check failed: {e}")
            # --- Repair Logic End ---

            # Use existing SchemaWithValueRAG to load table schemas
            if self.agent_config:
                rag = SchemaWithValueRAG(agent_config=self.agent_config)

                # Determine target database for search
                # If multi-db mode, use empty string to search across all databases
                target_database = "" if is_multi_db else (task.database_name or "")

                schemas, values = rag.search_tables(
                    tables=candidate_tables,
                    catalog_name=task.catalog_name or "",
                    database_name=target_database,
                    schema_name=task.schema_name or "",
                    dialect=task.database_type if hasattr(task, "database_type") else None,
                )

                if schemas:
                    # ✅ Use thread-safe context update
                    def update_schemas():
                        self.workflow.context.update_schema_and_values(schemas, values)
                        return {"loaded_count": len(schemas)}

                    result = safe_context_update(
                        self.workflow.context,
                        update_schemas,
                        operation_name="load_table_schemas",
                    )

                    if result["success"]:
                        logger.debug(f"Loaded schemas for {result['result']['loaded_count']} tables")
                    else:
                        logger.warning(f"Context update failed: {result.get('error')}")
                else:
                    logger.warning(f"No schemas found for candidate tables: {candidate_tables}")

                    # DDL Fallback: If storage is empty, retrieve DDL from database and populate storage
                    if candidate_tables:
                        logger.info("Schema storage empty, attempting DDL retrieval fallback...")
                        await self._ddl_fallback_and_retry(candidate_tables, task)

        except Exception as e:
            logger.warning(f"Failed to load table schemas: {e}")
            # Don't fail the entire node for schema loading issues

    async def _repair_metadata(self, table_names: List[str], schema_storage, task) -> int:
        """
        Attempt to repair missing metadata by fetching DDL directly from the database.
        """
        repaired_count = 0
        try:
            db_manager = get_db_manager()
            # Use the configured connection for the current task
            connector = db_manager.get_conn(self.agent_config.current_namespace, task.database_name)

            if not connector:
                return 0

            per_table_supported = hasattr(connector, "get_table_ddl") or hasattr(connector, "get_ddl")

            if per_table_supported:
                for table in table_names:
                    try:
                        ddl = None
                        if hasattr(connector, "get_table_ddl"):
                            ddl = connector.get_table_ddl(table)
                        elif hasattr(connector, "get_ddl"):
                            ddl = connector.get_ddl(table)

                        if ddl:
                            schema_storage.update_table_schema(table, ddl)
                            repaired_count += 1
                            logger.info(f"Repaired metadata for table: {table}")
                    except Exception as e:
                        logger.debug(f"Failed to repair metadata for table {table}: {e}")
            elif hasattr(connector, "get_tables_with_ddl"):
                tables_with_ddl = connector.get_tables_with_ddl()
                ddl_by_table = {}
                for tbl_info in tables_with_ddl:
                    table_name = tbl_info.get("table_name") or tbl_info.get("name")
                    if table_name:
                        ddl_by_table[str(table_name)] = tbl_info.get("ddl", "")
                for table in table_names:
                    ddl = ddl_by_table.get(table)
                    if ddl:
                        schema_storage.update_table_schema(table, ddl)
                        repaired_count += 1
                        logger.info(f"Repaired metadata for table: {table}")

        except Exception as e:
            logger.error(f"Metadata repair process failed: {e}")

        return repaired_count

    async def _ddl_fallback_and_retry(self, candidate_tables: List[str], task) -> None:
        """
        Fallback to retrieve DDL from database and populate storage when SchemaStorage is empty.
        Supports multi-database configuration by iterating through all configured databases.

        This method:
        1. Connects to the database(s) using the configured connector(s)
        2. Retrieves DDL for all tables (or candidate tables)
        3. Populates SchemaStorage with the retrieved DDL
        4. Retries schema search with populated storage

        Args:
            candidate_tables: List of table names to search for
            task: The current task with database connection info
        """
        try:
            db_manager = get_db_manager()
            
            # Check if multiple databases are configured
            db_configs = self.agent_config.current_db_configs()
            is_multi_db = len(db_configs) > 1

            # Initialize RAG for storage operations
            rag = SchemaWithValueRAG(agent_config=self.agent_config)
            schemas_to_store = []
            seen_identifiers = set()
            
            # Define list of (connector, db_name, catalog, schema) tuples to process
            connectors_to_process = []
            
            if is_multi_db:
                for logic_name, db_config in db_configs.items():
                    conn = db_manager.get_conn(self.agent_config.current_namespace, logic_name)
                    if conn:
                        connectors_to_process.append((conn, logic_name, db_config.catalog, db_config.schema))
            else:
                conn = db_manager.get_conn(self.agent_config.current_namespace, task.database_name)
                if conn:
                    connectors_to_process.append((conn, task.database_name, task.catalog_name, task.schema_name))
            
            if not connectors_to_process:
                logger.warning(f"Could not get any database connectors")
                return

            for connector, db_name, catalog, schema in connectors_to_process:
                try:
                    dialect = getattr(connector, "dialect", None) or getattr(task, "database_type", None)
                    db_candidates = self._filter_candidate_tables_for_db(candidate_tables, db_name, dialect)
                    if db_candidates and (hasattr(connector, "get_table_ddl") or hasattr(connector, "get_ddl")):
                        logger.info(f"Retrieving DDL for candidate tables individually from {db_name}...")
                        lookup_seen = set()
                        for table_name in db_candidates:
                            lookup_name = table_name.split(".")[-1] if "." in table_name else table_name
                            if lookup_name in lookup_seen:
                                continue
                            lookup_seen.add(lookup_name)
                            try:
                                ddl = None
                                if hasattr(connector, "get_table_ddl"):
                                    ddl = connector.get_table_ddl(lookup_name)
                                elif hasattr(connector, "get_ddl"):
                                    ddl = connector.get_ddl(lookup_name)

                                if ddl:
                                    identifier = f"{catalog or ''}.{db_name}..{lookup_name}.table"
                                    if identifier in seen_identifiers:
                                        continue
                                    seen_identifiers.add(identifier)
                                    schemas_to_store.append(
                                        {
                                            "identifier": identifier,
                                            "catalog_name": catalog or "",
                                            "database_name": db_name,
                                            "schema_name": schema or "",
                                            "table_name": lookup_name,
                                            "table_type": "table",
                                            "definition": ddl,
                                        }
                                    )
                            except Exception as e:
                                logger.debug(f"Failed to get DDL for table {lookup_name} from {db_name}: {e}")
                    elif hasattr(connector, "get_tables_with_ddl"):
                        if candidate_tables and not db_candidates:
                            logger.info(f"Skipping full DDL scan for {db_name}; no candidates for this database")
                            continue
                        logger.info(f"Using get_tables_with_ddl to retrieve all table DDLs from {db_name}...")
                        tables_with_ddl = connector.get_tables_with_ddl()
                        
                        for tbl_info in tables_with_ddl:
                            table_name = tbl_info.get("name", "")
                            if table_name:
                                identifier = f"{catalog or ''}.{db_name}..{table_name}.table"
                                if identifier in seen_identifiers:
                                    continue
                                seen_identifiers.add(identifier)
                                schemas_to_store.append(
                                    {
                                        "identifier": identifier,
                                        "catalog_name": catalog or "",
                                        "database_name": db_name,
                                        "schema_name": schema or "",
                                        "table_name": table_name,
                                        "table_type": "table",
                                        "definition": tbl_info.get("ddl", ""),
                                    }
                                )
                except Exception as e:
                    logger.warning(f"Error processing database {db_name}: {e}")

            # Store retrieved schemas in storage
            if schemas_to_store:
                logger.info(f"Populating SchemaStorage with {len(schemas_to_store)} schemas...")
                rag.store_batch(schemas_to_store, [])

                # Retry schema search with populated storage
                target_database = "" if is_multi_db else (task.database_name or "")
                target_catalog = "" if is_multi_db else (task.catalog_name or "")
                target_schema = "" if is_multi_db else (task.schema_name or "")
                
                schemas, values = rag.search_tables(
                    tables=candidate_tables,
                    catalog_name=target_catalog,
                    database_name=target_database,
                    schema_name=target_schema,
                    dialect=task.database_type if hasattr(task, "database_type") else None,
                )

                if schemas:
                    # Update workflow context with retrieved schemas
                    def update_schemas():
                        self.workflow.context.update_schema_and_values(schemas, values)
                        return {"loaded_count": len(schemas)}

                    result = safe_context_update(
                        self.workflow.context,
                        update_schemas,
                        operation_name="ddl_fallback_load_schemas",
                    )

                    if result["success"]:
                        logger.info(f"✅ Successfully loaded {len(schemas)} schemas via DDL fallback")
                    else:
                        logger.warning(f"Context update failed after DDL fallback: {result.get('error')}")
                else:
                    logger.warning(
                        f"DDL fallback retrieved schemas but still no match for candidates: {candidate_tables}"
                    )
            else:
                # Enhanced fallback when get_tables_with_ddl returns 0 tables
                for connector, db_name, catalog, schema in connectors_to_process:
                    await self._enhanced_ddl_fallback(
                        candidate_tables,
                        task,
                        connector,
                        db_name=db_name,
                        catalog_name=catalog,
                        schema_name=schema,
                    )

        except Exception as e:
            logger.warning(f"DDL fallback failed: {e}")

    def _filter_candidate_tables_for_db(
        self,
        candidate_tables: List[str],
        db_name: str,
        dialect: Optional[str],
    ) -> List[str]:
        """
        Filter candidate tables to those explicitly matching the current database.

        If no explicit database-qualified candidates match, return unqualified entries
        so we still attempt a best-effort fallback.
        """
        explicit_matches: List[str] = []
        implicit_matches: List[str] = []

        for table in candidate_tables:
            parts = table.split(".")
            db_candidate = None
            if len(parts) >= 4:
                db_candidate = parts[1]
            elif len(parts) == 3:
                if dialect == DBType.STARROCKS:
                    db_candidate = parts[1]
                else:
                    db_candidate = parts[0]
            elif len(parts) == 2:
                if dialect in (DBType.SQLITE, DBType.MYSQL, DBType.STARROCKS):
                    db_candidate = parts[0]

            if db_candidate:
                if db_candidate == db_name:
                    explicit_matches.append(table)
            else:
                implicit_matches.append(table)

        return explicit_matches if explicit_matches else implicit_matches

    async def _enhanced_ddl_fallback(
        self,
        candidate_tables: List[str],
        task,
        connector,
        db_name: str = None,
        catalog_name: str = "",
        schema_name: str = "",
    ) -> None:
        """
        Enhanced fallback strategy when get_tables_with_ddl returns 0 tables.

        This method implements a multi-tier fallback:
        1. Use get_tables() to get all table names
        2. Try to get DDL for each candidate table individually
        3. Try get_table_schema() to build DDL from column info
        4. Store minimal schema entries as last resort

        Args:
            candidate_tables: List of table names to search for
            task: The current task
            connector: The database connector instance
            db_name: Optional database name override (for multi-db support)
        """
        try:
            logger.info(f"Attempting enhanced DDL fallback strategy for database: {db_name or task.database_name}...")

            # Use provided db_name or fall back to task.database_name
            current_db = db_name or task.database_name or ""
            current_catalog = catalog_name or task.catalog_name or ""
            current_schema = schema_name or task.schema_name or ""
            dialect = getattr(connector, "dialect", None) or getattr(task, "database_type", None)

            # Check if multiple databases are configured (for target_database logic)
            db_configs = self.agent_config.current_db_configs()
            is_multi_db = len(db_configs) > 1

            # Strategy 1: Retrieve DDL for candidates (or limited tables if no candidates)
            try:
                candidate_limit = 50
                if self.agent_config and hasattr(self.agent_config, "schema_discovery_config"):
                    candidate_limit = self.agent_config.schema_discovery_config.fallback_table_limit

                if candidate_tables:
                    target_tables = [name.split(".")[-1] for name in candidate_tables]
                else:
                    all_tables = connector.get_tables(
                        catalog_name=current_catalog,
                        database_name=current_db,
                        schema_name=current_schema,
                    )
                    if len(all_tables) > candidate_limit:
                        logger.warning(
                            f"Too many tables ({len(all_tables)}), limiting to first {candidate_limit} for DDL fallback"
                        )
                        all_tables = all_tables[:candidate_limit]
                    target_tables = all_tables

                if target_tables:
                    logger.info(
                        f"Trying individual DDL retrieval for {len(target_tables)} tables in {current_db}..."
                    )

                    rag = SchemaWithValueRAG(agent_config=self.agent_config)
                    schemas_to_store = []
                    retrieved_count = 0

                    # Try to get DDL for each table
                    for table_name in target_tables:
                        try:
                            # Try different DDL retrieval methods
                            ddl = None

                            # Method 1: get_table_ddl
                            if hasattr(connector, "get_table_ddl"):
                                try:
                                    ddl = connector.get_table_ddl(table_name)
                                except Exception as e:
                                    logger.debug(f"get_table_ddl failed for {table_name}: {e}")

                            # Method 2: get_ddl
                            if not ddl and hasattr(connector, "get_ddl"):
                                try:
                                    ddl = connector.get_ddl(table_name)
                                except Exception as e:
                                    logger.debug(f"get_ddl failed for {table_name}: {e}")

                            # Method 3: Build DDL from schema
                            if not ddl and hasattr(connector, "get_schema"):
                                try:
                                    schema_info = connector.get_schema(table_name=table_name)
                                    if schema_info:
                                        ddl = self._build_ddl_from_schema(table_name, schema_info)
                                except Exception as e:
                                    logger.debug(f"get_schema failed for {table_name}: {e}")

                            if ddl:
                                schemas_to_store.append(
                                    {
                                        "identifier": f"{current_catalog or ''}.{current_db}..{table_name}.table",
                                        "catalog_name": current_catalog or "",
                                        "database_name": current_db,
                                        "schema_name": current_schema or "",
                                        "table_name": table_name,
                                        "table_type": "table",
                                        "definition": ddl,
                                    }
                                )
                                retrieved_count += 1

                        except Exception as e:
                            logger.debug(f"Failed to get DDL for table {table_name}: {e}")

                    if retrieved_count > 0:
                        logger.info(f"Enhanced fallback retrieved DDL for {retrieved_count} tables from {current_db}")
                        rag.store_batch(schemas_to_store, [])

                        # Retry schema search
                        target_database = "" if is_multi_db else (task.database_name or "")
                        target_catalog = "" if is_multi_db else (task.catalog_name or "")
                        target_schema = "" if is_multi_db else (task.schema_name or "")

                        schemas, values = rag.search_tables(
                            tables=candidate_tables,
                            catalog_name=target_catalog,
                            database_name=target_database,
                            schema_name=target_schema,
                            dialect=task.database_type if hasattr(task, "database_type") else None,
                        )

                        if schemas:
                            logger.info(
                                f"Schema search successful after enhanced fallback: found {len(schemas)} schemas"
                            )
                            return

            except Exception as e:
                logger.warning(f"Enhanced DDL fallback strategy failed for {current_db}: {e}")

            # Strategy 2: Final fallback - store just table names for LLM reference
            if candidate_tables:
                logger.warning(f"All DDL retrieval attempts failed for {current_db}, storing table names for LLM reference")
                rag = SchemaWithValueRAG(agent_config=self.agent_config)
                db_candidates = self._filter_candidate_tables_for_db(candidate_tables, current_db, dialect)
                candidate_lookup = {name.split(".")[-1] for name in db_candidates}

                # Create minimal schema entries with just table names
                minimal_schemas = []
                for table_name in candidate_lookup:
                    if table_name in candidate_lookup:
                        minimal_schemas.append(
                            {
                                "identifier": f"{current_catalog or ''}.{current_db}..{table_name}.table",
                                "catalog_name": current_catalog or "",
                                "database_name": current_db,
                                "schema_name": current_schema or "",
                                "table_name": table_name,
                                "table_type": "table",
                                "definition": f"CREATE TABLE {table_name} (-- DDL retrieval failed, table name only)",
                            }
                        )

                if minimal_schemas:
                    rag.store_batch(minimal_schemas, [])
                    logger.info(f"Stored {len(minimal_schemas)} minimal schema entries for {current_db}")

        except Exception as e:
            logger.error(f"Enhanced DDL fallback failed for {db_name}: {e}")

    def _build_ddl_from_schema(self, table_name: str, schema_info: List[Dict[str, Any]]) -> str:
        """
        Build a basic DDL statement from schema information.

        Args:
            table_name: Name of the table
            schema_info: List of column information dictionaries

        Returns:
            A basic CREATE TABLE DDL statement
        """
        try:
            columns = []
            for col in schema_info:
                col_name = col.get("name", "")
                col_type = col.get("type", "")
                nullable = col.get("nullable", True)
                default = col.get("default_value")

                column_def = f"`{col_name}` {col_type}"
                if not nullable:
                    column_def += " NOT NULL"
                if default is not None and default != "":
                    column_def += f" DEFAULT {default}"

                columns.append(column_def)

            if columns:
                return f"CREATE TABLE `{table_name}` (\n  " + ",\n  ".join(columns) + "\n)"
            else:
                return f"CREATE TABLE `{table_name}` (-- Schema retrieval incomplete)"

        except Exception as e:
            logger.warning(f"Failed to build DDL from schema: {e}")
            return f"CREATE TABLE `{table_name}` (-- DDL reconstruction failed)"

    def execute(self) -> BaseResult:
        """
        Execute schema discovery synchronously.

        Returns:
            BaseResult: The result of schema discovery execution
        """
        # execute_with_async_stream ensures self.result is properly set
        return execute_with_async_stream(self)

    async def execute_stream(
        self, action_history_manager: Optional[ActionHistoryManager] = None
    ) -> AsyncGenerator[ActionHistory, None]:
        """
        Execute schema discovery with streaming support.

        Args:
            action_history_manager: Manager for tracking action history

        Yields:
            ActionHistory: Progress and result actions during execution
        """
        # Set the action_history_manager if provided
        if action_history_manager:
            self.action_history_manager = action_history_manager

        # Delegate to the existing run method
        async for action in self.run():
            yield action

    def update_context(self, workflow: "Workflow") -> Dict[str, Any]:
        """
        Update workflow context with schema discovery results.

        ✅ Fixed: Uses thread-safe context update to prevent race conditions.

        Args:
            workflow: The workflow instance to update

        Returns:
            Dict with success status and message
        """
        try:
            if not self.result or not self.result.success:
                return {"success": False, "message": "Schema discovery failed, cannot update context"}

            # If result has schema information, update workflow context
            if hasattr(self.result, "data") and self.result.data:
                output = self.result.data

                # ✅ Thread-safe metadata update
                if "candidate_tables" in output:

                    def update_metadata():
                        # Store discovered tables in workflow metadata for downstream nodes
                        if not hasattr(workflow, "metadata"):
                            workflow.metadata = {}
                        workflow.metadata["discovered_tables"] = output["candidate_tables"]
                        if output.get("table_candidates"):
                            workflow.metadata["discovered_table_details"] = output["table_candidates"]
                        if output.get("discovery_stats"):
                            workflow.metadata["schema_discovery_stats"] = output["discovery_stats"]
                        return {"table_count": len(output["candidate_tables"])}

                    result = safe_context_update(
                        workflow,
                        update_metadata,
                        operation_name="update_discovered_tables",
                    )

                    if not result["success"]:
                        logger.warning(f"Metadata update failed: {result.get('error')}")

                # Note: Table schemas are already loaded via _load_table_schemas() which calls
                # workflow.context.update_schema_and_values(). The schema loading happens
                # during the run() method before this update_context() is called.
                if "table_schemas" in output and workflow.context:
                    # Schemas were already loaded by _load_table_schemas(), log for verification
                    schema_count = len(workflow.context.table_schemas)
                    value_count = len(workflow.context.table_values)
                    logger.debug(f"Schema discovery context contains {schema_count} schemas and {value_count} values")

            return {
                "success": True,
                "message": f"Schema discovery context updated with {len(self.result.data.get('candidate_tables', []))} tables",
            }

        except Exception as e:
            logger.error(f"Failed to update schema discovery context: {str(e)}")
            return {"success": False, "message": f"Schema discovery context update failed: {str(e)}"}

    # ===== SchemaLinkingNode Integration Methods (v2.5) =====

    def _apply_progressive_matching(self, base_rate: str = "fast") -> str:
        """
        Calculate final matching rate based on reflection round (from SchemaLinkingNode).

        Progressive matching strategy:
        - Round 0: fast (5 tables)
        - Round 1: medium (10 tables)
        - Round 2: slow (20 tables)
        - Round 3+: from_llm (LLM-based matching)

        Args:
            base_rate: Base matching rate (fast/medium/slow/from_llm)

        Returns:
            Final matching rate adjusted for reflection round
        """
        if not self.workflow or not self.agent_config:
            return base_rate

        # Check if progressive matching is enabled (with backward compatibility)
        if not hasattr(self.agent_config, "schema_discovery_config"):
            return base_rate

        if not self.agent_config.schema_discovery_config.progressive_matching_enabled:
            return base_rate

        # Get matching rates from configuration
        matching_rates = self.agent_config.schema_discovery_config.matching_rates

        # Find base rate index
        try:
            start_idx = matching_rates.index(base_rate) if base_rate in matching_rates else 0
        except ValueError:
            start_idx = 0

        # Get reflection round
        reflection_round = getattr(self.workflow, "reflection_round", 0)

        # Calculate final rate index
        final_idx = min(start_idx + reflection_round, len(matching_rates) - 1)
        final_rate = matching_rates[final_idx]

        logger.info(f"Progressive matching: base={base_rate}, round={reflection_round}, final={final_rate}")

        return final_rate

    async def _search_and_enhance_external_knowledge(
        self, user_query: str, subject_path: Optional[List[str]] = None
    ) -> str:
        """
        Search for relevant external knowledge based on user query (from SchemaLinkingNode).

        This method enhances the query context with domain knowledge before schema discovery.

        Args:
            user_query: The user's natural language query
            subject_path: Optional subject hierarchy path (e.g., ['Finance', 'Revenue', 'Q1'])

        Returns:
            Formatted string of relevant knowledge entries
        """
        try:
            if not self.agent_config:
                return ""

            # Check if external knowledge enhancement is enabled (with backward compatibility)
            if not hasattr(self.agent_config, "schema_discovery_config"):
                return ""

            if not self.agent_config.schema_discovery_config.external_knowledge_enabled:
                logger.debug("External knowledge enhancement is disabled")
                return ""

            context_search_tools = ContextSearchTools(self.agent_config)

            # Check if external knowledge store is available
            if not context_search_tools.has_ext_knowledge:
                logger.debug("External knowledge store is empty, skipping search")
                return ""

            # Convert subject_path to domain/layer1/layer2 format
            domain = subject_path[0] if len(subject_path) > 0 else ""
            layer1 = subject_path[1] if len(subject_path) > 1 else ""
            layer2 = subject_path[2] if len(subject_path) > 2 else ""

            # Get top_n from configuration (with backward compatibility)
            if hasattr(self.agent_config, "schema_discovery_config"):
                top_n = self.agent_config.schema_discovery_config.external_knowledge_top_n
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

    def _combine_knowledge(self, original: str, enhanced: str) -> str:
        """
        Combine original knowledge and searched knowledge (from SchemaLinkingNode).

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

    async def _llm_based_schema_matching(self, query: str, matching_rate: str = "from_llm") -> List[str]:
        """
        Use LLM-based schema matching for large datasets (from SchemaLinkingNode).

        This method leverages MatchSchemaTool for intelligent table selection
        when dealing with databases containing many tables.

        Args:
            query: User query text
            matching_rate: Matching rate strategy (should be "from_llm")

        Returns:
            List of selected table names
        """
        if not self.agent_config or matching_rate != "from_llm":
            return []

        # Check if LLM matching is enabled (with backward compatibility)
        if not hasattr(self.agent_config, "schema_discovery_config"):
            return []

        if not self.agent_config.schema_discovery_config.llm_matching_enabled:
            logger.debug("LLM-based schema matching is disabled")
            return []

        try:
            from datus.schemas.schema_linking_node_models import \
                SchemaLinkingInput
            from datus.tools.lineage_graph_tools.schema_lineage import \
                SchemaLineageTool

            # Get task from workflow
            task = self.workflow.task if self.workflow else None
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
            tool = SchemaLineageTool(agent_config=self.agent_config)
            result = tool.execute(linking_input, self.model)

            if result.success and result.table_schemas:
                table_names = [schema.table_name for schema in result.table_schemas]
                logger.info(f"LLM-based matching found {len(table_names)} tables")
                return table_names

            return []

        except Exception as e:
            logger.warning(f"LLM-based schema matching failed: {e}")
            return []
