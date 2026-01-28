# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details

"""
Utility functions for schema discovery.

This module provides helper functions for system resource detection,
Chinese text processing, and query enhancement.
"""

import os
import subprocess
import sys
from typing import Any, Dict, List
from datus.utils.query_utils import rewrite_fts_query_with_llm

logger = None


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


def get_available_cpu_count() -> int:
    """
    Get the number of available CPU cores using process affinity.

    Returns:
        Number of available CPU cores, or 0 if detection fails
    """
    try:
        return len(os.sched_getaffinity(0))
    except Exception:
        return os.cpu_count() or 0


def get_available_memory_bytes() -> int:
    """
    Get available system memory in bytes.

    Returns:
        Available memory in bytes, or 0 if detection fails
    """
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


def check_rerank_resources(
    model_path: str, min_cpu_count: int, min_memory_gb: float
) -> Dict[str, Any]:
    """
    Check if system has sufficient resources for reranking.

    Args:
        model_path: Path to reranker model
        min_cpu_count: Minimum required CPU cores
        min_memory_gb: Minimum required memory in GB

    Returns:
        Dictionary with check results including 'ok' status and reasons
    """
    available_cpus = get_available_cpu_count()
    available_memory_bytes = get_available_memory_bytes()
    available_memory_gb = available_memory_bytes / (1024 ** 3) if available_memory_bytes else 0.0
    model_exists = bool(model_path) and os.path.exists(model_path)

    reasons = []
    if not model_exists:
        reasons.append(f"model_not_found:{model_path}")
    if min_cpu_count > 0 and available_cpus < min_cpu_count:
        reasons.append(f"cpu_insufficient:{available_cpus}<{min_cpu_count}")
    if min_memory_gb > 0 and available_memory_gb < min_memory_gb:
        reasons.append(f"memory_insufficient:{available_memory_gb:.2f}<{min_memory_gb}")

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


def rewrite_fts_query_with_llm_wrapper(
    query_text: str,
    agent_config,
    model: Any,
) -> str:
    """
    Rewrite FTS query using shared LLM-based query rewriting utility.

    Args:
        query_text: Original query text
        agent_config: Agent configuration
        model: LLM model instance

    Returns:
        Rewritten query text, or original if rewriting fails
    """
    if not query_text or not model:
        return query_text
    return rewrite_fts_query_with_llm(
        query=query_text,
        agent_config=agent_config,
        model_name="",
    )


def extract_potential_tables_from_text(text: str) -> List[str]:
    """
    Helper to extract potential table names (snake_case words) from text.

    Args:
        text: Text to parse for table names

    Returns:
        List of potential table names in snake_case format
    """
    import re
    words = re.findall(r"\b[a-z][a-z0-9_]{2,}\b", text)
    return [w for w in words if "_" in w]


def combine_knowledge(original: str, enhanced: str) -> str:
    """
    Combine original and enhanced external knowledge.

    Args:
        original: Original knowledge text
        enhanced: Enhanced knowledge text

    Returns:
        Combined knowledge text with duplicates removed
    """
    if not original:
        return enhanced
    if not enhanced:
        return original

    original_entries = set(original.split(chr(10)))
    enhanced_entries = set(enhanced.split(chr(10)))

    combined_entries = original_entries | enhanced_entries
    return chr(10).join(sorted(combined_entries))


def build_query_context(task) -> Dict[str, Any]:
    """
    Build query context from workflow metadata.

    Args:
        task: The SQL task with workflow metadata

    Returns:
        Dictionary with query context including business terms and entities
    """
    from datus.agent.workflow import Workflow

    context: Dict[str, Any] = {"task": task.task}
    if isinstance(task.workflow, Workflow) and hasattr(task.workflow, "metadata") and task.workflow.metadata:
        clarified_task = task.workflow.metadata.get("clarified_task")
        if clarified_task:
            context["clarified_task"] = clarified_task
        clarification = task.workflow.metadata.get("intent_clarification", {})
        if isinstance(clarification, dict):
            entities = clarification.get("entities", {})
            if entities:
                context["business_terms"] = entities.get("business_terms", [])
                context["dimensions"] = entities.get("dimensions", [])
                context["metrics"] = entities.get("metrics", [])
                context["time_range"] = entities.get("time_range")
    return context
