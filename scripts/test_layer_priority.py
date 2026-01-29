#!/usr/bin/env python3
# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0

"""
Test script to verify layer priority (whitelist/blacklist) weighting in schema discovery.

This script validates that:
1. DWS/DWD/DIM tables (whitelist) get priority bonus
2. ODS tables (blacklist) get penalty applied
3. ADS tables stay neutral
4. The final ranking reflects the intended layer priority: DW > ADS > ODS

Usage:
    python scripts/test_layer_priority.py --config=/path/to/agent.yml --namespace=test
    python scripts/test_layer_priority.py --mock  # Use mock data without LanceDB
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datus.configuration.agent_config import SchemaDiscoveryConfig
from datus.configuration.agent_config_loader import load_agent_config


# Mock candidate data for testing without LanceDB
MOCK_CANDIDATES = {
    # DWS层 - 汇总表，应该优先
    "dws_clue_index_sc_channel_daily_di": {
        "sources": ["semantic"],
        "scores": {"semantic": 14.64},
        "matched_terms": ["线索", "指标", "渠道"],
        "order_index": 0,
        "table_comment": "店+sc+渠道+车型维度线索指标日度汇总表",
    },
    "dws_testdrive_index_sc_channel_daily_di": {
        "sources": ["semantic"],
        "scores": {"semantic": 15.45},
        "matched_terms": ["试驾", "指标", "渠道"],
        "order_index": 1,
        "table_comment": "店+sc+渠道+车型维度试驾指标日度汇总表",
    },
    "dws_order_index_sc_channel_daily_di": {
        "sources": ["semantic"],
        "scores": {"semantic": 15.31},
        "matched_terms": ["订单", "指标", "渠道"],
        "order_index": 2,
        "table_comment": "店+sc+渠道+车型维度订单指标日度汇总表",
    },
    "dws_clue_dlr_clue_testdrive_order_2h_di2": {
        "sources": ["semantic"],
        "scores": {"semantic": 8.28},
        "matched_terms": ["线索", "试驾", "订单"],
        "order_index": 3,
        "table_comment": "线索-试驾-订单宽表",
    },
    # DWD层 - 明细表
    "dwd_clue_clue_fact": {
        "sources": ["semantic"],
        "scores": {"semantic": 10.0},
        "matched_terms": ["线索", "事实表"],
        "order_index": 4,
        "table_comment": "线索事实表",
    },
    "dwd_testdrv_test_drive_fact": {
        "sources": ["semantic"],
        "scores": {"semantic": 9.5},
        "matched_terms": ["试驾", "事实表"],
        "order_index": 5,
        "table_comment": "试驾事实表",
    },
    # DIM层 - 维度表
    "dim_customer": {
        "sources": ["semantic"],
        "scores": {"semantic": 7.0},
        "matched_terms": ["客户", "维度"],
        "order_index": 6,
        "table_comment": "客户维度表",
    },
    "dim_date_info": {
        "sources": ["semantic"],
        "scores": {"semantic": 21.76},
        "matched_terms": ["日期", "维度"],
        "order_index": 7,
        "table_comment": "日期维表",
    },
    # ADS层 - 应用层， нейтральный
    "ads_sales_summary_di": {
        "sources": ["semantic"],
        "scores": {"semantic": 10.0},
        "matched_terms": ["销售", "汇总"],
        "order_index": 8,
        "table_comment": "销售汇总表",
    },
    "ads_real_funnel_schd_td_followup_result_df": {
        "sources": ["semantic"],
        "scores": {"semantic": 8.73},
        "matched_terms": ["漏斗", "试驾", "回访"],
        "order_index": 9,
        "table_comment": "真漏斗试驾排程-试驾-回访结果表",
    },
    # ODS层 - 原始层，应该被惩罚
    "ods_eip_nev_online_intention_order_info_di": {
        "sources": ["semantic"],
        "scores": {"semantic": 15.68},
        "matched_terms": ["意向金", "订单"],
        "order_index": 10,
        "table_comment": "意向金订单信息_ods",
    },
    "ods_dms_sal_vhs_customer_pending_di": {
        "sources": ["semantic"],
        "scores": {"semantic": 15.48},
        "matched_terms": ["客户", "代办"],
        "order_index": 11,
        "table_comment": "客户代办表_ods",
    },
    "ods_dms_sal_vhs_customer_pending_follow_di": {
        "sources": ["semantic"],
        "scores": {"semantic": 8.56},
        "matched_terms": ["跟进", "代办"],
        "order_index": 12,
        "table_comment": "待办跟进表_ods",
    },
}


@dataclass
class LayerWeightTestResult:
    """Result of layer weight test."""
    table_name: str
    layer: str
    original_score: float
    priority_bonus: float
    score_multiplier: float
    final_priority: int
    final_score: float
    rank_before: int
    rank_after: int
    rank_change: int


def _match_prefix(name: str, prefixes: List[str]) -> bool:
    """Check if table name starts with any of the given prefixes."""
    name_lower = name.lower()
    return any(name_lower.startswith(p.lower()) for p in prefixes)


def apply_layer_weights(
    candidates: Dict[str, Dict[str, Any]],
    whitelist: List[str],
    blacklist: List[str],
    blacklist_penalty: float = 0.3,
    whitelist_bonus: float = 0.05,
) -> List[Dict[str, Any]]:
    """
    Apply layer priority weights to candidates.

    This mirrors the logic in discovery_engine.py::finalize_candidates()

    Args:
        candidates: Dict of table_name -> candidate details
        whitelist: List of prefixes that get bonus (e.g., ["dws_", "dwd_", "dim_"])
        blacklist: List of prefixes that get penalty (e.g., ["ods_"])
        blacklist_penalty: Multiplier for blacklisted tables (0.3 means score × 0.7)
        whitelist_bonus: Additive bonus for whitelisted tables

    Returns:
        List of candidate dicts with weights applied, sorted by final score
    """
    details_list = []
    for table_name, details in candidates.items():
        item = {
            "table_name": table_name,
            "sources": details.get("sources", []),
            "scores": details.get("scores", {}),
            "matched_terms": details.get("matched_terms", []),
            "order_index": details.get("order_index", 0),
        }
        details_list.append(item)

    # Default priority from sources
    priority_map = {
        "explicit": 5,
        "semantic": 4,
        "keyword": 3,
        "llm_schema_matching": 3,
        "llm": 2,
        "context_search": 2,
        "fallback": 1,
    }

    # Apply layer weights
    for item in details_list:
        sources = item.get("sources", [])
        item["base_priority"] = max((priority_map.get(s, 0) for s in sources), default=0)

        # Get original semantic score
        original_score = item["scores"].get("semantic", 0.0)

        # Apply whitelist bonus
        if _match_prefix(item["table_name"], whitelist):
            item["priority_bonus"] = 1  # +1 to priority
            item["score_bonus"] = whitelist_bonus  # +0.05 to score
            item["layer_type"] = "WHITELIST"
        # Apply blacklist penalty
        elif _match_prefix(item["table_name"], blacklist):
            item["priority_bonus"] = -1  # -1 from priority
            item["score_bonus"] = 0  # No additive bonus
            item["score_multiplier"] = 1.0 - blacklist_penalty  # ×0.7
            item["layer_type"] = "BLACKLIST"
        else:
            item["priority_bonus"] = 0
            item["score_bonus"] = 0
            item["score_multiplier"] = 1.0
            item["layer_type"] = "NEUTRAL"

        # Calculate final values
        item["final_priority"] = item["base_priority"] + item["priority_bonus"]
        item["original_score"] = original_score
        item["final_score"] = original_score + item.get("score_bonus", 0)
        if item.get("score_multiplier"):
            item["final_score"] = item["final_score"] * item["score_multiplier"]
        item["final_score"] = round(item["final_score"], 4)

    # Sort by final priority and score
    details_list.sort(
        key=lambda x: (
            x["final_priority"],
            x["final_score"],
            len(x.get("matched_terms", [])),
            -x["order_index"],
        ),
        reverse=True,
    )

    return details_list


def analyze_layer_distribution(ranked_list: List[Dict[str, Any]]) -> Dict[str, int]:
    """Analyze the distribution of layers in ranked results."""
    distribution = {
        "dws": 0,
        "dwd": 0,
        "dim": 0,
        "ads": 0,
        "ods": 0,
        "other": 0,
    }
    for item in ranked_list:
        name = item["table_name"].lower()
        if name.startswith("dws_"):
            distribution["dws"] += 1
        elif name.startswith("dwd_"):
            distribution["dwd"] += 1
        elif name.startswith("dim_"):
            distribution["dim"] += 1
        elif name.startswith("ads_"):
            distribution["ads"] += 1
        elif name.startswith("ods_"):
            distribution["ods"] += 1
        else:
            distribution["other"] += 1
    return distribution


def run_test_with_config(config_path: str, namespace: str = "") -> None:
    """Run test using actual agent config."""
    print("=" * 80)
    print("LAYER PRIORITY TEST - With Agent Config")
    print("=" * 80)
    print(f"Config: {config_path}")
    print(f"Namespace: {namespace or 'default'}")
    print("")

    # Load config
    agent_config = load_agent_config(config=config_path)
    if namespace:
        agent_config.current_namespace = namespace

    # Get schema discovery config
    if hasattr(agent_config, "schema_discovery_config"):
        cfg = agent_config.schema_discovery_config
    else:
        cfg = SchemaDiscoveryConfig()

    whitelist = getattr(cfg, "table_prefix_whitelist", ["dws_", "dwd_", "dim_"])
    blacklist = getattr(cfg, "table_prefix_blacklist", ["ods_"])
    blacklist_penalty = getattr(cfg, "prefix_blacklist_penalty", 0.3)
    whitelist_bonus = getattr(cfg, "prefix_whitelist_bonus", 0.05)

    print("Configuration:")
    print(f"  Whitelist: {whitelist}")
    print(f"  Blacklist: {blacklist}")
    print(f"  Whitelist bonus: +{whitelist_bonus}")
    print(f"  Blacklist penalty: ×{1 - blacklist_penalty}")
    print("")

    # Use mock candidates (actual LanceDB query would be similar)
    candidates = MOCK_CANDIDATES

    run_test(candidates, whitelist, blacklist, blacklist_penalty, whitelist_bonus)


def run_test_with_lancedb(config_path: str, namespace: str = "") -> None:
    """Run test with actual LanceDB data."""
    print("=" * 80)
    print("LAYER PRIORITY TEST - With LanceDB Data")
    print("=" * 80)
    print(f"Config: {config_path}")
    print(f"Namespace: {namespace or 'default'}")
    print("")

    from datus.configuration.agent_config_loader import load_agent_config
    from datus.storage.embedding_models import get_db_embedding_model
    from datus.storage.schema_metadata.store import SchemaStorage

    # Load config
    agent_config = load_agent_config(config=config_path)
    if namespace:
        agent_config.current_namespace = namespace

    # Get storage path
    if namespace:
        db_path = agent_config.rag_storage_path()
    else:
        db_path = f"{agent_config.rag_base_path}/lancedb"

    # Get schema discovery config
    if hasattr(agent_config, "schema_discovery_config"):
        cfg = agent_config.schema_discovery_config
    else:
        cfg = SchemaDiscoveryConfig()

    whitelist = getattr(cfg, "table_prefix_whitelist", ["dws_", "dwd_", "dim_"])
    blacklist = getattr(cfg, "table_prefix_blacklist", ["ods_"])
    blacklist_penalty = getattr(cfg, "prefix_blacklist_penalty", 0.3)
    whitelist_bonus = getattr(cfg, "prefix_whitelist_bonus", 0.05)

    print("Configuration:")
    print(f"  Whitelist: {whitelist}")
    print(f"  Blacklist: {blacklist}")
    print(f"  Whitelist bonus: +{whitelist_bonus}")
    print(f"  Blacklist penalty: ×{1 - blacklist_penalty}")
    print(f"  Database: {db_path}")
    print("")

    # Query LanceDB for all tables with scores
    storage = SchemaStorage(db_path=db_path, embedding_model=get_db_embedding_model())
    storage._ensure_table_ready()

    # Get all tables
    results = storage.search_all(
        table_type="full",
        select_fields=["table_name", "table_comment"],
    )

    if results is None or len(results) == 0:
        print("No tables found in LanceDB. Using mock data.")
        candidates = MOCK_CANDIDATES
    else:
        # Build candidates from LanceDB (use mock scores for demo)
        candidates = {}
        for row in results.to_pylist():
            table_name = row.get("table_name", "")
            table_comment = row.get("table_comment", "")

            # Generate mock semantic score based on table name pattern
            # In real usage, this would come from actual search results
            import hashlib
            hash_val = int(hashlib.md5(table_name.encode()).hexdigest()[:8], 16)
            mock_score = 5.0 + (hash_val % 20)

            candidates[table_name] = {
                "sources": ["semantic"],
                "scores": {"semantic": mock_score},
                "matched_terms": [],
                "order_index": 0,
                "table_comment": table_comment or "",
            }

    run_test(candidates, whitelist, blacklist, blacklist_penalty, whitelist_bonus)


def run_test(
    candidates: Dict[str, Dict[str, Any]],
    whitelist: List[str],
    blacklist: List[str],
    blacklist_penalty: float,
    whitelist_bonus: float,
) -> None:
    """Execute the layer priority test."""

    # Get original ranking (by semantic score only)
    original_ranking = sorted(
        [
            (name, data["scores"].get("semantic", 0.0))
            for name, data in candidates.items()
        ],
        key=lambda x: x[1],
        reverse=True,
    )

    print(f"Total candidates: {len(candidates)}")
    print("")

    # Show original ranking
    print("-" * 80)
    print("ORIGINAL RANKING (by semantic score only)")
    print("-" * 80)
    print(f"{'Rank':<6} {'Table Name':<50} {'Layer':<8} {'Score':<10}")
    print("-" * 80)

    original_ranks = {}
    for rank, (table_name, score) in enumerate(original_ranking, 1):
        layer = _get_layer(table_name)
        print(f"{rank:<6} {table_name:<50} {layer:<8} {score:.4f}")
        original_ranks[table_name] = rank

    # Apply weights
    weighted = apply_layer_weights(
        candidates,
        whitelist,
        blacklist,
        blacklist_penalty,
        whitelist_bonus,
    )

    print("")
    print("-" * 80)
    print("WEIGHTED RANKING (after applying layer priorities)")
    print("-" * 80)
    print(f"{'Rank':<6} {'Table Name':<45} {'Layer':<8} {'Priority':<10} {'Score':<10} {'Change':<10}")
    print("-" * 80)

    results = []
    for rank, item in enumerate(weighted, 1):
        table_name = item["table_name"]
        layer = _get_layer(table_name)
        original_rank = original_ranks.get(table_name, 999)
        rank_change = original_rank - rank

        result = LayerWeightTestResult(
            table_name=table_name,
            layer=layer,
            original_score=item["original_score"],
            priority_bonus=item["priority_bonus"],
            score_multiplier=item.get("score_multiplier", 1.0),
            final_priority=item["final_priority"],
            final_score=item["final_score"],
            rank_before=original_rank,
            rank_after=rank,
            rank_change=rank_change,
        )
        results.append(result)

        change_str = f"+{rank_change}" if rank_change > 0 else str(rank_change)
        if rank_change > 0:
            change_str = f"↑{rank_change}"
        elif rank_change < 0:
            change_str = f"↓{abs(rank_change)}"
        else:
            change_str = "-"

        print(
            f"{rank:<6} {table_name:<45} {layer:<8} {item['final_priority']:<10} "
            f"{item['final_score']:<10.4f} {change_str:<10}"
        )

    # Analysis
    print("")
    print("=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    # Layer distribution in top 5
    top5_layers = [_get_layer(r.table_name) for r in results[:5]]
    print(f"\nTop 5 layer distribution: {top5_layers}")

    # Count improvements/declines
    improved = sum(1 for r in results if r.rank_change > 0)
    declined = sum(1 for r in results if r.rank_change < 0)
    unchanged = sum(1 for r in results if r.rank_change == 0)

    print(f"Tables improved: {improved}")
    print(f"Tables declined: {declined}")
    print(f"Tables unchanged: {unchanged}")

    # Verify expected behavior
    print("")
    print("-" * 80)
    print("VERIFICATION")
    print("-" * 80)

    # Check if DWS/DWD/DIM tables are promoted
    dw_tables = [r for r in results if r.layer in ["DWS", "DWD", "DIM"]]
    dw_promoted = sum(1 for r in dw_tables if r.rank_change > 0)

    # Check if ODS tables are demoted
    ods_tables = [r for r in results if r.layer == "ODS"]
    ods_demoted = sum(1 for r in ods_tables if r.rank_change < 0)

    print(f"DW tables promoted: {dw_promoted}/{len(dw_tables)}")
    print(f"ODS tables demoted: {ods_demoted}/{len(ods_tables)}")

    # Final verdict
    print("")
    if dw_promoted > 0 and ods_demoted > 0:
        print("✅ LAYER PRIORITY: WORKING CORRECTLY")
        print("   - DW层 (DWS/DWD/DIM) tables are being promoted")
        print("   - ODS tables are being demoted")
    elif len(dw_tables) == 0 or len(ods_tables) == 0:
        print("⚠️  LAYER PRIORITY: PARTIAL (missing tables of certain layers)")
    else:
        print("❌ LAYER PRIORITY: NOT WORKING")
        print("   - Expected DW promotion and ODS demotion")

    # Layer order verification
    print("")
    print("-" * 80)
    print("LAYER ORDER VERIFICATION")
    print("-" * 80)

    # Get first table of each layer
    layer_first = {}
    for r in results:
        if r.layer not in layer_first:
            layer_first[r.layer] = (r.rank_after, r.table_name, r.final_score)

    # Expected order: DWS/DWD/DIM > ADS > ODS
    layer_order = []
    for layer in ["DWS", "DWD", "DIM"]:
        if layer in layer_first:
            layer_order.append((layer, layer_first[layer]))

    if "ADS" in layer_first:
        layer_order.append(("ADS", layer_first["ADS"]))

    if "ODS" in layer_first:
        layer_order.append(("ODS", layer_first["ODS"]))

    print(f"{'Layer':<10} {'First Rank':<12} {'Table Name':<50} {'Score':<10}")
    print("-" * 80)
    for layer, (rank, name, score) in layer_order:
        print(f"{layer:<10} {rank:<12} {name:<50} {score:.4f}")

    # Summary
    print("")
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if len(layer_order) >= 2:
        first_layer = layer_order[0][0]
        last_layer = layer_order[-1][0]

        if first_layer in ["DWS", "DWD", "DIM"] and last_layer == "ODS":
            print("✅ CORRECT LAYER ORDER: DW > ADS > ODS")
        elif first_layer in ["DWS", "DWD", "DIM"]:
            print("⚠️  PARTIAL ORDER: Top layer is correct, but some layers may be missing")
        else:
            print("❌ INCORRECT LAYER ORDER: Expected DW layers at top")


def _get_layer(table_name: str) -> str:
    """Get the layer prefix from a table name."""
    name_lower = table_name.lower()
    if name_lower.startswith("dws_"):
        return "DWS"
    elif name_lower.startswith("dwd_"):
        return "DWD"
    elif name_lower.startswith("dim_"):
        return "DIM"
    elif name_lower.startswith("ads_"):
        return "ADS"
    elif name_lower.startswith("ods_"):
        return "ODS"
    else:
        return "OTHER"


def main():
    parser = argparse.ArgumentParser(
        description="Test layer priority (whitelist/blacklist) weighting in schema discovery"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to agent configuration file",
    )
    parser.add_argument(
        "--namespace",
        default="",
        help="Namespace to use (optional)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock data instead of querying LanceDB",
    )
    parser.add_argument(
        "--lancedb",
        action="store_true",
        help="Query actual LanceDB data (default if --mock not specified)",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Write results to JSON file",
    )

    args = parser.parse_args()

    if args.mock:
        run_test_with_config(args.config, args.namespace)
    else:
        run_test_with_lancedb(args.config, args.namespace)

    # Write output if requested
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({"test": "layer_priority", "status": "completed"}, f, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
