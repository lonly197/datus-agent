#!/usr/bin/env python3
# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Schema Enrichment Verification Script

验证 Schema 增强效果，对比增强前后的元数据质量。

使用示例：
    python scripts/verify_schema_enrichment.py \
        --config=/root/.datus/conf/agent.yml \
        --namespace=test \
        --before-backup=/root/.datus/data/datus_db_test.backup_v0_20260128_120000
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import lancedb
except ImportError:
    print("Error: lancedb is required")
    sys.exit(1)

from datus.configuration.agent_config_loader import load_agent_config
from datus.utils.loggings import configure_logging, get_logger

logger = get_logger(__name__)


def load_schema_data(db_path: str) -> List[Dict[str, Any]]:
    """从 LanceDB 加载 schema 数据"""
    try:
        db = lancedb.connect(db_path)
        if "schema_metadata" not in db.table_names():
            logger.error(f"schema_metadata table not found in {db_path}")
            return []
        
        table = db.open_table("schema_metadata")
        df = table.to_pandas()
        return df.to_dict('records')
    except Exception as e:
        logger.error(f"Failed to load schema data from {db_path}: {e}")
        return []


def calculate_metadata_score(record: Dict) -> Dict[str, Any]:
    """计算单条记录的元数据质量分数"""
    score = {
        "table_name": record.get("table_name", ""),
        "has_table_comment": False,
        "table_comment_quality": 0,
        "has_column_comments": False,
        "column_comment_coverage": 0.0,
        "has_enums": False,
        "enum_coverage": 0.0,
        "has_business_tags": False,
        "business_tags_count": 0,
        "has_relationships": False,
        "overall_score": 0.0,
    }
    
    # 表注释质量
    table_comment = record.get("table_comment", "") or ""
    if table_comment:
        score["has_table_comment"] = True
        # 基于长度评分：少于10字符低质量，10-30中等，30+高质量
        comment_len = len(table_comment)
        if comment_len >= 30:
            score["table_comment_quality"] = 3  # 高
        elif comment_len >= 10:
            score["table_comment_quality"] = 2  # 中
        else:
            score["table_comment_quality"] = 1  # 低
    
    # 列注释覆盖率
    col_comments_str = record.get("column_comments", "{}") or "{}"
    try:
        col_comments = json.loads(col_comments_str)
    except json.JSONDecodeError:
        col_comments = {}
    
    if col_comments:
        score["has_column_comments"] = True
        # 从 definition 估算总列数（简化处理）
        definition = record.get("definition", "") or ""
        estimated_columns = estimate_column_count(definition)
        if estimated_columns > 0:
            score["column_comment_coverage"] = len(col_comments) / estimated_columns
    
    # 枚举值覆盖率
    enums_str = record.get("column_enums", "{}") or "{}"
    try:
        enums = json.loads(enums_str)
    except json.JSONDecodeError:
        enums = {}
    
    if enums:
        score["has_enums"] = True
        estimated_columns = estimate_column_count(definition)
        if estimated_columns > 0:
            score["enum_coverage"] = len(enums) / estimated_columns
    
    # 业务标签
    business_tags = record.get("business_tags", []) or []
    if business_tags:
        score["has_business_tags"] = True
        score["business_tags_count"] = len(business_tags)
    
    # 关系元数据
    rel_str = record.get("relationship_metadata", "{}") or "{}"
    try:
        rel = json.loads(rel_str)
    except json.JSONDecodeError:
        rel = {}
    
    if rel and (rel.get("foreign_keys") or rel.get("join_paths")):
        score["has_relationships"] = True
    
    # 综合评分（满分100）
    overall = 0
    if score["has_table_comment"]:
        overall += 20 * score["table_comment_quality"] / 3
    if score["has_column_comments"]:
        overall += 25 * min(score["column_comment_coverage"], 1.0)
    if score["has_enums"]:
        overall += 15 * min(score["enum_coverage"], 1.0)
    if score["has_business_tags"]:
        overall += 20 * min(score["business_tags_count"] / 5, 1.0)  # 5个标签满分
    if score["has_relationships"]:
        overall += 20
    
    score["overall_score"] = round(overall, 1)
    return score


def estimate_column_count(definition: str) -> int:
    """从 DDL 估算列数"""
    if not definition:
        return 0
    # 简单估算：统计 CREATE TABLE 后的逗号数量
    # 这是一个粗略估计，实际应用中可能需要更精确的解析
    matches = re.findall(r'`\w+`\s+\w+', definition)
    return len(matches)


def compare_scores(before: List[Dict], after: List[Dict]) -> Dict[str, Any]:
    """对比增强前后的分数"""
    before_scores = [calculate_metadata_score(r) for r in before]
    after_scores = [calculate_metadata_score(r) for r in after]
    
    # 建立表名索引
    before_by_table = {s["table_name"]: s for s in before_scores}
    after_by_table = {s["table_name"]: s for s in after_scores}
    
    # 计算平均分数
    before_avg = sum(s["overall_score"] for s in before_scores) / len(before_scores) if before_scores else 0
    after_avg = sum(s["overall_score"] for s in after_scores) / len(after_scores) if after_scores else 0
    
    # 找出改进最大的表
    improvements = []
    for table_name, after_score in after_by_table.items():
        before_score = before_by_table.get(table_name, {}).get("overall_score", 0)
        if after_score["overall_score"] > before_score:
            improvements.append({
                "table": table_name,
                "before": before_score,
                "after": after_score["overall_score"],
                "improvement": round(after_score["overall_score"] - before_score, 1),
            })
    
    improvements.sort(key=lambda x: x["improvement"], reverse=True)
    
    return {
        "before_avg_score": round(before_avg, 1),
        "after_avg_score": round(after_avg, 1),
        "improvement": round(after_avg - before_avg, 1),
        "tables_improved": len(improvements),
        "top_improvements": improvements[:10],
        "coverage_changes": {
            "table_comment": {
                "before": sum(1 for s in before_scores if s["has_table_comment"]),
                "after": sum(1 for s in after_scores if s["has_table_comment"]),
            },
            "column_comments": {
                "before": sum(1 for s in before_scores if s["has_column_comments"]),
                "after": sum(1 for s in after_scores if s["has_column_comments"]),
            },
            "business_tags": {
                "before": sum(1 for s in before_scores if s["has_business_tags"]),
                "after": sum(1 for s in after_scores if s["has_business_tags"]),
            },
        }
    }


def print_comparison_report(comparison: Dict[str, Any]):
    """打印对比报告"""
    print("\n" + "=" * 80)
    print("SCHEMA ENRICHMENT VERIFICATION REPORT")
    print("=" * 80)
    print()
    
    print("Overall Score Comparison:")
    print(f"  Before: {comparison['before_avg_score']}/100")
    print(f"  After:  {comparison['after_avg_score']}/100")
    print(f"  Improvement: +{comparison['improvement']}")
    print()
    
    print("Coverage Changes:")
    for field, counts in comparison['coverage_changes'].items():
        before_pct = counts['before'] / max(comparison['tables_improved'], 1) * 100
        after_pct = counts['after'] / max(comparison['tables_improved'], 1) * 100
        print(f"  {field}:")
        print(f"    Before: {counts['before']} tables ({before_pct:.1f}%)")
        print(f"    After:  {counts['after']} tables ({after_pct:.1f}%)")
    print()
    
    print(f"Tables Improved: {comparison['tables_improved']}")
    print()
    
    if comparison['top_improvements']:
        print("Top 10 Improvements:")
        print(f"  {'Table':<50} {'Before':>8} {'After':>8} {'+/-':>8}")
        print("  " + "-" * 76)
        for imp in comparison['top_improvements']:
            print(f"  {imp['table']:<50} {imp['before']:>8.1f} {imp['after']:>8.1f} +{imp['improvement']:>7.1f}")
    
    print()
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Verify schema enrichment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("--config", required=True, help="Path to agent configuration file")
    parser.add_argument("--namespace", required=True, help="Target namespace")
    parser.add_argument("--before-backup", help="Path to backup database (before enrichment)")
    parser.add_argument("--current-db", help="Override current database path")
    parser.add_argument("--output-json", help="Save detailed results to JSON file")
    
    args = parser.parse_args()
    
    configure_logging(debug=False)
    
    # 加载配置
    agent_config = load_agent_config(config=args.config)
    agent_config.current_namespace = args.namespace
    
    current_db_path = args.current_db or agent_config.rag_storage_path()
    
    # 加载当前数据
    logger.info(f"Loading current schema data from: {current_db_path}")
    current_data = load_schema_data(current_db_path)
    
    if not current_data:
        logger.error("No current data found")
        sys.exit(1)
    
    logger.info(f"Loaded {len(current_data)} current records")
    
    # 如果有备份，进行对比
    if args.before_backup:
        logger.info(f"Loading backup data from: {args.before_backup}")
        backup_data = load_schema_data(args.before_backup)
        
        if backup_data:
            logger.info(f"Loaded {len(backup_data)} backup records")
            comparison = compare_scores(backup_data, current_data)
            print_comparison_report(comparison)
            
            if args.output_json:
                import json
                Path(args.output_json).write_text(
                    json.dumps(comparison, ensure_ascii=False, indent=2),
                    encoding="utf-8"
                )
                logger.info(f"Detailed results saved to: {args.output_json}")
        else:
            logger.warning("No backup data found, showing current state only")
            scores = [calculate_metadata_score(r) for r in current_data]
            avg_score = sum(s["overall_score"] for s in scores) / len(scores)
            print(f"\nCurrent average metadata score: {avg_score:.1f}/100")
    else:
        # 仅显示当前状态
        scores = [calculate_metadata_score(r) for r in current_data]
        avg_score = sum(s["overall_score"] for s in scores) / len(scores)
        
        print("\n" + "=" * 80)
        print("CURRENT SCHEMA METADATA QUALITY")
        print("=" * 80)
        print(f"Total tables: {len(current_data)}")
        print(f"Average score: {avg_score:.1f}/100")
        print()
        
        # 分布统计
        score_ranges = {
            "Excellent (80-100)": sum(1 for s in scores if 80 <= s["overall_score"] <= 100),
            "Good (60-79)": sum(1 for s in scores if 60 <= s["overall_score"] < 80),
            "Fair (40-59)": sum(1 for s in scores if 40 <= s["overall_score"] < 60),
            "Poor (20-39)": sum(1 for s in scores if 20 <= s["overall_score"] < 40),
            "Very Poor (0-19)": sum(1 for s in scores if 0 <= s["overall_score"] < 20),
        }
        
        print("Score Distribution:")
        for range_name, count in score_ranges.items():
            pct = count / len(scores) * 100
            bar = "█" * int(pct / 2)
            print(f"  {range_name:20} {count:4} ({pct:5.1f}%) {bar}")
        
        print("=" * 80)


if __name__ == "__main__":
    import re  # 需要导入 re 模块
    main()
