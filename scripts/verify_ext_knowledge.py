#!/usr/bin/env python3
# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
验证 ext_knowledge 表数据导入情况

该脚本用于验证业务术语和指标是否成功导入到 LanceDB 的 ext_knowledge 表中，
并提供详细的统计信息和搜索测试。

使用方法:
    python scripts/verify_ext_knowledge.py --config=conf/agent.yml --namespace=test

输出信息:
    - 表存在性检查
    - 记录总数统计
    - 按分类统计
    - 样本数据展示
    - 搜索功能测试
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import lancedb
from rich.console import Console
from rich.table import Table

from datus.configuration.agent_config_loader import load_agent_config
from datus.storage.ext_knowledge.store import ExtKnowledgeStore
from datus.utils.loggings import configure_logging, get_logger

logger = get_logger(__name__)
console = Console()


def check_ext_knowledge(db_path: str, test_queries: list = None) -> dict:
    """
    检查 ext_knowledge 表的数据情况

    Args:
        db_path: LanceDB 数据库路径
        test_queries: 测试搜索的查询列表

    Returns:
        检查结果字典
    """
    results = {
        "table_exists": False,
        "total_records": 0,
        "categories": {},
        "sample_entries": [],
        "search_tests": [],
        "errors": [],
    }

    try:
        db = lancedb.connect(db_path)
        table_names = db.table_names()

        if "ext_knowledge" not in table_names:
            results["errors"].append("ext_knowledge 表不存在")
            return results

        results["table_exists"] = True
        table = db.open_table("ext_knowledge")

        # 获取总记录数
        results["total_records"] = table.count_rows()

        if results["total_records"] == 0:
            results["errors"].append("ext_knowledge 表为空")
            return results

        # 获取样本数据
        sample = table.search().limit(10).to_arrow()
        sample_list = sample.to_pylist()

        # 统计分类
        categories = {}
        for entry in sample_list:
            # 尝试从 subject_node_id 或其他字段推断分类
            subj_id = entry.get("subject_node_id", "unknown")
            categories[subj_id] = categories.get(subj_id, 0) + 1

        results["categories"] = categories
        results["sample_entries"] = sample_list[:5]  # 前5条样本

        # 测试搜索功能
        if test_queries:
            ext_store = ExtKnowledgeStore(db_path=db_path)
            for query in test_queries:
                try:
                    search_results = ext_store.search_knowledge(query, top_n=3)
                    results["search_tests"].append({
                        "query": query,
                        "found": len(search_results),
                        "top_result": search_results[0] if search_results else None,
                    })
                except Exception as e:
                    results["search_tests"].append({
                        "query": query,
                        "error": str(e),
                    })

    except Exception as e:
        results["errors"].append(f"检查失败: {e}")
        import traceback
        logger.error(f"检查详情: {traceback.format_exc()}")

    return results


def print_results(results: dict, namespace: str = ""):
    """打印检查结果"""
    console.print("\n" + "=" * 80)
    console.print(f"[bold cyan]EXT_KNOWLEDGE 数据验证报告[/bold cyan]"
                  f"{' - ' + namespace if namespace else ''}")
    console.print("=" * 80)

    # 表存在性
    if results["table_exists"]:
        console.print(f"\n[bold green]✓[/bold green] ext_knowledge 表存在")
    else:
        console.print(f"\n[bold red]✗[/bold red] ext_knowledge 表不存在")
        return

    # 记录数
    total = results["total_records"]
    if total == 0:
        console.print(f"[bold red]✗[/bold red] 表为空（0 条记录）")
        return
    elif total < 10:
        console.print(f"[bold yellow]⚠[/bold yellow] 记录数: {total}（可能不完整）")
    else:
        console.print(f"[bold green]✓[/bold green] 记录数: {total}")

    # 样本数据
    console.print("\n[bold]样本数据（前5条）:[/bold]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Name", style="cyan", width=25)
    table.add_column("Terminology", style="green", width=20)
    table.add_column("Explanation", style="yellow", width=40)

    for entry in results["sample_entries"]:
        name = str(entry.get("name", "N/A"))[:23]
        term = str(entry.get("terminology", "N/A"))[:18]
        expl = str(entry.get("explanation", "N/A"))[:38]
        table.add_row(name, term, expl)

    console.print(table)

    # 分类统计
    if results["categories"]:
        console.print("\n[bold]分类统计:[/bold]")
        for cat, count in results["categories"].items():
            console.print(f"  • {cat}: {count} 条")

    # 搜索测试
    if results["search_tests"]:
        console.print("\n[bold]搜索功能测试:[/bold]")
        test_table = Table(show_header=True, header_style="bold blue")
        test_table.add_column("Query", style="cyan", width=35)
        test_table.add_column("Found", style="green", width=8)
        test_table.add_column("Top Result", style="yellow", width=30)

        for test in results["search_tests"]:
            query = test.get("query", "N/A")
            if "error" in test:
                test_table.add_row(query[:33], "[red]ERROR[/red]", test["error"][:28])
            else:
                found = str(test.get("found", 0))
                top = test.get("top_result", {})
                top_name = str(top.get("name", "N/A"))[:28] if top else "None"
                test_table.add_row(query[:33], found, top_name)

        console.print(test_table)

    # 错误信息
    if results["errors"]:
        console.print("\n[bold red]错误:[/bold red]")
        for error in results["errors"]:
            console.print(f"  [red]• {error}[/red]")

    console.print("\n" + "=" * 80)

    # 总结
    console.print("\n[bold]验证总结:[/bold]")
    if total > 0 and not results["errors"]:
        console.print("[bold green]✓ ext_knowledge 数据导入正常[/bold green]")
    elif total > 0:
        console.print("[bold yellow]⚠ ext_knowledge 有数据但存在警告[/bold yellow]")
    else:
        console.print("[bold red]✗ ext_knowledge 数据缺失，需要重新导入[/bold red]")
        console.print("\n[bold]建议操作:[/bold]")
        console.print("  1. 检查指标清单 Excel 文件路径是否正确")
        console.print("  2. 运行 generate_business_config.py 导入指标:")
        console.print("     python scripts/generate_business_config.py \\")
        console.print("       --config=/root/.datus/conf/agent.yml \\")
        console.print("       --namespace=<namespace> \\")
        console.print("       --metrics-xlsx=/path/to/指标清单.xlsx \\")
        console.print("       --import-to-lancedb \\")
        console.print("       --verbose")

    console.print()


def main():
    parser = argparse.ArgumentParser(
        description="验证 ext_knowledge 表数据导入情况",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基础验证
  python scripts/verify_ext_knowledge.py --config=conf/agent.yml --namespace=test

  # 带搜索测试
  python scripts/verify_ext_knowledge.py --config=conf/agent.yml \\
    --namespace=test \\
    --test-queries="线索统计,试驾转化,订单金额"

  # 详细日志
  python scripts/verify_ext_knowledge.py --config=conf/agent.yml --verbose
        """,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="配置文件路径 (如: /root/.datus/conf/agent.yml)",
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default="",
        help="命名空间（用于确定存储路径）",
    )
    parser.add_argument(
        "--test-queries",
        type=str,
        default="线索,试驾,订单,漏斗,转化",
        help="逗号分隔的测试查询列表",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="启用详细日志输出",
    )

    args = parser.parse_args()

    # 配置日志
    log_level = "DEBUG" if args.verbose else "INFO"
    configure_logging(log_level)

    # 加载配置
    try:
        agent_config = load_agent_config(config=args.config, namespace=args.namespace or None)
        db_path = agent_config.rag_storage_path()
        logger.info(f"使用存储路径: {db_path}")
    except Exception as e:
        logger.error(f"加载配置失败: {e}")
        sys.exit(1)

    # 解析测试查询
    test_queries = [q.strip() for q in args.test_queries.split(",") if q.strip()]

    # 执行检查
    results = check_ext_knowledge(db_path, test_queries)

    # 打印结果
    print_results(results, args.namespace)

    # 返回状态码
    if results["total_records"] == 0 or not results["table_exists"]:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
