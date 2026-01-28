#!/usr/bin/env python3
# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Command-line interface for business configuration generation.
"""

import argparse
import sys
from pathlib import Path

from datus.configuration.agent_config_loader import load_agent_config
from datus.utils.loggings import configure_logging, get_logger

from .generators import BusinessTermsGenerator, MetricsCatalogGenerator
from .processors import DdlMerger, ExtKnowledgeImporter

logger = get_logger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description="Generate business terms configuration from design documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From Excel
  python scripts/generate_business_config.py \\
      --config=conf/agent.yml \\
      --namespace=test \\
      --arch-xlsx=/path/to/data.xlsx

  # From CSV
  python scripts/generate_business_config.py \\
      --config=conf/agent.yml \\
      --namespace=test \\
      --arch-csv=/path/to/data.csv
        """,
    )

    parser.add_argument("--config", required=True, help="Path to agent configuration file")
    parser.add_argument("--namespace", required=True, help="Target namespace")
    parser.add_argument("--arch-csv", help="Path to architecture CSV file")
    parser.add_argument("--metrics-csv", help="Path to metrics CSV file")
    parser.add_argument("--arch-xlsx", help="Path to architecture Excel file")
    parser.add_argument("--metrics-xlsx", help="Path to metrics Excel file")
    parser.add_argument("--output", default="conf/business_terms.yml", help="Output YAML file path")
    parser.add_argument("--merge-ddl", action="store_true", help="Merge with existing DDL comments")
    parser.add_argument("--min-term-length", type=int, default=2, help="Minimum term length")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Verbose output")
    parser.add_argument("--import-to-lancedb", action="store_true", help="Import metrics to ext_knowledge")

    return parser


class BusinessConfigCLI:
    """业务配置生成CLI"""

    def __init__(self, args):
        self.args = args
        self.min_term_length = args.min_term_length
        self.business_terms_gen = BusinessTermsGenerator(self.min_term_length)
        self.metrics_catalog_gen = MetricsCatalogGenerator(self.min_term_length)

    def run(self) -> int:
        """执行CLI主流程"""
        # 初始化日志
        configure_logging(debug=self.args.verbose >= 2)

        # 加载配置
        logger.info(f"Loading agent config from: {self.args.config}")
        agent_config = load_agent_config(config=self.args.config)
        agent_config.current_namespace = self.args.namespace

        # 生成业务术语
        business_terms = None

        if self.args.arch_xlsx:
            business_terms = self.business_terms_gen.generate_from_architecture_xlsx(
                Path(self.args.arch_xlsx), header_rows=3
            )
        elif self.args.arch_csv:
            business_terms = self.business_terms_gen.generate_from_architecture_csv(
                Path(self.args.arch_csv)
            )

        if self.args.metrics_xlsx:
            if business_terms is None:
                business_terms = {"term_to_table": {}, "term_to_schema": {}, "table_keywords": {}}
            business_terms = self.business_terms_gen.generate_from_metrics_xlsx(
                Path(self.args.metrics_xlsx), business_terms, header_rows=2
            )
        elif self.args.metrics_csv:
            if business_terms is None:
                business_terms = {"term_to_table": {}, "term_to_schema": {}, "table_keywords": {}}
            # TODO: Add CSV metrics support

        if business_terms is None:
            logger.error("No input files provided")
            return 1

        # 合并DDL
        if self.args.merge_ddl:
            merger = DdlMerger(None, self.min_term_length)
            business_terms = merger.merge(business_terms)

        # 导入到ExtKnowledge
        if self.args.import_to_lancedb and self.args.metrics_xlsx:
            importer = ExtKnowledgeImporter(agent_config.rag_storage_path())
            metrics_catalog = self.metrics_catalog_gen.generate(
                Path(self.args.metrics_xlsx), header_rows=2
            )
            if metrics_catalog:
                importer.import_metrics(metrics_catalog)

        logger.info("Business configuration generation complete!")
        return 0


def main():
    """CLI入口"""
    parser = create_parser()
    args = parser.parse_args()

    cli = BusinessConfigCLI(args)
    sys.exit(cli.run())
