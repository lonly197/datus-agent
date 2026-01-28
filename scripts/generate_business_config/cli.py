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
from datus.storage.embedding_models import get_db_embedding_model
from datus.storage.schema_metadata import SchemaStorage
from datus.utils.loggings import configure_logging, get_logger

from .generators import BusinessTermsGenerator, MetricsCatalogGenerator, LLMEnhancedBusinessTermsGenerator
from .processors import DdlMerger, ExtKnowledgeImporter
from .shared import TablePriority

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
    parser.add_argument("--arch-csv", help="Path to architecture CSV file (legacy)")
    parser.add_argument("--metrics-csv", help="Path to metrics CSV file (legacy, optional)")
    parser.add_argument("--arch-xlsx", help="Path to architecture Excel file (recommended, supports multi-row headers)")
    parser.add_argument("--metrics-xlsx", help="Path to metrics Excel file (recommended, supports multi-row headers)")
    parser.add_argument(
        "--arch-sheet-name",
        dest="arch_sheet_name",
        default=None,
        help="Sheet name or index for architecture Excel (default: first sheet or 'Sheet1')",
    )
    parser.add_argument(
        "--metrics-sheet-name",
        dest="metrics_sheet_name",
        default=None,
        help="Sheet name or index for metrics Excel (default: first sheet or 'Sheet1')",
    )
    parser.add_argument(
        "--output",
        default="conf/business_terms.yml",
        help="Output YAML file path (default: conf/business_terms.yml)",
    )
    parser.add_argument(
        "--merge-ddl",
        action="store_true",
        help="Merge with existing DDL comments from LanceDB",
    )
    parser.add_argument(
        "--min-term-length",
        type=int,
        default=2,
        help="Minimum term length (default: 2)",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Enable LLM enhancement for term disambiguation and conflict resolution",
    )
    parser.add_argument(
        "--include-confidence",
        action="store_true",
        help="Include confidence scores in output (for debugging)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="count",
        default=0,
        help="Verbose output (-v for INFO, -vv for DEBUG)",
    )
    parser.add_argument(
        "--import-to-lancedb",
        action="store_true",
        help="Import metrics catalog to ext_knowledge table in LanceDB",
    )
    parser.add_argument(
        "--max-table-priority",
        choices=["DIM", "DWD", "DWS", "ADS", "ODS"],
        default="ADS",
        help="Maximum table priority to include (default: ADS, excludes ODS). "
             "Priority order: DIM < DWD < DWS < ADS < ODS",
    )
    parser.add_argument(
        "--disable-text-cleaning",
        action="store_true",
        help="Disable Excel text cleaning (enabled by default)",
    )
    parser.add_argument(
        "--rewrite-with-llm",
        action="store_true",
        help="Use LLM to rewrite metric definitions for better searchability (requires --use-llm)",
    )

    return parser


class BusinessConfigCLI:
    """业务配置生成CLI"""

    def __init__(self, args):
        self.args = args
        self.min_term_length = args.min_term_length
        self.use_llm = args.use_llm
        self.include_confidence = args.include_confidence
        self.rewrite_with_llm = args.rewrite_with_llm
        
        # 表优先级设置
        priority_map = {
            "DIM": TablePriority.DIM,
            "DWD": TablePriority.DWD,
            "DWS": TablePriority.DWS,
            "ADS": TablePriority.ADS,
            "ODS": TablePriority.ODS,
        }
        self.max_table_priority = priority_map[args.max_table_priority]
        self.enable_text_cleaning = not args.disable_text_cleaning
        
        self.metrics_catalog_gen = MetricsCatalogGenerator(self.min_term_length)

        # 根据 use_llm 选择生成器
        if self.use_llm:
            logger.info("LLM enhancement enabled for term extraction and conflict resolution")
            if self.rewrite_with_llm:
                logger.info("LLM rewriting enabled for metric definitions")
            self.business_terms_gen = None  # 初始化时不需要，传agent_config
        else:
            self.business_terms_gen = BusinessTermsGenerator(
                min_term_length=self.min_term_length,
                max_table_priority=self.max_table_priority,
                enable_text_cleaning=self.enable_text_cleaning,
            )

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
            if self.use_llm:
                llm_gen = LLMEnhancedBusinessTermsGenerator(
                    agent_config=agent_config,
                    namespace=self.args.namespace,
                    min_term_length=self.min_term_length,
                    use_llm=True,
                    max_table_priority=self.max_table_priority,
                    enable_text_cleaning=self.enable_text_cleaning,
                )
                business_terms = llm_gen.generate_from_architecture_xlsx(
                    Path(self.args.arch_xlsx), header_rows=3, sheet_name=self.args.arch_sheet_name
                )
            else:
                business_terms = self.business_terms_gen.generate_from_architecture_xlsx(
                    Path(self.args.arch_xlsx), header_rows=3, sheet_name=self.args.arch_sheet_name
                )
        elif self.args.arch_csv:
            business_terms = self.business_terms_gen.generate_from_architecture_csv(
                Path(self.args.arch_csv)
            )

        if self.args.metrics_xlsx:
            if business_terms is None:
                business_terms = {"term_to_table": {}, "term_to_schema": {}, "table_keywords": {}}
            if self.use_llm:
                llm_gen = LLMEnhancedBusinessTermsGenerator(
                    agent_config=agent_config,
                    namespace=self.args.namespace,
                    min_term_length=self.min_term_length,
                    use_llm=True,
                    max_table_priority=self.max_table_priority,
                    enable_text_cleaning=self.enable_text_cleaning,
                )
                business_terms = llm_gen.generate_from_metrics_xlsx(
                    Path(self.args.metrics_xlsx), business_terms, header_rows=2, sheet_name=self.args.metrics_sheet_name
                )
            else:
                business_terms = self.business_terms_gen.generate_from_metrics_xlsx(
                    Path(self.args.metrics_xlsx), business_terms, header_rows=2, sheet_name=self.args.metrics_sheet_name
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
            schema_storage = SchemaStorage(
                db_path=agent_config.rag_storage_path(),
                embedding_model=get_db_embedding_model()
            )
            merger = DdlMerger(schema_storage, self.min_term_length)
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
