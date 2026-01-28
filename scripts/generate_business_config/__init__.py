#!/usr/bin/env python3
# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Business Configuration Generator - Modular Package

结合数据架构设计文档和DDL元数据，生成/更新业务术语配置文件 business_terms.yml。

支持两种输入格式：
1. Excel文件 (.xlsx) - 推荐，支持多行表头
2. CSV文件 (.csv) - 兼容，需确保表头正确

使用示例：
    # 从数据架构Excel生成
    python -m scripts.generate_business_config \\
        --config=conf/agent.yml \\
        --namespace=test \\
        --arch-xlsx=/path/to/数据架构详细设计.xlsx

    # 从数据架构和指标清单生成
    python -m scripts.generate_business_config \\
        --config=conf/agent.yml \\
        --namespace=test \\
        --arch-xlsx=/path/to/数据架构详细设计.xlsx \\
        --metrics-xlsx=/path/to/指标清单.xlsx
"""

import sys
from pathlib import Path

# Add repo root to path for imports
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Shared constants
from .shared import STOP_WORDS, SYNONYM_MAP

__version__ = "2.0.0"

# Readers
from .readers import ExcelReader, CsvReader, HeaderParser

# Extractors
from .extractors import TermExtractor, KeywordExtractor, BusinessTermMapping

# Generators
from .generators import BusinessTermsGenerator, MetricsCatalogGenerator, LLMEnhancedBusinessTermsGenerator

# Processors
from .processors import DdlMerger, ExtKnowledgeImporter

# CLI
from .cli import BusinessConfigCLI, main

__all__ = [
    # Version
    '__version__',
    # Shared constants
    'STOP_WORDS',
    'SYNONYM_MAP',
    # Readers
    'ExcelReader',
    'CsvReader',
    'HeaderParser',
    # Extractors
    'TermExtractor',
    'KeywordExtractor',
    'BusinessTermMapping',
    # Generators
    'BusinessTermsGenerator',
    'MetricsCatalogGenerator',
    'LLMEnhancedBusinessTermsGenerator',
    # Processors
    'DdlMerger',
    'ExtKnowledgeImporter',
    # CLI
    'BusinessConfigCLI',
    'main',
]
