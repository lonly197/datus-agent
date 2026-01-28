#!/usr/bin/env python3
# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Backward-compatible wrapper for generate_business_config.

This script provides full compatibility with the original API
while delegating to the new modular package.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Re-export all symbols for backward compatibility
from scripts.generate_business_config import (
    main,
    BusinessConfigCLI,
    BusinessTermsGenerator,
    MetricsCatalogGenerator,
    BusinessTermMapping,
    TermExtractor,
    KeywordExtractor,
    ExcelReader,
    CsvReader,
    HeaderParser,
    DdlMerger,
    ExtKnowledgeImporter,
    STOP_WORDS,
    SYNONYM_MAP,
    __version__,
)

if __name__ == "__main__":
    main()
