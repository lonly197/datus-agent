# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
LLM-Enhanced Extraction Module (Backward Compatible Entry Point)

This module provides re-exports of LLM-based extraction functions for:
- Enum extraction from comments
- Business metadata extraction from DDL
- Design requirement parsing

For individual module documentation, see:
- datus.storage.schema_metadata.enum_extractor
- datus.storage.schema_metadata.metadata_extractor
- datus.storage.schema_metadata.design_requirement_parser
"""

from .design_requirement_parser import (
    DEFAULT_DESIGN_REQ_PROMPT,
    parse_design_requirement_with_llm,
)
from .enum_extractor import (
    DEFAULT_ENUM_EXTRACT_PROMPT,
    EnhancedEnumExtractor,
    create_enhanced_enum_extractor,
    extract_enums_with_llm,
)
from .metadata_extractor import (
    DEFAULT_METADATA_EXTRACT_PROMPT,
    extract_business_metadata_with_llm,
)

__all__ = [
    # Enum extraction
    "DEFAULT_ENUM_EXTRACT_PROMPT",
    "extract_enums_with_llm",
    "EnhancedEnumExtractor",
    "create_enhanced_enum_extractor",
    # Business metadata
    "DEFAULT_METADATA_EXTRACT_PROMPT",
    "extract_business_metadata_with_llm",
    # Design requirement
    "DEFAULT_DESIGN_REQ_PROMPT",
    "parse_design_requirement_with_llm",
]
