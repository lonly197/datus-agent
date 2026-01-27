# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Execution scenario enumeration.

This module defines standardized scenario names as an enumeration to prevent
string typos and improve code maintainability.
"""

from enum import Enum


class ExecutionScenario(str, Enum):
    """
    Execution scenario enumeration.

    Using str as base class ensures compatibility with existing string comparisons
    and dictionary lookups (e.g., scenario == "text2sql" still works).

    Available scenarios:
    - TEXT2SQL: Natural language to SQL translation
    - SQL_REVIEW: SQL query review and validation
    - DATA_ANALYSIS: General data analysis tasks
    - SMART_QUERY: Intelligent query execution
    - DEEP_ANALYSIS: Deep analytical processing
    """

    TEXT2SQL = "text2sql"
    SQL_REVIEW = "sql_review"
    DATA_ANALYSIS = "data_analysis"
    SMART_QUERY = "smart_query"
    DEEP_ANALYSIS = "deep_analysis"

    def __str__(self) -> str:
        """Return the string value for compatibility."""
        return self.value
