# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details

"""
Search functionality for schema metadata storage.

This module provides full-text search (FTS), vector similarity search,
and Chinese text processing capabilities for metadata queries.
"""

import re
from typing import Any, List, Optional, Set

import pyarrow as pa
from lancedb.rerankers import Reranker

from datus.schemas.base import TABLE_TYPE
from datus.storage.base import BaseEmbeddingStore, WhereExpr
from datus.utils.query_utils import FTS_BUSINESS_TERMS, FTS_MAX_TOKENS, FTS_STOP_CHARS

# Import utilities
from datus.storage.schema_metadata.store_modules.utils import _build_where_clause


class SearchMixin:
    """
    Mixin class providing search functionality for metadata storage.

    This mixin can be used with any BaseEmbeddingStore subclass
    to add semantic search and full-text search capabilities.
    """

    @staticmethod
    def _sanitize_fts_query(query_text: str) -> str:
        """
        Sanitize FTS query by removing special characters.

        Keeps alphanumeric, whitespace, and Chinese characters.

        Args:
            query_text: Raw query text

        Returns:
            Sanitized query string
        """
        if not query_text:
            return ""
        # Keep word characters, spaces, and Chinese characters
        sanitized = re.sub(r"[^\w\s\u4e00-\u9fff]", " ", query_text)
        sanitized = re.sub(r"\s+", " ", sanitized).strip()
        return sanitized

    @staticmethod
    def _simplify_fts_query(query_text: str) -> str:
        """
        Simplify FTS query for Chinese text by extracting key tokens.

        This method implements a three-stage tokenization strategy:

        1. **Mixed tokens**: Preserve alphanumeric patterns like "铂智3X" or "3X"
        2. **Business terms**: Extract known domain-specific terms
        3. **Bigram fallback**: Generate character bigrams for remaining Chinese text

        Args:
            query_text: Raw query text in natural language

        Returns:
            Simplified query string with space-separated tokens
        """
        if not query_text:
            return ""
        sanitized = SearchMixin._sanitize_fts_query(query_text)
        if not sanitized:
            return ""

        has_chinese = any("\u4e00" <= ch <= "\u9fff" for ch in sanitized)
        if not has_chinese:
            return sanitized

        tokens: List[str] = []
        seen: set[str] = set()

        def _add(token: str) -> None:
            """Add unique non-empty token to results."""
            if not token:
                return
            token = token.strip()
            if not token or token in seen:
                return
            seen.add(token)
            tokens.append(token)

        # Stage 1: Extract mixed alphanumeric tokens (e.g., "铂智3X", "3X")
        mixed_tokens = re.findall(r"[\u4e00-\u9fff]+[A-Za-z0-9]+|[A-Za-z0-9]+", sanitized)
        for token in mixed_tokens:
            _add(token)

        # Stage 2: Extract common business terms from Chinese segments
        chinese_segments = re.findall(r"[\u4e00-\u9fff]+", sanitized)
        for segment in chinese_segments:
            for term in FTS_BUSINESS_TERMS:
                if term in segment:
                    _add(term)

        # Stage 3: Generate bigrams for remaining Chinese segments
        for segment in chinese_segments:
            SearchMixin._extract_chinese_bigrams(
                segment, FTS_STOP_CHARS, FTS_MAX_TOKENS, _add, tokens
            )
            if len(tokens) >= FTS_MAX_TOKENS:
                break

        if not tokens:
            return sanitized
        return " ".join(tokens[:FTS_MAX_TOKENS])

    @staticmethod
    def _extract_chinese_bigrams(
        segment: str,
        stop_chars: set[str],
        max_tokens: int,
        add_func: callable,
        tokens: List[str],
    ) -> None:
        """
        Extract character bigrams from a Chinese text segment.

        Generates overlapping 2-character tokens while skipping stop characters.
        Short segments (<=2 chars) are preserved as-is.

        Args:
            segment: Chinese text segment
            stop_chars: Characters to skip during bigram generation
            max_tokens: Maximum tokens to collect
            add_func: Callback function to add valid tokens
            tokens: Token list reference for length checking
        """
        if len(segment) <= 2:
            add_func(segment)
            return

        for i in range(len(segment) - 1):
            token = segment[i : i + 2]
            if any(ch in stop_chars for ch in token):
                continue
            add_func(token)
            if len(tokens) >= max_tokens:
                break

    def search_similar(
        self,
        query_text: str,
        catalog_name: str = "",
        database_name: str = "",
        schema_name: str = "",
        top_n: int = 5,
        table_type: TABLE_TYPE = "table",
        reranker: Optional[Reranker] = None,
    ) -> pa.Table:
        """
        Search using vector similarity with optional filters.

        Args:
            query_text: Query text to search for
            catalog_name: Optional catalog filter
            database_name: Optional database filter
            schema_name: Optional schema filter
            top_n: Number of results to return
            table_type: Table type filter
            reranker: Optional reranker for re-ranking results

        Returns:
            PyArrow Table with search results
        """
        where = _build_where_clause(
            catalog_name=catalog_name,
            database_name=database_name,
            schema_name=schema_name,
            table_type=table_type,
        )
        return self.do_search_similar(query_text, top_n=top_n, where=where, reranker=reranker)

    def do_search_similar(
        self,
        query_text: str,
        top_n: int = 5,
        where: WhereExpr = None,
        reranker: Optional[Reranker] = None,
    ) -> pa.Table:
        """
        Internal method for vector similarity search.

        Args:
            query_text: Query text
            top_n: Number of results
            where: WHERE clause for filtering
            reranker: Optional reranker

        Returns:
            PyArrow Table with search results
        """
        return self.search(
            query_text,
            top_n=top_n,
            where=where,
            reranker=reranker,
        )

    def search_all(
        self,
        catalog_name: str = "",
        database_name: str = "",
        schema_name: str = "",
        table_type: TABLE_TYPE = "full",
        select_fields: Optional[List[str]] = None,
    ) -> pa.Table:
        """
        Search all schemas matching the criteria (no vector search).

        Args:
            catalog_name: Optional catalog filter
            database_name: Optional database filter
            schema_name: Optional schema filter
            table_type: Table type filter
            select_fields: Fields to include in results

        Returns:
            PyArrow Table with all matching records
        """
        # Ensure table is ready before searching
        self._ensure_table_ready()

        where = _build_where_clause(
            catalog_name=catalog_name,
            database_name=database_name,
            schema_name=schema_name,
            table_type=table_type,
        )
        return self._search_all(where=where, select_fields=select_fields)

    def search_fts(
        self,
        query_text: str,
        catalog_name: str = "",
        database_name: str = "",
        schema_name: str = "",
        table_type: TABLE_TYPE = "table",
        top_n: int = 5,
        select_fields: Optional[List[str]] = None,
    ) -> pa.Table:
        """
        Search using Full-Text Search (FTS) with Chinese bigram fallback.

        Implements two-stage search strategy:
        1. Primary: Direct FTS search with sanitized query
        2. Fallback: Simplified query (Chinese bigrams) if primary returns no results

        Note: table_type is applied post-query rather than in the WHERE clause
        for better FTS performance on large datasets.

        Args:
            query_text: Search query text
            catalog_name: Optional catalog filter
            database_name: Optional database filter
            schema_name: Optional schema filter
            table_type: Table type filter (applied post-query)
            top_n: Maximum number of results
            select_fields: Fields to include in results

        Returns:
            PyArrow Table with matching records
        """
        from datus.storage.lancedb_conditions import build_where
        from datus.utils.loggings import get_logger

        logger = get_logger(__name__)

        self._ensure_table_ready()
        available_fields = set(self.table.schema.names)
        if select_fields:
            select_fields = [field for field in select_fields if field in available_fields]

        sanitized_query = self._sanitize_fts_query(query_text)
        if not sanitized_query:
            logger.warning("FTS query is empty after sanitization; skipping FTS search")
            return pa.table({})

        # Query all table types initially, filter post-query for FTS performance
        where = _build_where_clause(
            catalog_name=catalog_name,
            database_name=database_name,
            schema_name=schema_name,
            table_type="full",
            available_fields=available_fields,
        )
        where_clause = build_where(where)

        def _run_query(query: str, where_sql: str) -> pa.Table:
            """Execute FTS query."""
            query_builder = self.table.search(query, query_type="fts")
            query_builder = BaseEmbeddingStore._fill_query(query_builder, select_fields, where_sql)
            results = query_builder.limit(max(top_n * 5, top_n)).to_arrow()
            if self.vector_column_name in results.column_names:
                results = results.drop([self.vector_column_name])
            return results

        def _post_filter(results: pa.Table) -> pa.Table:
            """Filter results by table_type if needed."""
            if not results or results.num_rows == 0:
                return results
            if table_type and table_type != "full" and "table_type" in results.column_names:
                filtered = results.filter(results.column("table_type") == table_type)
                return filtered.slice(0, top_n)
            return results.slice(0, top_n)

        try:
            # Primary search with sanitized query
            results = _run_query(sanitized_query, where_clause)

            # Fallback: simplified query if no results
            if results.num_rows == 0:
                simplified = self._simplify_fts_query(query_text)
                if simplified and simplified != sanitized_query:
                    logger.debug(f"FTS fallback: '{sanitized_query}' -> '{simplified}'")
                    results = _run_query(simplified, where_clause)
                else:
                    logger.debug("FTS fallback skipped: simplification produced no new tokens")

            return _post_filter(results)
        except Exception as exc:
            logger.warning(f"FTS search failed: {exc}")
            return pa.table({})
