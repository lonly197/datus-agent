# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Enhanced metadata extraction from database connectors.

This module provides database-specific metadata extractors that go beyond DDL parsing
to extract row counts, column statistics, relationships, and business context.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from datus.utils.loggings import get_logger
from datus.utils.sql_utils import quote_identifier

logger = get_logger(__name__)


class BaseMetadataExtractor(ABC):
    """
    Abstract base class for database-specific metadata extractors.

    Each database type (DuckDB, Snowflake, MySQL, PostgreSQL) should implement
    this interface to provide enhanced metadata capabilities.
    """

    def __init__(self, connector, dialect: str):
        """
        Initialize the metadata extractor.

        Args:
            connector: Database connector instance
            dialect: Database type identifier
        """
        self.connector = connector
        self.dialect = dialect

    @abstractmethod
    def extract_row_count(self, table_name: str) -> int:
        """
        Get approximate row count for a table.

        Uses database statistics tables when available for efficiency.
        Falls back to COUNT(*) if statistics not available.

        Args:
            table_name: Name of the table

        Returns:
            Approximate row count (0 if unable to determine)
        """
        pass

    @abstractmethod
    def extract_column_statistics(self, table_name: str, sample_size: int = 10000) -> Dict[str, Dict]:
        """
        Extract statistical information for numeric columns.

        Args:
            table_name: Name of the table
            sample_size: Number of rows to sample for statistics

        Returns:
            Dictionary mapping column names to stats:
            {
                "price": {"min": 0, "max": 1000, "mean": 250.5, "std": 150.2},
                "quantity": {"min": 1, "max": 100, "mean": 25.3}
            }
        """
        pass

    @abstractmethod
    def detect_relationships(self, table_name: str) -> Dict[str, Any]:
        """
        Detect foreign key relationships from information_schema.

        Args:
            table_name: Name of the table

        Returns:
            Dictionary with relationship metadata:
            {
                "foreign_keys": [
                    {"from_column": "user_id", "to_table": "users", "to_column": "id"}
                ],
                "join_paths": [...]
            }
        """
        pass


class DuckDBMetadataExtractor(BaseMetadataExtractor):
    """
    DuckDB-specific metadata extractor using information_schema.
    """

    def extract_row_count(self, table_name: str) -> int:
        """
        Get approximate row count from DuckDB statistics.

        Uses pg_class.reltuples for approximate count (faster than COUNT(*)).
        """
        try:
            safe_table = quote_identifier(table_name, "postgres")

            # Try approximate count from statistics first
            query = f"""
                SELECT COALESCE(reltuples, 0)::BIGINT as row_count
                FROM pg_class
                WHERE relname = {quote_identifier(table_name, "postgres")}
            """

            result = self.connector.execute_sql(query)
            if result and len(result) > 0:
                row_count = result[0].get("row_count", 0)
                if row_count > 0:
                    return row_count

            # Fallback to COUNT(*) if statistics not available
            query = f"SELECT COUNT(*) as row_count FROM {safe_table}"
            result = self.connector.execute_sql(query)
            if result and len(result) > 0:
                return result[0].get("row_count", 0)

        except Exception as e:
            logger.warning(f"Failed to extract row count for {table_name}: {e}")

        return 0

    def extract_column_statistics(self, table_name: str, sample_size: int = 10000) -> Dict[str, Dict]:
        """
        Extract column statistics using sampling.

        Samples numeric columns and calculates min/max/mean/std.
        """
        import json

        try:
            safe_table = quote_identifier(table_name, "postgres")

            # Get numeric columns first
            query = f"""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = {quote_identifier(table_name, "postgres")}
                AND data_type IN ('INTEGER', 'BIGINT', 'FLOAT', 'DOUBLE', 'DECIMAL', 'NUMERIC')
            """

            columns_result = self.connector.execute_sql(query)
            if not columns_result:
                return {}

            numeric_columns = [row["column_name"] for row in columns_result]
            if not numeric_columns:
                return {}

            # Sample data and calculate statistics
            stats = {}
            for col in numeric_columns:
                try:
                    safe_col = quote_identifier(col, "postgres")
                    query = f"""
                        SELECT
                            MIN({safe_col}) as min_val,
                            MAX({safe_col}) as max_val,
                            AVG({safe_col}) as mean_val,
                            STDDEV({safe_col}) as std_val
                        FROM (
                            SELECT {safe_col}
                            FROM {safe_table}
                            LIMIT {sample_size}
                        )
                    """

                    result = self.connector.execute_sql(query)
                    if result and len(result) > 0:
                        row = result[0]
                        if row["min_val"] is not None:  # Valid statistics
                            stats[col] = {
                                "min": float(row["min_val"]),
                                "max": float(row["max_val"]),
                                "mean": float(row["mean_val"]) if row["mean_val"] else None,
                                "std": float(row["std_val"]) if row["std_val"] else None
                            }

                except Exception as e:
                    logger.debug(f"Failed to extract statistics for {table_name}.{col}: {e}")
                    continue

            return stats

        except Exception as e:
            logger.warning(f"Failed to extract column statistics for {table_name}: {e}")
            return {}

    def detect_relationships(self, table_name: str) -> Dict[str, Any]:
        """
        Detect foreign key relationships from information_schema.

        Uses DuckDB's information_schema to query foreign key constraints.
        """
        try:
            safe_table = quote_identifier(table_name, "postgres")

            # Query foreign key constraints
            query = f"""
                SELECT
                    kcu.column_name as from_column,
                    ccu.table_name as to_table,
                    ccu.column_name as to_column
                FROM information_schema.table_constraints AS tc
                JOIN information_schema.key_column_usage AS kcu
                    ON tc.constraint_name = kcu.constraint_name
                JOIN information_schema.constraint_column_usage AS ccu
                    ON ccu.constraint_name = tc.constraint_name
                WHERE tc.constraint_type = 'FOREIGN KEY'
                AND tc.table_name = {quote_identifier(table_name, "postgres")}
            """

            result = self.connector.execute_sql(query)
            foreign_keys = []
            if result:
                for row in result:
                    foreign_keys.append({
                        "from_column": row["from_column"],
                        "to_table": row["to_table"],
                        "to_column": row["to_column"]
                    })

            return {
                "foreign_keys": foreign_keys,
                "join_paths": self._infer_join_paths(foreign_keys)
            }

        except Exception as e:
            logger.warning(f"Failed to detect relationships for {table_name}: {e}")
            return {"foreign_keys": [], "join_paths": []}

    def _infer_join_paths(self, foreign_keys: List[Dict]) -> List[str]:
        """
        Infer common join paths from foreign key relationships.
        """
        join_paths = []
        for fk in foreign_keys:
            join_path = f"{fk['from_column']} -> {fk['to_table']}.{fk['to_column']}"
            join_paths.append(join_path)
        return join_paths


class SnowflakeMetadataExtractor(BaseMetadataExtractor):
    """
    Snowflake-specific metadata extractor using TABLE_STORAGE_METRICS.
    """

    def extract_row_count(self, table_name: str) -> int:
        """
        Get row count from Snowflake TABLE_STORAGE_METRICS view.

        This is much faster than COUNT(*) for large tables.
        """
        try:
            query = f"""
                SELECT row_count
                FROM SNOWFLAKE.ACCOUNT_USAGE.TABLE_STORAGE_METRICS
                WHERE table_name = {quote_identifier(table_name.upper(), "snowflake")}
                ORDER BY last_altered DESC
                LIMIT 1
            """

            result = self.connector.execute_sql(query)
            if result and len(result) > 0:
                return result[0].get("row_count", 0)

        except Exception as e:
            logger.warning(f"Failed to extract Snowflake row count for {table_name}: {e}")

        return 0

    def extract_column_statistics(self, table_name: str, sample_size: int = 10000) -> Dict[str, Dict]:
        """
        Extract column statistics using sampling.

        Note: Snowflake SAMPLE clause is efficient for large tables.
        """
        try:
            safe_table = quote_identifier(table_name, "snowflake")
            safe_db = quote_identifier(self.connector.database, "snowflake") if self.connector.database else ""
            safe_schema = quote_identifier(self.connector.schema, "snowflake") if self.connector.schema else ""

            # Get numeric columns
            query = f"""
                SELECT column_name, data_type
                FROM {safe_db}.INFORMATION_SCHEMA.COLUMNS
                WHERE table_name = {quote_identifier(table_name.upper(), "snowflake")}
                AND data_type IN ('NUMBER', 'FLOAT', 'DOUBLE', 'DECIMAL', 'NUMERIC')
            """

            columns_result = self.connector.execute_sql(query)
            if not columns_result:
                return {}

            numeric_columns = [row["COLUMN_NAME"] for row in columns_result]
            if not numeric_columns:
                return {}

            stats = {}
            for col in numeric_columns:
                try:
                    safe_col = quote_identifier(col, "snowflake")
                    qualified_table = f"{safe_db}.{safe_schema}.{safe_table}" if safe_db and safe_schema else safe_table
                    query = f"""
                        SELECT
                            MIN({safe_col}) as min_val,
                            MAX({safe_col}) as max_val,
                            AVG({safe_col}) as mean_val,
                            STDDEV({safe_col}) as std_val
                        FROM {qualified_table}
                        SAMPLE ({sample_size} ROWS)
                    """

                    result = self.connector.execute_sql(query)
                    if result and len(result) > 0:
                        row = result[0]
                        if row["MIN_VAL"] is not None:
                            stats[col] = {
                                "min": float(row["MIN_VAL"]),
                                "max": float(row["MAX_VAL"]),
                                "mean": float(row["MEAN_VAL"]) if row["MEAN_VAL"] else None,
                                "std": float(row["STD_VAL"]) if row["STD_VAL"] else None
                            }

                except Exception as e:
                    logger.debug(f"Failed to extract statistics for {table_name}.{col}: {e}")
                    continue

            return stats

        except Exception as e:
            logger.warning(f"Failed to extract Snowflake column statistics for {table_name}: {e}")
            return {}

    def detect_relationships(self, table_name: str) -> Dict[str, Any]:
        """
        Detect foreign key relationships from Snowflake information_schema.
        """
        try:
            safe_db = quote_identifier(self.connector.database, "snowflake") if self.connector.database else ""

            query = f"""
                SELECT
                    kcu.column_name as from_column,
                    ccu.table_name as to_table,
                    ccu.column_name as to_column
                FROM {safe_db}.INFORMATION_SCHEMA.TABLE_CONSTRAINTS AS tc
                JOIN {safe_db}.INFORMATION_SCHEMA.KEY_COLUMN_USAGE AS kcu
                    ON tc.constraint_name = kcu.constraint_name
                JOIN {safe_db}.INFORMATION_SCHEMA.CONSTRAINT_COLUMN_USAGE AS ccu
                    ON ccu.constraint_name = tc.constraint_name
                WHERE tc.constraint_type = 'FOREIGN KEY'
                AND tc.table_name = {quote_identifier(table_name.upper(), "snowflake")}
            """

            result = self.connector.execute_sql(query)
            foreign_keys = []
            if result:
                for row in result:
                    foreign_keys.append({
                        "from_column": row["FROM_COLUMN"],
                        "to_table": row["TO_TABLE"],
                        "to_column": row["TO_COLUMN"]
                    })

            return {
                "foreign_keys": foreign_keys,
                "join_paths": self._infer_join_paths(foreign_keys)
            }

        except Exception as e:
            logger.warning(f"Failed to detect Snowflake relationships for {table_name}: {e}")
            return {"foreign_keys": [], "join_paths": []}

    def _infer_join_paths(self, foreign_keys: List[Dict]) -> List[str]:
        """Infer common join paths from foreign key relationships."""
        join_paths = []
        for fk in foreign_keys:
            join_path = f"{fk['from_column']} -> {fk['to_table']}.{fk['to_column']}"
            join_paths.append(join_path)
        return join_paths


def get_metadata_extractor(connector, dialect: str) -> BaseMetadataExtractor:
    """
    Factory function to get the appropriate metadata extractor for a database type.

    Args:
        connector: Database connector instance
        dialect: Database type identifier

    Returns:
        Metadata extractor instance for the specified database type
    """
    extractors = {
        "duckdb": DuckDBMetadataExtractor,
        "snowflake": SnowflakeMetadataExtractor,
        # Future implementations:
        # "mysql": MySQLMetadataExtractor,
        # "postgres": PostgreSQLMetadataExtractor,
        # "bigquery": BigQueryMetadataExtractor,
    }

    extractor_class = extractors.get(dialect.lower())
    if extractor_class:
        return extractor_class(connector, dialect)
    else:
        logger.warning(f"No metadata extractor available for dialect '{dialect}', using base class")
        # Return base instance (will return empty results)
        return BaseMetadataExtractor(connector, dialect)
