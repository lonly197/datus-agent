# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
StarRocks Database Connector

This connector implements StarRocks database operations including:
- Table listing and DDL retrieval
- Schema information retrieval
- Query execution
- Sample data retrieval
"""

from typing import Any, Dict, List, Literal, Optional

try:
    from typing import override
except ImportError:
    from typing_extensions import override

from datus.schemas.base import TABLE_TYPE
from datus.schemas.node_models import ExecuteSQLResult
from datus.tools.db_tools.base import BaseSqlConnector
from datus.utils.constants import DBType
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class StarRocksConnector(BaseSqlConnector):
    """
    Connector for StarRocks databases using mysql-connector-python.

    StarRocks is a new-generation fast analytics database that can query petabyte-scale datasets with fast response times.
    """

    def __init__(self, config: dict):
        """
        Initialize StarRocks connector.

        Args:
            config: Configuration dictionary containing connection parameters:
                - host: Database host (default: localhost)
                - port: Database port (default: 9030)
                - user: Username
                - password: Password
                - database: Database name
                - catalog: Catalog name (optional, for StarRocks catalog feature)
        """
        # Set default dialect
        dialect = DBType.STARROCKS

        # Initialize base connector
        super().__init__(config, dialect=dialect)

        # Extract connection parameters
        self.host = config.get("host", "localhost")
        self.port = config.get("port", 9030)
        self.user = config.get("user", "")
        self.password = config.get("password", "")
        self.database = config.get("database", "")
        self.catalog = config.get("catalog", "")

        # Store namespace information
        if self.database:
            self.database_name = self.database
        if self.catalog:
            self.catalog_name = self.catalog

        # Connection will be initialized when needed
        self.connection = None

    def _fix_truncated_ddl(self, ddl: str) -> str:
        """
        Fix truncated DDL statements by detecting common truncation patterns and attempting completion.

        This method handles DDL that may have been truncated during retrieval from the database,
        such as when SHOW CREATE TABLE returns incomplete results due to length limits.

        Args:
            ddl: Potentially truncated DDL statement

        Returns:
            Fixed DDL statement or original if not recognized as truncated
        """
        if not ddl or not isinstance(ddl, str):
            return ddl

        # Check if DDL appears to be truncated
        ddl_upper = ddl.upper().strip()

        # Patterns that indicate truncation
        truncation_indicators = [
            ddl.rstrip().endswith(','),  # Ends with comma (incomplete column list)
            not ddl.rstrip().endswith(';'),  # No semicolon at end
            not ('ENGINE=' in ddl_upper or 'PARTITION BY' in ddl_upper),  # Missing StarRocks-specific clauses
        ]

        # Check if missing closing paren for CREATE TABLE
        open_parens = ddl.count('(')
        close_parens = ddl.count(')')
        missing_closing_paren = open_parens > close_parens

        # If any truncation indicators or missing closing paren, try to fix
        if sum(truncation_indicators) >= 1 or missing_closing_paren:
            logger.debug(f"Detected potentially truncated DDL for StarRocks table (indicators: {sum(truncation_indicators)}, missing paren: {missing_closing_paren})")

            # Try to complete the basic structure
            fixed_ddl = ddl

            # Remove trailing comma if present
            if fixed_ddl.rstrip().endswith(','):
                # Remove the trailing comma from the end of the DDL
                fixed_ddl = fixed_ddl.rstrip()[:-1].rstrip()  # Remove comma and any trailing whitespace

            # Add basic StarRocks table structure if missing
            if 'ENGINE=' not in fixed_ddl.upper():
                # If missing closing paren, add it before ENGINE
                if missing_closing_paren:
                    fixed_ddl += '\n) ENGINE=OLAP;'
                # If has closing paren but missing ENGINE
                elif not fixed_ddl.rstrip().endswith(';'):
                    fixed_ddl += ' ENGINE=OLAP;'
                else:
                    fixed_ddl = fixed_ddl.rstrip(';') + ' ENGINE=OLAP;'

            # Clean up
            fixed_ddl = fixed_ddl.strip()

            if fixed_ddl != ddl:
                logger.info(f"Fixed truncated DDL for StarRocks table (length: {len(ddl)} -> {len(fixed_ddl)})")
                return fixed_ddl

        return ddl

    @override
    def connect(self):
        """Establish connection to StarRocks database."""
        if self.connection:
            return

        try:
            import mysql.connector
            from mysql.connector import Error as MySQLError

            # Create connection
            self.connection = mysql.connector.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database if self.database else None,
                auth_plugin='mysql_native_password',
                connection_timeout=self.timeout_seconds,
            )

            logger.debug(f"Connected to StarRocks database: {self.host}:{self.port}/{self.database}")

        except ImportError:
            raise DatusException(
                ErrorCode.DB_CONNECTION_FAILED,
                message="mysql-connector-python is required for StarRocks connector. Install with: pip install mysql-connector-python"
            )
        except Exception as e:
            raise DatusException(
                ErrorCode.DB_CONNECTION_FAILED,
                message_args={"error_message": str(e)},
            ) from e

    @override
    def close(self):
        """Close the database connection."""
        if self.connection:
            try:
                self.connection.close()
                logger.debug("Closed StarRocks connection")
            except Exception as e:
                logger.warning(f"Error closing StarRocks connection: {e}")
            finally:
                self.connection = None

    @override
    def test_connection(self) -> bool:
        """Test the database connection."""
        opened_here = self.connection is None
        try:
            self.connect()
            cursor = self.connection.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            return True
        except Exception as e:
            raise DatusException(
                ErrorCode.DB_CONNECTION_FAILED,
                message_args={"error_message": str(e)},
            ) from e
        finally:
            if opened_here:
                self.close()

    def _handle_exception(self, e: Exception, sql: str = "") -> DatusException:
        """Handle MySQL exceptions and map to appropriate Datus ErrorCode."""
        if isinstance(e, DatusException):
            return e

        error_msg = str(e).lower()

        try:
            import mysql.connector
            from mysql.connector import Error as MySQLError

            if isinstance(e, MySQLError):
                # Connection errors
                if e.errno in (2003, 2006, 2013):  # Can't connect, MySQL server has gone away, Lost connection
                    return DatusException(
                        ErrorCode.DB_CONNECTION_FAILED,
                        message_args={"error_message": str(e)},
                    )
                # Table doesn't exist
                elif e.errno == 1146:
                    return DatusException(
                        ErrorCode.DB_TABLE_NOT_EXISTS,
                        message_args={"table_name": sql, "error_message": str(e)},
                    )
                # Syntax error
                elif e.errno == 1064:
                    return DatusException(
                        ErrorCode.DB_EXECUTION_SYNTAX_ERROR,
                        message_args={"sql": sql, "error_message": str(e)},
                    )
                # Lock timeout
                elif e.errno == 1205:
                    return DatusException(
                        ErrorCode.DB_CONNECTION_TIMEOUT,
                        message_args={"error_message": str(e)},
                    )
                else:
                    return DatusException(
                        ErrorCode.DB_EXECUTION_ERROR,
                        message_args={"sql": sql, "error_message": str(e)},
                    )
        except ImportError:
            pass

        # Generic error handling
        if "connection" in error_msg or "timeout" in error_msg:
            return DatusException(
                ErrorCode.DB_CONNECTION_TIMEOUT,
                message_args={"error_message": str(e)},
            )
        elif "syntax" in error_msg:
            return DatusException(
                ErrorCode.DB_EXECUTION_SYNTAX_ERROR,
                message_args={"sql": sql, "error_message": str(e)},
            )
        else:
            return DatusException(
                ErrorCode.DB_EXECUTION_ERROR,
                message_args={"sql": sql, "error_message": str(e)},
            )

    @override
    def execute_insert(self, sql: str) -> ExecuteSQLResult:
        """Execute an INSERT SQL statement."""
        try:
            self.connect()
            cursor = self.connection.cursor()
            cursor.execute(sql)
            self.connection.commit()
            return ExecuteSQLResult(
                success=True,
                sql_query=sql,
                sql_return=str(cursor.lastrowid),
                row_count=cursor.rowcount,
            )
        except Exception as e:
            ex = self._handle_exception(e, sql)
            return ExecuteSQLResult(
                success=False,
                error=str(ex),
                sql_query=sql,
            )

    @override
    def execute_update(self, sql: str) -> ExecuteSQLResult:
        """Execute an UPDATE SQL statement."""
        try:
            self.connect()
            cursor = self.connection.cursor()
            cursor.execute(sql)
            self.connection.commit()
            return ExecuteSQLResult(
                success=True,
                sql_query=sql,
                sql_return=str(cursor.rowcount),
                row_count=cursor.rowcount,
            )
        except Exception as e:
            ex = self._handle_exception(e, sql)
            return ExecuteSQLResult(
                success=False,
                error=str(ex),
                sql_query=sql,
            )

    @override
    def execute_delete(self, sql: str) -> ExecuteSQLResult:
        """Execute a DELETE SQL statement."""
        try:
            self.connect()
            cursor = self.connection.cursor()
            cursor.execute(sql)
            self.connection.commit()
            return ExecuteSQLResult(
                success=True,
                sql_query=sql,
                sql_return=str(cursor.rowcount),
                row_count=cursor.rowcount,
            )
        except Exception as e:
            ex = self._handle_exception(e, sql)
            return ExecuteSQLResult(
                success=False,
                error=str(ex),
                sql_query=sql,
            )

    @override
    def execute_ddl(self, sql: str) -> ExecuteSQLResult:
        """Execute a DDL SQL statement."""
        try:
            self.connect()
            cursor = self.connection.cursor()
            cursor.execute(sql)
            self.connection.commit()
            return ExecuteSQLResult(
                success=True,
                sql_query=sql,
                sql_return="Success",
                row_count=0,
            )
        except Exception as e:
            ex = self._handle_exception(e, sql)
            return ExecuteSQLResult(
                success=False,
                error=str(ex),
                sql_query=sql,
            )

    @override
    def execute_query(
        self, sql: str, result_format: Literal["csv", "arrow", "pandas", "list"] = "csv"
    ) -> ExecuteSQLResult:
        """Execute a SELECT query."""
        try:
            self.connect()
            cursor = self.connection.cursor(buffered=True)
            cursor.execute(sql)

            # Get column names
            columns = [desc[0] for desc in cursor.description] if cursor.description else []

            # Fetch results
            rows = cursor.fetchall()
            row_count = len(rows)

            # Format results
            if result_format == "csv":
                import csv
                from io import StringIO

                output = StringIO()
                writer = csv.writer(output)
                writer.writerow(columns)
                writer.writerows(rows)
                result = output.getvalue()
            elif result_format == "arrow":
                try:
                    import pyarrow as pa
                    import pandas as pd

                    df = pd.DataFrame(rows, columns=columns)
                    result = pa.Table.from_pandas(df)
                except ImportError:
                    raise DatusException(
                        ErrorCode.COMMON_CONFIG_ERROR,
                        message="pyarrow and pandas are required for arrow format"
                    )
            elif result_format == "pandas":
                import pandas as pd

                result = pd.DataFrame(rows, columns=columns)
            else:  # list
                result = [dict(zip(columns, row)) for row in rows]

            return ExecuteSQLResult(
                success=True,
                sql_query=sql,
                sql_return=result,
                row_count=row_count,
                result_format=result_format,
            )
        except Exception as e:
            ex = self._handle_exception(e, sql)
            return ExecuteSQLResult(
                success=False,
                error=str(ex),
                sql_query=sql,
            )

    @override
    def execute_pandas(self, sql: str) -> ExecuteSQLResult:
        """Execute query and return pandas DataFrame."""
        return self.execute_query(sql, result_format="pandas")

    @override
    def execute_csv(self, sql: str) -> ExecuteSQLResult:
        """Execute query and return CSV format."""
        return self.execute_query(sql, result_format="csv")

    @override
    def execute_queries(self, queries: List[str]) -> List[Any]:
        """Execute multiple queries."""
        results = []
        self.connect()
        try:
            cursor = self.connection.cursor(buffered=True)
            for query in queries:
                cursor.execute(query)
                if cursor.description:
                    rows = cursor.fetchall()
                    columns = [desc[0] for desc in cursor.description]
                    results.append([dict(zip(columns, row)) for row in rows])
                else:
                    results.append(cursor.rowcount)
            self.connection.commit()
        except Exception as e:
            self.connection.rollback()
            raise self._handle_exception(e, "\n".join(queries))
        return results

    @override
    def execute_content_set(self, sql_query: str) -> ExecuteSQLResult:
        """Execute SET/USE commands."""
        try:
            self.connect()
            cursor = self.connection.cursor()

            # Parse USE command
            if sql_query.strip().upper().startswith("USE "):
                database_name = sql_query.strip()[4:].strip()
                if database_name:
                    self.database = database_name
                    self.database_name = database_name
                    # Note: In MySQL, USE doesn't require explicit connection switch
                    # but we'll update our internal state
                    logger.info(f"Switched to database: {database_name}")

            cursor.execute(sql_query)
            self.connection.commit()

            return ExecuteSQLResult(
                success=True,
                sql_query=sql_query,
                sql_return="Success",
                row_count=0,
            )
        except Exception as e:
            ex = self._handle_exception(e, sql_query)
            return ExecuteSQLResult(
                success=False,
                error=str(ex),
                sql_query=sql_query,
            )

    @override
    def get_databases(self, catalog_name: str = "", include_sys: bool = False) -> List[str]:
        """Get list of database names."""
        self.connect()
        cursor = self.connection.cursor()

        # StarRocks doesn't have catalogs in the traditional sense
        # We'll return the current database and other databases if accessible
        query = "SHOW DATABASES"
        if not include_sys:
            query += " WHERE Database NOT IN ('information_schema', 'mysql', 'performance_schema')"

        cursor.execute(query)
        return [row[0] for row in cursor.fetchall()]

    @override
    def get_tables(self, catalog_name: str = "", database_name: str = "", schema_name: str = "") -> List[str]:
        """Get all table names."""
        self.connect()
        cursor = self.connection.cursor()

        # StarRocks doesn't support schema_name (no schema layer)
        # Use SHOW TABLE STATUS or information_schema
        db_name = database_name or self.database
        if not db_name:
            return []

        try:
            # Ensure we have the correct database context
            if self.database != db_name:
                cursor.execute(f"USE `{db_name}`")
                self.database = db_name
                logger.debug(f"Switched to database context: {db_name}")
        except Exception as e:
            logger.warning(f"Failed to switch to database {db_name}: {e}")

        # Fix: Remove FROM clause (database context already set)
        cursor.execute("SHOW TABLES")
        return [row[0] for row in cursor.fetchall()]

    @override
    def get_views(self, catalog_name: str = "", database_name: str = "", schema_name: str = "") -> List[str]:
        """Get all view names."""
        self.connect()
        cursor = self.connection.cursor()

        db_name = database_name or self.database
        if not db_name:
            return []

        # In StarRocks, views are also in SHOW TABLES but we can filter
        # For now, return empty list as StarRocks view handling may differ
        return []

    @override
    def full_name(
        self, catalog_name: str = "", database_name: str = "", schema_name: str = "", table_name: str = ""
    ) -> str:
        """Get the full name of the table."""
        # StarRocks uses catalog.database.table format
        parts = []
        if catalog_name:
            parts.append(f"`{catalog_name}`")
        if database_name:
            parts.append(f"`{database_name}`")
        # StarRocks doesn't have schema, so we skip schema_name
        if table_name:
            parts.append(f"`{table_name}`")

        return ".".join(parts) if parts else ""

    @override
    def do_switch_context(self, catalog_name: str = "", database_name: str = "", schema_name: str = ""):
        """Switch context for StarRocks."""
        self.connect()

        if database_name:
            self.database = database_name
            self.database_name = database_name

        if catalog_name:
            self.catalog = catalog_name
            self.catalog_name = catalog_name

        # StarRocks USE command
        if database_name:
            cursor = self.connection.cursor()
            cursor.execute(f"USE `{database_name}`")

    @override
    def get_tables_with_ddl(
        self, catalog_name: str = "", database_name: str = "", schema_name: str = "", tables: Optional[List[str]] = None
    ) -> List[Dict[str, str]]:
        """
        Get all tables with DDL definitions.

        This is a critical method for DDL RAG retrieval.
        Returns table schemas with their DDL definitions.
        """
        self.connect()
        cursor = self.connection.cursor()

        db_name = database_name or self.database
        if not db_name:
            logger.warning("No database name specified for get_tables_with_ddl")
            return []

        try:
            # Ensure we have the correct database context
            try:
                # Switch to the database context if not already there
                if self.database != db_name:
                    cursor.execute(f"USE `{db_name}`")
                    self.database = db_name
                    logger.debug(f"Switched to database context: {db_name}")
            except Exception as e:
                logger.warning(f"Failed to switch to database {db_name}: {e}")

            # Get table names
            if tables:
                # Get DDL for specific tables
                table_list = []
                for table in tables:
                    try:
                        # Fix: Remove database name from SHOW CREATE TABLE (database context already set)
                        cursor.execute(f"SHOW CREATE TABLE `{table}`")
                        result = cursor.fetchone()
                        if result and len(result) >= 2:
                            create_statement = result[1]

                            # Fix truncated DDL
                            create_statement = self._fix_truncated_ddl(create_statement)

                            table_list.append({
                                "identifier": self.identifier(
                                    catalog_name=catalog_name,
                                    database_name=db_name,
                                    table_name=table,
                                ),
                                "catalog_name": catalog_name,
                                "database_name": db_name,
                                "schema_name": "",  # StarRocks doesn't have schema
                                "table_name": table,
                                "definition": create_statement,
                                "table_type": "table",
                            })
                    except Exception as e:
                        logger.warning(f"Failed to get DDL for table {table}: {e}")

                # Log result count
                if table_list:
                    logger.debug(f"Retrieved DDL for {len(table_list)} tables from database")
                else:
                    logger.warning(
                        f"get_tables_with_ddl returned 0 tables. "
                        f"DB: {db_name}, "
                        f"Filter: {tables}"
                    )

                return table_list
            else:
                # Get all tables with DDL
                # First, get all table names
                cursor.execute("SHOW TABLES")  # Fix: Remove FROM clause (database context already set)
                all_tables = [row[0] for row in cursor.fetchall()]

                table_list = []
                for table in all_tables:
                    try:
                        # Fix: Remove database name from SHOW CREATE TABLE (database context already set)
                        cursor.execute(f"SHOW CREATE TABLE `{table}`")
                        result = cursor.fetchone()
                        if result and len(result) >= 2:
                            create_statement = result[1]

                            # Fix truncated DDL
                            create_statement = self._fix_truncated_ddl(create_statement)

                            table_list.append({
                                "identifier": self.identifier(
                                    catalog_name=catalog_name,
                                    database_name=db_name,
                                    table_name=table,
                                ),
                                "catalog_name": catalog_name,
                                "database_name": db_name,
                                "schema_name": "",
                                "table_name": table,
                                "definition": create_statement,
                                "table_type": "table",
                            })
                    except Exception as e:
                        logger.warning(f"Failed to get DDL for table {table}: {e}")

                # Log result count
                if table_list:
                    logger.debug(f"Retrieved DDL for {len(table_list)} tables from database")
                else:
                    logger.warning(
                        f"get_tables_with_ddl returned 0 tables. "
                        f"DB: {db_name}, "
                        f"Schema: {schema_name}, "
                        f"Filter: {tables}"
                    )

                return table_list

        except Exception as e:
            logger.error(f"Failed to get tables with DDL from database {db_name}: {e}")
            return []

    @override
    def get_schema(
        self, catalog_name: str = "", database_name: str = "", schema_name: str = "", table_name: str = ""
    ) -> List[Dict[str, str]]:
        """Get schema information for a table."""
        if not table_name:
            return []

        self.connect()
        cursor = self.connection.cursor()

        db_name = database_name or self.database
        if not db_name:
            return []

        try:
            # Get table structure
            cursor.execute(f"DESCRIBE `{db_name}`.`{table_name}`")
            rows = cursor.fetchall()

            schema_list = []
            for row in rows:
                # DESCRIBE output: Field, Type, Null, Key, Default, Extra
                schema_list.append({
                    "name": row[0],
                    "type": row[1],
                    "nullable": row[2].upper() == "YES" if row[2] else True,
                    "default_value": row[4],
                    "pk": row[5] if len(row) > 5 and row[5] else "",
                })

            return schema_list
        except Exception as e:
            logger.error(f"Failed to get schema for table {table_name}: {e}")
            return []

    @override
    def get_sample_rows(
        self,
        tables: Optional[List[str]] = None,
        top_n: int = 5,
        catalog_name: str = "",
        database_name: str = "",
        schema_name: str = "",
        table_type: TABLE_TYPE = "table",
    ) -> List[Dict[str, Any]]:
        """Get sample rows from tables."""
        self.connect()
        cursor = self.connection.cursor()

        db_name = database_name or self.database
        if not db_name:
            return []

        try:
            # Ensure we have the correct database context
            if self.database != db_name:
                cursor.execute(f"USE `{db_name}`")
                self.database = db_name
                logger.debug(f"Switched to database context: {db_name}")
        except Exception as e:
            logger.warning(f"Failed to switch to database {db_name}: {e}")

        samples = []

        if tables:
            for table_name in tables:
                try:
                    # Fix: Remove database name from query (database context already set)
                    query = f"SELECT * FROM `{table_name}` LIMIT {top_n}"
                    cursor.execute(query)

                    if cursor.description:
                        columns = [desc[0] for desc in cursor.description]
                        rows = cursor.fetchall()

                        # Convert to CSV format
                        import csv
                        from io import StringIO

                        output = StringIO()
                        writer = csv.writer(output)
                        writer.writerow(columns)
                        writer.writerows(rows)

                        samples.append({
                            "catalog_name": catalog_name,
                            "database_name": db_name,
                            "schema_name": "",
                            "table_name": table_name,
                            "sample_rows": output.getvalue(),
                        })
                except Exception as e:
                    logger.warning(f"Failed to get sample rows for table {table_name}: {e}")
        else:
            # Get all tables
            cursor.execute("SHOW TABLES")  # Fix: Remove FROM clause (database context already set)
            all_tables = [row[0] for row in cursor.fetchall()]

            for table_name in all_tables:
                try:
                    # Fix: Remove database name from query (database context already set)
                    query = f"SELECT * FROM `{table_name}` LIMIT {top_n}"
                    cursor.execute(query)

                    if cursor.description:
                        columns = [desc[0] for desc in cursor.description]
                        rows = cursor.fetchall()

                        import csv
                        from io import StringIO

                        output = StringIO()
                        writer = csv.writer(output)
                        writer.writerow(columns)
                        writer.writerows(rows)

                        samples.append({
                            "catalog_name": catalog_name,
                            "database_name": db_name,
                            "schema_name": "",
                            "table_name": table_name,
                            "sample_rows": output.getvalue(),
                        })
                except Exception as e:
                    logger.warning(f"Failed to get sample rows for table {table_name}: {e}")

        return samples

    def to_dict(self) -> Dict[str, Any]:
        """Convert connector to serializable dictionary."""
        return {
            "db_type": DBType.STARROCKS,
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "catalog": self.catalog,
        }

    def get_type(self) -> str:
        return DBType.STARROCKS
