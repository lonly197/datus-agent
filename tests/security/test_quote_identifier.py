# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details

"""
Security tests for quote_identifier() function.

Tests SQL injection prevention for various edge cases including:
- SQL injection attempts
- Reserved keywords
- Special characters
- Unicode identifiers
- Dialect-specific quoting
"""

import pytest

from datus.utils.sql_utils import quote_identifier


class TestQuoteIdentifierSQLInjection:
    """Test SQL injection prevention."""

    def test_sql_injection_drop_table(self):
        """Test that SQL injection attempts are properly quoted."""
        malicious = "users; DROP TABLE users--"
        result_postgres = quote_identifier(malicious, "postgres")
        result_mysql = quote_identifier(malicious, "mysql")

        # Should be quoted as a single identifier, not executable
        assert result_postgres == '"users; DROP TABLE users--"'
        assert result_mysql == "`users; DROP TABLE users--`"

    def test_sql_injection_union_select(self):
        """Test UNION SELECT injection is prevented."""
        malicious = "users' UNION SELECT * FROM passwords--"
        result = quote_identifier(malicious, "postgres")

        # Entire string should be quoted as identifier
        assert result == '"users\' UNION SELECT * FROM passwords--"'

    def test_sql_injection_comment_termination(self):
        """Test comment termination injection is prevented."""
        malicious = "users/* comment */--"
        result = quote_identifier(malicious, "postgres")

        assert result == '"users/* comment */--"'


class TestQuoteIdentifierReservedKeywords:
    """Test quoting of SQL reserved keywords."""

    def test_reserved_keyword_order(self):
        """Test ORDER keyword is properly quoted."""
        assert quote_identifier("order", "postgres") == '"order"'
        assert quote_identifier("order", "mysql") == "`order`"
        assert quote_identifier("order", "snowflake") == '"order"'

    def test_reserved_keyword_group(self):
        """Test GROUP keyword is properly quoted."""
        assert quote_identifier("group", "postgres") == '"group"'
        assert quote_identifier("group", "mysql") == "`group`"

    def test_reserved_keyword_select(self):
        """Test SELECT keyword is properly quoted."""
        assert quote_identifier("select", "postgres") == '"select"'
        assert quote_identifier("select", "mysql") == "`select`"

    def test_reserved_keyword_from(self):
        """Test FROM keyword is properly quoted."""
        assert quote_identifier("from", "postgres") == '"from"'
        assert quote_identifier("from", "mysql") == "`from`"


class TestQuoteIdentifierSpecialCharacters:
    """Test handling of special characters in identifiers."""

    def test_identifier_with_dash(self):
        """Test identifiers with dashes."""
        identifier = "table-with-dash"
        assert quote_identifier(identifier, "postgres") == '"table-with-dash"'
        assert quote_identifier(identifier, "snowflake") == '"table-with-dash"'

    def test_identifier_with_dots(self):
        """Test identifiers with dots."""
        identifier = "table.with.dots"
        assert quote_identifier(identifier, "postgres") == '"table.with.dots"'

    def test_identifier_with_spaces(self):
        """Test identifiers with spaces."""
        identifier = "table with spaces"
        assert quote_identifier(identifier, "postgres") == '"table with spaces"'

    def test_identifier_with_at_sign(self):
        """Test identifiers with @ sign (common in some databases)."""
        identifier = "table@name"
        assert quote_identifier(identifier, "postgres") == '"table@name"'


class TestQuoteIdentifierUnicode:
    """Test Unicode identifier support."""

    def test_chinese_characters(self):
        """Test Chinese characters in identifiers."""
        identifier = "用户表"
        result = quote_identifier(identifier, "postgres")
        assert result == '"用户表"'

    def test_cyrillic_characters(self):
        """Test Cyrillic characters in identifiers."""
        identifier = "таблица"
        result = quote_identifier(identifier, "snowflake")
        assert result == '"таблица"'

    def test_mixed_unicode_and_ascii(self):
        """Test mixed Unicode and ASCII characters."""
        identifier = "用户users表"
        result = quote_identifier(identifier, "postgres")
        assert result == '"用户users表"'


class TestQuoteIdentifierCaseSensitivity:
    """Test case sensitivity handling."""

    def test_mixed_case_identifier(self):
        """Test mixed case identifiers."""
        identifier = "MyTable"
        result_postgres = quote_identifier(identifier, "postgres")
        result_mysql = quote_identifier(identifier, "mysql")

        # Should preserve case when quoted
        assert result_postgres == '"MyTable"'
        assert result_mysql == "`MyTable`"

    def test_all_uppercase(self):
        """Test all uppercase identifiers."""
        identifier = "MYTABLE"
        assert quote_identifier(identifier, "postgres") == '"MYTABLE"'

    def test_all_lowercase(self):
        """Test all lowercase identifiers."""
        identifier = "mytable"
        assert quote_identifier(identifier, "postgres") == '"mytable"'


class TestQuoteIdentifierDialectCoverage:
    """Test dialect-specific quoting rules."""

    def test_postgres_dialect(self):
        """Test PostgreSQL uses double quotes."""
        assert quote_identifier("users", "postgres") == '"users"'
        assert quote_identifier("order", "postgres") == '"order"'

    def test_mysql_dialect(self):
        """Test MySQL uses backticks."""
        assert quote_identifier("users", "mysql") == "`users`"
        assert quote_identifier("order", "mysql") == "`order`"

    def test_snowflake_dialect(self):
        """Test Snowflake uses double quotes."""
        assert quote_identifier("users", "snowflake") == '"users"'
        assert quote_identifier("order", "snowflake") == '"order"'

    def test_duckdb_dialect(self):
        """Test DuckDB uses double quotes."""
        assert quote_identifier("users", "duckdb") == '"users"'

    def test_mssql_dialect(self):
        """Test SQL Server uses brackets."""
        # Note: sqlglot may handle this differently
        result = quote_identifier("users", "mssql")
        assert "[" in result or '"' in result  # Either brackets or quotes acceptable

    def test_sqlite_dialect(self):
        """Test SQLite uses brackets or double quotes."""
        result = quote_identifier("users", "sqlite")
        # Acceptable formats
        assert result in ['"users"', "[users]"]


class TestQuoteIdentifierEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_string(self):
        """Test empty string identifier."""
        result = quote_identifier("", "postgres")
        assert result == ""  # Should return empty string as-is

    def test_none_input(self):
        """Test None input."""
        result = quote_identifier(None, "postgres")
        assert result is None  # Should return None as-is

    def test_single_character(self):
        """Test single character identifier."""
        assert quote_identifier("x", "postgres") == '"x"'

    def test_very_long_identifier(self):
        """Test very long identifier (most DBs have limits)."""
        identifier = "a" * 1000
        result = quote_identifier(identifier, "postgres")
        assert '"' + identifier + '"' == result

    def test_identifier_starting_with_number(self):
        """Test identifiers starting with numbers."""
        identifier = "123table"
        result = quote_identifier(identifier, "postgres")
        assert result == '"123table"'

    def test_identifier_starting_with_underscore(self):
        """Test identifiers starting with underscore."""
        identifier = "_private_table"
        result = quote_identifier(identifier, "postgres")
        assert result == '"_private_table"'


class TestQuoteIdentifierIdempotency:
    """Test that double-quoting doesn't cause issues."""

    def test_double_quote_safe(self):
        """Test that already quoted identifiers are handled correctly."""
        # If an identifier is already quoted, quote_identifier should quote it again
        # This is expected behavior - caller should avoid double-quoting
        quoted = '"users"'
        result = quote_identifier(quoted, "postgres")

        # Should add another layer of quotes
        assert result == '"""users"""' or result == '"users"'

    def test_unquoted_then_quoted(self):
        """Test normal usage - unquoted identifier becomes quoted."""
        identifier = "users"
        result = quote_identifier(identifier, "postgres")

        assert result == '"users"'


class TestQuoteIdentifierIntegration:
    """Integration tests with SQL queries."""

    def test_select_query_construction(self):
        """Test using quoted identifier in SELECT query."""
        table = "order"
        safe_table = quote_identifier(table, "postgres")

        query = f"SELECT * FROM {safe_table}"
        assert query == 'SELECT * FROM "order"'

    def test_join_query_construction(self):
        """Test using quoted identifiers in JOIN query."""
        table1 = "users"
        table2 = "order"
        safe_table1 = quote_identifier(table1, "postgres")
        safe_table2 = quote_identifier(table2, "postgres")

        query = f"SELECT * FROM {safe_table1} JOIN {safe_table2} ON {safe_table1}.id = {safe_table2}.user_id"
        assert query == 'SELECT * FROM "users" JOIN "order" ON "users".id = "order".user_id'

    def test_column_reference_construction(self):
        """Test using quoted column identifier."""
        table = "users"
        column = "group"
        safe_table = quote_identifier(table, "postgres")
        safe_column = quote_identifier(column, "postgres")

        query = f"SELECT {safe_column} FROM {safe_table}"
        assert query == 'SELECT "group" FROM "users"'
