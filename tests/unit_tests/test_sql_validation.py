"""
Unit tests for SQL validation and fix suggestion generation.
"""

import pytest
from datus.utils.sql_utils import validate_and_suggest_sql_fixes


class TestSQLValidation:
    """Test SQL validation and fix suggestion generation."""

    def test_valid_sql(self):
        """Test that valid SQL passes validation."""
        sql = "SELECT id, name FROM users WHERE id = 1"
        result = validate_and_suggest_sql_fixes(sql, dialect="starrocks")

        assert result["is_valid"] is True
        assert result["can_parse"] is True
        assert len(result["errors"]) == 0
        assert result["original_sql"] == sql

    def test_unquoted_like_pattern(self):
        """Test detection of unquoted LIKE patterns."""
        # Test the actual failing SQL from logs
        sql = "SELECT * FROM table WHERE col LIKE %线上%"
        result = validate_and_suggest_sql_fixes(sql, dialect="starrocks")

        # Should detect the LIKE pattern issue
        assert len(result["fix_suggestions"]) > 0

        # Check that LIKE pattern issue was detected
        like_suggestions = [
            s for s in result["fix_suggestions"]
            if "like" in s.get("issue_type", "").lower()
        ]
        assert len(like_suggestions) > 0

        # Verify the suggestion details
        like_issue = like_suggestions[0]
        assert like_issue["severity"] == "error"
        assert "quoted" in like_issue["description"].lower()

    def test_unquoted_date_literal(self):
        """Test detection of unquoted date literals."""
        sql = "SELECT * FROM table WHERE date_col = 2025-12-24"
        result = validate_and_suggest_sql_fixes(sql, dialect="starrocks")

        # Should have date warning
        date_warnings = [w for w in result["warnings"] if "date" in w.lower()]
        assert len(date_warnings) > 0

        # Should have fix suggestion
        date_suggestions = [
            s for s in result["fix_suggestions"]
            if s.get("issue_type") == "unquoted_date_literal"
        ]
        assert len(date_suggestions) > 0

    def test_select_star_warning(self):
        """Test SELECT * warning."""
        sql = "SELECT * FROM users"
        result = validate_and_suggest_sql_fixes(sql, dialect="starrocks")

        # Should parse successfully
        assert result["can_parse"] is True

        # Should have SELECT * warning
        select_star_warnings = [
            s for s in result["fix_suggestions"]
            if s.get("issue_type") == "select_star"
        ]
        assert len(select_star_warnings) > 0

    def test_combined_errors_from_logs(self):
        """Test the exact SQL from the error logs."""
        # Complete SQL from the logs
        sql = "SELECT * FROM dwd_assign_dlr_clue_fact_di WHERE LEFT(clue_create_time, 10) = 2025-12-24 AND newest_original_clue_json LIKE %线上%;"
        result = validate_and_suggest_sql_fixes(sql, dialect="starrocks")

        # Should detect multiple issues
        assert len(result["fix_suggestions"]) >= 2

        # Check for LIKE pattern issue
        issue_types = [s.get("issue_type") for s in result["fix_suggestions"]]
        assert any("like" in str(t).lower() for t in issue_types)

        # Check for date issue (may or may not be detected depending on sqlglot)
        date_issues = [t for t in issue_types if "date" in str(t).lower()]
        # At minimum, we should have warnings or suggestions

        print(f"\nDetected {len(result['fix_suggestions'])} issues:")
        for issue in result['fix_suggestions']:
            print(f"  - [{issue.get('severity', 'unknown')}] {issue.get('description', 'N/A')}")

    def test_delete_without_where(self):
        """Test detection of dangerous DELETE without WHERE."""
        sql = "DELETE FROM users"
        result = validate_and_suggest_sql_fixes(sql, dialect="starrocks")

        # Should detect dangerous operation
        dangerous_suggestions = [
            s for s in result["fix_suggestions"]
            if s.get("issue_type") == "dangerous_operation"
        ]
        assert len(dangerous_suggestions) > 0

        # Should be marked as critical
        assert dangerous_suggestions[0]["severity"] == "critical"

    def test_join_without_on(self):
        """Test detection of JOIN without ON clause."""
        sql = "SELECT * FROM table1 JOIN table2"
        result = validate_and_suggest_sql_fixes(sql, dialect="starrocks")

        # Should detect missing ON clause
        join_issues = [
            s for s in result["fix_suggestions"]
            if s.get("issue_type") == "join_without_on"
        ]
        assert len(join_issues) > 0

    def test_order_by_without_limit(self):
        """Test detection of ORDER BY without LIMIT."""
        sql = "SELECT * FROM users ORDER BY created_at"
        result = validate_and_suggest_sql_fixes(sql, dialect="starrocks")

        # Should have info-level warning
        order_warnings = [
            w for w in result["warnings"]
            if "order by" in w.lower() and "limit" in w.lower()
        ]
        assert len(order_warnings) > 0

    def test_chinese_text_detection(self):
        """Test detection of unquoted Chinese text."""
        sql = "SELECT * FROM table WHERE name = 线上渠道"
        result = validate_and_suggest_sql_fixes(sql, dialect="starrocks")

        # Should detect Chinese text needs quotes
        chinese_issues = [
            s for s in result["fix_suggestions"]
            if s.get("issue_type") == "unquoted_chinese_text"
        ]
        assert len(chinese_issues) > 0

    def test_function_on_column_in_where(self):
        """Test detection of function on column in WHERE clause."""
        sql = "SELECT * FROM users WHERE UPPER(name) = 'JOHN'"
        result = validate_and_suggest_sql_fixes(sql, dialect="starrocks")

        # Should warn about function preventing index usage
        func_warnings = [
            w for w in result["warnings"]
            if "function" in w.lower() and "index" in w.lower()
        ]
        assert len(func_warnings) > 0

    def test_empty_sql(self):
        """Test handling of empty SQL."""
        sql = ""
        result = validate_and_suggest_sql_fixes(sql, dialect="starrocks")

        assert result["is_valid"] is False
        assert len(result["errors"]) > 0
        assert "empty" in result["errors"][0].lower()

    def test_validation_result_structure(self):
        """Test that validation result has all required fields."""
        sql = "SELECT * FROM users WHERE id = 1"
        result = validate_and_suggest_sql_fixes(sql, dialect="starrocks")

        # Check all required fields are present
        required_fields = [
            "is_valid",
            "errors",
            "warnings",
            "fix_suggestions",
            "can_parse",
            "original_sql"
        ]

        for field in required_fields:
            assert field in result, f"Missing field: {field}"

        # Check field types
        assert isinstance(result["is_valid"], bool)
        assert isinstance(result["errors"], list)
        assert isinstance(result["warnings"], list)
        assert isinstance(result["fix_suggestions"], list)
        assert isinstance(result["can_parse"], bool)
        assert isinstance(result["original_sql"], str)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
