
import unittest
from unittest.mock import MagicMock
from datus.api.event_converter import DeepResearchEventConverter

class TestDeepResearchEventConverter(unittest.TestCase):
    def setUp(self):
        self.converter = DeepResearchEventConverter()

    def test_escape_markdown_table_cell(self):
        """Test Markdown table cell escaping."""
        # Normal text
        self.assertEqual(self.converter._escape_markdown_table_cell("hello"), "hello")
        
        # Pipe escaping
        self.assertEqual(self.converter._escape_markdown_table_cell("a|b"), "a&#124;b")
        
        # Newline escaping
        self.assertEqual(self.converter._escape_markdown_table_cell("a\nb"), "a b")
        
        # Combined
        self.assertEqual(self.converter._escape_markdown_table_cell("a|b\nc"), "a&#124;b c")
        
        # None handling
        self.assertEqual(self.converter._escape_markdown_table_cell(None), "-")
        
        # Non-string input
        self.assertEqual(self.converter._escape_markdown_table_cell(123), "123")

    def test_add_field_comment_security(self):
        """Test security limits in _add_field_comment."""
        sql_line = "SELECT column FROM table"
        field_name = "column"
        comment = "test comment"
        
        # Normal case
        result = self.converter._add_field_comment(field_name, comment, sql_line)
        self.assertEqual(result, "SELECT column -- test comment FROM table")
        
        # Long field name (should be ignored)
        long_field = "a" * 300
        result = self.converter._add_field_comment(long_field, comment, sql_line)
        self.assertEqual(result, sql_line)  # Unchanged
        
        # Long SQL line (should be ignored)
        long_sql = "SELECT " + "a" * 5000 + " FROM table"
        result = self.converter._add_field_comment(field_name, comment, long_sql)
        self.assertEqual(result, long_sql)  # Unchanged

if __name__ == "__main__":
    unittest.main()
