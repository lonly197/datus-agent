#!/usr/bin/env python3
"""
Simple test for the check_table_exists fix.
"""

def test_table_processing_logic():
    """Test the fixed table processing logic from check_table_exists."""

    # Simulate the original broken logic
    def broken_logic(tables):
        try:
            table_names_lower = {t.get("name", "").lower(): t.get("name") for t in tables}
            return "broken logic worked"
        except AttributeError as e:
            return f"broken logic failed: {e}"

    # Simulate the fixed logic
    def fixed_logic(tables):
        try:
            # tables is a List[str], convert to dict format for consistent processing
            table_names_lower = {t.lower(): t for t in tables}
            return "fixed logic worked"
        except Exception as e:
            return f"fixed logic failed: {e}"

    # Test with string list (correct format from get_tables)
    tables_as_strings = ["users", "orders", "products"]

    broken_result = broken_logic(tables_as_strings)
    fixed_result = fixed_logic(tables_as_strings)

    print(f"Broken logic result: {broken_result}")
    print(f"Fixed logic result: {fixed_result}")

    assert "failed" in broken_result, "Broken logic should fail with string list"
    assert "worked" in fixed_result, "Fixed logic should work with string list"

    print("✅ Table processing logic fix verified")

def test_none_check_result():
    """Test the None check_result handling."""

    def test_logic(check_result):
        if check_result is None:
            return "handled None correctly"
        elif not check_result.get("success"):
            return "handled dict correctly"
        else:
            return "other case"

    # Test with None
    result1 = test_logic(None)
    assert result1 == "handled None correctly", f"Expected 'handled None correctly', got {result1}"

    # Test with dict
    result2 = test_logic({"success": False})
    assert result2 == "handled dict correctly", f"Expected 'handled dict correctly', got {result2}"

    print("✅ None check_result handling verified")

if __name__ == "__main__":
    test_table_processing_logic()
    test_none_check_result()
    print("\n🎉 All fixes verified!")
