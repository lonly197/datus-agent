
import unittest
from unittest.mock import MagicMock
import datus.api.service
from datus.api.service import DatusAPIService

import sys
import inspect
class TestApiService(unittest.TestCase):
    def setUp(self):
        print(f"DEBUG: datus.api.service file: {sys.modules['datus.api.service'].__file__}")
        print(f"DEBUG: _parse_csv_to_list source:\n{inspect.getsource(DatusAPIService._parse_csv_to_list)}")
        args = MagicMock()
        self.service = DatusAPIService(args)

    def test_parse_csv_limit(self):
        """Test CSV parsing size limit."""
        # Create a large CSV string > 10MB
        # 10MB = 10 * 1024 * 1024 bytes
        # "a,b\n1,2\n" is 8 bytes.
        large_csv = "col1,col2\n" + "1,2\n" * (1024 * 1024) # ~4MB
        # Let's make it definitely larger than 10MB
        large_csv = "col1,col2\n" + ("x" * 1024 + ",y\n") * 10240 # 10KB * 10240 = 100MB approx
        
        result = self.service._parse_csv_to_list(large_csv)
        self.assertEqual(result, [])

    def test_parse_csv_valid(self):
        """Test valid CSV parsing."""
        csv_data = "id,name\n1,Alice\n2,Bob"
        result = self.service._parse_csv_to_list(csv_data)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["name"], "Alice")
        self.assertEqual(result[1]["name"], "Bob")

if __name__ == "__main__":
    unittest.main()
