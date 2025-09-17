
import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestExample(unittest.TestCase):
    
    def test_example(self):
        """Example test case."""
        expected = 4
        
        result = 2 + 2
        
        self.assertEqual(result, expected)

if __name__ == '__main__':
    unittest.main()
