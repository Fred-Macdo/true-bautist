#!/usr/bin/env python3
"""
Test Runner for Alpaca Trading System Tests
Run with: python run_tests.py
"""

import unittest
import sys
import os
from unittest import TestLoader, TextTestRunner, TestSuite

def run_tests():
    """Run all tests in the test suite"""
    # Discover and load all test modules
    loader = TestLoader()
    
    # Get current directory to find test files
    start_dir = os.path.dirname(os.path.abspath(__file__))
    
    # First run the specific test files
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    # Create test runner
    runner = TextTestRunner(verbosity=2)
    
    # Run the test suite
    result = runner.run(suite)
    
    # Return the number of failures/errors
    return len(result.failures) + len(result.errors)

if __name__ == '__main__':
    # Run the tests
    print("Running Alpaca Trading System tests...")
    failures = run_tests()
    
    # Exit with appropriate code
    sys.exit(failures)
