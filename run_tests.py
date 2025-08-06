#!/usr/bin/env python3
"""
Comprehensive test runner for robot MCP project.
Runs all unit tests and provides detailed reporting.
"""

import unittest
import sys
import os
import time
from io import StringIO
from typing import Dict, List, Tuple

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Test modules to run
TEST_MODULES = [
    'tests.test_llm_providers',
    'tests.test_robot_controller', 
    'tests.test_agent',
    'tests.test_mcp_server',
    'tests.test_config'
]

class DetailedTestResult(unittest.TestResult):
    """Custom test result class that provides detailed reporting."""
    
    def __init__(self):
        super().__init__()
        self.successes = []
        self.test_timings = {}
        self.start_time = None
        
    def startTest(self, test):
        super().startTest(test)
        self.start_time = time.time()
        
    def stopTest(self, test):
        super().stopTest(test)
        if self.start_time:
            duration = time.time() - self.start_time
            self.test_timings[str(test)] = duration
            
    def addSuccess(self, test):
        super().addSuccess(test)
        self.successes.append(test)
        
    def addError(self, test, err):
        super().addError(test, err)
        
    def addFailure(self, test, err):
        super().addFailure(test, err)
        
    def addSkip(self, test, reason):
        super().addSkip(test, reason)


def run_test_module(module_name: str) -> Tuple[DetailedTestResult, int]:
    """Run tests for a specific module."""
    print(f"\n{'='*60}")
    print(f"Running tests for: {module_name}")
    print('='*60)
    
    try:
        # Import the test module
        __import__(module_name)
        module = sys.modules[module_name]
        
        # Create test suite
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(module)
        
        # Run tests with custom result
        result = DetailedTestResult()
        suite.run(result)
        
        # Print results
        tests_run = result.testsRun
        successes = len(result.successes)
        failures = len(result.failures)
        errors = len(result.errors)
        skipped = len(result.skipped)
        
        print(f"\nResults for {module_name}:")
        print(f"  Tests run: {tests_run}")
        print(f"  Successes: {successes}")
        print(f"  Failures: {failures}")
        print(f"  Errors: {errors}")
        print(f"  Skipped: {skipped}")
        
        # Print failures and errors
        if result.failures:
            print(f"\n❌ FAILURES ({len(result.failures)}):")
            for test, traceback in result.failures:
                error_msg = traceback.split('AssertionError: ')[-1].split('\n')[0]
                print(f"  - {test}: {error_msg}")
                
        if result.errors:
            print(f"\n💥 ERRORS ({len(result.errors)}):")
            for test, traceback in result.errors:
                error_msg = traceback.split('\\n')[-2] if '\\n' in traceback else traceback
                print(f"  - {test}: {error_msg}")
                
        if result.skipped:
            print(f"\n⏭️  SKIPPED ({len(result.skipped)}):")
            for test, reason in result.skipped:
                print(f"  - {test}: {reason}")
        
        # Print timing for slow tests
        slow_tests = [(test, time) for test, time in result.test_timings.items() if time > 0.1]
        if slow_tests:
            print(f"\n⏱️  SLOW TESTS (>0.1s):")
            for test, duration in sorted(slow_tests, key=lambda x: x[1], reverse=True)[:5]:
                print(f"  - {test}: {duration:.3f}s")
                
        return result, tests_run
        
    except ImportError as e:
        print(f"❌ Failed to import {module_name}: {e}")
        return None, 0
    except Exception as e:
        print(f"💥 Error running {module_name}: {e}")
        return None, 0


def check_dependencies() -> bool:
    """Check if all required dependencies are available."""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'unittest',
        'numpy',
        'anthropic',
        'google.genai',
        'mcp',
        'dotenv',
        'PIL'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} (missing)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Install them with: pip install -r requirements.txt")
        return False
    
    return True


def generate_coverage_report(results: Dict[str, Tuple[DetailedTestResult, int]]) -> None:
    """Generate a coverage report based on test results."""
    print(f"\n{'='*60}")
    print("TEST COVERAGE REPORT")
    print('='*60)
    
    total_tests = 0
    total_passed = 0
    total_failed = 0
    total_errors = 0
    total_skipped = 0
    
    for module_name, (result, tests_run) in results.items():
        if result:
            passed = len(result.successes)
            failed = len(result.failures)
            errors = len(result.errors)
            skipped = len(result.skipped)
            
            total_tests += tests_run
            total_passed += passed
            total_failed += failed
            total_errors += errors
            total_skipped += skipped
            
            success_rate = (passed / tests_run * 100) if tests_run > 0 else 0
            status = "✅" if success_rate == 100 else "⚠️" if success_rate >= 80 else "❌"
            
            print(f"{status} {module_name}: {passed}/{tests_run} ({success_rate:.1f}%)")
    
    print(f"\n📊 OVERALL SUMMARY:")
    print(f"  Total tests: {total_tests}")
    print(f"  Passed: {total_passed}")
    print(f"  Failed: {total_failed}")
    print(f"  Errors: {total_errors}")
    print(f"  Skipped: {total_skipped}")
    
    if total_tests > 0:
        overall_success_rate = (total_passed / total_tests * 100)
        print(f"  Success rate: {overall_success_rate:.1f}%")
        
        if overall_success_rate == 100:
            print(f"\n🎉 ALL TESTS PASSED! 🎉")
        elif overall_success_rate >= 90:
            print(f"\n✅ Excellent test coverage!")
        elif overall_success_rate >= 80:
            print(f"\n⚠️  Good test coverage, but room for improvement.")
        else:
            print(f"\n❌ Test coverage needs improvement.")


def main():
    """Main test runner function."""
    print("🧪 Robot MCP Project Test Suite")
    print("=" * 60)
    
    # Check dependencies first
    if not check_dependencies():
        print("\n❌ Cannot run tests due to missing dependencies.")
        sys.exit(1)
    
    # Set environment variables for testing
    os.environ['ANTHROPIC_API_KEY'] = 'test_key'
    os.environ['GEMINI_API_KEY'] = 'test_key'
    
    start_time = time.time()
    results = {}
    
    # Run each test module
    for module_name in TEST_MODULES:
        result, tests_run = run_test_module(module_name)
        results[module_name] = (result, tests_run)
    
    # Generate coverage report
    generate_coverage_report(results)
    
    # Print timing
    total_time = time.time() - start_time
    print(f"\n⏱️  Total execution time: {total_time:.2f}s")
    
    # Determine exit code
    has_failures = any(
        result and (result.failures or result.errors) 
        for result, _ in results.values()
    )
    
    if has_failures:
        print("\n❌ Some tests failed. Please review the output above.")
        sys.exit(1)
    else:
        print("\n✅ All tests completed successfully!")
        sys.exit(0)


if __name__ == '__main__':
    main() 