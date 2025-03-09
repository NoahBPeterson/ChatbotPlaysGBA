#!/usr/bin/env python3
"""
mGBA-http Test Harness

A simple test script to verify the functionality of mGBA-http.
This script tests various API endpoints and displays the responses.
"""

import requests
import time
import json
from pprint import pprint
import argparse
import sys

# Default API base URL
API_BASE_URL = "http://localhost:5000"
DEFAULT_TIMEOUT = 5  # Default timeout in seconds

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_colored(message, color):
    print(f"{color}{message}{Colors.ENDC}")

def print_test_header(test_name):
    print("\n" + "="*80)
    print_colored(f"TEST: {test_name}", Colors.HEADER + Colors.BOLD)
    print("="*80)

def print_response(response):
    try:
        print_colored("Status Code:", Colors.BOLD)
        status_color = Colors.GREEN if response.status_code == 200 else Colors.FAIL
        print_colored(f"  {response.status_code} {response.reason}", status_color)
        
        print_colored("Response Headers:", Colors.BOLD)
        for header, value in response.headers.items():
            print(f"  {header}: {value}")
        
        print_colored("Response Body:", Colors.BOLD)
        try:
            pprint(response.json())
        except json.JSONDecodeError:
            print(response.text)
    except Exception as e:
        print_colored(f"Error printing response: {e}", Colors.FAIL)

def test_connection():
    """Test basic connection to the mGBA-http server."""
    print_test_header("Basic Connection Test")
    try:
        response = requests.get(f"{API_BASE_URL}/memory/romtitle", timeout=DEFAULT_TIMEOUT)
        print_response(response)
        if response.status_code == 200:
            print_colored("\n✅ Connection test PASSED", Colors.GREEN)
            return True
        else:
            print_colored("\n❌ Connection test FAILED", Colors.FAIL)
            return False
    except requests.exceptions.RequestException as e:
        print_colored(f"\n❌ Connection test FAILED: {e}", Colors.FAIL)
        return False

def test_button_press():
    """Test button press functionality."""
    print_test_header("Button Press Test")
    # Test pressing the A button
    try:
        response = requests.post(f"{API_BASE_URL}/mgba-http/button/tap", params={"key": "A"}, timeout=DEFAULT_TIMEOUT)
        print_response(response)
        if response.status_code == 200:
            print_colored("\n✅ Button press test PASSED", Colors.GREEN)
            return True
        else:
            print_colored("\n❌ Button press test FAILED", Colors.FAIL)
            return False
    except requests.exceptions.RequestException as e:
        print_colored(f"\n❌ Button press test FAILED: {e}", Colors.FAIL)
        return False

def test_memory_read():
    """Test memory reading functionality."""
    print_test_header("Memory Read Test")
    try:
        # The memory domain endpoints require the domain name parameter
        memory_domain = "internal"  # Default memory domain in mGBA
        
        # Get memory at address 0 (size 16 bytes)
        response = requests.get(
            f"{API_BASE_URL}/memorydomain/readrange", 
            params={"domain": memory_domain, "address": 0, "size": 16},
            timeout=DEFAULT_TIMEOUT
        )
        print_response(response)
        if response.status_code == 200:
            print_colored("\n✅ Memory read test PASSED", Colors.GREEN)
            return True
        else:
            print_colored("\n❌ Memory read test FAILED", Colors.FAIL)
            return False
    except requests.exceptions.RequestException as e:
        print_colored(f"\n❌ Memory read test FAILED: {e}", Colors.FAIL)
        return False

def test_multiple_button_sequence():
    """Test a sequence of button presses."""
    print_test_header("Button Sequence Test")
    try:
        # Press Up, then A
        print_colored("Pressing Up...", Colors.CYAN)
        up_response = requests.post(f"{API_BASE_URL}/mgba-http/button/tap", params={"key": "Up"}, timeout=DEFAULT_TIMEOUT)
        print_response(up_response)
        time.sleep(0.5)  # Short delay between button presses
        
        print_colored("\nPressing A...", Colors.CYAN)
        a_response = requests.post(f"{API_BASE_URL}/mgba-http/button/tap", params={"key": "A"}, timeout=DEFAULT_TIMEOUT)
        print_response(a_response)
        
        if up_response.status_code == 200 and a_response.status_code == 200:
            print_colored("\n✅ Button sequence test PASSED", Colors.GREEN)
            return True
        else:
            print_colored("\n❌ Button sequence test FAILED", Colors.FAIL)
            return False
    except requests.exceptions.RequestException as e:
        print_colored(f"\n❌ Button sequence test FAILED: {e}", Colors.FAIL)
        return False

def test_multiple_buttons():
    """Test pressing multiple buttons simultaneously."""
    print_test_header("Multiple Buttons Test")
    try:
        # Press Up+A simultaneously
        response = requests.post(
            f"{API_BASE_URL}/mgba-http/button/tapmany", 
            params={"keys": ["Up", "A"]},
            timeout=DEFAULT_TIMEOUT
        )
        print_response(response)
        if response.status_code == 200:
            print_colored("\n✅ Multiple buttons test PASSED", Colors.GREEN)
            return True
        else:
            print_colored("\n❌ Multiple buttons test FAILED", Colors.FAIL)
            return False
    except requests.exceptions.RequestException as e:
        print_colored(f"\n❌ Multiple buttons test FAILED: {e}", Colors.FAIL)
        return False

def test_core_reset():
    """Test core reset functionality."""
    print_test_header("Core Reset Test")
    try:
        response = requests.post(f"{API_BASE_URL}/coreadapter/reset", timeout=DEFAULT_TIMEOUT)
        print_response(response)
        if response.status_code == 200:
            print_colored("\n✅ Core reset test PASSED", Colors.GREEN)
            return True
        else:
            print_colored("\n❌ Core reset test FAILED", Colors.FAIL)
            return False
    except requests.exceptions.RequestException as e:
        print_colored(f"\n❌ Core reset test FAILED: {e}", Colors.FAIL)
        return False

def test_memory_domain_info():
    """Test memory domain info retrieval."""
    print_test_header("Memory Domain Info Test")
    try:
        # The memory domain endpoints require the domain name parameter
        memory_domain = "internal"  # Default memory domain in mGBA
        
        name_response = requests.get(
            f"{API_BASE_URL}/memorydomain/name", 
            params={"domain": memory_domain},
            timeout=DEFAULT_TIMEOUT
        )
        
        base_response = requests.get(
            f"{API_BASE_URL}/memorydomain/base", 
            params={"domain": memory_domain},
            timeout=DEFAULT_TIMEOUT
        )
        
        size_response = requests.get(
            f"{API_BASE_URL}/memorydomain/size", 
            params={"domain": memory_domain},
            timeout=DEFAULT_TIMEOUT
        )
        
        print_colored("Memory Domain Name:", Colors.CYAN)
        print_response(name_response)
        
        print_colored("\nMemory Domain Base:", Colors.CYAN)
        print_response(base_response)
        
        print_colored("\nMemory Domain Size:", Colors.CYAN)
        print_response(size_response)
        
        passed = all(r.status_code == 200 for r in [name_response, base_response, size_response])
        
        if passed:
            print_colored("\n✅ Memory domain info test PASSED", Colors.GREEN)
            return True
        else:
            print_colored("\n❌ Memory domain info test FAILED", Colors.FAIL)
            return False
    except requests.exceptions.RequestException as e:
        print_colored(f"\n❌ Memory domain info test FAILED: {e}", Colors.FAIL)
        return False

def test_swagger_docs():
    """Test access to the Swagger documentation."""
    print_test_header("Swagger Documentation Test")
    try:
        response = requests.get(f"{API_BASE_URL}/swagger/v0.4/swagger.json", timeout=DEFAULT_TIMEOUT)
        print_colored("Status Code:", Colors.BOLD)
        status_color = Colors.GREEN if response.status_code == 200 else Colors.FAIL
        print_colored(f"  {response.status_code} {response.reason}", status_color)
        
        # Don't print the full swagger json as it's large
        print_colored("Response Body (truncated):", Colors.BOLD)
        try:
            swagger_json = response.json()
            print(f"  Swagger document found with {len(str(swagger_json))} characters")
            # Print some basic info about the API
            print(f"  API Title: {swagger_json.get('info', {}).get('title', 'Unknown')}")
            print(f"  API Description: {swagger_json.get('info', {}).get('description', 'Unknown')}")
            print(f"  Available Paths: {len(swagger_json.get('paths', {}))} endpoints")
        except json.JSONDecodeError:
            print(response.text)

        if response.status_code == 200:
            print_colored("\n✅ Swagger documentation test PASSED", Colors.GREEN)
            return True
        else:
            print_colored("\n❌ Swagger documentation test FAILED", Colors.FAIL)
            return False
    except requests.exceptions.RequestException as e:
        print_colored(f"\n❌ Swagger documentation test FAILED: {e}", Colors.FAIL)
        return False

def main():
    global API_BASE_URL, DEFAULT_TIMEOUT
    
    parser = argparse.ArgumentParser(description="Test mGBA-http API functionality")
    parser.add_argument("--url", default=API_BASE_URL, help=f"mGBA-http API base URL (default: {API_BASE_URL})")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help=f"Timeout in seconds for API calls (default: {DEFAULT_TIMEOUT})")
    parser.add_argument("--test", choices=["all", "connection", "button", "memory", "sequence", "multiple", "reset", "domain", "swagger"],
                         default="all", help="Run specific test (default: all)")
    args = parser.parse_args()
    
    API_BASE_URL = args.url
    DEFAULT_TIMEOUT = args.timeout
    
    # Welcome message
    print_colored("\n🎮 mGBA-http Test Harness 🎮", Colors.BOLD + Colors.HEADER)
    print(f"API Base URL: {API_BASE_URL}")
    print(f"Request Timeout: {DEFAULT_TIMEOUT} seconds")
    
    tests = {
        "connection": test_connection,
        "button": test_button_press,
        "memory": test_memory_read,
        "sequence": test_multiple_button_sequence,
        "multiple": test_multiple_buttons,
        "reset": test_core_reset,
        "domain": test_memory_domain_info,
        "swagger": test_swagger_docs
    }
    
    results = {}
    
    # Check if API is available before running tests
    try:
        requests.get(API_BASE_URL, timeout=DEFAULT_TIMEOUT)
    except requests.exceptions.RequestException:
        print_colored(f"\n❌ Could not connect to mGBA-http at {API_BASE_URL}", Colors.FAIL)
        print("Please ensure mGBA and mGBA-http are running properly.")
        sys.exit(1)
    
    if args.test == "all":
        # Run all tests
        for name, test_func in tests.items():
            results[name] = test_func()
    else:
        # Run specific test
        if args.test in tests:
            results[args.test] = tests[args.test]()
        else:
            print_colored(f"Unknown test: {args.test}", Colors.FAIL)
            sys.exit(1)
    
    # Print summary
    print("\n" + "="*80)
    print_colored("TEST SUMMARY", Colors.BOLD + Colors.HEADER)
    print("="*80)
    
    all_passed = True
    for name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        color = Colors.GREEN if passed else Colors.FAIL
        print_colored(f"{name.upper()}: {status}", color)
        if not passed:
            all_passed = False
    
    exit_code = 0 if all_passed else 1
    overall_color = Colors.GREEN if all_passed else Colors.FAIL
    overall_status = "✅ ALL TESTS PASSED" if all_passed else "❌ SOME TESTS FAILED"
    print("\n" + "="*80)
    print_colored(f"OVERALL: {overall_status}", Colors.BOLD + overall_color)
    print("="*80 + "\n")
    
    return exit_code

if __name__ == "__main__":
    sys.exit(main()) 