#!/usr/bin/env python3
"""
Comprehensive test runner for RLHF Phi-3 Pipeline.

This script provides a convenient interface for running different types of tests
with appropriate configurations and reporting.
"""

import argparse
import sys
import subprocess
import os
from pathlib import Path
from typing import List, Optional

def run_command(cmd: List[str], description: str, check: bool = True) -> int:
    """Run a command and return the exit code."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=check)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        return e.returncode
    except FileNotFoundError:
        print(f"Command not found: {cmd[0]}")
        return 1

def run_unit_tests(verbose: bool = False, coverage: bool = True) -> int:
    """Run unit tests."""
    cmd = ["pytest", "tests/unit/", "-m", "unit"]
    
    if verbose:
        cmd.append("-v")
    
    if coverage:
        cmd.extend(["--cov=rlhf_phi3", "--cov-report=term-missing"])
    
    return run_command(cmd, "Unit Tests")

def run_property_tests(verbose: bool = False, profile: str = "default") -> int:
    """Run property-based tests."""
    cmd = ["pytest", "tests/property/", "-m", "property"]
    
    if verbose:
        cmd.append("-v")
    
    env = os.environ.copy()
    env["HYPOTHESIS_PROFILE"] = profile
    
    print(f"Using Hypothesis profile: {profile}")
    
    try:
        result = subprocess.run(cmd, env=env, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        return e.returncode

def run_integration_tests(verbose: bool = False, include_slow: bool = False) -> int:
    """Run integration tests."""
    cmd = ["pytest", "tests/integration/", "-m", "integration"]
    
    if not include_slow:
        cmd.extend(["-m", "integration and not slow"])
    
    if verbose:
        cmd.append("-v")
    
    return run_command(cmd, "Integration Tests")

def run_all_tests(verbose: bool = False, coverage: bool = True, include_slow: bool = False) -> int:
    """Run all tests."""
    cmd = ["pytest", "tests/"]
    
    if not include_slow:
        cmd.extend(["-m", "not slow"])
    
    if verbose:
        cmd.append("-v")
    
    if coverage:
        cmd.extend([
            "--cov=rlhf_phi3",
            "--cov-report=html:htmlcov",
            "--cov-report=xml:coverage.xml",
            "--cov-report=term-missing",
            "--cov-fail-under=90"
        ])
    
    return run_command(cmd, "All Tests")

def run_fast_tests(verbose: bool = False) -> int:
    """Run fast tests only (excluding slow, gpu, network tests)."""
    cmd = [
        "pytest", "tests/",
        "-m", "not slow and not gpu and not network",
        "--tb=short"
    ]
    
    if verbose:
        cmd.append("-v")
    
    return run_command(cmd, "Fast Tests")

def run_linting() -> int:
    """Run code linting checks."""
    exit_codes = []
    
    # Black formatting check
    exit_codes.append(run_command(
        ["black", "--check", "--diff", "rlhf_phi3/", "tests/"],
        "Black Formatting Check",
        check=False
    ))
    
    # isort import sorting check
    exit_codes.append(run_command(
        ["isort", "--check-only", "--diff", "rlhf_phi3/", "tests/"],
        "Import Sorting Check",
        check=False
    ))
    
    # Flake8 linting
    exit_codes.append(run_command(
        ["flake8", "rlhf_phi3/", "tests/"],
        "Flake8 Linting",
        check=False
    ))
    
    # MyPy type checking (optional)
    exit_codes.append(run_command(
        ["mypy", "rlhf_phi3/"],
        "MyPy Type Checking",
        check=False
    ))
    
    return max(exit_codes) if exit_codes else 0

def run_security_checks() -> int:
    """Run security checks."""
    exit_codes = []
    
    # Safety check for known vulnerabilities
    exit_codes.append(run_command(
        ["safety", "check"],
        "Safety Vulnerability Check",
        check=False
    ))
    
    # Bandit security linting
    exit_codes.append(run_command(
        ["bandit", "-r", "rlhf_phi3/"],
        "Bandit Security Linting",
        check=False
    ))
    
    return max(exit_codes) if exit_codes else 0

def format_code() -> int:
    """Format code using black and isort."""
    exit_codes = []
    
    # Black formatting
    exit_codes.append(run_command(
        ["black", "rlhf_phi3/", "tests/"],
        "Black Code Formatting",
        check=False
    ))
    
    # isort import sorting
    exit_codes.append(run_command(
        ["isort", "rlhf_phi3/", "tests/"],
        "Import Sorting",
        check=False
    ))
    
    return max(exit_codes) if exit_codes else 0

def run_coverage_report() -> int:
    """Generate detailed coverage report."""
    cmd = [
        "pytest", "tests/",
        "--cov=rlhf_phi3",
        "--cov-report=html:htmlcov",
        "--cov-report=xml:coverage.xml",
        "--cov-report=term-missing",
        "--cov-fail-under=90",
        "-q"
    ]
    
    exit_code = run_command(cmd, "Coverage Report Generation")
    
    if exit_code == 0:
        print(f"\nCoverage report generated:")
        print(f"  HTML: {Path('htmlcov/index.html').absolute()}")
        print(f"  XML:  {Path('coverage.xml').absolute()}")
    
    return exit_code

def run_ci_pipeline() -> int:
    """Run the complete CI pipeline."""
    print("Running complete CI pipeline...")
    
    steps = [
        ("Linting", run_linting),
        ("Fast Tests", lambda: run_fast_tests(verbose=True)),
        ("Coverage Report", run_coverage_report),
        ("Security Checks", run_security_checks)
    ]
    
    failed_steps = []
    
    for step_name, step_func in steps:
        print(f"\n{'#'*80}")
        print(f"CI Step: {step_name}")
        print(f"{'#'*80}")
        
        exit_code = step_func()
        if exit_code != 0:
            failed_steps.append(step_name)
            print(f"❌ {step_name} failed with exit code {exit_code}")
        else:
            print(f"✅ {step_name} passed")
    
    if failed_steps:
        print(f"\n❌ CI Pipeline failed. Failed steps: {', '.join(failed_steps)}")
        return 1
    else:
        print(f"\n✅ CI Pipeline completed successfully!")
        return 0

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test runner for RLHF Phi-3 Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py unit                    # Run unit tests
  python run_tests.py property --profile ci  # Run property tests with CI profile
  python run_tests.py all --coverage         # Run all tests with coverage
  python run_tests.py fast -v                # Run fast tests verbosely
  python run_tests.py lint                   # Run linting checks
  python run_tests.py format                 # Format code
  python run_tests.py ci                     # Run complete CI pipeline
        """
    )
    
    parser.add_argument(
        "test_type",
        choices=["unit", "property", "integration", "all", "fast", "lint", "security", "format", "coverage", "ci"],
        help="Type of tests to run"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "--no-coverage",
        action="store_true",
        help="Disable coverage reporting"
    )
    
    parser.add_argument(
        "--include-slow",
        action="store_true",
        help="Include slow tests"
    )
    
    parser.add_argument(
        "--profile",
        choices=["default", "fast", "thorough", "ci"],
        default="default",
        help="Hypothesis profile for property tests"
    )
    
    args = parser.parse_args()
    
    # Set up environment
    os.environ["PYTHONPATH"] = str(Path.cwd())
    
    # Run the appropriate test type
    if args.test_type == "unit":
        exit_code = run_unit_tests(args.verbose, not args.no_coverage)
    elif args.test_type == "property":
        exit_code = run_property_tests(args.verbose, args.profile)
    elif args.test_type == "integration":
        exit_code = run_integration_tests(args.verbose, args.include_slow)
    elif args.test_type == "all":
        exit_code = run_all_tests(args.verbose, not args.no_coverage, args.include_slow)
    elif args.test_type == "fast":
        exit_code = run_fast_tests(args.verbose)
    elif args.test_type == "lint":
        exit_code = run_linting()
    elif args.test_type == "security":
        exit_code = run_security_checks()
    elif args.test_type == "format":
        exit_code = format_code()
    elif args.test_type == "coverage":
        exit_code = run_coverage_report()
    elif args.test_type == "ci":
        exit_code = run_ci_pipeline()
    else:
        parser.print_help()
        exit_code = 1
    
    sys.exit(exit_code)

if __name__ == "__main__":
    main()