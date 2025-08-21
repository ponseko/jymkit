#!/usr/bin/env python3
"""Convenient test runner for the improved test suite."""

import subprocess
import sys
from typing import List, Optional


def run_tests(
    test_type: Optional[str] = None,
    verbose: bool = True,
    external: bool = True,
    extra_args: Optional[List[str]] = None
) -> int:
    """
    Run tests with convenient options.
    
    Args:
        test_type: Type of tests to run ('unit', 'integration', 'performance', 'fast', 'slow')
        verbose: Whether to run with verbose output
        external: Whether to include external library tests
        extra_args: Additional arguments to pass to pytest
    
    Returns:
        Exit code from pytest
    """
    cmd = ["python", "-m", "pytest", "tests/improved/"]
    
    if verbose:
        cmd.append("-v")
    
    if test_type:
        if test_type in ["unit", "integration", "performance", "fast", "slow"]:
            cmd.extend(["-m", test_type])
        else:
            cmd.append(f"tests/improved/{test_type}/")
    
    if not external:
        cmd.extend(["-m", "not external"])
    
    if extra_args:
        cmd.extend(extra_args)
    
    print(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run improved jymkit tests")
    parser.add_argument(
        "--type", 
        choices=["unit", "integration", "performance", "fast", "slow"],
        help="Type of tests to run"
    )
    parser.add_argument("--no-external", action="store_true", help="Skip external library tests")
    parser.add_argument("--quiet", action="store_true", help="Run without verbose output")
    parser.add_argument("args", nargs="*", help="Additional pytest arguments")
    
    args = parser.parse_args()
    
    exit_code = run_tests(
        test_type=args.type,
        verbose=not args.quiet,
        external=not args.no_external,
        extra_args=args.args
    )
    
    sys.exit(exit_code)