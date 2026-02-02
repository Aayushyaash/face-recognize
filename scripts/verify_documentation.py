#!/usr/bin/env python3
"""
Documentation verification script.

This script verifies that examples in the README.md file work correctly.
"""

import re
import sys
from pathlib import Path


def extract_python_examples_from_readme() -> list[str]:
    """Extract Python code examples from README.md."""
    readme_path = Path("README.md")
    if not readme_path.exists():
        print("README.md not found!")
        return []

    content = readme_path.read_text()
    # Find all Python code blocks in the README
    pattern = r"```python\n(.*?)\n```"
    matches = re.findall(pattern, content, re.DOTALL)
    return matches


def test_python_example(example_code: str) -> bool:
    """Test a single Python example by executing it."""
    try:
        # Execute the example code in a separate namespace
        exec(example_code, {})
        return True
    except Exception as e:
        print(f"Error executing example: {str(e)}")
        return False


def main() -> int:
    """Main function to run documentation verification."""
    print("Verifying README.md examples...")

    examples = extract_python_examples_from_readme()

    if not examples:
        print("No Python examples found in README.md")
        return 0  # Not an error, just no examples to test

    print(f"Found {len(examples)} Python examples in README.md")

    all_passed = True
    for i, example in enumerate(examples, 1):
        print(f"Testing example {i}...")
        if not test_python_example(example.strip()):
            print(f"Example {i} failed!")
            all_passed = False
        else:
            print(f"Example {i} passed!")

    if all_passed:
        print("All README examples passed!")
        return 0
    else:
        print("Some README examples failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
