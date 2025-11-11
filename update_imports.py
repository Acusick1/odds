#!/usr/bin/env python3
"""
Script to update imports after workspace migration.

Handles the complex mapping where 'core' splits into odds_core and odds_lambda.
"""

import re
from pathlib import Path

# Core modules that belong to odds-core package
ODDS_CORE_MODULES = {"models", "database", "config", "api_models", "time"}

# Mapping of old import prefixes to new ones
# Special handling for 'core' based on specific module
IMPORT_MAPPINGS = {
    "from storage.": "from odds_lambda.storage.",
    "from jobs.": "from odds_lambda.jobs.",
    "from analytics.": "from odds_analytics.",
    "from cli.": "from odds_cli.",
    "from alerts.": "from odds_cli.alerts.",
}


def update_core_import(line: str) -> str:
    """Handle 'from core.X' imports specially."""
    # Pattern: from core.MODULE import ...
    match = re.match(r"from core\.(\w+)", line)
    if not match:
        return line

    module = match.group(1)
    if module in ODDS_CORE_MODULES:
        # This belongs to odds-core
        return line.replace("from core.", "from odds_core.")
    elif module == "scheduling":
        # scheduling and its submodules belong to odds-lambda
        return line.replace("from core.scheduling", "from odds_lambda.scheduling")
    else:
        # Other core modules belong to odds-lambda
        return line.replace("from core.", "from odds_lambda.")


def update_imports_in_file(file_path: Path):
    """Update imports in a single Python file."""
    try:
        content = file_path.read_text()
        lines = content.splitlines()
        modified = False

        new_lines = []
        for line in lines:
            new_line = line

            # Handle 'from core.' imports specially
            if line.strip().startswith("from core."):
                new_line = update_core_import(line)
                if new_line != line:
                    modified = True
            else:
                # Handle other import mappings
                for old_prefix, new_prefix in IMPORT_MAPPINGS.items():
                    if old_prefix in line:
                        new_line = line.replace(old_prefix, new_prefix)
                        if new_line != line:
                            modified = True
                        break

            new_lines.append(new_line)

        if modified:
            file_path.write_text("\n".join(new_lines) + "\n")
            print(f"✓ Updated: {file_path}")
            return True
        return False

    except Exception as e:
        print(f"✗ Error updating {file_path}: {e}")
        return False


def main():
    """Update all Python files in packages/ and tests/."""
    # Process both packages and tests directories
    dirs_to_process = [Path("packages"), Path("tests")]

    all_py_files = []
    for directory in dirs_to_process:
        if directory.exists():
            all_py_files.extend(list(directory.rglob("*.py")))

    print(f"Found {len(all_py_files)} Python files to process\n")

    updated_count = 0
    for py_file in all_py_files:
        if update_imports_in_file(py_file):
            updated_count += 1

    print(f"\nCompleted: {updated_count} files updated")


if __name__ == "__main__":
    main()
