"""Protect the contracts package from heavy runtime dependencies."""

import ast
import unittest
from pathlib import Path

_ALLOWED_STANDARD_LIBRARY_ROOTS = {
    "__future__",
    "collections",
    "dataclasses",
    "enum",
    "math",
    "pathlib",
    "re",
    "typing",
}


class ContractDependencyBoundaryTest(unittest.TestCase):
    def test_contracts_only_import_stdlib_or_sibling_modules(self) -> None:
        repository_root = Path(__file__).resolve().parents[2]
        contracts_dir = repository_root / "src" / "embodied_ai" / "contracts"
        violations: list[str] = []

        for source_path in sorted(contracts_dir.rglob("*.py")):
            tree = ast.parse(source_path.read_text(encoding="utf-8"), source_path.as_posix())
            for node in ast.walk(tree):
                imported_roots: list[str] = []
                if isinstance(node, ast.Import):
                    imported_roots = [alias.name.split(".", 1)[0] for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    imported_roots = [node.module.split(".", 1)[0]]
                for root in imported_roots:
                    if root not in _ALLOWED_STANDARD_LIBRARY_ROOTS:
                        violations.append(f"{source_path.name}:{node.lineno}: {root}")

        self.assertEqual(violations, [], f"non-lightweight imports found: {violations}")


if __name__ == "__main__":
    unittest.main()
