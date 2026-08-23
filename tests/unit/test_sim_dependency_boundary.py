"""Protect the Isaac simulation package from VLA-stack imports."""

import ast
import unittest
from pathlib import Path

_FORBIDDEN_IMPORT_ROOTS = {"accelerate", "datasets", "lerobot", "peft", "transformers"}


class SimulationDependencyBoundaryTest(unittest.TestCase):
    def test_simulation_code_does_not_import_vla_dependencies(self) -> None:
        repository_root = Path(__file__).resolve().parents[2]
        sim_dir = repository_root / "src" / "embodied_ai" / "sim"
        violations: list[str] = []

        for source_path in sorted(sim_dir.rglob("*.py")):
            tree = ast.parse(source_path.read_text(encoding="utf-8"), source_path.as_posix())
            for node in ast.walk(tree):
                imported_roots: list[str] = []
                if isinstance(node, ast.Import):
                    imported_roots = [alias.name.split(".", 1)[0] for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported_roots = [node.module.split(".", 1)[0]]
                for root in imported_roots:
                    if root in _FORBIDDEN_IMPORT_ROOTS:
                        violations.append(f"{source_path.name}:{node.lineno}: {root}")

        self.assertEqual(violations, [], f"VLA imports found in simulation code: {violations}")


if __name__ == "__main__":
    unittest.main()
