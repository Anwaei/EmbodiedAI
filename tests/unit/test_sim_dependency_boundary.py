"""Protect the Isaac simulation package from VLA-stack imports."""

import ast
import re
import tomllib
import unittest
from pathlib import Path

_FORBIDDEN_IMPORT_ROOTS = {"accelerate", "datasets", "lerobot", "peft", "transformers"}


class SimulationDependencyBoundaryTest(unittest.TestCase):
    def test_simulation_code_does_not_import_vla_dependencies(self) -> None:
        repository_root = Path(__file__).resolve().parents[2]
        source_roots = (
            repository_root / "src" / "embodied_ai" / "sim",
            repository_root / "scripts" / "sim",
        )
        violations: list[str] = []

        for source_root in source_roots:
            for source_path in sorted(source_root.rglob("*.py")):
                tree = ast.parse(
                    source_path.read_text(encoding="utf-8"),
                    source_path.as_posix(),
                )
                for node in ast.walk(tree):
                    imported_roots: list[str] = []
                    if isinstance(node, ast.Import):
                        imported_roots = [
                            alias.name.split(".", 1)[0] for alias in node.names
                        ]
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        imported_roots = [node.module.split(".", 1)[0]]
                    for root in imported_roots:
                        if root in _FORBIDDEN_IMPORT_ROOTS:
                            relative_path = source_path.relative_to(repository_root)
                            violations.append(
                                f"{relative_path}:{node.lineno}: {root}"
                            )

        self.assertEqual(violations, [], f"VLA imports found in simulation code: {violations}")

    def test_isaac_project_does_not_declare_vla_dependencies(self) -> None:
        repository_root = Path(__file__).resolve().parents[2]
        project_path = repository_root / "env" / "isaac" / "pyproject.toml"
        project = tomllib.loads(project_path.read_text(encoding="utf-8"))
        dependencies = project["project"]["dependencies"]
        dependency_roots = {
            re.split(r"[\[<>=!~ ]", dependency, maxsplit=1)[0].lower()
            for dependency in dependencies
        }

        self.assertEqual(dependency_roots & _FORBIDDEN_IMPORT_ROOTS, set())


if __name__ == "__main__":
    unittest.main()
