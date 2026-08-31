"""Keep Stage 9 contracts/configuration dependency-light and runtimes isolated."""

import ast
from pathlib import Path


def test_dependency_light_rl_modules_do_not_import_heavy_runtimes() -> None:
    repository = Path(__file__).resolve().parents[2]
    paths = [repository / "src/embodied_ai/contracts/rl.py"]
    paths.extend(sorted((repository / "src/embodied_ai/rl").glob("*.py")))
    forbidden = {"isaaclab", "isaacsim", "lerobot", "peft", "transformers"}
    violations: list[str] = []

    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), path.as_posix())
        for node in ast.walk(tree):
            roots: list[str] = []
            if isinstance(node, ast.Import):
                roots = [alias.name.split(".", 1)[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                roots = [node.module.split(".", 1)[0]]
            for root in roots:
                if root in forbidden:
                    violations.append(f"{path.name}:{node.lineno}:{root}")

    assert violations == []


def test_isaac_rl_task_does_not_import_vla_stack() -> None:
    repository = Path(__file__).resolve().parents[2]
    root = repository / "src/embodied_ai/sim/tasks/franka_pick_place"
    forbidden = {"lerobot", "peft", "transformers", "datasets", "accelerate"}
    violations: list[str] = []

    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), path.as_posix())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports = [node.module]
            else:
                imports = []
            for imported in imports:
                if imported.split(".", 1)[0] in forbidden:
                    violations.append(f"{path.name}:{node.lineno}:{imported}")

    assert violations == []
