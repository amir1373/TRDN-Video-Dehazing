"""Audit third-party imports against pinned runtime requirement files."""

import ast
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
IMPORT_TO_DISTRIBUTION = {
    "PIL": "Pillow",
    "cv2": "opencv-python",
    "skimage": "scikit-image",
    "yaml": "PyYAML",
}
LOCAL_MODULES = {"src", "scripts"}
RUNTIME_STRING_DEPENDENCIES = {"tensorboard", "xformers"}


def _imports_from_source(source: str, filename: str) -> set[str]:
    tree = ast.parse(source, filename=filename)
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imports.add(node.module.split(".", 1)[0])
    return imports


def collect_imports() -> set[str]:
    imports = set()
    for root in (REPO_ROOT / "src", REPO_ROOT / "scripts"):
        for path in root.rglob("*.py"):
            imports.update(
                _imports_from_source(path.read_text(encoding="utf-8"), str(path))
            )
    notebook_path = REPO_ROOT / "notebooks" / "TRDN_REVIDE_Colab.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            imports.update(
                _imports_from_source(
                    "".join(cell["source"]),
                    f"{notebook_path}:cell-{index}",
                )
            )
    return imports


def _requirements(path: Path) -> set[str]:
    names = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = re.match(r"([A-Za-z0-9_.-]+)", stripped)
        if match:
            names.add(match.group(1))
    return names


def _unversioned(path: Path) -> list[str]:
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        requirement = stripped.split(";", 1)[0].strip()
        if not re.search(r"(===|==|~=|>=|<=|>|<)", requirement):
            result.append(requirement)
    return result


def audit() -> dict:
    imports = collect_imports()
    third_party_modules = {
        name
        for name in imports
        if name not in sys.stdlib_module_names
        and name not in LOCAL_MODULES
        and not name.startswith("_")
    }
    required = {
        IMPORT_TO_DISTRIBUTION.get(name, name)
        for name in third_party_modules
    }
    required.update(RUNTIME_STRING_DEPENDENCIES)
    full = _requirements(REPO_ROOT / "requirements.txt")
    colab = _requirements(REPO_ROOT / "requirements_colab.txt")
    return {
        "third_party_import_modules": sorted(third_party_modules),
        "required_distributions": sorted(required),
        "undeclared_in_requirements": sorted(required - full),
        "missing_from_notebook_install": sorted(required - colab),
        "unused_declarations": sorted(full - required),
        "unused_colab_declarations": sorted(colab - required),
        "unversioned_declarations": _unversioned(REPO_ROOT / "requirements.txt"),
        "unversioned_colab_declarations": _unversioned(
            REPO_ROOT / "requirements_colab.txt"
        ),
    }


def main() -> None:
    report = audit()
    print(json.dumps(report, indent=2))
    if (
        report["undeclared_in_requirements"]
        or report["missing_from_notebook_install"]
        or report["unversioned_declarations"]
        or report["unversioned_colab_declarations"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
