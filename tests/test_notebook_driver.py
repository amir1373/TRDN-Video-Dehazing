import json
import re
from pathlib import Path


def test_runpod_notebook_is_a_stateless_script_driver():
    notebook_path = Path(__file__).parents[1] / "notebooks" / "TRDN_REVIDE_Colab.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    code_cells = [cell for cell in cells if cell["cell_type"] == "code"]
    sources = ["".join(cell["source"]) for cell in cells]
    combined = "\n".join(sources)

    assert all(cell.get("outputs", []) == [] for cell in code_cells)
    assert all(cell.get("execution_count") is None for cell in code_cells)
    for index, source in enumerate(sources):
        if cells[index]["cell_type"] == "code" and source.strip():
            compile(source, f"{notebook_path.name}:cell-{index}", "exec")
    assert "/content" not in combined
    assert "/workspace" in "".join(cells[1]["source"])
    assert all("/workspace" not in source for source in sources[2:])
    assert "DATASET_ROOT.is_dir()" in sources[2]
    assert 'DATASET_ROOT.rglob("*")' in sources[2]
    assert not re.search(r"\btrain_mode\s*=", combined)
    assert not re.search(r"\bresume_from_checkpoint\s*=", combined)

    required_scripts = [
        "preflight.py",
        "train_colab.py",
        "evaluate_full_test.py",
        "make_paper_figures.py",
        "emit_paper_tables.py",
    ]
    positions = [
        next(
            index
            for index, source in enumerate(sources)
            if cells[index]["cell_type"] == "code" and script in source
        )
        for script in required_scripts
    ]
    assert positions == sorted(positions)


def test_runpod_notebook_has_ordered_detached_workflow():
    notebook_path = (
        Path(__file__).parents[1] / "notebooks" / "TRDN_REVIDE_RunPod.ipynb"
    )
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    code_cells = [cell for cell in cells if cell["cell_type"] == "code"]
    combined = "\n".join("".join(cell["source"]) for cell in cells)

    assert all(cell.get("outputs", []) == [] for cell in code_cells)
    assert all(cell.get("execution_count") is None for cell in code_cells)
    assert all(
        index > 0 and cells[index - 1]["cell_type"] == "markdown"
        for index, cell in enumerate(cells)
        if cell["cell_type"] == "code"
    )
    for index, cell in enumerate(cells):
        if cell["cell_type"] == "code":
            compile(
                "".join(cell["source"]),
                f"{notebook_path.name}:cell-{index}",
                "exec",
            )

    headings = [
        "".join(cell["source"]).splitlines()[0]
        for cell in cells
        if cell["cell_type"] == "markdown"
    ]
    assert [heading.split(".", 1)[0] for heading in headings] == [
        f"# {letter}" for letter in "ABCDEFGHIJKLMNOPQ"
    ]
    assert "/workspace" in "".join(cells[1]["source"])
    assert "/workspace" not in "\n".join(
        "".join(cell["source"]) for cell in cells[2:]
    )
    assert "launch-training" in combined
    assert "--resume-if-interrupted" in combined
    assert '"list"' in combined and '"monitor"' in combined and '"stop"' in combined
    assert "shared_sample_selection.json" in combined
    assert "make_paper_figures.py" in combined
    assert "bundle" in combined
