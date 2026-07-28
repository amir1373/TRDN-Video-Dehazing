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
