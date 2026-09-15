"""Append repository-backed automatic completion and SSH cells idempotently."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / 'notebooks/TRDN_REVIDE_RunPod.ipynb'
notebook = nbf.read(path, as_version=4)
notebook.cells = [cell for cell in notebook.cells if not cell.metadata.get('automatic_completion')]
for cell in [
    nbf.v4.new_markdown_cell('''# Automatic training through final visual reports
After completing environment, dataset, VAE ceiling and benchmark/preset setup above, this alternative runs all four variants sequentially and then full-test evaluation, paper figures, tables and the ZIP bundle. Do not run it alongside the detached per-variant training cells. It calls repository code and regenerates artifacts on resume. The final figure checklist must say PASS.'''),
    nbf.v4.new_code_cell('''RUN_COMPLETE_WORKFLOW = False
complete_command = [sys.executable, str(REPO_ROOT / 'scripts/runpod_complete.py'), '--dataset-root', str(DATASET_ROOT), '--session-root', str(SESSION_ROOT), '--preset', str(PRESET_PATH), '--vae-ceiling-json', str(VAE_JSON), '--num-epochs', str(NUM_EPOCHS), '--seq-len', str(SEQ_LEN), '--crop-size', str(CROP_SIZE), '--seed', str(SEED), '--num-steps', str(INFERENCE_STEPS)]
import shlex
print(shlex.join(complete_command))
if RUN_COMPLETE_WORKFLOW:
    subprocess.run(complete_command, cwd=REPO_ROOT, check=True)'''),
    nbf.v4.new_markdown_cell('''# SSH training on RunPod
Copy the SSH host/port command from RunPod Connect and use it on your computer. Inside the pod:

```bash
cd /workspace/TRDN-Video-Dehazing
tmux new -s trdn
python scripts/runpod_complete.py --dataset-root /workspace/datasets/REVIDE --session-root /workspace/trdn_paper_run --preset configs/a40.yaml --vae-ceiling-json /workspace/trdn_paper_run/reports/vae_ceiling.json --num-epochs 30
```

Fill the measured preset and produce the VAE ceiling JSON using the preceding setup cells first. Detach with Ctrl+B then D and reattach with `tmux attach -t trdn`. Do not run a second trainer against the same output directory. tmux survives SSH disconnects; persistent storage and checkpoints are needed for pod termination/restart. Re-run the same command to resume.

The automatic workflow generates all existing paper figures: variant metrics, shared qualitative comparisons, error maps, zooms, failures, temporal consistency, reference weights, training curves and VAE ceiling. Figures are PNG/PDF with provenance sidecars. Final tables, full-test JSON and the ZIP bundle are retained on the configured persistent volume.''')
]:
    cell.metadata['automatic_completion'] = True
    notebook.cells.append(cell)
nbf.validate(notebook)
nbf.write(notebook, path)
