# Automatic GPU training and final figures

Use `notebooks/TRDN_REVIDE_RunPod.ipynb`. Complete its environment, dataset,
VAE ceiling and benchmark/locked-preset sections first. The appended automatic
workflow is an alternative to the detached per-variant launch cells; it trains
all four variants sequentially, evaluates the complete test set, generates the
existing complete figure set and tables, verifies the checklist, and bundles
artifacts. Do not launch both paths simultaneously.

## SSH

Use the SSH command shown by your pod's Connect panel from your computer.
Inside the pod, after the notebook setup/benchmark steps:

```bash
cd /workspace/TRDN-Video-Dehazing
git pull --ff-only
tmux new -s trdn
python scripts/runpod_complete.py --dataset-root /workspace/datasets/REVIDE --session-root /workspace/trdn_paper_run --preset configs/a40.yaml --vae-ceiling-json /workspace/trdn_paper_run/reports/vae_ceiling.json --num-epochs 30
```

The preset must contain measured values, not TODO placeholders. Detach with
Ctrl+B then D; reattach using `tmux attach -t trdn`. tmux keeps the process alive
across SSH disconnections, not pod shutdown. Persistent storage is required.
Rerunning the same command resumes checkpoints and regenerates evaluation and
figures. Training commands finish sequentially before evaluation begins.

## Shared data with VideoEENet

Both accept an extracted sequence dataset at a configurable absolute root:
`Train/hazy/<sequence>/<frame>.png`, `Train/gt/<sequence>/<frame>.png`, and
the same layout under `Test`. Sequence-first hazy/gt folders are also supported.
Known clean-folder aliases and modality-normalized frame names are supported.
The default whole-sequence validation partition is derived from Train by a
SHA256 ranking with seed 1234 and fraction 0.1. Keep split membership fixed and
compare dataset-check JSON sequence lists from both repositories before a study.
TRDN crops images whereas VideoEENet currently resizes: do not treat their scores
as preprocessing-matched results. No current pod host/path has been verified.

## Output completeness

`paper_artifacts/figure_checklist.json` must report PASS. It checks PNG/PDF and
provenance sidecars for variant metrics, shared qualitative comparisons, error
maps, zooms, temporal consistency, reference weights, failures, training curves,
and VAE ceiling. Tables are written to Markdown/CSV and a ZIP bundle contains
the reports and supporting metadata. Missing artifacts cause a failing exit code.
These outputs depend on all four independently trained variants and the VAE
ceiling report; absent experiment results are never replaced with fabricated plots.
