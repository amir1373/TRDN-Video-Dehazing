#!/bin/bash
# Waits for the latent cache, then trains the decoder and evaluates it. Runs unattended so the
# GPU is never idle between stages.
#
# Completion is judged by the CACHE_DONE flag, which run_cache.sh writes only after counting
# real files in both splits — not by an exit code.
set -uo pipefail
export PYTHONPATH=/workspace/pylibs
D=/workspace/trdn_paper_run

echo "=== waiting for latent cache  $(date -u +%FT%TZ) ==="
waited=0
while [ ! -f $D/CACHE_DONE ]; do
  if ! tmux has-session -t cache 2>/dev/null; then
    echo "!!! cache session gone without CACHE_DONE after ${waited}s — aborting"
    echo "    train=$(ls $D/latent_cache/train/*.pt 2>/dev/null | wc -l) test=$(ls $D/latent_cache/test/*.pt 2>/dev/null | wc -l)"
    exit 1
  fi
  sleep 60; waited=$((waited+60))
  [ $((waited % 900)) -eq 0 ] && echo "  still waiting (${waited}s) train=$(ls $D/latent_cache/train/*.pt 2>/dev/null|wc -l) test=$(ls $D/latent_cache/test/*.pt 2>/dev/null|wc -l)"
done
echo "=== cache ready after ${waited}s: train=$(ls $D/latent_cache/train/*.pt|wc -l) test=$(ls $D/latent_cache/test/*.pt|wc -l) ==="

echo "=== training decoder  $(date -u +%FT%TZ) ==="
cd /workspace/repos/TRDN-Video-Dehazing
python3 /root/train_decoder.py --epochs 12 --batch-size 4 --lr 1e-5 --w-lpips 0.25
echo "--- decoder training exit=$? ---"

echo "=== evaluating on cached TEST latents  $(date -u +%FT%TZ) ==="
python3 /root/eval_decoder.py
echo "--- eval exit=$? ---"

echo "=== bootstrap CIs across everything ==="
python3 /root/bootstrap_ci2.py 2>&1 | tee $D/logs/bootstrap_ci_decoder.log | tail -30

# Claim completion only on real artifacts, never on an exit code.
if [ -f $D/decoder_ft/summary.json ] && [ -f $D/evaluations/trdn_decoderft.json ]; then
  touch $D/DECODER_DONE; echo "=== DECODER EXPERIMENT COMPLETE ==="
else
  echo "=== INCOMPLETE — summary.json or the eval JSON is missing ==="
fi
