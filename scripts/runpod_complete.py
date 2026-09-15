"""Sequential GPU training, full-test evaluation and complete paper figures."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.runpod_workflow import VARIANTS, _latest_step_checkpoint


def run(args):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this RunPod workflow')
    session = args.session_root.resolve()
    runs, reports, evaluations, figures = [session / name for name in ('runs', 'reports', 'evaluations', 'paper_artifacts')]
    for path in (runs, reports, evaluations, figures): path.mkdir(parents=True, exist_ok=True)
    def call(script, *arguments):
        subprocess.run([sys.executable, str(ROOT / 'scripts' / script), *map(str, arguments)], cwd=ROOT, check=True)
    shared = ['--dataset-root', args.dataset_root, '--seq-len', args.seq_len, '--crop-size', args.crop_size, '--seed', args.seed]
    call('runpod_workflow.py', 'dataset-check', *shared, '--output', reports / 'dataset_check.json', '--force')
    for variant in VARIANTS:
        project = runs / variant
        command = [*shared, '--project-root', project, '--run-name', variant, '--model-variant', variant, '--preset', args.preset, '--num-epochs', args.num_epochs]
        last = project / 'checkpoints' / 'last'
        checkpoint = last if last.is_dir() else _latest_step_checkpoint(project)
        if checkpoint is not None: command += ['--resume-from-checkpoint', checkpoint]
        call('train_colab.py', *command)
    call('runpod_workflow.py', 'evaluate-all', *shared, '--runs-root', runs, '--eval-dir', evaluations, '--preset', args.preset, '--num-steps', args.num_steps, '--force')
    eval_paths = [evaluations / f'{variant}.json' for variant in VARIANTS]
    selected = runs / 'full' / 'checkpoints' / 'best_psnr'
    call('make_paper_figures.py', '--checkpoint', selected, '--eval-json', *eval_paths, '--dataset-root', args.dataset_root, '--output-dir', figures, '--seed', args.seed, '--metric-log', *[runs / variant / 'logs' / 'runs' / variant / 'metrics.jsonl' for variant in VARIANTS], '--vae-ceiling-json', args.vae_ceiling_json, '--force')
    call('emit_paper_tables.py', '--eval-json', *eval_paths, '--output-dir', figures)
    checklist = json.loads((figures / 'figure_checklist.json').read_text())
    if checklist.get('status') != 'PASS': raise RuntimeError('Incomplete figure checklist')
    call('runpod_workflow.py', 'bundle', '--workspace-root', session, '--runs-root', runs, '--eval-dir', evaluations, '--artifacts-dir', figures, '--preset', args.preset, '--output', session / 'paper_bundle.zip', '--force')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--session-root', type=Path, required=True)
    parser.add_argument('--preset', type=Path, required=True, help='Measured and filled numerics YAML from existing benchmark workflow')
    parser.add_argument('--vae-ceiling-json', type=Path, required=True)
    parser.add_argument('--num-epochs', type=int, default=30)
    parser.add_argument('--seq-len', type=int, default=10)
    parser.add_argument('--crop-size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--num-steps', type=int, default=50)
    args = parser.parse_args()
    if not args.vae_ceiling_json.is_file(): parser.error('Run the VAE ceiling cell first')
    run(args)


if __name__ == '__main__': main()
