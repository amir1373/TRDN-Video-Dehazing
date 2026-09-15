from types import SimpleNamespace
import json
import subprocess

from scripts import runpod_complete


def test_completion_order_and_arguments(tmp_path, monkeypatch):
    calls = []
    args = SimpleNamespace(session_root=tmp_path, dataset_root=tmp_path / 'data', preset=tmp_path / 'preset.yaml', vae_ceiling_json=tmp_path / 'vae.json', seq_len=10, crop_size=256, seed=1234, num_epochs=30, num_steps=50)
    monkeypatch.setattr('torch.cuda.is_available', lambda: True)
    def fake_run(command, **kwargs):
        calls.append(command)
        if command[1].endswith('make_paper_figures.py'):
            (tmp_path / 'paper_artifacts/figure_checklist.json').write_text(json.dumps({'status': 'PASS'}))
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(runpod_complete.subprocess, 'run', fake_run)
    runpod_complete.run(args)
    train_calls = [c for c in calls if c[1].endswith('train_colab.py')]
    assert len(train_calls) == 4
    assert [c[c.index('--model-variant') + 1] for c in train_calls] == list(runpod_complete.VARIANTS)
    from scripts.train_colab import build_parser
    for call in train_calls: build_parser().parse_args(call[2:])
    assert calls[-1][2] == 'bundle'
    assert calls[-3][1].endswith('make_paper_figures.py')
