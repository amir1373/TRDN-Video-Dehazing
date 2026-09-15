from io import StringIO

from src.progress import ProgressReporter


def test_redirected_progress_uses_periodic_eta_lines():
    output = StringIO()
    progress = ProgressReporter(2, "Smoke stage", file=output)
    progress.set_postfix({"loss": "1.0", "lr": "1e-5"})
    progress.update()
    progress.update()
    progress.close()

    text = output.getvalue()
    assert "Smoke stage: start 0/2" in text
    assert "Smoke stage: complete 2/2" in text
    assert "ETA" in text
    assert "loss=1.0" in text
    assert "\r" not in text
