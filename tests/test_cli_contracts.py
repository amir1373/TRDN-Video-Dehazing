from pathlib import Path

import pytest

from scripts.evaluate_full_test import build_parser as evaluation_parser
from scripts import evaluate_full_test
from scripts.make_paper_figures import build_parser as figure_parser
from scripts.emit_paper_tables import build_parser as table_parser
from scripts.verify_cli_contracts import verify


def test_notebook_and_readme_cli_contracts_match():
    rows = verify()

    assert rows
    assert all(row["status"] == "OK" for row in rows)


@pytest.mark.parametrize(
    "parser_factory",
    [evaluation_parser, figure_parser, table_parser],
)
def test_required_cli_arguments_fail_with_argparse_message(parser_factory, capsys):
    with pytest.raises(SystemExit) as exc_info:
        parser_factory().parse_args([])

    assert exc_info.value.code == 2
    stderr = capsys.readouterr().err
    assert "required" in stderr
    assert "Traceback" not in stderr


def test_evaluation_requires_step_setting_without_traceback(monkeypatch, capsys):
    monkeypatch.setattr(
        "sys.argv",
        ["evaluate_full_test.py", "--checkpoint", "not-loaded"],
    )

    with pytest.raises(SystemExit) as exc_info:
        evaluate_full_test.main()

    assert exc_info.value.code == 2
    stderr = capsys.readouterr().err
    assert "--num-steps or --step-sweep is required" in stderr
    assert "Traceback" not in stderr
