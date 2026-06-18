"""Fast tests for the CLI parser and lightweight commands."""

from pathlib import Path

import pytest

from bellwether.cli import build_parser, main


def test_parser_requires_a_command() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_version_command(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["version"]) == 0
    assert "bellwether" in capsys.readouterr().out


def test_dashboard_command_writes_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "dash.html"
    assert main(["dashboard", "-o", str(out)]) == 0
    assert out.read_text(encoding="utf-8").startswith("<!doctype html>")


def test_benchmark_quick_runs(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["benchmark", "--quick"]) == 0
    assert "precision" in capsys.readouterr().out
