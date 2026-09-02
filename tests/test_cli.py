"""Tests for the top-level `hbb2obb` command line, mostly its help output."""

import re
from pathlib import Path

import pytest

from hbb2obb import cli


def run_help(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["hbb2obb", "--help"])
    with pytest.raises(SystemExit) as exc:
        cli.main_hbb2obb()
    assert exc.value.code == 0
    return capsys.readouterr().out


def test_help_lists_every_entry_point(monkeypatch, capsys):
    out = run_help(monkeypatch, capsys)
    for command in ("hbb2obb-detect", "hbb2obb-convert", "hbb2obb-view", "hbb2obb-eval", "hbb2obb-optimize"):
        assert command in out


def test_help_nudges_the_reader_to_explore(monkeypatch, capsys):
    assert "hbb2obb-detect --help" in run_help(monkeypatch, capsys)


def test_provenance_lands_beside_the_labels_not_among_them(tmp_path):
    """
    A PROVENANCE.txt inside a label directory is read as a frame by any `labels/*.txt` glob, which
    is how most tooling that is not this one reads YOLO labels. It belongs one level up.
    """
    labels = tmp_path / "train" / "labels"
    assert cli.provenance_path(labels) == tmp_path / "train" / "PROVENANCE.txt"


def test_the_optimizer_keeps_its_own_placement():
    """A sweep's output folder holds runs, not labels, so its record stays inside it."""
    from hbb2obb import optimizer

    assert optimizer.BENCHMARK_PROVENANCE_NAME == "PROVENANCE.txt"
    assert "provenance_path" not in Path("hbb2obb/optimizer.py").read_text()


def test_the_epilog_stays_in_sync_with_the_registered_scripts():
    """Every console script in pyproject is named in the epilog, and nothing extra is."""
    section = re.search(r"\[project\.scripts\]\n(.*?)\n\[", Path("pyproject.toml").read_text(), re.S).group(1)
    scripts = set(re.findall(r'^"?(hbb2obb[\w-]*)"?\s*=', section, re.M))
    listed = {line.split()[0] for line in cli.ENTRY_POINTS_EPILOG.splitlines() if line.startswith("  hbb2obb")}
    assert listed == scripts
