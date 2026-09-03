"""Tests for the top-level `hbb2obb` command line, mostly its help output."""

import argparse
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


def test_detection_and_conversion_records_do_not_overwrite_each_other(tmp_path):
    """
    labels_hbb and labels_obb share a parent, so one level up is one path for both commands. The
    detection record therefore has its own name; under the plain one the conversion that reads
    those HBBs would erase the detector's checkpoint hash and settings.
    """
    root = tmp_path / "dataset"
    detection = cli.provenance_path(root / "labels_hbb", cli.DETECTION_PROVENANCE_NAME)
    conversion = cli.provenance_path(root / "labels_obb")
    assert detection.parent == conversion.parent == root
    assert detection != conversion


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


def test_precision_without_normalize_is_refused(monkeypatch, capsys):
    """
    Absolute output is integral and has no decimals to set, so accepting --precision on its own
    would look like it did something. Same rule the confidence side-car follows.
    """
    monkeypatch.setattr("sys.argv", ["hbb2obb", "images", "--precision", "6"])
    with pytest.raises(SystemExit) as exc:
        cli.main_hbb2obb()
    assert exc.value.code == 2
    assert "--precision applies to normalized output" in capsys.readouterr().err


@pytest.mark.parametrize("value", ["-1", "0", "18"])
def test_an_unusable_precision_is_refused(monkeypatch, capsys, value):
    """
    A negative precision is a format-string error rather than a coordinate, and zero decimals
    write every normalized coordinate as 0 or 1. Both used to reach the writer.
    """
    monkeypatch.setattr("sys.argv", ["hbb2obb", "images", "--normalize", "--precision", value])
    with pytest.raises(SystemExit) as exc:
        cli.main_hbb2obb()
    assert exc.value.code == 2
    assert "--precision must be between 1 and 17" in capsys.readouterr().err


def test_a_precision_that_cannot_round_trip_is_flagged(capsys):
    """
    Reading a normalized set multiplies by the frame size, so too few decimals move the boxes.
    Written labels are still valid, hence a warning rather than a refusal.
    """
    assert cli.warn_if_precision_loses_pixels(3, (3840, 2160)) is True
    out = capsys.readouterr().out
    assert "3840x2160" in out and "Use 6 or more" in out

    assert cli.warn_if_precision_loses_pixels(6, (3840, 2160)) is False
    assert capsys.readouterr().out == ""


def test_zero_precision_is_allowed_for_absolute_pixels():
    """Whole pixels are a real request; only normalized output needs a decimal to survive."""
    parser = argparse.ArgumentParser()
    cli.check_precision(parser, 0, normalize=False)  # must not raise

    with pytest.raises(SystemExit):
        cli.check_precision(parser, 0, normalize=True)


def test_normalize_is_offered_by_the_conversion_command(monkeypatch, capsys):
    """`hbb2obb-detect` and `hbb2obb-convert` have had it; the conversion itself was the gap."""
    out = run_help(monkeypatch, capsys)
    assert "--normalize" in out
