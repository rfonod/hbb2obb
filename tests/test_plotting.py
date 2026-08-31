"""
Tests for hbb2obb.plotting.

The per-run plot is a verbatim port and is pinned by comparing renders, not by unit tests.
What is tested here is the comparison plot, which is new: it has to stay readable as a
benchmark grows from a handful of runs to a couple of dozen.
"""

from pathlib import Path

import pytest

from hbb2obb import plotting

pytest.importorskip("matplotlib")


def row(name, models, iou, seconds):
    return {
        "name": name,
        "sam_models": models,
        "avg_iou": iou,
        "std_iou": 0.07,
        "execution_time": seconds,
        "imgsz": 1280,
        "scale_factor": 0.05,
        "opening_kernel_percentage": 0.15,
        "total_matches": 200,
        "total_gt": 201,
        "n_points": 36,
    }


def test_a_wide_cost_spread_goes_logarithmic():
    """Single small models against five-model ensembles is roughly a decade of cost."""
    assert plotting.use_log_scale([8.1, 8.4, 14.4, 43.4, 80.4])


def test_a_narrow_cost_spread_stays_linear():
    assert not plotting.use_log_scale([9.9, 10.0, 14.4])


def test_use_log_scale_survives_degenerate_input():
    assert not plotting.use_log_scale([])
    assert not plotting.use_log_scale([0.0, 5.0])  # a zero cost has no logarithm
    assert not plotting.use_log_scale([12.0])


def test_comparison_plot_renders_a_crowded_benchmark(tmp_path):
    """Seventeen runs, seven of them within two seconds of each other, is the shipped case."""
    rows = [
        row("mobile_sam", ["mobile_sam"], 0.8578, 8.1),
        row("sam2_t", ["sam2_t"], 0.8665, 8.5),
        row("sam2.1_t", ["sam2.1_t"], 0.8689, 8.4),
        row("sam2_s", ["sam2_s"], 0.8652, 8.4),
        row("sam2.1_s", ["sam2.1_s"], 0.8648, 8.7),
        row("sam_b", ["sam_b"], 0.8964, 14.4),
        row("sam_l", ["sam_l"], 0.8989, 25.4),
        row("sam_l-sam_b", ["sam_l", "sam_b"], 0.9005, 43.4),
        row("big", ["sam_l", "sam_b", "sam2_b", "sam2.1_b"], 0.9079, 66.8),
        row("bigger", ["sam_l", "sam_b", "sam2_b", "sam2.1_b", "sam2_s"], 0.8937, 80.4),
    ]
    out = plotting.comparison_plot(rows, tmp_path / "comparison.png")
    assert out.is_file() and out.stat().st_size > 0


def test_comparison_plot_survives_runs_that_all_cost_the_same(tmp_path):
    """A zero span would divide by zero in the axis padding and render nothing."""
    rows = [row("a", ["sam_b"], 0.88, 10.0), row("b", ["sam_l"], 0.89, 10.0)]
    assert plotting.comparison_plot(rows, Path(tmp_path / "flat.png")).is_file()
