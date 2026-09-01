"""
Tests for hbb2obb.plotting.

The per-run plot is a verbatim port and is pinned by comparing renders, not by unit tests, and
a sweep over one opening kernel still renders byte for byte what it always did. What is tested
here is the comparison plot, which is new, and the colour ladder a sweep over several opening
kernels needs so that no two of its series come out the same colour.
"""

import colorsys
from pathlib import Path

import pytest
import yaml

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


def _write_run(tmp_path, results, name="sam_l-sam_b"):
    """A run folder as the optimizer leaves it, ready for ``run_plot``."""
    run_dir = tmp_path / name
    run_dir.mkdir(exist_ok=True)
    best = max(results, key=lambda r: r["avg_iou"])
    with open(run_dir / "results.yaml", "w") as f:
        yaml.safe_dump({"all_results": results, "best_parameters": best}, f)
    with open(run_dir / "run_config.yaml", "w") as f:
        yaml.safe_dump({"run_name": name}, f)
    return run_dir


def _grid(image_sizes, kernels, scale_factors=(-0.01, 0.0, 0.05, 0.1)):
    """One sweep result per grid point, the IoU rising with the image size."""
    return [
        {
            "imgsz": imgsz,
            "opening_kernel_percentage": kernel,
            "scale_factor": sf,
            "avg_iou": 0.70 + 0.05 * i + 0.01 * j - abs(sf - 0.05),
            "std_iou": 0.07,
            "execution_time": 8.0 + 4.0 * i,
            "total_matches": 200,
            "total_gt": 201,
        }
        for i, imgsz in enumerate(image_sizes)
        for j, kernel in enumerate(kernels)
        for sf in scale_factors
    ]


def test_a_single_kernel_keeps_the_base_colour_untouched(tmp_path, monkeypatch):
    """With nothing to ladder, a two-axis sweep keeps exactly the palette it always had."""
    used = []
    monkeypatch.setattr(plotting.plt, "plot", lambda *a, **k: used.append(k.get("color")))
    plotting.run_plot(_write_run(tmp_path, _grid([640, 960, 1280], [0.15])), output=tmp_path / "p.png")
    assert used == plotting.COLORS[:3]


def test_each_kernel_gets_its_own_shade_of_one_hue(tmp_path, monkeypatch):
    """The complaint that started this: three kernels at one image size drew in one colour."""
    used = []
    monkeypatch.setattr(plotting.plt, "plot", lambda *a, **k: used.append(k.get("color")))
    plotting.run_plot(_write_run(tmp_path, _grid([640], [0.05, 0.15, 0.3])), output=tmp_path / "p.png")

    assert len(set(used)) == 3
    hues, lightnesses = zip(*[colorsys.rgb_to_hls(*c)[:2] for c in used])
    assert len(set(round(h, 6) for h in hues)) == 1  # one image size stays one hue
    assert list(lightnesses) == sorted(lightnesses, reverse=True)  # and darkens with the kernel


def test_ladder_position_centres_a_lone_kernel():
    """One kernel has no ladder to sit on, so it takes the middle rather than dividing by zero."""
    assert plotting.ladder_position(0, 1) == 0.5
    assert (plotting.ladder_position(0, 3), plotting.ladder_position(2, 3)) == (0.0, 1.0)


def test_a_three_axis_sweep_renders_one_plot(tmp_path):
    """Three axes in a single figure, not a grid of subplots."""
    out = plotting.run_plot(_write_run(tmp_path, _grid([640, 960, 1280], [0.0, 0.15, 0.3])))
    assert out.name == "plot.png" and out.is_file() and out.stat().st_size > 0


def test_no_time_mode_still_renders_a_kernel_sweep(tmp_path):
    """Fixed marker areas leave the shape and the shade as the only kernel cues."""
    run_dir = _write_run(tmp_path, _grid([640, 1280], [0.1, 0.2]))
    assert plotting.run_plot(run_dir, no_time=True).stat().st_size > 0
