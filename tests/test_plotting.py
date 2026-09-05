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


# ------------------------------------------------------------------------ metrics other than IoU
def _grid_with_metrics(image_sizes, kernels, scale_factors=(0.0, 0.025, 0.05)):
    """
    A grid where the mean IoU and the orientation error disagree about the winner.

    That disagreement is the reason the option exists, so the fixture builds it in: IoU peaks at
    the largest image size, the angle error bottoms out at the smallest.
    """
    return [
        {
            "imgsz": imgsz,
            "opening_kernel_percentage": kernel,
            "scale_factor": sf,
            "avg_iou": 0.80 + 0.02 * i - abs(sf - 0.025),
            "std_iou": 0.07,
            "median_iou": 0.82 + 0.02 * i,
            "iou_fractions": {"0.50": 1.0, "0.75": 0.9 - 0.05 * i, "0.85": 0.7, "0.90": 0.40 + 0.05 * i},
            "median_angle_error": 2.0 + 0.5 * i + 0.1 * j,
            "avg_angle_error": 2.4 + 0.5 * i,
            "std_angle_error": 1.1,
            "p90_angle_error": 6.0 + 0.5 * i,
            "execution_time": 8.0 + 4.0 * i,
            "total_matches": 200,
            "total_gt": 201,
        }
        for i, imgsz in enumerate(image_sizes)
        for j, kernel in enumerate(kernels)
        for sf in scale_factors
    ]


def test_the_default_metric_renders_exactly_what_it_always_did(tmp_path):
    """
    The shipped plots were measured by an older version and are not to be regenerated, so the
    default path has to survive the metric option byte for byte.
    """
    results = _grid([640, 960, 1280], [0.15])
    run_dir = _write_run(tmp_path, results)
    default = plotting.run_plot(run_dir, output=tmp_path / "default.png")
    named = plotting.run_plot(run_dir, output=tmp_path / "named.png", metric="avg_iou")
    assert default.read_bytes() == named.read_bytes()


def test_an_unknown_metric_names_the_ones_that_exist(tmp_path):
    run_dir = _write_run(tmp_path, _grid([640], [0.15]))
    with pytest.raises(ValueError, match="median_angle_error"):
        plotting.run_plot(run_dir, output=tmp_path / "p.png", metric="orientation")


@pytest.mark.parametrize("metric", ["median_iou", "median_angle_error", "p90_angle_error", "iou_at_75", "iou_at_90"])
def test_every_reported_metric_renders_a_run(tmp_path, metric):
    run_dir = _write_run(tmp_path, _grid_with_metrics([640, 960], [0.0, 0.15]), name=f"run-{metric}")
    assert plotting.run_plot(run_dir, output=tmp_path / f"{metric}.png", metric=metric).stat().st_size > 0


@pytest.mark.parametrize(
    "value, expected",
    [
        (0.0, "0.000"),
        (-0.01, "-0.010"),
        (0.05, "0.050"),
        (0.1, "0.100"),
        (0.15, "0.150"),
        (0.0125, "0.0125"),
        (0.0375, "0.0375"),
        (0.00625, "0.00625"),
    ],
)
def test_a_swept_value_is_never_drawn_as_one_that_was_not_swept(value, expected):
    """
    Three decimals turn the 0.0125 of a refined grid into 0.013, a point nobody measured.

    Worse, 0.0375 comes out as 0.037, so the label is not even the nearest three-decimal value.
    Everything a two- or three-decimal grid contains has to keep the width it has always had,
    since that is every figure shipped so far.
    """
    assert plotting.exact(value) == expected


def test_the_wider_summary_format_is_also_left_alone(tmp_path):
    """The run summary asks for four decimals; the shipped grids must still get exactly four."""
    for value in (-0.01, 0.0, 0.05, 0.1, 0.15):
        assert plotting.exact(value, 4) == f"{value:.4f}"


def test_a_lower_is_better_metric_marks_its_own_winner(tmp_path):
    """
    The star must sit on the grid point the drawn metric likes, not on the objective's winner.

    Reading an angle plot whose star marks the best IoU would be worse than not drawing it.
    """
    results = _grid_with_metrics([640, 960], [0.0])
    run_dir = _write_run(tmp_path, results)
    marked = []
    monkey = plotting.plt.scatter

    def record(x, y, *a, **k):
        if k.get("marker") == "*":
            marked.append(y)
        return monkey(x, y, *a, **k)

    plotting.plt.scatter = record
    try:
        plotting.run_plot(run_dir, output=tmp_path / "angle.png", metric="median_angle_error")
    finally:
        plotting.plt.scatter = monkey
    assert marked == [min(r["median_angle_error"] for r in results)]


def test_a_run_measured_before_a_metric_existed_is_left_out(tmp_path):
    """
    Older grid points record no angle at all. Plotting them as zero would invent a perfect run.
    """
    old = _grid([640], [0.15])
    new = _grid_with_metrics([960], [0.15])
    run_dir = _write_run(tmp_path, old + new)
    series = plotting.organize_data_by_series(old + new, plotting.METRICS["median_angle_error"])
    assert {imgsz for imgsz, _ in series} == {960}
    assert plotting.run_plot(run_dir, output=tmp_path / "mixed.png", metric="median_angle_error").is_file()


def test_the_pareto_front_flips_for_an_error_metric(tmp_path):
    """Cheaper and *smaller* dominates when the metric is an error rather than a score."""
    rows = [
        {**row("cheap-accurate", ["sam_l"], 0.88, 10.0), "median_angle_error": 1.5},
        {**row("dear-sloppy", ["sam_l", "sam_b"], 0.89, 90.0), "median_angle_error": 4.0},
    ]
    drawn = []
    monkey = plotting.plt.plot
    plotting.plt.plot = lambda *a, **k: drawn.append(k.get("label")) or monkey(*a, **k)
    try:
        plotting.comparison_plot(rows, tmp_path / "cmp.png", metric="median_angle_error")
    finally:
        plotting.plt.plot = monkey
    # The dear run is beaten on both cost and angle, so nothing is left to join: no front is drawn
    assert "Pareto front" not in drawn


def test_comparing_on_a_metric_no_run_recorded_says_so(tmp_path):
    with pytest.raises(ValueError, match="nothing to compare"):
        plotting.comparison_plot([row("a", ["sam_b"], 0.88, 10.0)], tmp_path / "x.png", metric="median_angle_error")
