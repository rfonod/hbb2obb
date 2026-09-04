# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
Plots for the hyperparameter sweeps produced by ``hbb2obb.optimizer``.

``run_plot`` renders one sweep: average IoU against the scale factor, with the hue encoding
the inference image size and, when the run swept more than one morphological opening kernel,
the lightness of that hue and the marker shape encoding the kernel, so no two of the swept
combinations share a colour. Marker area encodes the execution time, so the accuracy and the
cost of a grid point are readable at once.

``comparison_plot`` renders a whole benchmark: one point per run, its best IoU against what
that point cost, which is the accuracy-versus-compute trade-off the per-run plots cannot show.

Both are called by ``hbb2obb-optimize``; there is no separate plotting command.
"""

from __future__ import annotations

import colorsys
import os
from pathlib import Path
from typing import Callable, NamedTuple, Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # a sweep writes files, and a workstation run has no display

import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter  # noqa: E402

COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
MARKERS = ['s', '^', 'o', 'D', 'P', '*', 'X']

# The lightness a kernel ladder runs between, lightest kernel first. Kept clear of both ends so
# no shade washes out against the white ground or goes dark enough to lose its hue.
SHADE_RANGE = (0.70, 0.30)


class Metric(NamedTuple):
    """
    One plottable quantity: how to read it out of a grid point and how to label it.

    A sweep records several scores per grid point and only one of them is the objective. Average
    IoU saturates on tight boxes, so two settings can tie on it while differing plainly in
    orientation or in the share of boxes they get nearly right; the plot has to be able to show
    those. Selection is unaffected: the optimizer still ranks by average IoU, and this only
    decides what a figure draws.
    """

    value: Callable[[dict], Optional[float]]
    error: Callable[[dict], Optional[float]]
    axis_label: str
    annotation: Callable[[dict], str]
    higher_is_better: bool
    ylim_pad: float


def _fraction_at(threshold: str) -> Callable[[dict], Optional[float]]:
    """Read one rung of the IoU ladder out of a grid point, tolerating a row that has none."""

    def read(row: dict) -> Optional[float]:
        fractions = row.get("iou_fractions") or {}
        return fractions.get(threshold)

    return read


METRICS = {
    "avg_iou": Metric(
        value=lambda r: r.get("avg_iou"),
        error=lambda r: r.get("std_iou"),
        axis_label="Average IoU (with ±std)",
        annotation=lambda r: f"IoU={r['avg_iou']:.4f}±{r['std_iou']:.4f}",
        higher_is_better=True,
        ylim_pad=0.05,
    ),
    "median_iou": Metric(
        value=lambda r: r.get("median_iou"),
        error=lambda r: None,
        axis_label="Median IoU",
        annotation=lambda r: f"median IoU={r['median_iou']:.4f}",
        higher_is_better=True,
        ylim_pad=0.05,
    ),
    "median_angle_error": Metric(
        value=lambda r: r.get("median_angle_error"),
        error=lambda r: None,
        axis_label="Median orientation error (degrees)",
        annotation=lambda r: f"angle={r['median_angle_error']:.2f}°",
        higher_is_better=False,
        ylim_pad=0.5,
    ),
    "p90_angle_error": Metric(
        value=lambda r: r.get("p90_angle_error"),
        error=lambda r: None,
        axis_label="90th percentile orientation error (degrees)",
        annotation=lambda r: f"p90 angle={r['p90_angle_error']:.2f}°",
        higher_is_better=False,
        ylim_pad=0.5,
    ),
    "iou_at_75": Metric(
        value=_fraction_at("0.75"),
        error=lambda r: None,
        axis_label="Share of matched boxes at IoU ≥ 0.75",
        annotation=lambda r: f"IoU≥0.75: {_fraction_at('0.75')(r):.1%}",
        higher_is_better=True,
        ylim_pad=0.05,
    ),
    "iou_at_90": Metric(
        value=_fraction_at("0.90"),
        error=lambda r: None,
        axis_label="Share of matched boxes at IoU ≥ 0.90",
        annotation=lambda r: f"IoU≥0.90: {_fraction_at('0.90')(r):.1%}",
        higher_is_better=True,
        ylim_pad=0.05,
    ),
}

DEFAULT_METRIC = "avg_iou"


def resolve_metric(metric: Optional[str]) -> Metric:
    """Look up a metric by name, naming the alternatives when the name is not one of them."""
    name = metric or DEFAULT_METRIC
    if name not in METRICS:
        raise ValueError(f"unknown plot metric {name!r}; choose one of {', '.join(sorted(METRICS))}")
    return METRICS[name]


def best_row(results: Sequence[dict], metric: Metric) -> Optional[dict]:
    """The grid point a metric likes most, skipping rows that never recorded it."""
    scored = [r for r in results if metric.value(r) is not None]
    if not scored:
        return None
    return (max if metric.higher_is_better else min)(scored, key=metric.value)


def load_results(benchmark_dir: Path) -> dict:
    """Load the results of one sweep."""
    results_path = benchmark_dir / "results.yaml"
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    with open(results_path, 'r') as f:
        return yaml.safe_load(f)


def run_name_of(benchmark_dir: Path) -> str:
    """
    The name a sweep recorded for itself, falling back to its directory name.

    ``run_config.yaml`` is the current name; folders written before the rename carry
    ``config.yaml`` and still render.
    """
    for filename in ("run_config.yaml", "config.yaml"):
        path = benchmark_dir / filename
        if path.exists():
            with open(path, 'r') as f:
                config = yaml.safe_load(f) or {}
            return config.get('run_name', os.path.basename(benchmark_dir))
    return os.path.basename(benchmark_dir)


def organize_data_by_series(results: Sequence[dict], metric: Optional[Metric] = None) -> dict:
    """Organize sweep results into one series per (image size, opening kernel) pair.

    Results produced before the opening kernel became a swept axis have no
    'opening_kernel_percentage' key; those series are keyed with a kernel of None and
    are labelled by image size alone, exactly as they were before.

    The plotted quantity is read through ``metric``, so the series carry a neutral 'value' and
    'error' rather than one score's own names. A grid point measured before a metric existed
    reports None for it and is left out of that metric's series instead of plotting as zero.
    """
    metric = metric or METRICS[DEFAULT_METRIC]
    data_by_series = {}

    for result in results:
        value = metric.value(result)
        if value is None:
            continue
        key = (result['imgsz'], result.get('opening_kernel_percentage'))
        if key not in data_by_series:
            data_by_series[key] = {'scale_factors': [], 'value': [], 'error': [], 'execution_time': []}

        error = metric.error(result)
        data_by_series[key]['scale_factors'].append(result['scale_factor'])
        data_by_series[key]['value'].append(value)
        data_by_series[key]['error'].append(0.0 if error is None else error)
        data_by_series[key]['execution_time'].append(result['execution_time'])

    # Sort data points by scale factor
    for data in data_by_series.values():
        idx = np.argsort(data['scale_factors'])
        data['scale_factors'] = np.array(data['scale_factors'])[idx]
        data['value'] = np.array(data['value'])[idx]
        data['error'] = np.array(data['error'])[idx]
        data['execution_time'] = np.array(data['execution_time'])[idx]

    return data_by_series


def series_label(imgsz: int, kernel: Optional[float], multiple_kernels: bool) -> str:
    """Label a series, naming the opening kernel only when the run swept more than one."""
    return f"{imgsz}px, k={kernel:g}" if multiple_kernels else f"{imgsz}px"


def shade(color: str, position: float) -> tuple:
    """
    Lighten or darken a base colour, ``position`` 0.0 lightest to 1.0 darkest.

    Hue carries the image size and lightness the opening kernel, so a sweep over both axes gives
    every combination a colour of its own while the reader still reads the two axes out of it
    separately. Marker shape says the same thing as lightness, for a reader who cannot rely on
    the colour. Without this, every kernel at one image size drew in one colour and their lines
    and error bars overprinted each other.
    """
    hue, _, saturation = colorsys.rgb_to_hls(*mcolors.to_rgb(color))
    lightest, darkest = SHADE_RANGE
    return colorsys.hls_to_rgb(hue, lightest + (darkest - lightest) * position, min(1.0, saturation * 1.10))


def ladder_position(index: int, count: int) -> float:
    """Where the ``index``-th of ``count`` kernels sits on the lightness ladder."""
    return index / (count - 1) if count > 1 else 0.5


def legend_handle(marker: str, color, label: str) -> Line2D:
    """A marker-only legend entry, its shape and colour carrying the encoding it stands for."""
    return Line2D(
        [0],
        [0],
        marker=marker,
        color='w',
        markerfacecolor=color,
        markeredgecolor='black',
        markeredgewidth=0.5,
        markersize=10,
        label=label,
    )


def create_plot(
    data_by_series: dict,
    best_params: dict,
    title: str,
    output: Optional[Path] = None,
    dpi: int = 300,
    no_time: bool = False,
    metric: Optional[Metric] = None,
) -> None:
    """Render one sweep and write it to ``output``."""
    metric = metric or METRICS[DEFAULT_METRIC]
    best_value = metric.value(best_params)

    plt.figure()

    # Colour encodes the image size, marker shape the opening kernel (when several were swept)
    image_sizes = sorted({imgsz for imgsz, _ in data_by_series})
    kernels = sorted({kernel for _, kernel in data_by_series}, key=lambda k: (k is None, k))
    multiple_kernels = len(kernels) > 1

    colors = COLORS
    markers = MARKERS

    legend_elements = []

    # Determine marker size range based on execution time
    if not no_time:
        all_times = []
        for series_data in data_by_series.values():
            all_times.extend(series_data['execution_time'])
        min_time, max_time = min(all_times), max(all_times)
        # A zero span means every point cost the same, so every marker takes the minimum size.
        # Without the guard the division makes them all NaN and matplotlib draws none of them.
        time_range = (max_time - min_time) or 1.0

        # Marker size range (min and max size)
        min_size, max_size = 50, 150

    # Plot data for each (image size, opening kernel) series
    for (imgsz, kernel), data in sorted(data_by_series.items(), key=lambda kv: (kv[0][0], kv[0][1] is None, kv[0][1])):
        imgsz_idx = image_sizes.index(imgsz)
        base = colors[imgsz_idx % len(colors)]
        # With a single kernel there is no ladder to walk, so the base colour is used untouched
        color = shade(base, ladder_position(kernels.index(kernel), len(kernels))) if multiple_kernels else base
        # With a single kernel the marker keeps cycling with the image size, as it always has
        marker = markers[(kernels.index(kernel) if multiple_kernels else imgsz_idx) % len(markers)]
        label = series_label(imgsz, kernel, multiple_kernels)

        # Calculate marker sizes based on execution time if requested
        if not no_time:
            marker_sizes = min_size + (data['execution_time'] - min_time) / time_range * (max_size - min_size)
        else:
            marker_sizes = [80] * len(data['scale_factors'])  # Fixed size

        # Plot the values with error bars, where the metric has an error to draw
        if np.any(data['error']):
            plt.errorbar(data['scale_factors'], data['value'], yerr=data['error'], fmt='none', ecolor=color, alpha=0.3)

        plt.scatter(
            data['scale_factors'],
            data['value'],
            s=marker_sizes,
            color=color,
            marker=marker,
            alpha=0.7,
            edgecolors='black',
            linewidths=0.5,
            label=label,
        )

        # Add lines connecting points
        plt.plot(data['scale_factors'], data['value'], color=color, alpha=0.6, linestyle='-', linewidth=1.5)

        # Add to legend
        legend_elements.append(
            Line2D([0], [0], marker=marker, color='w', markerfacecolor=color, markersize=10, label=label)
        )

    # Mark the best parameters with a star
    if not no_time:
        # Find the execution time of the winning point itself
        best_key = (best_params['imgsz'], best_params.get('opening_kernel_percentage'))
        best_series = data_by_series.get(best_key, {})
        best_time = None
        for sf, exec_time in zip(best_series.get('scale_factors', []), best_series.get('execution_time', [])):
            if sf == best_params['scale_factor']:
                best_time = exec_time
                break

        if best_time is not None:
            best_marker_size = min_size + (best_time - min_time) / time_range * (max_size - min_size)
        else:
            best_marker_size = max_size  # Default to max size if can't find the time
    else:
        best_marker_size = 80  # Fixed size, same as other markers

    # Scale the marker up a bit to make it stand out, but keep it proportional
    best_marker_size *= 2.0

    plt.scatter(
        best_params['scale_factor'],
        best_value,
        s=best_marker_size,
        color='gold',
        marker='*',
        edgecolors='black',
        linewidths=1.5,
        zorder=10,
        label="Best",
        alpha=0.7,
    )

    # Add text annotation for best parameters with adaptive placement
    x_best = best_params['scale_factor']
    y_best = best_value

    # Collect points in each quadrant
    quadrant_points = {'top_right': [], 'top_left': [], 'bottom_right': [], 'bottom_left': []}

    for data in data_by_series.values():
        for sf, iou in zip(data['scale_factors'], data['value']):
            if sf == x_best and iou == y_best:
                continue  # Skip the best point itself

            # Determine which quadrant this point is in
            if sf >= x_best and iou >= y_best:
                quadrant_points['top_right'].append((sf, iou))
            elif sf < x_best and iou >= y_best:
                quadrant_points['top_left'].append((sf, iou))
            elif sf >= x_best and iou < y_best:
                quadrant_points['bottom_right'].append((sf, iou))
            else:
                quadrant_points['bottom_left'].append((sf, iou))

    # Find the quadrant with the fewest points
    min_quadrant = min(quadrant_points, key=lambda q: len(quadrant_points[q]))

    # Set annotation position based on the least crowded quadrant
    if min_quadrant == 'top_right':
        ha, va = 'left', 'bottom'
        xytext = (20, 20)
    elif min_quadrant == 'top_left':
        ha, va = 'right', 'bottom'
        xytext = (-20, 20)
    elif min_quadrant == 'bottom_right':
        ha, va = 'left', 'top'
        xytext = (20, -20)
    else:  # bottom_left
        ha, va = 'right', 'top'
        xytext = (-20, -20)

    best_label = series_label(best_params['imgsz'], best_params.get('opening_kernel_percentage'), multiple_kernels)
    plt.annotate(
        f"Best: {best_label}, SF={best_params['scale_factor']:.3f}\n" + metric.annotation(best_params),
        xy=(best_params['scale_factor'], best_value),
        xytext=xytext,
        textcoords='offset points',
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        fontsize=9,
        ha=ha,
        va=va,
        arrowprops=dict(arrowstyle='->', lw=1.5, color='black', alpha=0.7),
    )

    # Add legend for image sizes
    if multiple_kernels:
        # One entry per series is nine rows of "640px, k=0.05" that matplotlib can only park on
        # top of the data, so the two encodings are legended apart into two short blocks: hues
        # for the image size, a grey lightness ladder for the kernel. Fixed corners rather than
        # 'best', which places each legend without knowing where the other one went.
        imgsz_handles = [
            legend_handle('o', shade(colors[i % len(colors)], 0.5), f"{imgsz}px") for i, imgsz in enumerate(image_sizes)
        ]
        kernel_handles = [
            legend_handle(
                markers[i % len(markers)],
                shade('#8a8a8a', ladder_position(i, len(kernels))),
                f"k={kernel:g}" if kernel is not None else "k=default",
            )
            for i, kernel in enumerate(kernels)
        ]
        legend1 = plt.legend(handles=imgsz_handles, loc='upper left', title="Image Size", framealpha=0.7)
        plt.gca().add_artist(legend1)
        plt.gca().add_artist(
            plt.legend(handles=kernel_handles, loc='center right', title="Opening Kernel", framealpha=0.7)
        )
    else:
        legend1 = plt.legend(handles=legend_elements, loc='best', title="Image Size")
        plt.gca().add_artist(legend1)

    # Add marker size legend for execution time if requested
    if not no_time:
        # Create custom handles for marker size legend
        size_handles = []
        size_labels = []

        # Use 3 sizes for the legend
        for size_fraction, label_fraction in zip([0, 0.5, 1], [0, 0.5, 1]):
            size = min_size + size_fraction * (max_size - min_size)
            time = min_time + label_fraction * time_range
            size_handles.append(
                Line2D([0], [0], linestyle='none', marker='o', markersize=np.sqrt(size / 3), color='gray')
            )
            size_labels.append(f"{time:.1f}s")

        # Add execution time legend
        plt.legend(size_handles, size_labels, loc='lower left', title="Execution Time", framealpha=0.7)

    plt.title(title)
    plt.xlabel("Scale Factor")
    plt.ylabel(metric.axis_label)
    plt.grid(True, linestyle='--', alpha=0.6)

    # Set decent axis limits
    plt.ylim(bottom=max(0, min([min(d['value']) for d in data_by_series.values()]) - metric.ylim_pad))

    # Add vertical line at scale_factor = 0
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

    # Add plot annotations in the bottom right
    plt.text(
        0.98,
        0.02,
        f"Total GT: {best_params['total_gt']}, Matches: {best_params['total_matches']}",
        transform=plt.gca().transAxes,
        fontsize=8,
        alpha=0.7,
        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.5),
        horizontalalignment='right',
        verticalalignment='bottom',
    )

    # Adjust layout
    plt.tight_layout()

    plt.savefig(output, dpi=dpi, bbox_inches='tight')
    plt.close()


def run_plot(
    benchmark_dir: Path,
    output: Optional[Path] = None,
    dpi: int = 300,
    no_time: bool = False,
    metric: Optional[str] = None,
) -> Path:
    """
    Render ``benchmark_dir/results.yaml`` into ``benchmark_dir/plot.png``.

    The default metric is the one the sweep optimised, and its plot is what it always was. Any
    other metric marks the grid point *that* metric likes best, which is the whole point of
    asking for it: the interesting question is whether it is the same point.
    """
    spec = resolve_metric(metric)
    data = load_results(benchmark_dir)
    results = data.get('all_results', [])
    best_params = data.get('best_parameters', {})
    if not results or not best_params:
        raise ValueError(f"{benchmark_dir}/results.yaml holds no usable results")

    # The file's own best point is the objective's winner, so it is used unchanged for the
    # objective and recomputed for anything else.
    if metric not in (None, DEFAULT_METRIC):
        best_params = best_row(results, spec)
        if best_params is None:
            raise ValueError(f"{benchmark_dir}/results.yaml holds no grid point recording {metric!r}")

    series = organize_data_by_series(results, spec)
    if not series:
        raise ValueError(f"{benchmark_dir}/results.yaml holds no grid point recording {metric!r}")

    output = output or benchmark_dir / "plot.png"
    create_plot(
        series,
        best_params,
        title=f"HBB2OBB Benchmark Results: {run_name_of(benchmark_dir)}",
        output=output,
        dpi=dpi,
        no_time=no_time,
        metric=spec,
    )
    return output


def use_log_scale(times: Sequence[float]) -> bool:
    """
    Whether the cost axis should be logarithmic.

    A benchmark that sweeps single models against ensembles spans close to an order of
    magnitude, and on a linear axis the cheap half collapses into the leftmost tenth of the
    plot. Below a four-fold spread the linear axis reads better and stays.
    """
    return bool(times) and min(times) > 0 and max(times) / min(times) >= 4


def comparison_plot(rows: Sequence[dict], output: Path, dpi: int = 300, metric: Optional[str] = None) -> Path:
    """
    Render one point per run: its best score against what that grid point cost.

    This is the picture a benchmark of several model ensembles is actually for. A per-run plot
    can only say which scale factor won inside one ensemble; only this one says whether the
    extra models were worth their time.

    Rows that never recorded the requested metric are left out rather than drawn at zero, so a
    folder holding runs from before a metric existed still renders the runs that have it.
    """
    spec = resolve_metric(metric)
    rows = [r for r in rows if spec.value(r) is not None]
    if not rows:
        raise ValueError(f"no run recorded {metric!r}; nothing to compare")
    # Wider than the per-run plots: a run is labelled by every model in it, so a five-model
    # ensemble carries a forty-character name that needs somewhere to go.
    plt.figure(figsize=(11, 6.2))

    times = [r['execution_time'] for r in rows]
    ious = [spec.value(r) for r in rows]
    sizes = [40 + 45 * len(r['sam_models']) for r in rows]
    colors = [COLORS[(len(r['sam_models']) - 1) % len(COLORS)] for r in rows]

    plt.scatter(times, ious, s=sizes, c=colors, alpha=0.75, edgecolors='black', linewidths=0.6, zorder=3)

    # A benchmark of ensembles spans an order of magnitude in cost: the single small models all
    # land within a second or two of each other while a five-model run takes ten times as long.
    # On a linear axis that packs half the runs into the leftmost tenth of the plot, so the axis
    # goes logarithmic once the spread is wide enough to warrant it.
    log_x = use_log_scale(times)
    y_span = (max(ious) - min(ious)) or 1.0
    # Room for the labels, which are long: a five-model run is named by all five models.
    if log_x:
        plt.xscale('log')
        x_lo, x_hi = min(times) / 1.25, max(times) * 1.18
        # Decade ticks alone would label almost nothing over a single decade, and the default
        # minor labels collide with them. A fixed ladder of round numbers reads as seconds.
        ladder = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100, 150, 200, 300, 500, 700, 1000]
        axis = plt.gca().xaxis
        axis.set_major_locator(FixedLocator([t for t in ladder if x_lo <= t <= x_hi]))
        axis.set_minor_locator(NullLocator())
        axis.set_major_formatter(ScalarFormatter())
    else:
        x_span = (max(times) - min(times)) or 1.0
        x_lo, x_hi = min(times) - 0.10 * x_span, max(times) + 0.14 * x_span
    plt.xlim(x_lo, x_hi)
    plt.ylim(min(ious) - 0.16 * y_span, max(ious) + 0.10 * y_span)

    def unit_x(value: float) -> float:
        """Position along the axis in 0-1, so collisions are judged as the reader sees them."""
        if log_x:
            return (np.log10(value) - np.log10(x_lo)) / (np.log10(x_hi) - np.log10(x_lo))
        return (value - x_lo) / (x_hi - x_lo)

    # Label each point on the side that has room, and step a label down when it would land on
    # one already placed. Two ensembles of similar cost and similar accuracy are exactly the
    # comparison this plot exists to make, so their labels must not overprint each other.
    midpoint = 10 ** ((np.log10(x_lo) + np.log10(x_hi)) / 2) if log_x else (min(times) + max(times)) / 2
    placed = []
    for row, x, y in sorted(zip(rows, times, ious), key=lambda r: -r[2]):
        offset_y = 5.0
        while any(
            abs(unit_x(x) - unit_x(px)) < 0.20 and abs(y + offset_y * y_span / 300 - py) < 0.035 * y_span
            for px, py in placed
        ):
            offset_y -= 11.0
        placed.append((x, y + offset_y * y_span / 300))
        right = x > midpoint
        plt.annotate(
            row['name'],
            xy=(x, y),
            xytext=(-7 if right else 7, offset_y),
            textcoords='offset points',
            fontsize=7.5,
            alpha=0.85,
            ha='right' if right else 'left',
        )

    # The Pareto front: the runs no other run beats on both accuracy and time
    def dominated(row):
        # "Better" follows the metric: a run is off the front when another is both cheaper and
        # better, and for orientation error better means smaller.
        if spec.higher_is_better:
            beats = lambda other: spec.value(other) > spec.value(row)  # noqa: E731
        else:
            beats = lambda other: spec.value(other) < spec.value(row)  # noqa: E731
        return any(beats(o) and o['execution_time'] < row['execution_time'] for o in rows)

    front = sorted((r for r in rows if not dominated(r)), key=lambda r: r['execution_time'])
    pareto_legend = None
    if len(front) > 1:
        plt.plot(
            [r['execution_time'] for r in front],
            [spec.value(r) for r in front],
            color='gray',
            linestyle='--',
            linewidth=1.2,
            alpha=0.7,
            zorder=2,
            label="Pareto front",
        )
        pareto_legend = plt.legend(loc='upper left', fontsize=8, framealpha=0.7)

    counts = sorted({len(r['sam_models']) for r in rows})
    handles = [
        Line2D(
            [0],
            [0],
            linestyle='none',
            marker='o',
            markersize=np.sqrt((40 + 45 * n) / 3),
            markerfacecolor=COLORS[(n - 1) % len(COLORS)],
            markeredgecolor='black',
            color='w',
            label=f"{n} model{'s' if n > 1 else ''}",
        )
        for n in counts
    ]
    plt.legend(handles=handles, loc='lower right', title="Ensemble size", fontsize=8, framealpha=0.7)
    if pareto_legend is not None:
        plt.gca().add_artist(pareto_legend)  # the second legend() call would otherwise replace it

    plt.xlabel("Execution time of the best grid point (s)")
    plt.ylabel(
        "Average IoU at the best grid point"
        if spec is METRICS[DEFAULT_METRIC]
        else f"{spec.axis_label} at the best grid point"
    )
    plt.title("HBB2OBB Benchmark: accuracy against compute")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(output, dpi=dpi, bbox_inches='tight')
    plt.close()
    return output
