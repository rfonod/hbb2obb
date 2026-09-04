# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
Hyperparameter sweeps for the HBB to OBB conversion.

One *run* is a grid search over inference image size x scale factor x morphological opening
kernel for one set of SAM models, scored by average IoU against ground-truth OBBs. One
*benchmark* is several runs described by a single YAML file, which is what makes a multi-hour
comparison of model ensembles reproducible from one command rather than from a shell script
nobody kept.

Every grid point is a complete SAM pass over the whole image set, so the cost multiplies fast:
3 image sizes x 12 scale factors x 1 kernel is already 36 passes per run. The passes run
in-process and share ``converter``'s model cache, so each checkpoint is loaded once per run
rather than once per grid point; ``execution_time`` therefore measures the conversion with the
models already resident, which is the number worth comparing between grid points.

Driven by ``hbb2obb-optimize``; see ``hbb2obb-optimize --help``.
"""

from __future__ import annotations

import copy
import datetime
import tempfile
import time
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml

DEFAULT_IMGSZ = [640, 960, 1280]
DEFAULT_SCALE_FACTORS = [-0.01, 0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
DEFAULT_OPENING_KERNELS = [0.15]

RUN_CONFIG_NAME = "run_config.yaml"
RESULTS_NAME = "results.yaml"
SUMMARY_NAME = "summary.txt"
PLOT_NAME = "plot.png"

BENCHMARK_SUMMARY_NAME = "summary.md"
BENCHMARK_PLOT_NAME = "comparison.png"
BENCHMARK_PROVENANCE_NAME = "PROVENANCE.txt"
AUTO_PLOT_METRICS = ("median_angle_error", "iou_at_90")

# What a benchmark YAML may say, at each level. Anything else is a typo, and a typo in a file
# that drives a six-hour unattended run should fail immediately rather than silently do the
# wrong sweep.
TOP_LEVEL_KEYS = {"img_source", "gt_dir", "hbb_dir", "output_folder", "defaults", "runs"}
RUN_KEYS = {
    "name",
    "sam_models",
    "imgsz",
    "scale_factors",
    "opening_kernels",
    "excluded_classes",
    "iou_threshold",
    "class_agnostic",
    "exclude_edge_cases",
    "edge_tolerance",
    "img_width",
    "img_height",
    "model_kwargs",
    "device",
}


@dataclass
class RunSpec:
    """One grid search: a set of models, the axes to sweep, and how to score the result."""

    name: str
    sam_models: List[str]
    imgsz: List[int] = field(default_factory=lambda: list(DEFAULT_IMGSZ))
    scale_factors: List[float] = field(default_factory=lambda: list(DEFAULT_SCALE_FACTORS))
    opening_kernels: List[float] = field(default_factory=lambda: list(DEFAULT_OPENING_KERNELS))
    excluded_classes: List[int] = field(default_factory=list)
    iou_threshold: float = 0.1
    class_agnostic: bool = False
    exclude_edge_cases: bool = False
    edge_tolerance: int = 1
    img_width: Optional[int] = None
    img_height: Optional[int] = None
    model_kwargs: Optional[str] = None
    device: Optional[str] = None

    @property
    def grid(self) -> List[tuple]:
        return list(product(self.imgsz, self.scale_factors, self.opening_kernels))

    def describe_grid(self) -> str:
        points = len(self.grid)
        return (
            f"{len(self.imgsz)} image size(s) x {len(self.scale_factors)} scale factor(s) "
            f"x {len(self.opening_kernels)} opening kernel(s) = {points} point{'s' if points != 1 else ''}"
        )


# ------------------------------------------------------------------------------------- config
def load_config(path: Path) -> dict:
    """Read and validate a benchmark YAML."""
    if not path.is_file():
        raise SystemExit(f"benchmark config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise SystemExit(f"{path} must hold a mapping at the top level")

    unknown = set(cfg) - TOP_LEVEL_KEYS
    if unknown:
        raise SystemExit(f"{path}: unknown key(s) {sorted(unknown)}; expected any of {sorted(TOP_LEVEL_KEYS)}")
    for required in ("img_source", "gt_dir", "runs"):
        if not cfg.get(required):
            raise SystemExit(f"{path}: '{required}' is required")
    if not isinstance(cfg["runs"], list):
        raise SystemExit(f"{path}: 'runs' must be a list")
    return cfg


def write_config_copy(output_folder: Path, config_path: Path, config_text: str) -> Path:
    """
    Keep the benchmark configuration beside the numbers it produced.

    A results folder is usually read somewhere else entirely, inside a dataset archive most of
    all, where the repository holding the config is not at hand. Copying it here is what lets
    the folder be re-run from itself rather than from a file the reader has to be told about.
    """
    destination = output_folder / config_path.name
    if destination.resolve() != config_path.resolve():
        destination.write_text(config_text, encoding="utf-8")
    return destination


def expand_runs(
    cfg: dict, supported_models: Optional[Sequence[str]] = None, config_path: str = "config"
) -> List[RunSpec]:
    """
    Turn the config's defaults plus its per-run overrides into one RunSpec per run.

    A run that names no ``name`` takes one from its models, so ``[sam_l, sam_b]`` writes into
    ``sam_l-sam_b``, which is both readable and unique for as long as no two runs share a model
    set.
    """
    base = dict(cfg.get("defaults") or {})
    unknown = set(base) - RUN_KEYS
    if unknown:
        raise SystemExit(f"{config_path}: unknown key(s) in 'defaults': {sorted(unknown)}")
    if "name" in base:
        raise SystemExit(f"{config_path}: 'name' belongs to a run, not to 'defaults'")

    specs: List[RunSpec] = []
    for i, entry in enumerate(cfg["runs"]):
        if not isinstance(entry, dict):
            raise SystemExit(f"{config_path}: run #{i + 1} must be a mapping")
        unknown = set(entry) - RUN_KEYS
        if unknown:
            raise SystemExit(f"{config_path}: unknown key(s) in run #{i + 1}: {sorted(unknown)}")

        merged = {**base, **entry}
        models = merged.get("sam_models")
        if not models:
            raise SystemExit(f"{config_path}: run #{i + 1} names no 'sam_models'")
        if isinstance(models, str):
            models = [models]
        if supported_models is not None:
            bad = [m for m in models if m not in supported_models]
            if bad:
                raise SystemExit(f"{config_path}: run #{i + 1} names unsupported model(s) {bad}")

        merged["sam_models"] = list(models)
        merged.setdefault("name", "-".join(models))
        specs.append(RunSpec(**merged))

    names = [s.name for s in specs]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise SystemExit(f"{config_path}: duplicate run name(s) {duplicates}; give them an explicit 'name'")
    return specs


# -------------------------------------------------------------------------------------- sweep
def is_complete(folder: Path, spec: RunSpec) -> bool:
    """
    Whether a run folder already holds this run's full grid of results, for --resume.

    The grid points on disk must be the ones the spec asks for, not merely as many of them.
    A benchmark whose axes were edited between attempts, an image size dropped after it ran out
    of memory most of all, would otherwise resume straight past the runs that measured the old
    grid and report their numbers under the new config.
    """
    results_file = folder / RESULTS_NAME
    if not results_file.is_file():
        return False
    try:
        with open(results_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except yaml.YAMLError:
        return False

    measured = {
        (int(r["imgsz"]), float(r["scale_factor"]), float(r["opening_kernel_percentage"]))
        for r in (data.get("all_results") or [])
        if {"imgsz", "scale_factor", "opening_kernel_percentage"} <= set(r)
    }
    return measured == {(int(i), float(sf), float(ok)) for i, sf, ok in spec.grid}


def sweep(
    spec: RunSpec,
    img_source: Path,
    gt_dir: Path,
    hbb_dir: Optional[Path] = None,
    no_bar: bool = True,
    quiet: bool = False,
) -> dict:
    """
    Run one grid search and return its results.

    Each grid point converts the whole image set into a temporary directory and evaluates that
    against the ground truth, so nothing the caller owns is written to or overwritten.
    """
    from hbb2obb import converter
    from hbb2obb.evaluator import evaluate_obb
    from hbb2obb.utils import get_hbb_dir, get_image_paths, process_ultralytics_kwargs

    hbb_dir = get_hbb_dir(img_source, hbb_dir)
    image_paths = get_image_paths(img_source)
    model_kwargs = process_ultralytics_kwargs(spec.model_kwargs)

    # Load every checkpoint once, up front, so the grid points are timed on equal footing and
    # the first one does not carry the whole ensemble's load time.
    load_start = time.time()
    for model_name in spec.sam_models:
        converter.load_sam_model(model_name)
    model_load_seconds = time.time() - load_start

    if not quiet:
        print(f"Loaded {len(spec.sam_models)} model(s) in {model_load_seconds:.1f}s: {' '.join(spec.sam_models)}")
        print("-" * 124)
        print(
            f"{'Image Size':<12} {'Scale Factor':<15} {'Kernel':<10} {'Avg IoU':<17} {'Angle':<8} "
            f"{'Matches':<10} {'GT Total':<10} {'Pred Total':<10} {'IoU Threshold':<15} {'Time (s)':<10}"
        )
        print("-" * 124)

    results: List[dict] = []
    best_iou = -1.0
    best_params: Optional[dict] = None
    sweep_start = time.time()

    with tempfile.TemporaryDirectory() as temp_dir:
        for imgsz, sf, ok in spec.grid:
            run_dir = Path(temp_dir) / f"imgsz_{imgsz}_sf_{sf}_ok_{ok}"
            run_dir.mkdir()

            start_time = time.time()
            for img_path in image_paths:
                # A malformed HBB file is a per-image problem, not a reason to lose the rest of an
                # hours-long sweep: skip it and keep going, the same way the viewer warns and moves on.
                try:
                    obb_annotations = converter.hbb2obb(
                        img_path=img_path,
                        hbb_dir=hbb_dir,
                        sam_models=spec.sam_models,
                        imgsz=imgsz,
                        scale_factors=sf,
                        opening_kernel_percentage=ok,
                        model_kwargs=model_kwargs,
                        device=spec.device,
                    )
                except ValueError as e:
                    print(f"Warning: skipping {img_path.name}: {e}")
                    continue
                converter.save_obb_annotations(obb_annotations, run_dir, img_path)
            execution_time = time.time() - start_time

            eval_results = evaluate_obb(
                gt_dir=gt_dir,
                pred_dir=run_dir,
                excluded_classes=spec.excluded_classes,
                iou_threshold=spec.iou_threshold,
                class_agnostic=spec.class_agnostic,
                exclude_edge_cases=spec.exclude_edge_cases,
                edge_tolerance=spec.edge_tolerance,
                img_width=spec.img_width,
                img_height=spec.img_height,
                debug=False,
                no_bar=no_bar,
            )

            param_result = {
                "imgsz": int(imgsz),
                "scale_factor": float(sf),
                "opening_kernel_percentage": float(ok),
                "avg_iou": float(eval_results["avg_iou"]),
                "std_iou": float(eval_results["std_iou"]),
                "median_iou": float(eval_results["median_iou"]),
                "iou_fractions": {k: float(v) for k, v in eval_results["iou_fractions"].items()},
                "median_angle_error": float(eval_results["median_angle_error"]),
                "avg_angle_error": float(eval_results["avg_angle_error"]),
                "std_angle_error": float(eval_results["std_angle_error"]),
                "p90_angle_error": float(eval_results["p90_angle_error"]),
                "total_matches": int(eval_results["total_matches"]),
                "total_gt": int(eval_results["total_gt"]),
                "total_pred": int(eval_results["total_pred"]),
                "class_agnostic": spec.class_agnostic,
                "iou_threshold": spec.iou_threshold,
                "excluded_classes": list(spec.excluded_classes),
                "execution_time": float(execution_time),
            }
            results.append(param_result)

            if param_result["avg_iou"] > best_iou:
                best_iou = param_result["avg_iou"]
                best_params = param_result

            if not quiet:
                print(
                    f"{imgsz:<12} {sf:^15.3f} {ok:^10.3f} {param_result['avg_iou']:<1.4f} ± "
                    f"{param_result['std_iou']:<1.4f}   {param_result['median_angle_error']:<7.2f} "
                    f"{param_result['total_matches']:<10} "
                    f"{param_result['total_gt']:<10} {param_result['total_pred']:<10} "
                    f"{spec.iou_threshold:<15} {execution_time:<10.2f}"
                )

    return {
        "best_parameters": best_params,
        "all_results": copy.deepcopy(results),
        "model_load_seconds": float(model_load_seconds),
        "sweep_seconds": float(time.time() - sweep_start),
    }


# ------------------------------------------------------------------------------------ writing
def run_config_dict(spec: RunSpec, img_source: Path, gt_dir: Path, hbb_dir: Path) -> dict:
    """The resolved configuration of one run, as recorded beside its results."""
    from hbb2obb.utils import get_system_metadata

    config = {
        "run_name": spec.name,
        "img_source": str(img_source),
        "gt_dir": str(gt_dir),
        "hbb_dir": str(hbb_dir),
        "sam_models": list(spec.sam_models),
        "scale_factors": list(spec.scale_factors),
        "imgsz": list(spec.imgsz),
        "opening_kernels": list(spec.opening_kernels),
        "excluded_classes": list(spec.excluded_classes),
        "iou_threshold": spec.iou_threshold,
        "class_agnostic": spec.class_agnostic,
        "model_kwargs": spec.model_kwargs,
        "device": spec.device,
    }
    config["system_metadata"] = get_system_metadata()
    return config


def write_run(
    folder: Path, spec: RunSpec, outcome: dict, config: dict, plot: bool = True, metric: Optional[str] = None
) -> None:
    """Write run_config.yaml, results.yaml, summary.txt and plot.png for one run."""
    folder.mkdir(parents=True, exist_ok=True)

    with open(folder / RUN_CONFIG_NAME, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    with open(folder / RESULTS_NAME, "w", encoding="utf-8") as f:
        yaml.dump(
            {
                "best_parameters": outcome["best_parameters"],
                "all_results": outcome["all_results"],
                "model_load_seconds": outcome["model_load_seconds"],
                "sweep_seconds": outcome["sweep_seconds"],
            },
            f,
            default_flow_style=False,
            sort_keys=False,
        )

    best = outcome["best_parameters"]
    with open(folder / SUMMARY_NAME, "w", encoding="utf-8") as f:
        f.write(f"Benchmark Run: {spec.name}\n")
        f.write(f"End Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("BEST PARAMETERS:\n")
        f.write(f"  Image Size: {best['imgsz']}\n")
        f.write(f"  Scale Factor: {best['scale_factor']:.4f}\n")
        f.write(f"  Opening Kernel: {best['opening_kernel_percentage']:.4f}\n")
        f.write(f"  Average IoU: {best['avg_iou']:.4f} ± {best['std_iou']:.4f}\n")
        f.write(f"  Median IoU: {best['median_iou']:.4f}\n")
        f.write(
            "  Matched Boxes Above Threshold: "
            + "  ".join(f"IoU>={t}: {share:.1%}" for t, share in sorted(best['iou_fractions'].items()))
            + "\n"
        )
        f.write(
            f"  Orientation Error: median {best['median_angle_error']:.2f} deg, "
            f"mean {best['avg_angle_error']:.2f} +/- {best['std_angle_error']:.2f} deg, "
            f"p90 {best['p90_angle_error']:.2f} deg\n"
        )
        f.write(f"  Total Matches: {best['total_matches']}\n")
        f.write(f"  Total GT: {best['total_gt']}\n")
        f.write(f"  Total Pred: {best['total_pred']}\n")
        f.write(f"  Execution Time: {best['execution_time']:.2f} seconds\n")
        f.write("\nALL RESULTS (sorted by Average IoU):\n")
        for i, result in enumerate(sorted(outcome["all_results"], key=lambda x: x["avg_iou"], reverse=True)):
            f.write(
                f"{i + 1:4d}. ImgSz: {result['imgsz']:5d}, SF: {result['scale_factor']:7.4f}, "
                f"K: {result['opening_kernel_percentage']:7.4f}, IoU: {result['avg_iou']:7.4f} ± "
                f"{result['std_iou']:7.4f}, Angle: {result['median_angle_error']:6.2f} deg, "
                f"IoU>=0.90: {result['iou_fractions']['0.90']:6.1%}, "
                f"Time: {result['execution_time']:5.2f}s\n"
            )

    if plot:
        from hbb2obb import plotting

        plotting.run_plot(folder, metric=metric)


# -------------------------------------------------------------------------------- aggregation
def collect_rows(output_folder: Path, names: Sequence[str]) -> List[dict]:
    """
    Read the best grid point of each finished run.

    Runs with no results yet are skipped rather than reported as zero, so a partial benchmark
    summarises what it actually has.
    """
    rows = []
    for name in names:
        folder = output_folder / name
        results_file = folder / RESULTS_NAME
        if not results_file.is_file():
            continue
        with open(results_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        best = data.get("best_parameters")
        if not best:
            continue

        models: List[str] = []
        for filename in (RUN_CONFIG_NAME, "config.yaml"):
            path = folder / filename
            if path.is_file():
                with open(path, "r", encoding="utf-8") as f:
                    models = (yaml.safe_load(f) or {}).get("sam_models") or []
                break
        rows.append(
            {
                "name": name,
                "sam_models": models or name.split("-"),
                "n_points": len(data.get("all_results") or []),
                "sweep_seconds": data.get("sweep_seconds"),
                **best,
            }
        )
    return sorted(rows, key=lambda r: r["avg_iou"], reverse=True)


def write_summary(
    output_folder: Path,
    rows: Sequence[dict],
    img_source: Path,
    hbb_dir: Path,
    gt_dir: Path,
    command: str,
    elapsed_seconds: Optional[float] = None,
    plot: bool = True,
    provenance: bool = False,
    config_name: Optional[str] = None,
    metric: Optional[str] = None,
) -> Path:
    """
    Write the one document that reads every run together.

    The per-run folders answer "which scale factor won for this ensemble"; only this answers
    "which ensemble, and was it worth its time".
    """
    from hbb2obb import plotting
    from hbb2obb.__version__ import __version__

    # A summary for a metric other than the ranking one is a second document beside the first,
    # never a replacement for it, so it is named after its metric exactly as its figure is.
    out = output_folder / plotting.plot_filename(Path(BENCHMARK_SUMMARY_NAME).stem, metric, "md")
    if not rows:
        out.write_text("# HBB2OBB Benchmark Summary\n\nNo finished runs to summarise.\n", encoding="utf-8")
        return out

    grids = {r["n_points"] for r in rows}
    grid = f"{grids.pop()} grid points per run" if len(grids) == 1 else "a grid that varies by run"
    total_points = sum(r["n_points"] for r in rows)
    best = rows[0]

    lines = [
        "# HBB2OBB Benchmark Summary",
        "",
        f"Generated by `{command}` on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}, hbb2obb {__version__}.",
        "",
        f"- Images: `{img_source}`",
        f"- HBB inputs: `{hbb_dir}`",
        f"- Ground truth: `{gt_dir}`",
        f"- {len(rows)} run(s), {grid}, {total_points} conversions of the whole image set in total.",
    ]
    if elapsed_seconds:
        lines.append(f"- Total wall time: {elapsed_seconds / 3600:.2f} h.")
    else:
        # A --refresh has no wall time of its own, so fall back to what the runs recorded. The
        # summary must not silently lose the cost of the benchmark just because it was redrawn.
        recorded = [r["sweep_seconds"] for r in rows if r.get("sweep_seconds")]
        if len(recorded) == len(rows):
            lines.append(f"- Total sweep time: {sum(recorded) / 3600:.2f} h.")
    lines.append("")
    if provenance:
        lines += [f"See `{BENCHMARK_PROVENANCE_NAME}` beside this file for the checkpoint and input hashes.", ""]
    if config_name:
        lines += [
            f"The configuration that produced all of this is beside this file as `{config_name}`, "
            "so the folder re-runs without the repository.",
            "",
        ]
    lines += [
        "## Best grid point per run",
        "",
        "| Models | Image size | Scale factor | Kernel | Average IoU | Angle err | IoU>=0.9 | Matches / GT | Time |",
        "| :--- | ---: | ---: | ---: | :--- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        # A run measured before these metrics existed still belongs in the table, so the columns it
        # cannot fill say so rather than crashing the summary of a folder that holds both.
        angle = f"{row['median_angle_error']:.2f}°" if row.get("median_angle_error") is not None else "-"
        high = row.get("iou_fractions") or {}
        high = f"{high['0.90']:.1%}" if "0.90" in high else "-"
        lines.append(
            f"| `{' '.join(row['sam_models'])}` | {row['imgsz']} | {row['scale_factor']:g} | "
            f"{row['opening_kernel_percentage']:g} | {row['avg_iou']:.4f} ± {row['std_iou']:.4f} | "
            f"{angle} | {high} | "
            f"{row['total_matches']} / {row['total_gt']} | {row['execution_time']:.1f} s |"
        )

    lines += [
        "",
        f"**Best overall:** `{' '.join(best['sam_models'])}` at {best['imgsz']} px, scale factor "
        f"{best['scale_factor']:g}, opening kernel {best['opening_kernel_percentage']:g}, "
        f"average IoU {best['avg_iou']:.4f} ± {best['std_iou']:.4f} in {best['execution_time']:.1f} s.",
        "",
        "Runs are ranked by average IoU, which is the quantity the grid search optimises. The angle",
        "and IoU>=0.9 columns are reported, never optimised: the mean saturates on tight boxes and",
        "cannot separate settings that these two can, so they are what to read when two runs tie.",
        "",
    ]

    if plot:
        comparison_name = plotting.plot_filename(Path(BENCHMARK_PLOT_NAME).stem, metric)
        # The ranking metric keeps the caption it has always had; another metric names itself.
        caption = (
            "Best IoU" if metric in (None, plotting.DEFAULT_METRIC) else plotting.resolve_metric(metric).axis_label
        )
        # No run recording the requested metric is a fact about the folder, not an error: the
        # summary still has a table to write, so it says why the figure is missing and goes on.
        try:
            plotting.comparison_plot(rows, output_folder / comparison_name, metric=metric)
        except ValueError as e:
            lines += [f"No accuracy-against-compute figure: {e}.", ""]
        else:
            lines += [
                "## Accuracy against compute",
                "",
                f"![{caption} against execution time]({comparison_name})",
                "",
                "Each point is one run at its best grid point. The dashed line is the Pareto front:",
                "the runs that no other run beats on both accuracy and time.",
                "",
            ]

    lines += [
        "## Per-run detail",
        "",
        f"Each `<run>/` folder holds `{RUN_CONFIG_NAME}` (the resolved settings and the host it ran on),",
        f"`{RESULTS_NAME}` (every grid point), `{SUMMARY_NAME}` (the same, human readable) and",
        f"`{PLOT_NAME}` (average IoU against scale factor, coloured by image size).",
        "",
        "The time column is one pass over the image set with the models already loaded. Checkpoint",
        "loading is timed separately, as `model_load_seconds` in each `results.yaml`.",
        "",
    ]

    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def print_best(best: Dict[str, Any], run_folder: Path) -> None:
    """The block a finished run prints, unchanged from the script this replaced."""
    print("\n" + "=" * 116)
    print("BEST PARAMETERS:")
    print(f"  Image Size: {best['imgsz']}")
    print(f"  Scale Factor: {best['scale_factor']:.4f}")
    print(f"  Opening Kernel: {best['opening_kernel_percentage']:.4f}")
    print(f"  Average IoU: {best['avg_iou']:.4f} ± {best['std_iou']:.4f}")
    print(f"  Total Matches: {best['total_matches']}")
    print(f"  Total GT: {best['total_gt']}")
    print(f"  Total Pred: {best['total_pred']}")
    print(f"  Execution Time: {best['execution_time']:.2f} seconds")
    print(f"\nResults saved to: {run_folder}")
    print("=" * 116)
