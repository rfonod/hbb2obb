# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
Where a conversion's average comes from, cut by the properties of the box it was fitted to.

``hbb2obb-eval`` reports one number per metric over a whole set. That number can be carried by a
majority of easy boxes: an aerial survey flown square to a road grid produces a ground truth that
is mostly axis-aligned, and on those boxes the correct answer is the input box. The cuts here take
the same matched pairs apart by ground-truth orientation, size, frame edge, the annotator's
``difficult`` flag and class, so the easy majority and the hard remainder report separately.

Every cut is a property of the ground-truth box, never of the prediction, so no cut is chosen by
the result it produces.

Two comparisons come with them:

- **The identity baseline.** A horizontal box is already a valid oriented box, so emitting the
  prompt unchanged is the null conversion. Scoring it says how much of the reported IoU the
  conversion earned and how much came free with the prompt.
- **The side ratios.** For each off-axis band, how wide and how long the converted box is against
  the reference. A step in one of them across a boundary the geometry crosses continuously is a
  property of the reference rather than of the conversion, which is how a ground truth built by
  two instruments (a detector for most boxes, a hand for the rest) shows itself.

Matching is the evaluator's own, so a pair scored here is the pair ``hbb2obb-eval`` scored.
"""

from __future__ import annotations

import math
import statistics as st
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import tqdm
from shapely.geometry import Polygon

from hbb2obb.formats import looks_normalized

from hbb2obb.evaluator import (
    HIGH_IOU_THRESHOLD,
    calculate_obb_iou,
    is_edge_box,
    match_boxes,
    obb_orientation,
    orientation_error,
    parse_obb_file,
)

# Degrees off 0 or 90 that still count as square to the image. Tight enough that only a box whose
# corners are exactly axis-aligned falls inside it, which is what a detector emits.
AXIS_TOLERANCE = 0.005

# Off-axis bands for the orientation cut. The off-axis distance is min(a, |a - 90|), so 45 is the
# most a box can be turned away from the image axes.
OFF_AXIS_EDGES = (0.0, 5.0, 15.0, 30.0, 45.001)

# The orientation bands as a partition: the axis-aligned spike on its own, then every rotated box
# by how far it is turned. The cut above overlaps by design, since "rotated" and "0 to 5 degrees"
# both answer questions a reader asks, but a figure needs each box in exactly one bar.
PARTITION_EDGES = (AXIS_TOLERANCE, 5.0, 15.0, 30.0, 45.001)

# Finer bands either side of the axis-aligned spike, for the side-ratio table. A reference built
# by one instrument varies smoothly across these; one built by two steps at the boundary.
BOUNDARY_EDGES = (AXIS_TOLERANCE, 1.0, 3.0, 5.0, 15.0, 45.001)

# Short side of the ground-truth box, in pixels. A vehicle in an aerial frame is a few tens of
# pixels across and the conversion has less to work with at the bottom of that range.
SHORT_SIDE_EDGES = (0, 25, 35, 45, 60, math.inf)


@dataclass
class Pair:
    """One matched ground-truth and converted box, with the properties every cut reads."""

    frame: str
    cls: int
    iou: float
    angle: float
    off_axis: float
    gt_short: float
    gt_long: float
    pred_short: float
    pred_long: float
    area_ratio: float
    edge: Optional[bool] = None
    difficult: Optional[int] = None
    identity_iou: Optional[float] = None
    identity_angle: Optional[float] = None


def box_sides(points) -> Tuple[float, float]:
    """The short and long side of a box, in pixels."""
    corners = np.asarray(points, dtype=float)[:4]
    edges = corners[[1, 2]] - corners[[0, 1]]
    lengths = np.hypot(edges[:, 0], edges[:, 1])
    return float(lengths.min()), float(lengths.max())


def off_axis_angle(points) -> float:
    """How far a box is turned away from the image axes, in degrees, in [0, 45]."""
    angle = obb_orientation(points)
    return float(min(angle, abs(angle - 90.0), 180.0 - angle))


def read_difficult(path: Path) -> List[int]:
    """
    The ``difficult`` column of a DOTA file, by row index.

    DOTA is the only format here that carries the flag, so a set that wants it cut on ships its
    ``.dota`` files beside its ``.txt`` ones. Header lines (``key:value``) are skipped.
    """
    if not path.exists():
        return []
    flags = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) >= 10 and ":" not in fields[0]:
            flags.append(int(fields[9]))
    return flags


def collect_pairs(
    gt_dir: Path,
    pred_dir: Path,
    hbb_dir: Optional[Path] = None,
    image_sizes: Optional[Dict[str, Tuple[int, int]]] = None,
    excluded_classes: Sequence[int] = (),
    iou_threshold: float = 0.1,
    class_agnostic: bool = False,
    no_bar: bool = False,
) -> List[Pair]:
    """
    Match every converted box to its ground truth and record what each cut needs.

    Args:
        gt_dir: Directory of ground-truth OBB annotations
        pred_dir: Directory of converted OBB annotations
        hbb_dir: Directory of the horizontal boxes the conversion was prompted with, for the
                 identity baseline; without it that comparison is left out
        image_sizes: Frame stem to (width, height), for the frame-edge cut; without it that cut
                 is left out
        excluded_classes: Class ids to drop before matching
        iou_threshold: Minimum IoU for a match, as in ``hbb2obb-eval``
        class_agnostic: Match regardless of class label
        no_bar: Disable the progress bar

    Returns:
        One Pair per matched box.
    """
    gt_files = sorted(Path(gt_dir).glob("*.txt"))
    if not gt_files:
        raise SystemExit(f"no ground-truth .txt files in {gt_dir}")

    pairs: List[Pair] = []
    warned_difficult = False
    for gt_file in tqdm.tqdm(gt_files, desc="Analyzing files", leave=True, disable=no_bar):
        pred_file = Path(pred_dir) / gt_file.name
        if not pred_file.exists():
            print(f"Warning: no converted file for {gt_file.name}")
            continue

        gt_boxes = [b for b in parse_obb_file(gt_file) if b["label"] not in excluded_classes]
        pred_boxes = [b for b in parse_obb_file(pred_file) if b["label"] not in excluded_classes]
        matches, _, _ = match_boxes(gt_boxes, pred_boxes, iou_threshold, class_agnostic)

        # The flag is stored per row of the DOTA file, so it can only be attached when that file
        # holds the same boxes in the same order as the ground truth being scored.
        flags = read_difficult(gt_file.with_suffix(".dota"))
        if flags and len(flags) != len(gt_boxes) and not warned_difficult:
            print(
                f"Warning: {gt_file.with_suffix('.dota').name} has {len(flags)} rows, not {len(gt_boxes)}; "
                "the difficult flag is left out"
            )
            warned_difficult = True
        flag_by_index = dict(enumerate(flags)) if len(flags) == len(gt_boxes) else {}
        index_of = {id(b): i for i, b in enumerate(gt_boxes)}

        identity = {}
        if hbb_dir is not None:
            hbb_file = Path(hbb_dir) / gt_file.name
            if hbb_file.exists():
                hbb_boxes = read_hbb_as_boxes(hbb_file, image_sizes, gt_file.stem)
                hbb_boxes = [b for b in hbb_boxes if b["label"] not in excluded_classes]
                for gt_box, hbb_box, _ in match_boxes(gt_boxes, hbb_boxes, iou_threshold, class_agnostic)[0]:
                    identity[id(gt_box)] = hbb_box

        size = (image_sizes or {}).get(gt_file.stem)
        for gt_box, pred_box, iou in matches:
            gt_short, gt_long = box_sides(gt_box["points"])
            pred_short, pred_long = box_sides(pred_box["points"])
            gt_area = gt_box["polygon"].area
            hbb_box = identity.get(id(gt_box))
            pairs.append(
                Pair(
                    frame=gt_file.stem,
                    cls=gt_box["label"],
                    iou=float(iou),
                    angle=orientation_error(gt_box["points"], pred_box["points"]),
                    off_axis=off_axis_angle(gt_box["points"]),
                    gt_short=gt_short,
                    gt_long=gt_long,
                    pred_short=pred_short,
                    pred_long=pred_long,
                    area_ratio=float(pred_box["polygon"].area / gt_area) if gt_area > 0 else 0.0,
                    edge=is_edge_box(gt_box, size[0], size[1]) if size else None,
                    difficult=flag_by_index.get(index_of[id(gt_box)]) if flag_by_index else None,
                    identity_iou=(
                        calculate_obb_iou(gt_box["polygon"], hbb_box["polygon"]) if hbb_box is not None else None
                    ),
                    identity_angle=(
                        orientation_error(gt_box["points"], hbb_box["points"]) if hbb_box is not None else None
                    ),
                )
            )
    return pairs


def read_hbb_as_boxes(path: Path, image_sizes: Optional[Dict[str, Tuple[int, int]]], stem: str) -> List[dict]:
    """
    Read a YOLO HBB file as the evaluator's box dictionaries, so the identity baseline is scored
    by exactly the code that scores the conversion.
    """
    size = (image_sizes or {}).get(stem)
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 5:
            continue
        rows.append((int(fields[0]), *(float(v) for v in fields[1:5])))

    values = [v for row in rows for v in row[1:]]
    if looks_normalized(values):
        if size is None:
            raise SystemExit(f"{path} holds relative coordinates; give --img_source so they can be scaled")
        width, height = size
        rows = [(cls, xc * width, yc * height, w * width, h * height) for cls, xc, yc, w, h in rows]

    boxes = []
    for cls, xc, yc, w, h in rows:
        x0, y0, x1, y1 = xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2
        points = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        boxes.append({"label": cls, "polygon": Polygon(points), "points": points})
    return boxes


# --------------------------------------------------------------------------------- the cuts
def group_stats(pairs: Sequence[Pair]) -> dict:
    """The numbers reported for one group of matched pairs."""
    if not pairs:
        return {"boxes": 0}
    ious = [p.iou for p in pairs]
    angles = [p.angle for p in pairs]
    identity = [p.identity_iou for p in pairs if p.identity_iou is not None]
    stats = {
        "boxes": len(pairs),
        "mean_iou": st.mean(ious),
        "std_iou": st.pstdev(ious),
        "sem_iou": st.pstdev(ious) / math.sqrt(len(ious)),
        "median_iou": st.median(ious),
        "below_075": sum(i < 0.75 for i in ious) / len(ious),
        "at_high": sum(i >= HIGH_IOU_THRESHOLD for i in ious) / len(ious),
        "angle_p90": float(np.percentile(angles, 90)),
        "angle_mean": st.mean(angles),
    }
    if identity:
        stats["identity_iou"] = st.mean(identity)
        stats["identity_angle_p90"] = float(
            np.percentile([p.identity_angle for p in pairs if p.identity_angle is not None], 90)
        )
    return stats


def bands(edges: Sequence[float], label) -> List[Tuple[str, float, float]]:
    """Consecutive [low, high) bands over ``edges``, named by ``label(low, high)``."""
    return [(label(lo, hi), lo, hi) for lo, hi in zip(edges, edges[1:])]


def orientation_cut(pairs: Sequence[Pair]) -> List[Tuple[str, List[Pair]]]:
    """Axis-aligned against rotated, then the rotated boxes by how far they are turned."""
    groups = [
        (f"axis-aligned (<{AXIS_TOLERANCE:g} deg)", [p for p in pairs if p.off_axis < AXIS_TOLERANCE]),
        ("rotated", [p for p in pairs if p.off_axis >= AXIS_TOLERANCE]),
    ]
    for name, lo, hi in bands(OFF_AXIS_EDGES, lambda lo, hi: f"off-axis {lo:g} to {min(hi, 45):g} deg"):
        groups.append((name, [p for p in pairs if lo <= p.off_axis < hi]))
    return groups


def orientation_partition(pairs: Sequence[Pair]) -> List[Tuple[str, List[Pair]]]:
    """The orientation cut with each box in exactly one band, for plotting."""
    groups = [("axis-aligned", [p for p in pairs if p.off_axis < AXIS_TOLERANCE])]
    for name, lo, hi in bands(PARTITION_EDGES, lambda lo, hi: f"{lo:g} to {min(hi, 45):g}"):
        groups.append((name.replace(f"{AXIS_TOLERANCE:g}", "0"), [p for p in pairs if lo <= p.off_axis < hi]))
    return groups


def size_cut(pairs: Sequence[Pair]) -> List[Tuple[str, List[Pair]]]:
    """By the short side of the ground-truth box, which is the vehicle across."""
    groups = []
    for name, lo, hi in bands(
        SHORT_SIDE_EDGES, lambda lo, hi: f"{lo:g} to {hi:g} px" if hi < math.inf else f"{lo:g} px and up"
    ):
        groups.append((name, [p for p in pairs if lo <= p.gt_short < hi]))
    return groups


def class_cut(pairs: Sequence[Pair], label_map: Optional[dict] = None) -> List[Tuple[str, List[Pair]]]:
    """By class label."""
    return [
        ((label_map or {}).get(c, str(c)), [p for p in pairs if p.cls == c]) for c in sorted({p.cls for p in pairs})
    ]


def controlled(pairs: Sequence[Pair]) -> List[Pair]:
    """
    The boxes that carry neither of the other two handicaps: interior, and not flagged difficult.

    Rotated vehicles are over-represented among edge and difficult boxes, so the plain orientation
    cut mixes three effects. On this subset the orientation cut is orientation alone.
    """
    return [p for p in pairs if not p.edge and not p.difficult]


def side_ratios(pairs: Sequence[Pair], classes: Optional[Sequence[int]] = None) -> List[dict]:
    """
    Converted-to-reference side ratios, by how far the reference is turned off the image axes.

    Reported for the short side and the long side separately. The long side is the control: a
    conversion that is simply wider than the reference moves both, while a change in the reference
    itself, such as a band of boxes carrying a detector's extent rather than a hand's, moves the
    one the detector was loose on and leaves the other where it is.
    """
    subset = [p for p in pairs if classes is None or p.cls in classes]
    rows = [("exactly aligned", [p for p in subset if p.off_axis < AXIS_TOLERANCE])]
    for name, lo, hi in bands(BOUNDARY_EDGES, lambda lo, hi: f"{lo:g} to {min(hi, 45):g} deg"):
        rows.append((name, [p for p in subset if lo <= p.off_axis < hi]))
    out = []
    for name, group in rows:
        if not group:
            out.append({"band": name, "boxes": 0})
            continue
        out.append(
            {
                "band": name,
                "boxes": len(group),
                "short_ratio": st.mean([p.pred_short / p.gt_short for p in group if p.gt_short > 0]),
                "long_ratio": st.mean([p.pred_long / p.gt_long for p in group if p.gt_long > 0]),
                "area_ratio": st.mean([p.area_ratio for p in group]),
            }
        )
    return out


def summarise(
    pairs: Sequence[Pair], label_map: Optional[dict] = None, boundary_classes: Optional[Sequence[int]] = None
) -> dict:
    """Every cut and comparison, as plain data a report or a plot can render."""
    clean = controlled(pairs)
    summary = {
        "boxes": len(pairs),
        "has_identity": any(p.identity_iou is not None for p in pairs),
        "has_edge": any(p.edge is not None for p in pairs),
        "has_difficult": any(p.difficult is not None for p in pairs),
        "overall": group_stats(pairs),
        "orientation": [(name, group_stats(g)) for name, g in orientation_cut(pairs)],
        "orientation_controlled": [(name, group_stats(g)) for name, g in orientation_cut(clean)],
        "orientation_partition": [(name, group_stats(g)) for name, g in orientation_partition(pairs)],
        "orientation_partition_controlled": [(name, group_stats(g)) for name, g in orientation_partition(clean)],
        "controlled_boxes": len(clean),
        "size": [(name, group_stats(g)) for name, g in size_cut(pairs)],
        "klass": [(name, group_stats(g)) for name, g in class_cut(pairs, label_map)],
        "side_ratios": side_ratios(clean, boundary_classes),
        "boundary_classes": list(boundary_classes) if boundary_classes else None,
    }
    if summary["has_edge"]:
        summary["edge"] = [
            ("touches an edge", group_stats([p for p in pairs if p.edge])),
            ("interior", group_stats([p for p in pairs if p.edge is False])),
        ]
    if summary["has_difficult"]:
        summary["difficult"] = [
            ("difficult 1", group_stats([p for p in pairs if p.difficult == 1])),
            ("difficult 0", group_stats([p for p in pairs if p.difficult == 0])),
        ]
    return summary


# ------------------------------------------------------------------------------- the report
STAT_COLUMNS = (
    ("boxes", "Boxes", "{:d}"),
    ("mean_iou", "IoU (mean)", "{:.4f}"),
    ("median_iou", "IoU (median)", "{:.4f}"),
    ("below_075", "IoU<0.75", "{:.1%}"),
    ("at_high", f"IoU>={HIGH_IOU_THRESHOLD:g}", "{:.1%}"),
    ("angle_p90", "Angle p90", "{:.2f}"),
    ("angle_mean", "Angle mean", "{:.2f}"),
)


def format_row(name: str, stats: dict) -> List[str]:
    cells = [name]
    for key, _, fmt in STAT_COLUMNS:
        cells.append(fmt.format(stats[key]) if key in stats else "")
    return cells


def markdown_table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def section(title: str, groups: Sequence[Tuple[str, dict]]) -> str:
    header = ["", *(label for _, label, _ in STAT_COLUMNS)]
    return f"### {title}\n\n" + markdown_table(header, [format_row(n, s) for n, s in groups]) + "\n"


def render_markdown(summary: dict, title: str = "Conversion breakdown") -> str:
    """The whole analysis as one Markdown document."""
    overall = summary["overall"]
    out = [f"# {title}", ""]
    out.append(
        f"{summary['boxes']} matched boxes. Mean IoU {overall['mean_iou']:.5f} "
        f"± {overall['std_iou']:.5f} (SEM {overall['sem_iou']:.5f}), median {overall['median_iou']:.5f}, "
        f"{overall['at_high']:.1%} at or above IoU {HIGH_IOU_THRESHOLD:g}, "
        f"orientation error p90 {overall['angle_p90']:.2f} deg."
    )
    out.append("")
    out.append(
        "Two runs of the same set are the same measurement only if their SEM says so: a difference "
        "smaller than one standard error is a tie."
    )
    out.append("")
    out.append(section("By ground-truth orientation", summary["orientation"]))
    out.append(
        section(
            f"By ground-truth orientation, interior and not difficult ({summary['controlled_boxes']} boxes)",
            summary["orientation_controlled"],
        )
    )
    out.append(section("By ground-truth short side", summary["size"]))
    if "edge" in summary:
        out.append(section("By frame edge", summary["edge"]))
    if "difficult" in summary:
        out.append(section("By the annotator's difficult flag", summary["difficult"]))
    out.append(section("By class", summary["klass"]))

    if summary["has_identity"]:
        out.append("### Against doing nothing\n")
        out.append(
            "A horizontal box is already a valid oriented box, so emitting the prompt unchanged is the "
            "null conversion. This is what it scores, beside the conversion, on the same matched pairs.\n"
        )
        rows = []
        for name, stats in [("all", summary["overall"])] + summary["orientation"]:
            if not stats.get("boxes") or "identity_iou" not in stats:
                continue
            rows.append(
                [
                    name,
                    f"{stats['boxes']:d}",
                    f"{stats['identity_iou']:.4f}",
                    f"{stats['mean_iou']:.4f}",
                    f"{stats['mean_iou'] - stats['identity_iou']:+.4f}",
                    f"{stats['identity_angle_p90']:.2f}",
                    f"{stats['angle_p90']:.2f}",
                ]
            )
        out.append(
            markdown_table(
                ["", "Boxes", "IoU as-is", "IoU converted", "Gain", "Angle p90 as-is", "Angle p90 converted"],
                rows,
            )
            + "\n"
        )

    out.append("### Converted-to-reference side ratios\n")
    scope = "all classes" if not summary["boundary_classes"] else f"classes {summary['boundary_classes']}"
    out.append(
        f"Interior boxes not flagged difficult, {scope}. The long side is the control. A step confined to "
        "the short side, across a boundary the geometry crosses continuously, is a property of the "
        "reference rather than of the conversion.\n"
    )
    out.append(
        markdown_table(
            ["Reference off-axis", "Boxes", "Short side", "Long side", "Area"],
            [
                [
                    r["band"],
                    f"{r['boxes']:d}",
                    f"{r['short_ratio']:.4f}" if r["boxes"] else "",
                    f"{r['long_ratio']:.4f}" if r["boxes"] else "",
                    f"{r['area_ratio']:.4f}" if r["boxes"] else "",
                ]
                for r in summary["side_ratios"]
            ],
        )
        + "\n"
    )
    return "\n".join(out)


def print_analysis(summary: dict) -> None:
    """The same report on the terminal, without the Markdown pipes."""
    text = render_markdown(summary)
    for line in text.splitlines():
        if line.startswith("|---") or set(line.strip()) <= {"|", "-"} and line.strip():
            continue
        print(line.replace("| ", "  ").replace(" |", "").rstrip())


def plot_analysis(summary: dict, out_path: Path) -> None:
    """
    Three panels: where the average comes from, what the prompt already gave, and the side ratios.

    Forces the Agg backend, as the sweep plots do, so this runs on a headless machine.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    converted, prompt, grid = "#3b6ea5", "#c0c0c0", "#cccccc"
    panels = 3 if summary["has_identity"] else 2
    fig, axes = plt.subplots(1, panels, figsize=(5.6 * panels, 4.6))
    axes = np.atleast_1d(axes)

    def ticks(ax, names, counts):
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([f"{n}\n({c})" for n, c in zip(names, counts)], fontsize=8)
        ax.grid(axis="y", color=grid, linewidth=0.6)
        ax.set_axisbelow(True)

    named = [(n, st_) for n, st_ in summary["orientation_partition_controlled"] if st_.get("boxes")]
    labels = [n for n, _ in named]
    values = [s["mean_iou"] for _, s in named]

    ax = axes[0]
    ax.bar(labels, values, color=converted)
    ax.errorbar(labels, values, yerr=[s["sem_iou"] for _, s in named], fmt="none", ecolor="black", capsize=3)
    ax.set_ylabel("Mean IoU")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Reference orientation, degrees off the image axes")
    ax.set_title("Where the average comes from\n(interior, not flagged difficult)")
    ticks(ax, labels, [s["boxes"] for _, s in named])

    if summary["has_identity"]:
        ax = axes[1]
        rows = [(n, s) for n, s in summary["orientation_partition"] if s.get("boxes") and "identity_iou" in s]
        x = np.arange(len(rows))
        ax.bar(x - 0.2, [s["identity_iou"] for _, s in rows], 0.4, label="prompt as-is", color=prompt)
        ax.bar(x + 0.2, [s["mean_iou"] for _, s in rows], 0.4, label="converted", color=converted)
        ax.set_ylabel("Mean IoU")
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Reference orientation, degrees off the image axes")
        ax.set_title("Against doing nothing")
        ax.legend(loc="lower left", frameon=False, fontsize=9)
        ticks(ax, [n for n, _ in rows], [s["boxes"] for _, s in rows])

    ax = axes[-1]
    rows = [r for r in summary["side_ratios"] if r["boxes"]]
    x = np.arange(len(rows))
    ax.plot(x, [r["short_ratio"] for r in rows], "o-", label="short side", color="#b5651d")
    ax.plot(x, [r["long_ratio"] for r in rows], "s-", label="long side", color=converted)
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle=":")
    ax.set_ylabel("Converted / reference")
    ax.set_xlabel("Reference orientation, degrees off the image axes")
    scope = "all classes" if not summary["boundary_classes"] else f"class {summary['boundary_classes']}"
    ax.set_title(f"Side ratios by reference orientation\n(interior, not flagged difficult, {scope})")
    ax.legend(frameon=False, fontsize=9)
    ticks(
        ax,
        [r["band"].replace(" deg", "").replace("exactly aligned", "aligned") for r in rows],
        [r["boxes"] for r in rows],
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def to_yaml_safe(summary: dict) -> dict:
    """The summary with tuples flattened, so yaml.safe_dump can write it."""
    out = {}
    for key, value in summary.items():
        if isinstance(value, list) and value and isinstance(value[0], tuple):
            out[key] = [
                {"group": name, **{k: int(v) if k == "boxes" else float(v) for k, v in stats.items()}}
                for name, stats in value
            ]
        else:
            out[key] = value
    return out
