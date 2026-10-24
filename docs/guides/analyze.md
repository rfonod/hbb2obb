# Analyze an evaluation

[`hbb2obb-eval`](evaluate.md) gives one number per metric. `hbb2obb-analyze` shows where it comes from, by cutting the same matched pairs by what the ground-truth box looks like:

```bash
hbb2obb-analyze /path/to/ground_truth /path/to/predictions -i /path/to/images -o /path/to/analysis
```

Boxes are matched exactly as `hbb2obb-eval` matches them, with the same `--iou_threshold`, `--class_agnostic` and `--excluded_classes`, so the pairs are the ones it scored. Every cut is a property of the ground-truth box, never of the prediction.

## Inputs

Only the two label directories are required. The rest switch tables on:

| Input | Enables |
| :--- | :--- |
| `--img_source` / `-i`, the images | the frame-edge cut; without it the *interior* table keeps edge boxes in. Also denormalizes relative HBBs |
| `--hbb_dir` / `-hd`, the HBB prompts | *Against doing nothing*; defaults to `<gt_dir>/../labels_hbb` when that exists |
| a `.dota` file beside each ground-truth `.txt` | the `difficult` cut, read row by row |
| `--map_path` / `-mp` | class names instead of IDs |

`--out_dir` / `-o` also writes the report as `analysis.md`, the numbers as `analysis.yaml` and a three-panel figure as `analysis.png`.

## The tables

Each row reports the number of boxes, mean and median IoU, the share below IoU 0.75 and at or above 0.9, and the p90 and mean [orientation error](metrics.md#orientation-error).

| Table | Rows |
| :--- | :--- |
| By ground-truth orientation | `axis-aligned`, `rotated`, and bands of off-axis angle |
| ... interior and not difficult | the same, on boxes clear of the frame edge and not flagged `difficult` |
| By ground-truth short side | bands of the shorter side: under 25, 25 to 35, 35 to 45, 45 to 60, 60 px and up |
| By frame edge | boxes with a corner within 1 px of the border, against the rest |
| By the annotator's difficult flag | flagged against not flagged |
| By class | one row per class |
| Against doing nothing | the identity baseline, below |
| Converted-to-reference side ratios | below |

The **off-axis angle** is how far the ground-truth box's longer side is turned from the nearest image axis, from 0° to 45°. `axis-aligned` is under 0.005°, `rotated` everything else; the two partition the set, and the bands of 0 to 5°, 5 to 15°, 15 to 30° and 30 to 45° split it again.

The second orientation table leaves out edge and difficult boxes, so its rows differ by orientation alone.

## The identity baseline

An HBB is already a valid oriented box, so *Against doing nothing* scores each ground-truth box's HBB as if it were the answer, beside the conversion. The HBBs are matched to the ground truth the same way the predictions are. On the 4,245 boxes of the [Songdo Vision OBB](songdo.md) tuning set:

```text
### Against doing nothing

                              Boxes   IoU as-is   IoU converted      Gain   Angle p90 as-is   Angle p90 converted
  all                          4245      0.8617          0.8868   +0.0251             22.69                  3.50
  axis-aligned (<0.005 deg)    2900      0.9816          0.9252   -0.0564              0.00                  0.00
  rotated                      1345      0.6032          0.8039   +0.2007             27.77                 16.00
  off-axis 0 to 5 deg          3159      0.9725          0.9228   -0.0498              0.00                  1.20
  off-axis 5 to 15 deg          162      0.7034          0.8421   +0.1387             13.57                  8.43
  off-axis 15 to 30 deg         820      0.5187          0.7745   +0.2558             25.68                 21.31
  off-axis 30 to 45 deg         104      0.4464          0.7495   +0.3031             43.50                 17.14
```

There, the HBBs were the envelopes of the ground-truth boxes, so on the axis-aligned two thirds the HBB already *was* the answer and the conversion could only lose. On the rotated third it gained 0.20 IoU. A set whose HBBs come from a detector, like the [sample data](quickstart.md#4-analyze), gains everywhere.

## The side ratios

The converted box's short side, long side and area, each divided by the ground truth's, for interior boxes not flagged difficult, in bands of the ground truth's off-axis angle. A ratio below 1 means the conversion is tighter than the reference. A jump confined to the short side between two adjacent bands, across a boundary the geometry crosses smoothly, points at how the reference was drawn rather than at the conversion. `--boundary_classes` / `-bc` restricts this table to some classes.

## The controlled table on Songdo Vision OBB

```text
### By ground-truth orientation, interior and not difficult (3840 boxes)

                              Boxes   IoU (mean)   IoU (median)   IoU<0.75   IoU>=0.9   Angle p90   Angle mean
  axis-aligned (<0.005 deg)    2694       0.9265         0.9351       0.2%      82.3%        0.00         0.14
  rotated                      1146       0.8366         0.8554      13.5%      21.2%        5.35         2.82
  off-axis 0 to 5 deg          2941       0.9239         0.9328       0.2%      79.2%        1.22         0.29
  off-axis 5 to 15 deg          149       0.8479         0.8546       9.4%      25.5%        7.01         3.56
  off-axis 15 to 30 deg         665       0.8171         0.8339      17.3%      12.3%        5.84         3.02
  off-axis 30 to 45 deg          85       0.7966         0.8231      30.6%      12.9%        4.95         2.46
```

## Python API

```python
from pathlib import Path

from hbb2obb.analysis import collect_pairs, print_analysis, summarise
from hbb2obb.formats import image_sizes

pairs = collect_pairs(
    Path("data/labels_obb_gt"),
    Path("data/labels_obb"),
    hbb_dir=Path("data/labels_hbb"),
    image_sizes=image_sizes(Path("data/images")),
)
summary = summarise(pairs, label_map={0: "Car", 1: "Bus", 2: "Truck", 3: "Motorcycle"})
print_analysis(summary)
```

Each [`Pair`](../reference/api/analysis.md#hbb2obb.analysis.Pair) holds one matched box with every property the cuts read, for cuts of your own.

All flags: [`hbb2obb-analyze` reference](../reference/cli/hbb2obb-analyze.md).
