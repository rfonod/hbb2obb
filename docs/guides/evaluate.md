# Evaluate

`hbb2obb-eval` scores predicted OBBs against ground-truth OBBs:

```bash
hbb2obb-eval /path/to/ground_truth /path/to/predictions -mp label_map.yaml
```

Both directories hold one YOLO OBB `.txt` per frame, paired by file name. Coordinates may be absolute or normalized, but the same in both; a trailing confidence column is ignored. Ground truth in another format converts first with [`hbb2obb-convert`](formats.md).

## The report

On the [sample data](quickstart.md#2-evaluate):

```text
=== Overall Results ===
Evaluation Mode: Class-Specific (matches only boxes with same class label)
Total Ground Truth Boxes: 201
Total Predicted/Converted Boxes: 201
Total Matched Boxes: 200
Total Unmatched GT Boxes: 1
Total Unmatched Pred Boxes: 1
Average IoU: 0.89642 ± 0.06834 (SEM 0.00483)
Median IoU: 0.91080
Matched Boxes Above Threshold: IoU>=0.50: 100.0%  IoU>=0.75: 98.0%  IoU>=0.85: 80.5%  IoU>=0.90: 57.0%
Orientation Error: p90 4.40°, median 1.04°, mean 2.11° ± 4.88°

=== Results by Class ===
     Class  GT  Pred  Matches IoU (mean ± std) IoU (median) IoU>=0.9 Angle p90 (°) Angle p50 (°)
       Car 185   186      185  0.8982 ± 0.0606       0.9106    57.3%          4.36          1.00
       Bus   6     6        6  0.9358 ± 0.0353       0.9390    66.7%          1.66          1.33
     Truck   8     8        8  0.8533 ± 0.1469       0.8884    50.0%         13.87          1.56
Motorcycle   2     1        1  0.6746 ± 0.0000       0.6746     0.0%          9.95          9.95
```

- The counts say how many boxes each side has and how many were paired. An unmatched ground-truth box has no prediction overlapping it above the IoU threshold; an unmatched prediction is the best match of no ground-truth box.
- **IoU**: the mean with its box-to-box spread and its standard error (SEM), the median, and the IoU ladder: the share of matched pairs at or above 0.5, 0.75, 0.85 and 0.9.
- **Orientation error**: the angle between the longer sides of each matched pair, from 0° to 90°.
- **Results by class**: the same per ground-truth class. Without `-mp` the classes are shown by ID.

[Metrics](metrics.md) defines each number and which to read when. Here the mean IoU hides an orientation tail (median 1.04°, p90 4.40°), and the one motorcycle converts far worse than the cars.

To see where the score comes from, run [`hbb2obb-analyze`](analyze.md) on the same two directories.

## Matching options

| Flag | Effect |
| :--- | :--- |
| `--iou_threshold` / `-t` | minimum IoU for a pair to match (default 0.1) |
| `--class_agnostic` / `-ca` | match regardless of class; the report then counts the pairs whose classes differ |
| `--excluded_classes` / `-e` | class IDs to leave out of both sides |
| `--debug` / `-d` | print up to three unmatched boxes on each side, for every frame that has any |

`--class_agnostic` is the option to use when the ground truth was labelled with different classes from the predictions.

## Leaving out boxes at the frame edge

To score only the boxes clear of the frame border:

```bash
hbb2obb-eval gt pred --exclude_edge_cases --img_width 3840 --img_height 2160 --edge_tolerance 2
```

A ground-truth box is left out when any corner lies within `--edge_tolerance` / `-et` pixels of the border (default 1), and the report says how many were. Predictions are not filtered, so the prediction of a left-out box counts as unmatched. The frame size applies to every frame, so the set must share one resolution, and the coordinates must be absolute. [`hbb2obb-analyze`](analyze.md) reads each frame's size from the images instead, and reports edge boxes as their own group.

## Python API

```python
from pathlib import Path

from hbb2obb.evaluator import evaluate_obb, print_results

results = evaluate_obb(
    gt_dir=Path("data/labels_obb_gt"),
    pred_dir=Path("data/labels_obb"),
    iou_threshold=0.1,
    class_agnostic=False,
)
print(f"{results['avg_iou']:.4f} ± {results['sem_iou']:.4f} (SEM), p90 {results['p90_angle_error']:.2f}°")

print_results(results, Path("data/classes.yaml"))  # the report above
```

[`evaluate_obb()`](../reference/api/evaluator.md#hbb2obb.evaluator.evaluate_obb) lists every key of the returned dictionary; `class_stats` holds the per-class values.

All flags: [`hbb2obb-eval` reference](../reference/cli/hbb2obb-eval.md).
