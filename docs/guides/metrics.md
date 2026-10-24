# Metrics

[`hbb2obb-eval`](evaluate.md), [`hbb2obb-analyze`](analyze.md) and [`hbb2obb-optimize`](optimize.md) all score converted OBBs against ground-truth OBBs the same way. This page defines what they report.

## Matching

Files are paired by name: `gt_dir/0001.txt` with `pred_dir/0001.txt`. A ground-truth file with no prediction file is skipped with a warning, and its boxes are not counted.

Within a frame, each ground-truth box is paired with the prediction of the same class that overlaps it most, and the pair counts as a match when that IoU exceeds the threshold (`--iou_threshold` / `-t`, default 0.1). `--class_agnostic` / `-ca` drops the class condition. Ground-truth boxes are matched independently, so in a dense frame one prediction can be the best match of two of them.

Every metric below is computed over matched pairs only.

## IoU

The intersection over union of the two rotated boxes, as polygons.

| Reported as | Meaning |
| :--- | :--- |
| `Average IoU: 0.89642 ± 0.06834` | mean over matched pairs, ± the standard deviation: the box-to-box spread |
| `(SEM 0.00483)` | standard error of the mean: how precisely the mean is known |
| `Median IoU` | half of the pairs score at least this |
| `IoU>=0.50 ... IoU>=0.90` | the **IoU ladder**: the share of pairs at or above each rung |

The ± and the SEM answer different questions. The ± stays wide however many boxes are scored. The SEM shrinks with the number of boxes and decides whether two runs differ: two mean IoUs less than one SEM apart are a tie.

## Orientation error

A box's orientation is the direction of its longer side, in `[0°, 180°)`. The error between two boxes is the difference of their orientations, wrapped so that a box turned end for end is the same box, which makes 90° the largest possible error. It is reported as its 90th percentile (p90), median and mean ± standard deviation.

The orientation of a near-square box is decided by a pixel or two, so its error can be large while its IoU stays high.

## Which number to read

**Mean IoU saturates.** On tight boxes it sits near 0.9 for every reasonable setting, and two settings can tie on it while recovering visibly different headings. The ladder's upper rungs and the orientation error keep separating them.

**On axis-aligned ground truth, read p90, not the median.** Where most objects lie square to the image, as in drone surveys flown along a road grid, most pairs have an orientation error of exactly 0°, and so does the median at every setting. The p90 sits above that mass.

**Precision and recall are not reported.** When the converted boxes come one for one from HBBs of the ground-truth objects, the match counts are fixed by the inputs rather than by the conversion. The unmatched counts are printed instead.

## Names in files and flags

[`evaluate_obb()`](../reference/api/evaluator.md#hbb2obb.evaluator.evaluate_obb) returns these as `avg_iou`, `std_iou`, `sem_iou`, `median_iou`, `iou_fractions` (the ladder, keyed `"0.50"` to `"0.90"`), `avg_angle_error`, `std_angle_error`, `median_angle_error` and `p90_angle_error`, and `hbb2obb-optimize` records every grid point's values under the same keys in `results.yaml`. Its `--plot_metric` draws one of:

| `--plot_metric` | Metric | Better |
| :--- | :--- | :--- |
| `avg_iou` | mean IoU, the quantity the search ranks by (default) | higher |
| `median_iou` | median IoU | higher |
| `iou_at_75`, `iou_at_90` | share of pairs at or above IoU 0.75, 0.90 | higher |
| `median_angle_error` | median orientation error, degrees | lower |
| `p90_angle_error` | 90th percentile orientation error, degrees | lower |

[`orientation_error()`](../reference/api/evaluator.md#hbb2obb.evaluator.orientation_error) and [`calculate_obb_iou()`](../reference/api/evaluator.md#hbb2obb.evaluator.calculate_obb_iou) score a single pair.
