# Annotation formats

`hbb2obb` reads and writes YOLO TXT: one file per image, named after it (`0001.jpg` → `0001.txt`), one object per line, fields separated by spaces. [`hbb2obb-convert`](formats.md) moves these to and from DOTA, Pascal VOC, COCO and LabelMe.

## HBBs (input)

```text
class_id x_center y_center width height [confidence]
```

- Coordinates are relative to the image (`0` to `1`) or in pixels. The convention is detected from all coordinates in the file.
- The optional 6th column is a detector confidence, which `--confidence_source` can carry into the output (see [Confidence scores](confidence.md)). [`hbb2obb-detect`](detect.md) writes it.
- Blank lines are skipped. An empty file is a frame with no objects and produces an empty output file. An image with no file at all stops the run.

## OBBs (output)

```text
class_id x1 y1 x2 y2 x3 y3 x4 y4 [confidence]
```

- The four corners of the rotated box, in integer pixels by default. `--normalize` writes them relative to `[0, 1]` at 10 decimals, enough that reading them back lands on the same pixel for any frame size; `--precision` sets fewer and warns when that is too coarse for the frame.
- The 10th column is the [confidence score](confidence.md), written with `--save_confidence`.
- Line *i* describes the object on line *i* of the HBB file, with its class.

!!! note "Training with Ultralytics"
    Ultralytics reads OBB labels only in the normalized form and rejects a 10th column, so for a training set pass `--normalize` and put the scores in a side-car with `--confidence_dir` rather than `--save_confidence`.

Every reader in HBB2OBB, `hbb2obb-eval` included, detects absolute or normalized coordinates on its own and ignores a trailing confidence column.

## Polygons (optional output)

With `--save_polygon`, the contour each OBB was fitted to goes to a parallel directory, `labels_polygon/` by default, in the same coordinate convention as the OBB file, with the same trailing score under `--save_confidence`:

```text
class_id x1 y1 x2 y2 ... xN yN [confidence]
```

The polygon file is row-aligned with the OBB file. A [fallback](how-it-works.md#fallback) is written as a four-point rectangle identical to its OBB line, never skipped. `--polygon_epsilon` / `-pe` simplifies the contours, the value being a fraction of the contour perimeter: `0.01` typically drops about 90% of the vertices. The default `0` writes the raw contour.

## Confidence side-car

With `--confidence_dir`, `labels_confidence/<image stem>.txt` holds one score per line, row-aligned with the label file:

```text
0.7820
0.7548
```

## Label map

A YAML file mapping class IDs to names:

```yaml
0: Car
1: Bus
2: Truck
```

`-mp` passes it to any command that names classes. Without it, `hbb2obb-detect`, `hbb2obb-convert` and `hbb2obb-view` look for a `classes.yaml`, or a `names.txt` with one name per line, beside the labels or one level up. The conversion itself never needs one: an OBB keeps its HBB's class ID.

## Directory layout

Every command defaults to directories beside the images, so a project laid out like this needs no path flags:

```text
project/
├── images/              # the frames
├── labels_hbb/          # HBBs: written by hbb2obb-detect, read by hbb2obb
├── labels_obb/          # OBBs: written by hbb2obb, read by hbb2obb-view
├── labels_polygon/      # contours: hbb2obb --save_polygon
├── labels_confidence/   # scores: hbb2obb --confidence_dir
├── labels_obb_gt/       # ground truth, if you have it
├── benchmark_results/   # hbb2obb-optimize output
└── classes.yaml         # label map
```

The [sample data](https://github.com/rfonod/hbb2obb/tree/main/data) follows it.
