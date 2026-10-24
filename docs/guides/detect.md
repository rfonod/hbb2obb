# Detect HBBs

`hbb2obb` needs horizontal boxes to prompt SAM with. When your images have none, `hbb2obb-detect` runs an Ultralytics detector over them and writes exactly the YOLO TXT the conversion reads, with the detector confidence in the 6th column:

```bash
hbb2obb-detect data/images -mp data/classes.yaml --overwrite
```

```text
Wrote 201 boxes over 3 frames to data/labels_hbb
  0 Car: 186
  1 Bus: 6
  2 Truck: 8
  3 Motorcycle: 1
```

On the [sample data](quickstart.md) this regenerates the shipped `data/labels_hbb/`. The boxes land in `<img_source>/../labels_hbb`, where `hbb2obb` looks for them, or in `--hbb_dir` / `-hd`. A directory that already holds labels is never overwritten without `--overwrite`. The label map only names the classes in the summary.

Then convert, keeping the detector confidence alongside the conversion score:

```bash
hbb2obb data/images --save_confidence --confidence_source combined
```

Detected boxes are a starting point, not ground truth: review them with [`hbb2obb-view`](view.md) before relying on them.

## Choosing the model

`--model` / `-m` takes any of:

| Value | Example |
| :--- | :--- |
| a registered detector | `geotrax` (the default) |
| a local `.pt` file | `weights/best.pt` |
| a Hugging Face file, as a link | `https://huggingface.co/<user>/<repo>/resolve/<revision>/<path>.pt`, or the `/blob/` page link |
| a Hugging Face file, short form | `<user>/<repo>/<path>.pt`, with no `huggingface.co/` prefix |
| any other `http(s)` link to a checkpoint | `https://example.org/model.pt` |
| an Ultralytics model name | `yolo11n.pt` |

An existing local file always wins, and a path that exists nowhere is an error. Downloaded weights are kept in the [models directory](install.md#model-weights) beside the SAM checkpoints.

The registered detector, [`geotrax`](https://github.com/rfonod/geo-trax), is a YOLOv8s vehicle detector for high-altitude drone imagery. It runs with the settings it was validated at: `--imgsz 1920`, `--conf 0.25`, `--iou 0.45`, and its four reliable classes, 0 car, 1 bus, 2 truck and 3 motorcycle. Any other model, including the same weights given by link, starts from the Ultralytics defaults (`--imgsz 640`, `--conf 0.25`, `--iou 0.7`, every class).

## Detection settings

| Flag | Sets |
| :--- | :--- |
| `--imgsz` | inference resolution |
| `--conf` | minimum detection confidence |
| `--iou` | NMS IoU threshold |
| `--classes` | class IDs to keep, in the detector's own numbering |
| `--max_det` | maximum detections per image |
| `--device` | e.g. `cpu`, `0`, `mps` |

Any other Ultralytics predictor argument passes through `--model_kwargs` / `-k`, unchecked; values are read as Python literals where they are one:

```bash
hbb2obb-detect /path/to/images --model_kwargs 'agnostic_nms=True,augment=True'
```

A setting that has its own flag can be given that way or in `--model_kwargs`, not both, and a malformed string stops the run.

## Renumbering classes

`--class_map` / `-cm` maps the detector's class IDs onto yours and drops every class it does not name. For a COCO-trained model:

```bash
hbb2obb-detect /path/to/images --model yolo11x.pt --class_map '2=0,5=1,7=2,3=3'
```

keeps car, bus, truck and motorcycle as `0`, `1`, `2` and `3`.

## Output options

- `--normalize` / `-n` writes coordinates relative to `[0, 1]`; `--precision` / `-p` sets the decimals (default 2).
- `--no_confidence` leaves out the 6th column.
- `--save_provenance` writes `PROVENANCE_hbb.txt` one level above the labels: the command that reproduces them, the settings that ran, the SHA-256 of the checkpoint and, for weights on the Hugging Face Hub, the licence and references the Hub declares. See [Provenance](provenance.md).

## Scoring hand-drawn boxes

If you already have hand-drawn HBBs and want only the confidence a detector gives them, `--merge_with` / `-mw` keeps your boxes and attaches the score of the detection covering each one:

```bash
hbb2obb-detect /path/to/images --merge_with /path/to/labels_hbb --hbb_dir /path/to/labels_hbb_scored --extras_dir /path/to/extras
```

- Your boxes keep their geometry, class and order. Each takes the confidence of the detection that overlaps it most, at IoU of at least `--merge_iou` (default 0.5), whatever either side's class; a box no detection covers gets `1.0`.
- The summary counts the boxes scored by a detection, the boxes kept at `1.0`, and the pairs whose classes differ.
- Detections that back none of your boxes are counted and, with `--extras_dir`, written as an HBB set of their own to review:

```bash
hbb2obb-view /path/to/images --hbb_dir /path/to/extras
```

They are never added to your boxes.

## Python API

```python
from pathlib import Path

import cv2

from hbb2obb.detector import detect_hbb, save_hbb_annotations

img_path = Path("data/images/04433.jpg")
boxes = detect_hbb(cv2.imread(str(img_path)), model="geotrax")  # (N, 6): class, xc, yc, w, h, conf in px
save_hbb_annotations(boxes, Path("data/labels_hbb_new"), img_path, scores=boxes[:, 5])
```

[`merge_detections()`](../reference/api/detector.md#hbb2obb.detector.merge_detections) is the merge behind `--merge_with`.

All flags: [`hbb2obb-detect` reference](../reference/cli/hbb2obb-detect.md).
