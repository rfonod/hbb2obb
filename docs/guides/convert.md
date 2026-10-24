# Convert HBBs to OBBs

`hbb2obb` reads a YOLO HBB file per image, prompts SAM with the boxes and writes a YOLO OBB file per image:

```bash
hbb2obb /path/to/images
```

`img_source` is one image or a directory of them (`.jpg`, `.jpeg`, `.png`, `.bmp`). Every image needs an HBB file named after it; a frame with no objects needs an empty one.

## Where files go

Without flags, `hbb2obb` reads and writes beside the image directory:

```text
project/
├── images/              # img_source: 0001.jpg, 0002.jpg, ...
├── labels_hbb/          # read: 0001.txt, 0002.txt, ...      --hbb_dir / -hd
├── labels_obb/          # written: 0001.txt, ...             --obb_dir / -od
├── labels_polygon/      # written with --save_polygon         --polygon_dir / -pd
└── labels_confidence/   # written with bare --confidence_dir
```

Overlays from `--save_img` go to `--obb_dir` unless `--viz_dir` says otherwise. The file layouts are on [Annotation formats](data-format.md).

## Choosing the models

`--sam_models` / `-sm` takes one model or several (default `sam_b`):

| Family | Names |
| :--- | :--- |
| SAM | `sam_b`, `sam_l` |
| Mobile SAM | `mobile_sam` |
| SAM 2 | `sam2_t`, `sam2_s`, `sam2_b`, `sam2_l` |
| SAM 2.1 | `sam2.1_t`, `sam2.1_s`, `sam2.1_b`, `sam2.1_l` |
| SAM 3 | `sam3` (download by hand, see [Installation](install.md#sam-3)) |
| FastSAM | `FastSAM-s`, `FastSAM-x` |

Several models are combined by a per-pixel majority vote:

```bash
hbb2obb /path/to/images --sam_models sam_l sam_b sam2.1_b
```

An ensemble costs the sum of its members' run times. A pixel survives when more than half the models claim it: two of two, two of three, three of four. A pair therefore keeps only the pixels both masks share; see [the vote](how-it-works.md#4-vote).

`--imgsz` sets the SAM inference resolution (default 1280). `--device` picks the hardware, e.g. `cpu`, `0`, `cuda:0` or `mps` (default: the first CUDA GPU if there is one, else the CPU). Any other Ultralytics predictor argument goes through `--model_kwargs` / `-k`, as `key=value` pairs separated by commas; values are read as Python literals where they are one.

## Fitting the box

Three settings shape the mask before the box is fitted to it. [How the conversion works](how-it-works.md) describes each step in full.

**Scale the prompts** with `--scale_factors` / `-sf` (default `0.05`). Each edge moves by that fraction of its own side: positive expands a box to recover parts the HBB cut off, negative shrinks a box drawn loosely. Two values scale the short and long sides separately:

```bash
hbb2obb /path/to/images --scale_factors 0.1      # expand uniformly
hbb2obb /path/to/images --scale_factors -0.02    # shrink uniformly
hbb2obb /path/to/images --scale_factors 0.1 0.05 # short side / long side
```

**Refine the mask** with `--opening_kernel_percentage` / `-okp` (default `0.15`), a kernel size as a fraction of the object's shorter side. Positive opens (removes thin protrusions), negative closes (fills holes and rejoins a mask split by glare or a pole), `0` turns it off.

**Rejoin fragments** with `--fragment_ratio` / `-fr` (default `0.1`). Any other mask piece holding at least this fraction of the largest piece's area is fitted together with it. `0` fits the largest piece alone.

The best values depend on the data. [Tune hyperparameters](optimize.md) searches them against ground truth, and [Best practices](practice.md) lists what held on a 4,245-box study.

## Output options

**Normalized coordinates.** OBBs are written in absolute pixels by default. `--normalize` / `-n` writes them relative to `[0, 1]`, which Ultralytics requires for training; `--precision` / `-p` sets the decimals (default 10).

**Confidence.** `--save_confidence` appends each OBB's score as a 10th column. `--confidence_dir` / `-cd` writes the scores to a side-car directory instead, leaving the label files standard. `--confidence_source` / `-cs` chooses between the conversion score, a detector confidence read from the HBB input, or their product. See [Confidence scores](confidence.md).

**Polygons.** `--save_polygon` also writes the segmentation contour behind each OBB, row-aligned with the OBB file, to `--polygon_dir`. `--polygon_epsilon` / `-pe` simplifies them. See [Polygons](data-format.md#polygons-optional-output).

**Provenance.** `--save_provenance` writes `PROVENANCE_obb.txt` one level above the labels, with the command that reproduces them. See [Provenance](provenance.md).

## Visualizations

`--save_img` writes an overlay per image, to `--viz_dir` or else beside the labels:

| Layer | Hide with |
| :--- | :--- |
| the source HBB | `--hide_hbb` |
| the aggregated mask | `--hide_masks` |
| the contour the box was fitted to | `--hide_segments` |
| the OBB, tinted green to red by confidence | `--hide_obb` |
| class names | `--hide_class_labels` |

`--show_confidence` also prints the score beside each OBB. To inspect the result interactively instead, use [`hbb2obb-view`](view.md).

## Fallbacks

A box with no usable mask is written as its scaled HBB, with confidence `0.0`; see [Fallback](how-it-works.md#fallback). Every run ends with the total:

```text
Wrote 201 boxes over 3 frames to data/labels_obb
Fallback to HBB: 0 boxes (0.00%)
```

Two flags act on the share of fallbacks:

- `--fallback_warn_share` / `-fw` warns on stderr for each image where more than this share of boxes fell back (default 0.5; `0` warns on any fallback, `1` never).
- `--fail_on_fallback_share` / `-ff` exits non-zero when more than this share of the whole run fell back. The annotations and the provenance are written either way, so a script can check the exit code before publishing.

## Python API

[`hbb2obb()`](../reference/api/converter.md#hbb2obb.converter.hbb2obb) converts one image and returns its OBBs as an `(N, 9)` NumPy array of `class, x1, y1, ..., x4, y4` in pixels. The writers produce the same files as the command:

```python
from pathlib import Path

from hbb2obb.converter import hbb2obb, save_obb_annotations

hbb_dir = Path("project/labels_hbb")
for img_path in sorted(Path("project/images").glob("*.jpg")):
    obbs = hbb2obb(
        img_path=img_path,
        hbb_dir=hbb_dir,
        sam_models=["sam_l", "sam_b", "sam2.1_b"],
        imgsz=1280,
        scale_factors=[0.05, 0.03],  # short side / long side
        opening_kernel_percentage=-0.1,
    )
    save_obb_annotations(obbs, Path("project/labels_obb"), img_path)
```

Flags add return values, in the order confidences, contours, stats:

```python
obbs, confidences, contours, stats = hbb2obb(
    img_path=img_path,
    hbb_dir=hbb_dir,
    return_confidence=True,  # list of floats, one per OBB
    return_contours=True,  # list of contours, None for a fallback
    return_stats=True,  # ConversionStats(boxes, fallbacks, masks_missing)
)
print(f"{stats.fallbacks} of {stats.boxes} boxes fell back")
```

[`save_confidence_annotations`](../reference/api/converter.md#hbb2obb.converter.save_confidence_annotations) and [`save_polygon_annotations`](../reference/api/converter.md#hbb2obb.converter.save_polygon_annotations) write the side-car and polygon files. Loaded models are cached for the life of the process; [`clear_model_cache()`](../reference/api/converter.md#hbb2obb.converter.clear_model_cache) releases them.

All flags: [`hbb2obb` reference](../reference/cli/hbb2obb.md).
