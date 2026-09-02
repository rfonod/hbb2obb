# HBB2OBB: Horizontal to Oriented Bounding Box Conversion and Evaluation Tool

[![GitHub Release](https://img.shields.io/github/v/release/rfonod/hbb2obb?include_prereleases)](https://github.com/rfonod/hbb2obb/releases) [![PyPI Version](https://img.shields.io/pypi/v/hbb2obb)](https://pypi.org/project/hbb2obb/) [![PyPi - Total Downloads](https://img.shields.io/pepy/dt/hbb2obb?label=total%20downloads)](https://pepy.tech/project/hbb2obb) [![CI](https://img.shields.io/github/actions/workflow/status/rfonod/hbb2obb/ci.yml?branch=main&label=CI)](https://github.com/rfonod/hbb2obb/actions/workflows/ci.yml) [![Python](https://img.shields.io/badge/python-3.9--3.13-blue)](https://www.python.org/) [![License](https://img.shields.io/github/license/rfonod/hbb2obb)](https://github.com/rfonod/hbb2obb/blob/main/LICENSE) [![GitHub Issues](https://img.shields.io/github/issues/rfonod/hbb2obb)](https://github.com/rfonod/hbb2obb/issues) [![DOI](https://zenodo.org/badge/960660341.svg)](https://doi.org/10.5281/zenodo.15151143) [![Development Status](https://img.shields.io/badge/development-active-brightgreen)](https://github.com/rfonod/hbb2obb)

**HBB2OBB** converts horizontal (axis-aligned) bounding boxes (HBBs) into oriented (rotated) bounding boxes (OBBs) by using your existing HBB annotations as prompts for segmentation models from the [SAM (Segment Anything Model) family](https://docs.ultralytics.com/models/sam/). It targets object detection tasks where objects appear at arbitrary orientations, such as aerial imagery, satellite data, or traffic monitoring, producing OBBs that tightly encapsulate non-upright objects. Beyond conversion, it ships evaluation, hyperparameter optimization, and annotation format-conversion tools, with both a command-line interface and a Python API.

![HBB to OBB Conversion Example](https://raw.githubusercontent.com/rfonod/hbb2obb/main/assets/hbb2obb_illustration.webp)

## Why HBB2OBB

- 🎯 **Accurate OBBs from HBBs**: prompts SAM-family segmentation models with your existing horizontal boxes to fit tight oriented boxes around non-upright objects, with no re-annotation required.
- 🚗 **No HBBs? Detect them**: `hbb2obb-detect` runs an Ultralytics detector over your images and writes the horizontal boxes the conversion consumes, confidence column included ([details](#detecting-hbbs)).
- 🧩 **Model ensemble**: combines masks from multiple SAM variants through majority voting for more robust, accurate results (see [Usage](#usage)).
- 🛡️ **Spatially constrained & safe**: region-specific masking and contour refinement keep segmentation inside the object, and a fallback keeps the original HBB when no valid mask is found.
- 🔎 **Confidence-scored output**: every OBB gets a quality score in `[0, 1]` that flags silent fallbacks and low-confidence conversions, so you know which boxes to trust; your detector's own confidence can be carried through instead of, or on top of, that score (see [Confidence scores](#confidence-scores)).
- 📐 **Flexible scaling**: positive or negative scale factors (optionally different for the short and long sides) recover cropped object parts or tighten overly conservative annotations.
- 📊 **Evaluate & optimize**: built-in IoU evaluation against ground truth plus `hbb2obb-optimize`, a hyperparameter search over SAM inference resolution × scale factors × opening kernel, driven by a config file so a whole benchmark is one reproducible command ([details](#tuning-hyperparameters)).
- 🔄 **Six annotation formats**: read and write YOLO, DOTA, Pascal VOC, COCO and LabelMe, for horizontal and oriented boxes alike, with a check that proves every format encodes the same boxes ([details](#converting-between-formats)).
- 🔍 **Interactive viewer**: `hbb2obb-view` pans and zooms over your annotations, in any format, coloring boxes by confidence and overlaying predictions against ground truth ([details](#inspecting-annotations)).
- ⚙️ **CLI + Python API**: `hbb2obb`, `hbb2obb-detect`, `hbb2obb-eval`, `hbb2obb-convert`, `hbb2obb-view` and `hbb2obb-optimize` commands plus an importable API, with transparent visualizations of every step.

<details>
<summary><b>📋 Full Feature Overview</b></summary>

- **HBB to OBB conversion**: converts YOLO-format horizontal bounding boxes to oriented bounding boxes.
- **HBB detection**: produce the horizontal boxes in the first place with any Ultralytics detector, local, from the Ultralytics catalogue, or from Hugging Face ([details](#detecting-hbbs)).
- **Segmentation-based**: uses state-of-the-art SAM models for accurate object boundary detection.
- **Multiple model support**: SAM, SAM2, SAM2.1, SAM3, Mobile SAM, and FastSAM families ([details](https://docs.ultralytics.com/models/sam/)).
- **Model ensemble**: combine multiple models via majority voting for enhanced accuracy.
- **Confidence scoring**: a per-OBB quality score flags fallbacks and low-confidence conversions for triage, optionally combined with the detector confidence from the input ([details](#confidence-scores)).
- **Polygon output**: optionally save the segmentation contour behind each OBB, row-aligned with the OBB file, as a tighter object outline for downstream masking ([details](#data-format)).
- **Evaluation tools**: assess OBB accuracy against ground truth using IoU metrics.
- **Hyperparameter optimization**: search SAM inference resolutions, HBB scale factors and opening kernels for the best settings on your data, one sweep at a time or a whole benchmark from a config file.
- **Provenance records**: `--save_provenance` writes the command, the versions, a digest of the source that ran and the SHA-256 of every checkpoint, so a released annotation set can be regenerated rather than trusted.
- **Visualization tools**: render HBBs, segmentation masks, derived contours, and resulting OBBs.
- **Interactive viewer**: pan and zoom over annotated frames, toggle each layer, and compare two annotation sets side by side.
- **Format conversion utilities**: convert between YOLO, DOTA, Pascal VOC, COCO and LabelMe annotations, in either direction, for both box kinds.

</details>

<details>
<summary><b>🚀 Planned Enhancements</b></summary>

- **Improved morphological operations**: more advanced operations for better mask refinement.
- **Support for other segmentation models**: extend compatibility beyond the SAM/FastSAM families.

</details>

<details>
<summary><b>🔗 Related Projects</b></summary>

HBB2OBB integrates with and complements several specialized tools:

- **[Geo-trax](https://github.com/rfonod/geo-trax) 🚀**: georeferenced vehicle trajectory extraction pipeline for high-altitude drone imagery, built on YOLO detection and multi-object tracking. Its vehicle detector supplies the HBB inputs for vehicle use cases (car, bus, truck, motorcycle), and `hbb2obb-detect` runs it by default ([details](#detecting-hbbs)).

- **[Stabilo](https://github.com/rfonod/stabilo) ⚖️**: Python library for video and trajectory stabilization using robust homography transformations. Supports various feature detectors, RANSAC algorithms, and user-defined masks.

- **[Stabilo-Optimize](https://github.com/rfonod/stabilo-optimize) 🎯**: benchmarking and hyperparameter optimization framework for Stabilo. Evaluates stabilization performance through ground truth-free assessment using random perturbations.

</details>

## Install

Create and activate a **Python virtual environment** (Python 3.9–3.13), then install from [PyPI](https://pypi.org/project/hbb2obb/):

```bash
python3.11 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install hbb2obb
```

Also works with [uv](https://docs.astral.sh/uv/) (`uv pip install hbb2obb`) and conda.

> [!NOTE]
> SAM model weights are downloaded automatically by [Ultralytics](https://docs.ultralytics.com/models/sam/) on first use into a `models/` directory relative to your **current working directory**: run commands from a consistent location so weights are reused. Detector weights for `hbb2obb-detect` land there too. The one exception is SAM 3, which must be downloaded manually (see below).

Every command checks PyPI once a day, in the background, for a newer HBB2OBB release and prints a one-line notice if one exists. The check never blocks, never fails a run, and is silent when offline. Set `HBB2OBB_DISABLE_UPDATE_CHECK=1` to turn it off.

<details>
<summary><b>Alternatives: conda or uv</b></summary>

**[Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install):**

```bash
conda create -n hbb2obb python=3.11 -y
conda activate hbb2obb
```

**[uv](https://docs.astral.sh/uv/getting-started/installation/) (fastest; then `uv pip install hbb2obb`):**

```bash
uv venv --python 3.11
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

</details>

<details>
<summary><b>Install from source (development / editable)</b></summary>

```bash
git clone https://github.com/rfonod/hbb2obb.git
cd hbb2obb && pip install -e '.[dev]'
```

The `-e` flag makes your local changes take effect without reinstalling; the `[dev]` extra adds `pytest` and `ruff`. For a plain install, use `pip install .` instead.

</details>

<details>
<summary><b>SAM 3 model weights (manual download required)</b></summary>

Unlike other SAM models, SAM 3 weights (`sam3.pt`) are **not** auto-downloaded by Ultralytics. To use SAM 3:

1. Request access on the [SAM 3 model page on Hugging Face](https://huggingface.co/facebook/sam3).
2. Once approved, download [`sam3.pt`](https://huggingface.co/facebook/sam3/resolve/main/sam3.pt?download=true).
3. Place `sam3.pt` in the `models/` directory (relative to where you run the conversion).

See the [Ultralytics SAM 3 documentation](https://docs.ultralytics.com/models/sam-3/) for more.

</details>

## Quick Start

A small sample dataset ships in [`data/`](data/). From the repository root:

```bash
# The sample already ships with detected HBBs, so start here. To redo that step yourself:
#   hbb2obb-detect data/images --overwrite

# Convert the sample HBBs to OBBs (default single model) and save visualizations
hbb2obb data/images --save_img

# Higher accuracy with a model ensemble
hbb2obb data/images --sam_models sam_b sam_l sam2_b sam2.1_b

# Evaluate the converted OBBs against ground truth
hbb2obb-eval data/labels_obb_gt data/labels_obb -mp data/classes.yaml

# Look at the result: pan, zoom, step through frames, q to quit
hbb2obb-view data/images --compare data/labels_obb_gt
```

Converted OBB annotations are written to `data/labels_obb/`. Every command takes `--help`.

## Usage

### Converting HBB to OBB

```bash
# Default single model (sam_b), HBBs read from <img_source>/../labels_hbb
hbb2obb /path/to/images --hbb_dir /path/to/hbb/annotations

# Model ensemble (majority voting across models)
hbb2obb /path/to/images --sam_models sam_b sam_l sam2_b sam2.1_b

# Scale HBBs: expand to recover cropped parts, or shrink conservative boxes
hbb2obb /path/to/images --scale_factors 0.1      # expand uniformly
hbb2obb /path/to/images --scale_factors -0.02    # shrink uniformly
hbb2obb /path/to/images --scale_factors 0.1 0.05 # short side / long side

# Save visualization images of the conversion
hbb2obb /path/to/images --save_img

# Also save the segmentation polygon of each object, a tighter outline than its OBB
hbb2obb /path/to/images --save_polygon --polygon_dir /path/to/save/polygons
```

### Evaluating OBB Predictions

```bash
hbb2obb-eval /path/to/ground_truth /path/to/predictions
```

<details>
<summary><b>More CLI arguments</b></summary>

Run `hbb2obb --help` / `hbb2obb-eval --help` for the full list. Key conversion arguments:

- `--hbb_dir` / `-hd`: directory of HBB annotations in YOLO TXT format (default: `<img_source>/../labels_hbb`).
- `--obb_dir` / `-od`: directory to save OBB annotations (default: `<img_source>/../labels_obb`).
- `--sam_models` / `-sm`: SAM model(s) to use (e.g. `sam_b`, `sam_l`, `sam2_b`, `sam2.1_b`, `sam3`, `mobile_sam`, `FastSAM-s`).
- `--imgsz`: SAM inference resolution (default: 1280).
- `--scale_factors` / `-sf`: factor(s) to scale HBBs (single value, or two values for short/long sides).
- `--opening_kernel_percentage` / `-okp`: morphological opening kernel size as a percentage of the mask's smaller dimension.
- `--save_confidence`: append a per-OBB [confidence score](#confidence-scores) as a 10th column in the output TXT files.
- `--confidence_dir` / `-cd`: write those scores to their own directory instead, one score per line, row-aligned with the labels. Use it when the label files have to stay strictly standard, since Ultralytics and other YOLO OBB readers reject a 10th column. Give it bare for `img_source/../labels_confidence`.
- `--save_img`, `--viz_dir`, `--show_confidence`, and `--hide_hbb` / `--hide_obb` / `--hide_masks` / `--hide_segments` / `--hide_class_labels`: visualization controls.
- `--device`: inference device for the SAM model(s), e.g. `cpu`, `0`, `cuda:0`, `mps` (default: Ultralytics picks).
- `--model_kwargs` / `-k`: extra Ultralytics inference kwargs as `key1=value1,key2=value2`.

Key evaluation arguments:

- `--excluded_classes` / `-e`: class IDs to exclude from evaluation.
- `--iou_threshold` / `-t`: IoU threshold for a match (default: 0.1).
- `--class_agnostic` / `-ca`: ignore class-label matching (useful for re-classified GT).
- `--exclude_edge_cases` / `-exc`, `--edge_tolerance` / `-et`, `--img_width` / `-iw`, `--img_height` / `-ih`: edge-case handling.
- `--map_path` / `-mp`: path to a label map YAML mapping class IDs to names.

</details>

<details>
<summary><b>Python API</b></summary>

**Converting HBB to OBB**: `hbb2obb()` processes a single image and returns the OBB annotations as a NumPy array:

```python
from pathlib import Path
from hbb2obb.converter import hbb2obb, save_obb_annotations, save_polygon_annotations

img_path = Path("/path/to/images/img1.jpg")

# Single SAM model
obb_annotations = hbb2obb(
    img_path=img_path,
    hbb_dir="/path/to/hbb/annotations",
    sam_models="sam_b",
    imgsz=1280,
    scale_factors=0.05,
    opening_kernel_percentage=0.15,
    save_img=True,
    viz_dir="/path/to/save/visualizations",
)

# Model ensemble with per-side scale factors
obb_annotations = hbb2obb(
    img_path=img_path,
    hbb_dir="/path/to/hbb/annotations",
    sam_models=["sam_b", "sam_l", "sam2_b", "sam2.1_b"],
    scale_factors=[0.1, 0.05],  # short side / long side
)

# Writes <obb_dir>/img1.txt, deriving the filename from img_path
save_obb_annotations(obb_annotations, "/path/to/save/obb/annotations", img_path)

# Also return per-OBB confidence scores, and write them as a 10th column
obb_annotations, confidences = hbb2obb(img_path=img_path, sam_models="sam_b", return_confidence=True)
save_obb_annotations(obb_annotations, "/path/to/save/obb/annotations", img_path, confidences=confidences)

# Also return the segmentation contours, and write them as polygon annotations
obb_annotations, contours = hbb2obb(img_path=img_path, sam_models="sam_b", return_contours=True)
save_polygon_annotations(contours, obb_annotations, "/path/to/save/polygons", img_path)
```

**Evaluating OBB predictions:**

```python
from pathlib import Path
from hbb2obb.evaluator import evaluate_obb, print_results

results = evaluate_obb(
    gt_dir=Path("/path/to/ground_truth_annotations"),
    pred_dir=Path("/path/to/predictions"),
    iou_threshold=0.1,
    class_agnostic=True,  # optional: match regardless of class label
    exclude_edge_cases=True,  # optional: drop boxes at the image edge
    img_width=3840,
    img_height=2160,
)

print_results(results, "/path/to/label_map.yaml")
```

</details>

<details>
<summary><b>End-to-end workflow, from someone else's annotation format</b></summary>

Starting from HBB annotations and OBB ground truth in whatever format you were given:

```bash
# 1. Bring both into YOLO TXT, whatever they arrived as (--from is detected if omitted)
hbb2obb-convert project/voc_hbb --to yolo -o project/labels_hbb -mp project/label_map.yaml
hbb2obb-convert project/gt.json --from coco --to yolo -o project/labels_obb_gt -mp project/label_map.yaml

# 2. Optimize hyperparameters to find the best settings (add -ok to also sweep the opening kernel)
hbb2obb-optimize project/images project/labels_obb_gt -sm sam_b sam_l sam2_b -n multi_sam

# 3. Inspect the best parameters, then convert with them
cat project/benchmark_results/multi_sam/summary.txt
hbb2obb project/images --hbb_dir project/labels_hbb --obb_dir project/labels_obb \
  --sam_models sam_b sam_l --imgsz 1280 --scale_factors 0.05 \
  --opening_kernel_percentage 0.15 --save_confidence --save_provenance

# 4. Evaluate against ground truth, then look at where it went wrong
hbb2obb-eval project/labels_obb_gt project/labels_obb -mp project/label_map.yaml
hbb2obb-view project/images --compare project/labels_obb_gt --show_confidence

# 5. Ship the result in every format your consumers want
hbb2obb-convert project/labels_obb --to dota coco voc -o project/release -mp project/label_map.yaml
```

</details>

### Detecting HBBs

No HBBs yet? `hbb2obb-detect` runs an Ultralytics detector over your images and writes exactly the YOLO TXT the conversion reads, with the detector confidence in the 6th column:

```bash
# geo-trax is the default detector, tuned for vehicles in high-altitude drone imagery (weights downloaded on first use)
hbb2obb-detect /path/to/images

# Any other Hugging Face model: all three parts, '<user>/<repo>/<file>.pt' (no "huggingface.co/" prefix)
hbb2obb-detect /path/to/images --model rfonod/geo-trax/geotrax_hbb_yolov8s_1920_v1.pt

# Then convert, carrying the detector confidence into the OBBs alongside the conversion score
hbb2obb /path/to/images --save_confidence --confidence_source combined
```

<details>
<summary><b>More about <code>--model</code>, class maps, and merging with hand-drawn boxes</b></summary>

`--model` takes a registered detector ([`geotrax`](https://github.com/rfonod/geo-trax) today, the default), any Ultralytics model name or `.pt` path, or a Hugging Face reference written as `<user>/<repo>/<file>.pt`. All three parts are required, even when the repo holds only one file: Hugging Face's download API always names the exact file rather than picking one for you. Weights land in `models/` beside the SAM checkpoints. A registered detector brings the settings it was validated at, so `geotrax` runs at `--imgsz 1920` over its four reliable classes; anything else starts from the Ultralytics defaults and is yours to set. A detector trained on COCO numbers its classes differently from yours, which is what `--class_map` is for: `--class_map '2=0,5=1,7=2,3=3'` turns COCO's car, bus, truck and motorcycle into `0,1,2,3` and drops every other class.

Detected boxes are a starting point, not ground truth. If you have hand-drawn boxes already and only want the confidence a detector would give them, `--merge_with` keeps your geometry untouched and only attaches the score of the detection covering each box:

```bash
hbb2obb-detect /path/to/images --merge_with /path/to/labels_hbb --extras_dir /tmp/extras --overwrite
```

Your boxes stay exactly as they are, in their own order; a box no detection covers keeps `1.0`, because a box somebody drew by hand is not less certain than one a model proposed. Detections that back no box of yours are counted and, with `--extras_dir`, written as their own set so you can look at them (`hbb2obb-view /path/to/images --hbb_dir /tmp/extras`) and decide whether they are objects you missed. The merge never adds them for you, and `--overwrite` is required before anything writes into a directory that already holds labels.

</details>

## Inspecting Annotations

`hbb2obb-view` opens your annotations over the images they belong to. A vehicle in a 4K aerial frame is forty pixels across, and whether its box is correctly oriented is simply not visible at fit-to-screen scale, so the window pans and zooms:

```bash
# Defaults: images in <dir>, boxes from ../labels_obb and ../labels_hbb
hbb2obb-view data/images

# Color the OBBs by confidence and print the score, to find the boxes worth checking
hbb2obb-view data/images --show_confidence

# Overlay ground truth in blue over the converted boxes in green
hbb2obb-view data/images --compare data/labels_obb_gt

# Read a different format, or write annotated images instead of opening a window
hbb2obb-view data/images --obb_format dota
hbb2obb-view data/images -o /path/to/annotated
```

![The hbb2obb annotation viewer](https://raw.githubusercontent.com/rfonod/hbb2obb/main/assets/hbb2obb_viewer.jpg)

<details>
<summary><b>Color legend and keyboard shortcuts</b></summary>

Green is the OBB, white its source HBB, red the segmentation polygon it was fitted to, orange a box flagged `difficult`; with `--show_confidence` the OBB is tinted green→red by score, the same gradient `--save_img` uses. The last two need the conversion to have been run with `--save_polygon` and `--save_confidence`, which is how the sample data in `data/` was produced (see [`data/README.md`](data/README.md) for the exact commands behind every file there). Labels that carry no confidence column still color by score if the scores are in a side-car directory: the viewer reads `labels_confidence/` beside them, or wherever `--confidence_dir` points.

| Key | | Key | |
| :--- | :--- | :--- | :--- |
| `q` / `Esc` | quit | `h` | show or hide the HBBs |
| `n` / `p`, arrows | next / previous frame | `l` | show or hide the class labels |
| wheel, `+` / `-` | zoom, about the cursor | `d` | show or hide boxes flagged `difficult` |
| `f` / `0` | fit the frame | `c` | color by confidence, and print it |
| `1` | zoom to 100% | `g` | show or hide the segmentation polygons |
| `s` | save the current view | `x` | cycle the comparison overlay |

Drag with the left mouse button to pan. `--crops` writes a contact sheet of the individual objects instead, which is the faster way to review a whole frame box by box.

</details>

## Converting Between Formats

`hbb2obb-convert` moves annotations between the six formats below, in either direction, for horizontal and oriented boxes alike:

| Format | HBB | OBB | Shape |
| :--- | :---: | :---: | :--- |
| `yolo` | ✅ | ✅ | one `.txt` per frame, with an optional trailing confidence column |
| `dota` | | ✅ | one file per frame, `x1 y1 … x4 y4 name difficult`, integer px |
| `voc` | ✅ | | one Pascal VOC `.xml` per frame, integer px |
| `coco` | ✅ | ✅ | one `.json` for the whole set; the quad goes in `segmentation`, a confidence in `score` |
| `labelme` | ✅ | ✅ | one [LabelMe](https://github.com/wkentaro/labelme) `.json` per frame |

```bash
# Write several formats in one pass; --from is detected from the files if omitted
hbb2obb-convert /path/to/labels_obb --to dota coco voc -o /path/to/release -mp label_map.yaml

# The reverse, into the YOLO TXT the tool consumes
hbb2obb-convert /path/to/instances.json --from coco --to yolo -o /path/to/labels_hbb --normalize

# Check that every format present under a directory encodes the same boxes
hbb2obb-convert /path/to/dataset --verify -mp label_map.yaml
```

<details>
<summary><b>Format details and edge cases</b></summary>

`--verify` compares the formats by exact equality after rounding rather than by a tolerance, because they are all one common rounding of a single canonical source. That is the property worth testing: rounding full precision and rounding an already-rounded canonical disagree wherever a coordinate lands on a half pixel, and an envelope then silently stops matching the box it was derived from.

Only DOTA and Pascal VOC can express a per-box `difficult` flag, so writing either one from YOLO or COCO resets it; `--difficult_from dota` carries the flags across. `--difficult_from confidence` instead derives the flag from the conversion score, flagging everything below `--difficult_below` (fallback boxes score 0.0, so they are always flagged); the scores come from a trailing column on the source labels, or from `--confidence_dir` when the labels are standard ones with the scores in a side-car. The scores themselves stay out of the output, so the flag can be set without the coordinates or the other formats beside them carrying anything extra. Only YOLO and COCO can carry a confidence, so DOTA and Pascal VOC drop it. Image dimensions come from `--images`, or from `--img_width` / `--img_height`, and are needed to denormalize relative YOLO coordinates. LabelMe stores class names rather than ids, so pass `-mp` when round-tripping through it to pin the ids.

A COCO file is named `coco_annotations_<kind>.json` unless `--coco_name` says otherwise, which is also how `--verify` pairs one with its directory: `labels_<name>/` goes with `coco_annotations_<name>.json` beside it. Where a directory holds the canonical YOLO files next to derived ones, `--from` is detected as `yolo`, so a second pass through the converter never rounds an already rounded file.

</details>

## Tuning Hyperparameters

`hbb2obb-optimize` grid-searches inference resolution x scale factor x opening kernel for a set of SAM models, scoring each point by average IoU against ground-truth OBBs:

```bash
hbb2obb-optimize /path/to/images /path/to/ground_truth -sm sam_b sam_l sam2_b sam2.1_b -n multi_sam
```

<details>
<summary><b>Grid size, outputs, and sweeping the opening kernel</b></summary>

The grid is the full product of `--imgsz` x `--scale_factors` x `--opening_kernels`, and each grid point is a complete SAM pass over the whole image set, so the cost multiplies quickly: the defaults (3 image sizes, 12 scale factors, 1 opening kernel) already amount to 36 passes, and sweeping three kernels instead of one triples that to 108. `--opening_kernels` / `-ok` defaults to the single value `0.15`, so omitting it leaves the two-axis sweep and its grid size unchanged.

```bash
# Add the opening kernel as a third axis (2 x 3 x 3 = 18 grid points)
hbb2obb-optimize /path/to/images /path/to/ground_truth -iz 960 1280 -sf 0.03 0.05 0.07 -ok 0.0 0.15 0.3
```

A run writes `run_config.yaml`, `results.yaml`, `summary.txt` and `plot.png` into `<output_folder>/<name>`, and `summary.md`, `comparison.png` and `PROVENANCE.txt` into the output folder itself. The plot gives each series a hue by image size and, when more than one opening kernel was swept, a lightness of that hue and a marker shape by kernel, so no two of the swept combinations share a colour. Marker area is the execution time.

`--device` (e.g. `cpu`, `0`, `cuda:0`, `mps`) applies to every run and overrides any `device` set in the config, so the same benchmark runs on whatever machine is free without editing the file.

</details>

### Comparing Several Sweeps

Comparing model ensembles means running the same grid many times, which is a job for a file rather than for shell history. A YAML lists the runs, and one command produces all of them:

```bash
hbb2obb-optimize -c benchmark.yaml             # all runs in benchmark.yaml
hbb2obb-optimize -c benchmark.yaml --resume    # continue one that was interrupted
```

<details>
<summary><b>Config file, dry runs, and refreshing plots</b></summary>

```yaml
# benchmark.yaml
img_source: data/images
gt_dir: data/labels_obb_gt
output_folder: data/benchmark_results

defaults:
  imgsz: [640, 960, 1280]
  scale_factors: [0.03, 0.04, 0.05, 0.06, 0.07]

runs:
  - sam_models: [sam_b]
  - sam_models: [sam_l]
  - sam_models: [sam_l, sam_b, sam2_b, sam2.1_b]
```

```bash
hbb2obb-optimize -c benchmark.yaml --dry_run   # the runs, the grid size, the total cost
hbb2obb-optimize -c benchmark.yaml --refresh   # only redraw the plots and the summary
```

Each run takes the `defaults` and overrides whatever it names; a run with no `name` takes one from its models, so `[sam_l, sam_b]` writes into `sam_l-sam_b`. Alongside `summary.md` and `PROVENANCE.txt`, which every sweep writes, a config-driven benchmark also leaves a **copy of the configuration itself** in the output folder, so the results re-run on their own once they have been moved into an archive or a dataset release, away from the repository that held the original. `--resume` skips runs that already hold a complete grid, which is what makes a multi-hour unattended sweep restartable; when it does skip some, `PROVENANCE.txt` says which runs this invocation measured and which it kept, so a partially resumed benchmark is legible as one. `--refresh` redraws the plots and the summary from the results on disk and rewrites no provenance at all, since that file describes the code and the checkpoints that produced the numbers, not the code that last redrew them.

[`data/benchmark.yaml`](data/benchmark.yaml) is a working example: it is the file behind [`data/benchmark_results/`](data/benchmark_results/), so that whole folder is one command away.

</details>

### Recording Provenance

`--save_provenance` writes a `PROVENANCE.txt` beside the annotations, so a release can be regenerated rather than taken on trust:

```bash
hbb2obb /path/to/images --sam_models sam_l sam_b sam2_b sam2.1_b --save_confidence --save_provenance
```

<details>
<summary><b>What gets recorded, and why a commit hash is not enough</b></summary>

```bash
# Works the same for detection
hbb2obb-detect /path/to/images --save_provenance
```

It records the exact command, the versions of `ultralytics`, `torch`, OpenCV, NumPy, Shapely and matplotlib, and the **SHA-256 of every checkpoint used**. That last part is the one worth insisting on: `ultralytics` resolves a bare model name by downloading it, and the file behind a name can change between asset releases, so the name alone does not pin the model that actually ran. The settings come from the run that happened rather than from arguments repeated afterwards, and a benchmark additionally hashes the label sets its numbers were measured against.

The code is pinned three ways, because a commit hash alone is not enough: the release (`pip install hbb2obb==<version>`), the commit with `git describe` when there is a checkout, and a **SHA-256 over the package source**. The record says outright whether that commit can be checked out to get the code that ran, and it decides that by comparing the package directory alone, so a sweep writing its results back into the repository does not mark its own record as untrustworthy. The digest is the durable one. A commit hash identifies a point in a history, and history is editable: squash a branch, rebase it, or delete it after the merge, and the hash stops resolving, while an uncommitted tree never had one to record. The source digest is computed from the bytes that ran, so it still matches years later. Use the commit to find the change; use the digest to prove you have the same code.

</details>

## Data Format

**HBB annotations (input)**: YOLO TXT, one file per image; coordinates relative (0–1) or absolute px:

```text
class_id x_center y_center width height
```

An optional 6th column holding the detector confidence is accepted and can be carried into the output (see [Confidence scores](#confidence-scores)), which is the shape most detectors write:

```text
class_id x_center y_center width height confidence
```

Blank lines are skipped, and an empty label file (a frame with no objects) is valid input: it simply produces an empty output file, so a whole directory converts without interruption.

You bring your own HBB annotations. If you don't have any, `hbb2obb-detect` produces them with an Ultralytics detector and writes this exact format, confidence column included (see [Detecting HBBs](#detecting-hbbs)).

**OBB annotations (output)**: YOLO TXT, one file per image; four corners in absolute px:

```text
class_id x1 y1 x2 y2 x3 y3 x4 y4
```

With `--save_confidence`, a 10th column holds the per-OBB [confidence score](#confidence-scores):

```text
class_id x1 y1 x2 y2 x3 y3 x4 y4 confidence
```

`hbb2obb-eval` ignores the trailing confidence column, so evaluation works on either variant.

**Polygon annotations (optional output)**: with `--save_polygon`, the segmentation contour each OBB was fitted to is written to a parallel directory (`labels_polygon` by default), one file per image; a variable number of corners in absolute px:

```text
class_id x1 y1 x2 y2 ... xN yN
```

With `--save_confidence`, a trailing column holds the same per-object score written to the OBB file:

```text
class_id x1 y1 x2 y2 ... xN yN confidence
```

The polygon is a tighter outline of the object than its OBB, which makes it useful as a mask for downstream work. It is row-aligned with the OBB file: line *i* of both files describes the same object, so the two can be joined by line number. Objects that fell back to the HBB are written as a four-point rectangle identical to their OBB line rather than skipped, which is what keeps that alignment exact. `--polygon_epsilon` simplifies the polygons, the value being a fraction of the contour perimeter (`0.01` typically drops about 90% of the vertices); the default `0` writes the raw contour.

**Label map (optional)**: YAML mapping class IDs to names:

```yaml
0: Car
1: Bus
2: Truck
```

## How It Works

HBB2OBB fits each OBB by prompting SAM with your HBBs, refining the resulting mask, and wrapping it in a minimum-area rotated box.

<details>
<summary><b>Conversion pipeline</b></summary>

1. **Load HBB annotations** from YOLO TXT.
2. **Scale bounding boxes**: positive factors expand HBBs (recover cropped parts), negative factors shrink them (tighten conservative boxes); short and long sides can be scaled differently.
3. **Segmentation**: run SAM model(s) with the HBBs as prompts.
4. **Mask aggregation**: with an ensemble, combine masks by majority voting; clip to the scaled HBB region; apply morphological opening.
5. **Contour extraction**: extract the largest refined mask contour per object (optionally saved with `--save_polygon`).
6. **OBB computation**: fit a minimum-area oriented bounding box.
7. **Fallback**: if no valid mask is found inside an HBB, keep the original HBB as the OBB (confidence `0.0`).
8. **Confidence**: score each OBB in `[0, 1]` (see [Confidence Scores](#confidence-scores)).
9. **Visualization (optional)**: overlay HBBs, masks, contours, and OBBs (colored by confidence).

Key characteristics:

- **Label preservation**: OBBs inherit the class label of their source HBB (no re-classification).
- **Corrective effects**: the transformation can recover cropped parts (positive scaling) and produce tighter boxes through precise segmentation.

</details>

## Confidence Scores

Each OBB carries a heuristic quality score in `[0, 1]` that helps triage a converted dataset: high scores are trustworthy SAM fits, low scores warrant a look, and `0.0` marks a fallback where the original HBB was kept. It is a heuristic, not a calibrated probability. The detail crop below shows the score that `--show_confidence` prints next to each converted box.

![Confidence-scored OBBs](https://raw.githubusercontent.com/rfonod/hbb2obb/main/assets/hbb2obb_confidence_scores.jpg)

The score is the product of two factors:

- **Rectangularity**: the fitted contour area divided by the area of its minimum-area rotated rectangle, i.e. how tightly the OBB wraps the segmented shape (`1.0` for a perfectly rectangular object).
- **Ensemble consensus**: the fraction of the per-model mask union that survived the majority vote, i.e. how strongly the SAM models agree. This is `1.0` for a single model and equals the mask IoU for two models.

Enable it with `--save_confidence` (writes a 10th column to each output file). In the visualization, OBBs are always tinted on a green→red gradient by score, and `--show_confidence` prints the numeric value next to each box. When using the Python API, pass `return_confidence=True` to `hbb2obb()` to get the scores back alongside the OBBs.

**Choosing which score is reported.** If your HBB files carry a detector confidence in a 6th column, `--confidence_source` (or the `confidence_source` argument of `hbb2obb()`) selects what the reported score means:

| Value | Reported score |
| :--- | :--- |
| `conversion` (default) | the heuristic conversion quality described above |
| `detector` | the detector confidence read from the HBB input |
| `combined` | the product of the two |

The two measure different things: `conversion` says how well the OBB fits the segmented shape, `detector` says how sure the detector was that there is an object there at all. Boxes whose input line carried no confidence column fall back to the conversion score, so no output is ever left without one.

`hbb2obb-detect` writes that 6th column, either for boxes it found itself or, with `--merge_with`, for boxes you drew by hand (see [Detecting HBBs](#detecting-hbbs)). The sample HBBs in `data/` are detector output and carry it, so all three sources can be tried there directly; the OBBs shipped in `data/labels_obb/` are scored `combined`.

## Best Practices

- For optimal results, combine multiple SAM models of comparable strength, e.g. `--sam_models sam_b sam_l sam2_b sam2.1_b`. Adding a weaker model is not free: with majority voting, two weak members can outvote a strong one, so measure rather than assume.
- Experiment with scale factors and inference resolutions based on your dataset.
- Run `hbb2obb-optimize` to find the best settings for your data, and `--save_provenance` to record the ones you settled on.
- Use class-agnostic evaluation when comparing against manually annotated ground truth with different class labels.
- Visualize the conversion to understand how the model interprets your HBBs.

## Limitations

- Results depend on the quality of the input HBBs and the SAM models used; poor annotations or weak segmentation lead to inaccurate OBBs.
- Highly occluded or complex objects, where the HBB gives insufficient context, may not convert well.

## Citation

If you use **HBB2OBB** in your research or software, please cite the archived release:

```bibtex
@software{fonod2026hbb2obb,
  author  = {Fonod, Robert},
  title   = {HBB2OBB: Horizontal to Oriented Bounding Box Conversion and Evaluation Tool},
  year    = {2026},
  license = {MIT},
  doi     = {10.5281/zenodo.15151143},
  url     = {https://github.com/rfonod/hbb2obb}
}
```

Each GitHub release is automatically archived to [Zenodo](https://doi.org/10.5281/zenodo.15151143) via the Zenodo–GitHub integration; see [`CITATION.cff`](CITATION.cff) for the latest version and DOI.

## Contributing

Contributions are welcome! If you encounter issues or have suggestions, please open a [GitHub Issue](https://github.com/rfonod/hbb2obb/issues) or submit a pull request.

## License

This project is distributed under the MIT License. See the [LICENSE](LICENSE) file for details.
