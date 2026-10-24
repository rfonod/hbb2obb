# Troubleshooting

## Inputs

**`Error: no HBB directory at .../labels_hbb`**
:   `hbb2obb` looks for the HBBs in `labels_hbb` beside the image directory. Pass `--hbb_dir`, or produce them with [`hbb2obb-detect`](detect.md).

**`Annotation file not found: .../0001.txt`**
:   Every image needs an HBB file of the same name. For a frame with no objects, create an empty one.

**`... already holds annotations; pass --overwrite to replace them or --hbb_dir to write elsewhere`**
:   `hbb2obb-detect` never overwrites labels unless told to.

**`DOTA files name their classes, so a label map is required to read them`**
:   Pass `-mp label_map.yaml` to `hbb2obb-convert`, or put a `classes.yaml` or `names.txt` beside the files. See [Label map](data-format.md#label-map).

**`--precision applies to normalized output; add --normalize or drop it`**
:   Absolute output is written in whole pixels, so `--precision` only applies with `--normalize`.

## Models

**Weights download again in every directory**
:   Without a setting, checkpoints go to `models/` in the working directory. Set `HBB2OBB_MODELS_DIR` once; see [Model weights](install.md#model-weights).

**`sam3` cannot be found**
:   SAM 3 is gated and is not downloaded automatically; see [SAM 3](install.md#sam-3).

**CUDA out of memory**
:   Memory grows with `--imgsz` and with the number of objects in a frame, and every model of an ensemble stays loaded for the whole run. Lower `--imgsz`, use fewer or smaller models, or pass `--device cpu`.

**Slow on a Mac**
:   Ultralytics uses the CPU unless told otherwise. Pass `--device mps` on Apple silicon.

## Many boxes fall back

A fallback is a box for which no usable mask was found; it is written as its HBB with confidence `0.0`. See [Fallback](how-it-works.md#fallback). To find the cause, convert a few frames with `--save_img` and look at the overlays.

- **`Warning: model sam_b returned 40 mask(s) for 56 prompt(s) in 0001.jpg`**: Ultralytics dropped masks below its quality threshold. `--model_kwargs conf=0.1` lowers it from the default 0.25.
- **Prompts too tight**: a mask must cover at least 10% of its scaled box. A negative `--scale_factors` shrinks the prompt and can cut the object off; try the default `0.05`.
- **Objects too small for the resolution**: raise `--imgsz`.
- **The model**: another SAM model, or an ensemble, may segment your imagery better. [Tune hyperparameters](optimize.md) compares them on ground truth.

`--fallback_warn_share` warns per image and `--fail_on_fallback_share` fails the run above a share of fallbacks; see [Fallbacks](convert.md#fallbacks).

## Evaluation

**`Warning: ground truth appear to be normalized to [0, 1] while the predictions are in absolute pixels`**
:   Both sides must use the same convention, or every pair misses. Convert one of them with [`hbb2obb-convert`](formats.md): `--normalize` with `-i images` to make it relative, or `-i images` alone to make it absolute.

**`Warning: No prediction file found for 0001.txt`**
:   Ground truth and predictions are paired by file name, and a frame without a prediction file is left out of every number.

**Many unmatched boxes with the right geometry**
:   The classes may differ between the two sets. Pass `--class_agnostic`, or renumber one set with a label map.

**The numbers differ slightly from another machine**
:   SAM inference is not bit-reproducible across PyTorch builds and hardware, so a box or two can move by a pixel. Compare runs against the [standard error](metrics.md#iou), not the last digit.

## Training

**Ultralytics rejects the converted labels**
:   Ultralytics reads OBB labels only relative to `[0, 1]`, with nine fields per line. Convert with `--normalize`, and write scores with `--confidence_dir` rather than `--save_confidence`. See [Annotation formats](data-format.md#obbs-output).

## Viewer

**`No annotations found next to images; pass --obb_dir or --hbb_dir`**
:   The viewer looks in `labels_obb`, `labels_obb_gt`, `labels` and `labels_hbb` beside the image directory. Name the directory explicitly.

**No display, e.g. over SSH**
:   `hbb2obb-view images -o annotated` writes the frames as image files instead of opening a window.

## Still stuck

Search the [issues](https://github.com/rfonod/hbb2obb/issues), or open one with the command you ran, its full output, `hbb2obb --version` and your platform.
