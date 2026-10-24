# HBB2OBB

**HBB2OBB** converts horizontal (axis-aligned) bounding boxes (HBBs) into oriented (rotated) bounding boxes (OBBs). It prompts a segmentation model from the [SAM family](https://docs.ultralytics.com/models/sam/) with each HBB you already have, and fits a minimum-area rotated box to the mask that comes back. No box is redrawn by hand.

Around the conversion it covers the rest of the job: detecting HBBs when you have none, scoring the result against ground truth and breaking that score down, tuning the settings on your own data, viewing the boxes, and writing them in the format the next tool expects.

![HBB to OBB conversion example](https://raw.githubusercontent.com/rfonod/hbb2obb/main/assets/hbb2obb_illustration.webp)

## What it does

<div class="grid cards" markdown>

- **Convert**

    One SAM model or a majority-vote ensemble of several. The mask is clipped to the prompt and refined before the box is fitted, and a mask an occluder split in two is fitted as one object.

    [Convert HBBs to OBBs](guides/convert.md) · [How the conversion works](guides/how-it-works.md)

- **Detect**

    No HBBs yet? Run any Ultralytics detector over the images. The default is the Geo-trax vehicle detector for high-altitude drone imagery.

    [Detect HBBs](guides/detect.md)

- **Triage**

    Every OBB carries a confidence in `[0, 1]`. A box that fell back to its HBB scores `0.0` and is counted at the end of the run.

    [Confidence scores](guides/confidence.md)

- **Evaluate and analyze**

    Mean and median IoU, the IoU ladder and the orientation error, then the same evaluation cut by orientation, size, frame edge, difficulty and class.

    [Evaluate](guides/evaluate.md) · [Analyze](guides/analyze.md) · [Metrics](guides/metrics.md)

- **Tune**

    A grid search over SAM resolution, scale factor and morphological kernel, or a whole benchmark of model sets from one config file.

    [Tune hyperparameters](guides/optimize.md)

- **View and ship**

    YOLO, DOTA, Pascal VOC, COCO and LabelMe in either direction, a check that every format holds the same boxes, and an interactive viewer that reads them all.

    [Convert between formats](guides/formats.md) · [View annotations](guides/view.md)

</div>

`--save_provenance` records the command, the library versions, a digest of the source that ran and the SHA-256 of every checkpoint, so a released annotation set can be regenerated. See [Provenance](guides/provenance.md).

## Install

HBB2OBB needs Python 3.9 to 3.13:

```bash
pip install hbb2obb
```

SAM weights download on first use into `models/` in the working directory. [Installation](guides/install.md) covers conda, uv and source installs, where to keep the weights, and SAM 3, which has to be downloaded by hand.

## Try it

A small sample dataset ships in [`data/`](https://github.com/rfonod/hbb2obb/tree/main/data) of the repository. From a clone:

```bash
hbb2obb data/images --save_img                                        # convert the sample HBBs, save overlays
hbb2obb-eval data/labels_obb_gt data/labels_obb -mp data/classes.yaml     # score against ground truth
hbb2obb-view data/images --compare data/labels_obb_gt                 # pan, zoom, q to quit
```

The [Quick start](guides/quickstart.md) walks through these and the rest of the toolkit on the same data, with the output to expect at each step.

## Where to go next

| If you want to | Read |
| :--- | :--- |
| convert your own boxes | [Convert HBBs to OBBs](guides/convert.md), [Annotation formats](guides/data-format.md) |
| produce boxes for images that have none | [Detect HBBs](guides/detect.md) |
| know how good a conversion is | [Evaluate](guides/evaluate.md), [Metrics](guides/metrics.md) |
| pick settings for your data | [Tune hyperparameters](guides/optimize.md), [Best practices](guides/practice.md) |
| publish a converted dataset | [End-to-end workflow](guides/workflow.md), [Provenance](guides/provenance.md) |
| look up a flag | [CLI reference](reference/cli/index.md) |
| call it from Python | [API reference](reference/api/index.md) |
| fix an error | [Troubleshooting](guides/troubleshooting.md) |

HBB2OBB is MIT licensed and developed on [GitHub](https://github.com/rfonod/hbb2obb); [releases](https://github.com/rfonod/hbb2obb/releases) list what changed in each version. If you use it in research, see [Citation](citation.md).
