# API reference

Every command is a front-end over a module of the `hbb2obb` package, and the same functions can be imported directly:

| Module | What it holds | Command |
| :--- | :--- | :--- |
| [`hbb2obb.converter`](converter.md) | The conversion: one image in, its OBBs out, and the writers for them. | `hbb2obb` |
| [`hbb2obb.detector`](detector.md) | Detection of horizontal boxes, and merging detections with hand-drawn ones. | `hbb2obb-detect` |
| [`hbb2obb.evaluator`](evaluator.md) | Matching and scoring OBBs against ground truth. | `hbb2obb-eval` |
| [`hbb2obb.analysis`](analysis.md) | The breakdown of an evaluation by what the ground-truth box looks like. | `hbb2obb-analyze` |
| [`hbb2obb.formats`](formats.md) | Readers and writers for the five annotation formats over one box representation. | `hbb2obb-convert`, `hbb2obb-view` |

Coordinates are in absolute pixels throughout, and paths are `pathlib.Path` objects. Each guide ends with a worked example: [conversion](../../guides/convert.md#python-api), [detection](../../guides/detect.md#python-api), [evaluation](../../guides/evaluate.md#python-api), [analysis](../../guides/analyze.md#python-api) and [formats](../../guides/formats.md#python-api).

The modules not listed here (`cli`, `optimizer`, `plotting`, `provenance`, `viewer`, `utils`) serve the commands and may change between releases.
