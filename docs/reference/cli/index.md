# CLI reference

Installing hbb2obb provides seven commands:

| Command | What it does | Guide |
| :--- | :--- | :--- |
| [`hbb2obb`](hbb2obb.md) | Converts HBB annotations to OBBs by prompting SAM. | [Convert](../../guides/convert.md) |
| [`hbb2obb-detect`](hbb2obb-detect.md) | Detects HBBs with an Ultralytics model, for images that have none. | [Detect](../../guides/detect.md) |
| [`hbb2obb-eval`](hbb2obb-eval.md) | Scores predicted OBBs against ground truth. | [Evaluate](../../guides/evaluate.md) |
| [`hbb2obb-analyze`](hbb2obb-analyze.md) | Breaks an evaluation down by orientation, size, frame edge, difficulty and class. | [Analyze](../../guides/analyze.md) |
| [`hbb2obb-convert`](hbb2obb-convert.md) | Converts annotations between YOLO, DOTA, Pascal VOC, COCO and LabelMe. | [Formats](../../guides/formats.md) |
| [`hbb2obb-view`](hbb2obb-view.md) | Shows HBB and OBB annotations over their images, interactively or as image files. | [View](../../guides/view.md) |
| [`hbb2obb-optimize`](hbb2obb-optimize.md) | Searches the conversion hyperparameters, as one sweep or a benchmark of several. | [Tune](../../guides/optimize.md) |

The help on each page is generated from the parsers when the site is built, so it matches the release the site was built from. On the command line, `<command> --help` prints the same text and `--version` / `-V` the installed version.

## Conventions

- **Default paths.** A command given an image directory reads and writes sibling directories by default (`labels_hbb`, `labels_obb`, `labels_polygon`, `labels_confidence`, `benchmark_results`), so a project in that [layout](../../guides/data-format.md#directory-layout) needs no path flags.
- **Short flags.** Most options have a short form, listed beside the long one, e.g. `-sm` for `--sam_models`.
- **Lists.** Options that take several values take them space-separated: `-sm sam_l sam_b`, `-sf 0.03 0.05`.
- **Label maps.** `-mp` / `--map_path` names a YAML of class IDs to names wherever classes are named; see [Label map](../../guides/data-format.md#label-map).
- **Predictor arguments.** `--model_kwargs` / `-k` passes any other Ultralytics argument through as `key=value,key=value`.
- **Progress bars.** `--no_bar` / `-nb` turns them off, e.g. for log files.
- **Environment.** `HBB2OBB_MODELS_DIR` sets the models directory and `HBB2OBB_DISABLE_UPDATE_CHECK=1` turns off the daily update check; see [Installation](../../guides/install.md#environment-variables).
