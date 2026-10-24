# End-to-end workflow

This page takes a dataset from horizontal boxes, in whatever format they arrived in, to a released set of oriented boxes. It assumes this layout; every command also takes explicit paths.

```text
project/
├── images/              # every frame
├── voc_hbb/             # horizontal boxes for every frame, as received (Pascal VOC here)
├── label_map.yaml       # class id to name
└── tune/
    ├── images/          # copies of a few dozen representative frames
    └── gt.json          # oriented boxes drawn by hand for those frames (COCO here)
```

The tuning frames are a sample of the dataset with hand-drawn OBBs. Without them, see [No ground truth](#no-ground-truth).

## 1. Bring the inputs into YOLO TXT

```bash
hbb2obb-convert project/voc_hbb --to yolo -o project/labels_hbb -mp project/label_map.yaml
hbb2obb-convert project/tune/gt.json --to yolo -o project/tune/labels_obb_gt -mp project/label_map.yaml
```

`--from` is detected from the files, and the label map gives both sets the same class IDs. See [Convert between formats](formats.md).

**No HBBs at all?** `hbb2obb-detect project/images -mp project/label_map.yaml` writes `project/labels_hbb` instead. See [Detect HBBs](detect.md).

## 2. Tune on the ground truth

```bash
hbb2obb-optimize project/tune/images project/tune/labels_obb_gt -hd project/labels_hbb -sm sam_l --dry_run
hbb2obb-optimize project/tune/images project/tune/labels_obb_gt -hd project/labels_hbb -sm sam_l
```

The sweep converts every image in `tune/images` once per grid point, so keep that directory to the frames that have ground truth. To compare model sets, list them in a [benchmark file](optimize.md#benchmarks-several-sweeps-from-one-file).

Read the winner from `project/tune/benchmark_results/sam_l/summary.txt`, and check it against the [standard error](metrics.md#iou) before preferring a costlier setting.

## 3. Convert every frame

```bash
hbb2obb project/images --sam_models sam_l --imgsz 1280 --scale_factors 0.02 \
    --opening_kernel_percentage -0.1 --confidence_dir --save_provenance --fail_on_fallback_share 0.01
```

This writes `project/labels_obb/`, the scores in `project/labels_confidence/` and the record in `project/PROVENANCE_obb.txt`. With `--fail_on_fallback_share 0.01` the command exits non-zero when more than 1% of the boxes fell back to their HBB, which a script can check before going on. Add `--normalize` if the set will train an Ultralytics model.

## 4. Check the result

```bash
hbb2obb-eval project/tune/labels_obb_gt project/labels_obb -mp project/label_map.yaml
hbb2obb-analyze project/tune/labels_obb_gt project/labels_obb -hd project/labels_hbb -i project/images -o project/analysis
hbb2obb-view project/images --show_confidence
```

The evaluation pairs files by name, so it scores the tuning frames only; the settings were chosen on them, so to report accuracy, score a second set of hand-drawn frames the tuning never saw. The viewer covers every frame: color by confidence and look at the lowest scores first. See [Evaluate](evaluate.md), [Analyze](analyze.md) and [View](view.md).

## 5. Ship it

Derive the other formats from the YOLO files, the DOTA files beside them and the COCO file one level up, then check that they agree:

```bash
hbb2obb-convert project/labels_obb -f yolo --to dota -o project/labels_obb -mp project/label_map.yaml -i project/images \
    --difficult_from confidence --confidence_dir project/labels_confidence
hbb2obb-convert project/labels_obb -f yolo --to coco -o project -mp project/label_map.yaml -i project/images
hbb2obb-convert project --verify -mp project/label_map.yaml
```

The DOTA files flag every box scoring below 0.5 as `difficult`. `--verify` pairs `labels_obb/` with `coco_annotations_obb.json` and checks that the YOLO, DOTA and COCO readings hold the same boxes. Publish them with the confidence side-car and `PROVENANCE_obb.txt`.

When the HBBs came from an existing COCO release, `--coco_from` keeps its image and annotation ids so the two join; see [COCO files](formats.md#coco-files).

## No ground truth

Skip steps 2 and 4. Convert with the defaults, or with settings that held on similar imagery ([Best practices](practice.md), [Songdo Vision OBB](songdo.md)), and triage by [confidence](confidence.md) in the viewer.
