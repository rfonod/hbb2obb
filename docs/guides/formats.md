# Convert between formats

`hbb2obb-convert` moves annotations between five formats, in either direction, for horizontal and oriented boxes alike:

| Format | HBB | OBB | Files | Confidence | `difficult` |
| :--- | :---: | :---: | :--- | :---: | :---: |
| `yolo` | ✅ | ✅ | one `.txt` per frame | ✅ trailing column | |
| `dota` | | ✅ | one `.dota` per frame (`--dota_ext` changes it), integer px | | ✅ |
| `voc` | ✅ | | one Pascal VOC `.xml` per frame, integer px | | ✅ |
| `coco` | ✅ | ✅ | one `.json` for the whole set, integer px; an OBB goes in `segmentation` | ✅ `score` | |
| `labelme` | ✅ | ✅ | one [LabelMe](https://github.com/wkentaro/labelme) `.json` per frame | | |

```bash
# Several formats in one pass; --from is detected from the files when omitted
hbb2obb-convert project/labels_obb --to dota coco -o project/release -mp label_map.yaml -i project/images

# A COCO file into the YOLO TXT hbb2obb reads
hbb2obb-convert project/instances.json --from coco --to yolo -o project/labels_hbb

# Check that every format of every set under a directory holds the same boxes
hbb2obb-convert project --verify -mp label_map.yaml
```

`source` is a directory of per-frame files, or the `.json` file itself for COCO. The output goes to `--out_dir` / `-o`, by default the source directory. The YOLO layouts are on [Annotation formats](data-format.md).

## What the conversion needs

**Image sizes.** Relative YOLO coordinates need each frame's size to become pixels, and VOC, COCO and LabelMe record it. `--images` / `-i` reads the sizes from the image headers; `--img_width` / `-iw` and `--img_height` / `-ih` give one size for every frame instead.

**Class names.** YOLO and COCO ids need names for the formats that store names (DOTA, VOC, LabelMe), and DOTA needs a label map to be read at all. Pass `-mp`, or keep a `classes.yaml` or `names.txt` beside the labels or one level up. Converting through LabelMe and back without one renumbers the classes in the order they first appear.

**Box kind.** `--kind` / `-k` writes `hbb` or `obb`; the default, `auto`, writes OBBs when any box is rotated. A format that cannot hold the kind, such as VOC for OBBs, stops the conversion.

**Normalized output.** `--normalize` / `-n` writes YOLO coordinates relative to `[0, 1]`; `--precision` / `-p` sets the decimals (default 10 normalized, 2 absolute).

## Rounding and verification

Coordinates are rounded to two decimals once, and every integer format (DOTA, VOC, COCO) rounds that value, so a COCO `bbox` is always the envelope of its own `segmentation`.

`--verify` reads every format present and compares them by exact equality after that rounding, not by a tolerance, so any disagreement it reports is real. Given a directory of label directories, it checks each one, pairing `labels_<name>/` with a `coco_annotations_<name>.json` beside it and `labels/` with `coco_annotations.json`. Image sizes come from `-i`, else from an `images/` directory beside the sets:

```text
labels_hbb: 3 frames, 201 boxes, 3 formats (coco, voc, yolo)
labels_obb: 3 frames, 201 boxes, 3 formats (coco, dota, yolo)
labels_obb_gt: 3 frames, 201 boxes, 3 formats (coco, dota, yolo)
OK: 603 boxes, every format encodes the same boxes
```

It exits non-zero on any disagreement and writes nothing.

When a directory holds YOLO files next to formats derived from them, `--from` is detected as `yolo`, so a set is always converted from its full-precision source.

## The difficult flag

DOTA and Pascal VOC carry a per-box `difficult` flag, so writing either from YOLO or COCO sets it to 0 unless one of these is given:

- `--difficult_from dota` (or `voc`) carries the flags across from that format in the source directory, row by row.
- `--difficult_from confidence` flags every box scoring below `--difficult_below` (default 0.5). A [fallback](how-it-works.md#fallback) scores 0.0, so it is always flagged. The scores come from a trailing column on every source label, or from `--confidence_dir` when the labels are standard ones with a side-car:

```bash
hbb2obb-convert project/labels_obb --to dota -i project/images -mp label_map.yaml \
    --difficult_from confidence --confidence_dir project/labels_confidence --difficult_below 0.4
```

The scores themselves stay out of the written labels.

## Confidence

YOLO keeps a confidence as a trailing column and COCO as `score`; DOTA, VOC and LabelMe drop it. [`hbb2obb-view`](view.md) reads it from either.

## COCO files

A COCO file is named `coco_annotations_<kind>.json`, `<kind>` being `hbb` or `obb`, unless `--coco_name` says otherwise. Name it after the directory it pairs with, `coco_annotations_<name>.json` for `labels_<name>/`, and `--verify` and the viewer find it.

A COCO file lists each frame as `<stem>.jpg`. Image and annotation ids are numbered from 1, in frame then row order, so an HBB and an OBB file written from the same boxes join by id; a category id is the class id plus 1. When the boxes are derived one for one from an existing COCO file, `--coco_from` takes its image and annotation ids, `licenses` and `supercategory` instead, so the two still join by id; the frame names, box counts and classes are checked to match, row by row, before anything is written. `--coco_info` reads the `info` object of the written file from a JSON file, for a description, version or DOI.

## Python API

[`hbb2obb.formats`](../reference/api/formats.md) reads and writes whole sets over one representation, a [`FrameAnnotations`](../reference/api/formats.md#hbb2obb.formats.FrameAnnotations) of [`Box`](../reference/api/formats.md#hbb2obb.formats.Box)es with four corners in pixels:

```python
from pathlib import Path

from hbb2obb import formats

names = ["Car", "Bus", "Truck", "Motorcycle"]
sizes = formats.image_sizes(Path("data/images"))  # frame stem -> (width, height)
frames, names, fmt = formats.read_set(Path("data/labels_obb"), names=names, sizes=sizes)
print(fmt, sum(len(f) for f in frames))  # yolo 201

frames = formats.canonicalize(frames)  # the single rounding point
formats.write_set(frames, Path("release"), "dota", names, kind="obb")
```

All flags: [`hbb2obb-convert` reference](../reference/cli/hbb2obb-convert.md).
