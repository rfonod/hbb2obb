# formats

Readers and writers for the five annotation formats, behind `hbb2obb-convert` and `hbb2obb-view`. Every format reads into, and writes from, one representation: a `FrameAnnotations` per frame holding a `Box` per object, with four corners in absolute pixels. A horizontal box is a `Box` whose corners are axis-aligned. The [guide](../../guides/formats.md#python-api) has a worked example.

```python
from pathlib import Path

from hbb2obb import formats

frames, names, fmt = formats.read_set(Path("data/labels_obb_gt"), sizes=formats.image_sizes(Path("data/images")))
box = frames[0].boxes[0]
print(box.cls, box.quad, box.xyxy, box.is_axis_aligned)
```

## Representation

::: hbb2obb.formats.Box

::: hbb2obb.formats.FrameAnnotations

::: hbb2obb.formats.canonicalize

## Whole sets

::: hbb2obb.formats.read_set

::: hbb2obb.formats.write_set

::: hbb2obb.formats.verify

::: hbb2obb.formats.detect_format

::: hbb2obb.formats.formats_present

## COCO identifiers

::: hbb2obb.formats.CocoIdentity

::: hbb2obb.formats.read_coco_identity

::: hbb2obb.formats.check_coco_identity

## Images

::: hbb2obb.formats.image_size
