# detector

The detection behind `hbb2obb-detect`: horizontal boxes for images that have none. See the [guide](../../guides/detect.md#python-api) for a worked example.

```python
from pathlib import Path

import cv2

from hbb2obb.detector import detect_hbb, save_hbb_annotations

img_path = Path("data/images/04433.jpg")
boxes = detect_hbb(cv2.imread(str(img_path)), model="geotrax")  # (N, 6): class, xc, yc, w, h, conf
save_hbb_annotations(boxes, Path("data/labels_hbb"), img_path, scores=boxes[:, 5])
```

## Detecting

::: hbb2obb.detector.detect_hbb

::: hbb2obb.detector.save_hbb_annotations

::: hbb2obb.detector.load_detector

::: hbb2obb.detector.resolve_weights

## Classes

::: hbb2obb.detector.parse_class_map

::: hbb2obb.detector.apply_class_map

## Merging with hand-drawn boxes

::: hbb2obb.detector.merge_detections

::: hbb2obb.detector.MergeReport
