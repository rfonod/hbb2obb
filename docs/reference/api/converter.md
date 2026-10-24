# converter

The conversion behind the `hbb2obb` command: one image in, its OBBs out. [How the conversion works](../../guides/how-it-works.md) describes the steps, and the [guide](../../guides/convert.md#python-api) has a worked example.

```python
from pathlib import Path

from hbb2obb.converter import hbb2obb, save_obb_annotations

img_path = Path("data/images/04433.jpg")
obbs = hbb2obb(img_path, sam_models=["sam_l", "sam_b"], scale_factors=0.05)  # (N, 9) array
save_obb_annotations(obbs, Path("data/labels_obb"), img_path)
```

## Converting an image

::: hbb2obb.converter.hbb2obb

::: hbb2obb.converter.ConversionStats

## Writing the results

::: hbb2obb.converter.save_obb_annotations

::: hbb2obb.converter.save_confidence_annotations

::: hbb2obb.converter.save_polygon_annotations

## Pipeline steps

::: hbb2obb.converter.scale_bounding_boxes

::: hbb2obb.converter.aggregate_masks_by_majority_vote

::: hbb2obb.converter.apply_morphological_opening

::: hbb2obb.converter.merge_fragment_contours

## Models

::: hbb2obb.converter.load_sam_model

::: hbb2obb.converter.clear_model_cache
