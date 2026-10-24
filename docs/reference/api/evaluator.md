# evaluator

The evaluation behind `hbb2obb-eval`. [Metrics](../../guides/metrics.md) defines what it reports, and the [guide](../../guides/evaluate.md#python-api) has a worked example.

```python
from pathlib import Path

from hbb2obb.evaluator import evaluate_obb, print_results

results = evaluate_obb(Path("data/labels_obb_gt"), Path("data/labels_obb"))
print(results["avg_iou"], results["sem_iou"], results["p90_angle_error"])
print_results(results, Path("data/classes.yaml"))
```

## Evaluating a set

::: hbb2obb.evaluator.evaluate_obb

::: hbb2obb.evaluator.print_results

## Matching and scoring

::: hbb2obb.evaluator.match_boxes

::: hbb2obb.evaluator.calculate_obb_iou

::: hbb2obb.evaluator.obb_orientation

::: hbb2obb.evaluator.orientation_error

::: hbb2obb.evaluator.iou_fractions
