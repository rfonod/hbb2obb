# analysis

The breakdown behind `hbb2obb-analyze`. The [guide](../../guides/analyze.md) describes each table and has a worked example.

```python
from pathlib import Path

from hbb2obb.analysis import collect_pairs, plot_analysis, render_markdown, summarise

pairs = collect_pairs(Path("data/labels_obb_gt"), Path("data/labels_obb"), hbb_dir=Path("data/labels_hbb"))
summary = summarise(pairs)
Path("analysis.md").write_text(render_markdown(summary))
plot_analysis(summary, Path("analysis.png"))
```

## Collecting pairs

::: hbb2obb.analysis.collect_pairs

::: hbb2obb.analysis.Pair

## Summarising and reporting

::: hbb2obb.analysis.summarise

::: hbb2obb.analysis.render_markdown

::: hbb2obb.analysis.print_analysis

::: hbb2obb.analysis.plot_analysis
