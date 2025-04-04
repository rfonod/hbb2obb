# Sample Data

This directory contains sample data for demonstrating and testing the HBB2OBB toolkit functionality. The data includes images, horizontal bounding box (HBB) annotations, ground truth oriented bounding box (OBB) annotations, and automatically converted OBB annotations.

## Data Attribution

The sample images provided in the `images` folder are sourced from the [Songdo Vision](https://doi.org/10.5281/zenodo.13828408) dataset and are re-used here under the Creative Commons Attribution 4.0 International (CC-BY-4.0) license.

### Original Dataset Details:
- **DOI:** [10.5281/zenodo.13828408](https://doi.org/10.5281/zenodo.13828408)
- **Title:** *Songdo Vision: Vehicle Annotations from High-Altitude BeV Drone Imagery in a Smart City*
- **Authors:** Robert Fonod, Haechan Cho, Hwasoo Yeo, Nikolas Geroliminis
- **Publisher:** Zenodo
- **Version:** v1
- **License:** [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/deed.en)

## Directory Structure

```
data/
├── images/                  # Sample drone imagery
├── labels_hbb/              # Horizontal Bounding Box annotations (YOLO format)
├── labels_obb_gt/           # Ground truth Oriented Bounding Box annotations (YOLO OBB format) 
├── labels_obb/              # Automatically converted OBB annotations using HBB2OBB
└── benchmark_results/       # Results of hyperparameter optimization runs
    └── sam_b/               # Results using SAM-B model
        ├── config.yaml      # Configuration used for the benchmark
        ├── plot.png         # Visualization of benchmark results
        ├── results.yaml     # Detailed benchmark results 
        └── summary.txt      # Human-readable summary
```

## Annotation Details

- **labels_hbb**: Contains modified versions of the original HBB annotations from the Songdo Vision dataset. These modifications improve upon the original semi-manual HBB annotation quality.

- **labels_obb_gt**: Contains manually annotated OBBs that serve as ground truth for HBB2OBB evaluation or parameter tuning.

- **labels_obb**: Contains automatically transformed versions of the HBB annotations using the `hbb2obb` function.

## Replicating the Annotations

To replicate the OBB annotations in the `labels_obb` directory, run the following command from the project root directory:

```bash
hbb2obb data/images --save_img
```

This command will process all images in the `data/images` directory, convert HBB annotations to OBB, and save visualizations using the default parameters. The converted OBB annotations will be saved in the `data/labels_obb` directory. (Note: Better results can be achieved by using multiple SAM models and hyperparameter tuning, as described in the main README.)

### Evaluating the Conversion Quality

You can evaluate the conversion quality against the ground truth using:

```bash
hbb2obb-eval data/labels_obb_gt data/labels_obb
```

For more evaluation options, such as excluding specific classes or setting different IoU thresholds, refer to the main README or run:

```bash
hbb2obb-eval --help
```

## Benchmark Results

The `benchmark_results` directory contains the output of hyperparameter optimization runs performed on the sample data and using default parameters. The results are organized into subfolders, each corresponding to a different set of SAM models.

For example, the results inside the `sam_b` folder can be reproduced by running:

```bash
python scripts/optimize_hbb2obb.py data/images data/labels_obb_gt -n sam_b
```

The visualization plot (`plot.png`) inside the `sam_b` folder can be reproduced by running:

```bash
python scripts/plot_optimization_results.py data/benchmark_results/sam_b
```
