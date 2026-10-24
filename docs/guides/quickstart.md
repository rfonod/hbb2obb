# Quick start

This page runs the whole toolkit once on the sample data that ships with the repository: three 4K drone frames with 201 vehicles. It takes a few minutes on a laptop CPU, plus the first download of the SAM weights.

## Set up

Clone the repository and install it, then run every command below from the repository root:

```bash
git clone https://github.com/rfonod/hbb2obb.git
cd hbb2obb && pip install -e .
```

The sample is in `data/`:

```text
data/
├── images/          # 3 drone frames
├── labels_hbb/      # horizontal boxes, one YOLO .txt per frame (the input)
├── labels_obb_gt/   # oriented boxes drawn by hand (the ground truth)
├── labels_obb/      # oriented boxes converted by hbb2obb (the output, regenerated below)
└── classes.yaml     # class id to name: Car, Bus, Truck, Motorcycle
```

[`data/README.md`](https://github.com/rfonod/hbb2obb/blob/main/data/README.md) describes every file and the command that produced it. A conversion of your own needs only the first two: images and horizontal boxes for them.

## 1. Convert

```bash
hbb2obb data/images --save_img
```

```text
Wrote 201 boxes over 3 frames to data/labels_obb
Fallback to HBB: 0 boxes (0.00%)
```

`hbb2obb` found the HBBs in `data/labels_hbb`, the sibling of the image directory, and wrote one OBB file per frame to `data/labels_obb`. Each line is a class and four corners in pixels:

```text
1 1622 1208 1572 1186 1645 1020 1695 1043
0 2013 1979 1974 1966 2005 1873 2044 1887
```

`--save_img` also wrote `data/labels_obb/<frame>.jpg`, an overlay of the HBB, the mask, its contour and the OBB for every object. The fallback count is the number of boxes for which no usable mask was found; [How the conversion works](how-it-works.md) explains both.

!!! tip
    This overwrites the shipped `data/labels_obb/`. `git checkout data` restores it.

## 2. Evaluate

Score the converted boxes against the hand-drawn ground truth:

```bash
hbb2obb-eval data/labels_obb_gt data/labels_obb -mp data/classes.yaml
```

```text
Total Matched Boxes: 200
Total Unmatched GT Boxes: 1
Total Unmatched Pred Boxes: 1
Average IoU: 0.89642 ± 0.06834 (SEM 0.00483)
Median IoU: 0.91080
Matched Boxes Above Threshold: IoU>=0.50: 100.0%  IoU>=0.75: 98.0%  IoU>=0.85: 80.5%  IoU>=0.90: 57.0%
Orientation Error: p90 4.40°, median 1.04°, mean 2.11° ± 4.88°
```

The detector that produced the HBBs missed one motorcycle and found one car the annotator left out, hence one unmatched box on each side. [Metrics](metrics.md) explains each number.

## 3. Try an ensemble

Several SAM models vote on every mask pixel. Write the result to its own directory and score it the same way:

```bash
hbb2obb data/images --sam_models sam_b sam_l sam2_b --obb_dir data/labels_obb_ensemble
hbb2obb-eval data/labels_obb_gt data/labels_obb_ensemble -mp data/classes.yaml
```

```text
Average IoU: 0.90609 ± 0.07081 (SEM 0.00501)
Median IoU: 0.92018
Matched Boxes Above Threshold: IoU>=0.50: 100.0%  IoU>=0.75: 98.0%  IoU>=0.85: 83.5%  IoU>=0.90: 62.5%
Orientation Error: p90 4.67°, median 0.92°, mean 2.02° ± 4.83°
```

Better on IoU, slightly worse on the orientation tail, at more than three times the compute. Whether an ensemble pays off depends on the data; on the 4,245-box [Songdo Vision OBB](songdo.md) study a single `sam_l` tied a five-model ensemble. SAM is not bit-reproducible across PyTorch builds and hardware, so expect the last digits to differ on your machine.

## 4. Analyze

Break the first evaluation down by what the ground-truth box looks like:

```bash
hbb2obb-analyze data/labels_obb_gt data/labels_obb -mp data/classes.yaml -i data/images
```

Among its tables, *Against doing nothing* scores the HBB itself as if it were the answer:

```text
                              Boxes   IoU as-is   IoU converted      Gain   Angle p90 as-is   Angle p90 converted
  all                           200      0.7742          0.8964   +0.1222             35.09                  4.40
  axis-aligned (<0.005 deg)      45      0.9112          0.9431   +0.0319              0.00                  0.36
  rotated                       155      0.7344          0.8829   +0.1484             36.37                  5.65
  off-axis 30 to 45 deg          27      0.5517          0.8521   +0.3004             40.95                  5.13
```

The more a vehicle is turned off the image axes, the more the conversion gains over its HBB. [Analyze an evaluation](analyze.md) describes every table.

## 5. View

```bash
hbb2obb-view data/images --compare data/labels_obb_gt
```

The converted OBBs are green, their HBBs white and the ground truth blue. Scroll to zoom, drag to pan, `n` and `p` to change frame, `x` to show one set at a time, `q` to quit. [View annotations](view.md) lists every key.

## 6. Export

Write the converted boxes as DOTA files and one COCO file:

```bash
hbb2obb-convert data/labels_obb -f yolo --to dota coco -o release -mp data/classes.yaml -i data/images
```

```text
Read 201 OBB boxes over 3 frames from data/labels_obb (yolo)
Wrote 3 dota file(s) to release
Wrote 1 coco file(s) to release
```

[Convert between formats](formats.md) covers the other formats and `--verify`, which checks that every format of a set holds the same boxes.

## 7. Tune

`hbb2obb-optimize` converts the images once per combination of settings and scores each against the ground truth. `--dry_run` shows what a sweep would cost without running it:

```bash
hbb2obb-optimize data/images data/labels_obb_gt -sm sam_b -iz 960 1280 -sf 0.03 0.05 -n quick --dry_run
```

```text
Runs           : 1
  quick                                        sam_b                                    2 image size(s) x 2 scale factor(s) x 1 opening kernel(s) = 4 points
Total          : 4 conversions of the whole image set
```

Drop `--dry_run` to run it; the results land in `data/benchmark_results/quick/`. [Tune hyperparameters](optimize.md) explains the grid and what it writes.

!!! warning "Do not tune on the sample"
    The three sample frames are test images of the Songdo Vision dataset, and 201 boxes are far too few to choose settings on. They are here to show what the tools produce.

## Next steps

- Convert your own boxes: [Convert HBBs to OBBs](convert.md), and [Annotation formats](data-format.md) for the files it reads.
- No boxes yet: [Detect HBBs](detect.md).
- Take a dataset from someone else's format to a release: [End-to-end workflow](workflow.md).
