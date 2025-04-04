#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
HBB2OBB Hyperparameter Optimization Tool

This script finds optimal hyperparameters for HBB2OBB conversion by evaluating
different combinations of inference image sizes and scale factors using grid search.

It supports customization of evaluation criteria, filtering options, and provides
detailed outputs of performance metrics including IoU scores and execution time.

Usage:
    python optimize_hbb2obb.py <img_source> <gt_dir> [options]

Required Arguments:
    img_source        Path to image(s) - can be a single file or directory
    gt_dir            Directory containing ground truth OBB annotations

Common Options:
    --hbb_dir, -hd       HBB annotation directory (default: img_source/../labels_hbb)
    --sam_models, -sm    SAM model variant (default: sam_b)
    --scale_factors, -sf Scale factors to test (default: range from -0.01 to 0.1)
    --imgsz, -iz         Image sizes to test (default: [640, 960, 1280])
    --output_folder, -o  Results save location (default: img_source/../benchmark_results)
    --run_name, -n       Custom name for this benchmark run (default: timestamp)

Evaluation Options:
    --excluded_classes, -e     Class labels to exclude (default: [])
    --iou_threshold, -t        IoU threshold for evaluation (default: 0.1)
    --class_agnostic, -ca      Evaluate without considering class labels (default: False)
    --exclude_edge_cases, -exc Skip boxes at image edges (default: False)
    --edge_tolerance, -et      Pixel tolerance for edge detection (default: 1)
    --img_width, -iw           Image width for edge case detection
    --img_height, -ih          Image height for edge case detection

Advanced Options:
    --model_kwargs, -k  Additional model parameters in 'key1=value1,key2=value2' format

Examples:
    # Basic usage
    python scripts/optimize_hbb2obb.py data/images data/labels_obb_gt

    # Custom scale factors and image sizes
    python scripts/optimize_hbb2obb.py data/images data/labels_obb_gt --scale_factors 0.05 0.1 --imgsz 640 960

    # Specify multiple SAM models
    python scripts/optimize_hbb2obb.py data/images data/labels_obb_gt --sam_models sam_b sam_l sam2_b sam2.1_b

    # Exclude certain classes and use class-agnostic evaluation
    python scripts/optimize_hbb2obb.py data/images data/labels_obb_gt --excluded_classes 0 3 --class_agnostic

Output:
    The script creates a results directory containing:
    - summary.txt: Human-readable results summary and best parameters
    - results.yaml: Detailed results for all parameter combinations
    - config.yaml: Complete configuration of the optimization run
"""

import argparse
import copy
import datetime
import subprocess
import tempfile
import time
from itertools import product
from pathlib import Path

import yaml

from hbb2obb.cli import SUPPORTED_SAM_MODELS
from hbb2obb.evaluator import evaluate_obb
from hbb2obb.utils import get_hbb_dir, get_system_metadata


def parse_cli_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Find optimal hyperparameters for HBB2OBB conversion")

    # Required arguments
    parser.add_argument("img_source", type=Path, help="Path to an image or directory containing images")
    parser.add_argument("gt_dir", type=Path, help="Directory containing ground truth OBB annotations")

    # Optional arguments
    parser.add_argument("--hbb_dir", "-hd", type=Path, help="Directory containing HBB annotations (default: img_source/../labels_hbb)")
    parser.add_argument("--sam_models", "-sm", type=str, default=["sam_b"], nargs='+', choices=SUPPORTED_SAM_MODELS, help="SAM model(s) to use")
    parser.add_argument("--opening_kernel_percentage", "-okp", type=float, default=0.15,
        help="Percentage of mask's smaller dimension for morphological opening. Ignored if <= 0.")

    # Hyperparameters to optimize
    parser.add_argument("--scale_factors", "-sf", type=float, nargs='+', default=[-0.01, 0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1],
                        help="List of scale factors to test")
    parser.add_argument("--imgsz", "-iz", type=int, nargs='+', default=[640, 960, 1280], help="List of image sizes to test")

    # Evaluation options
    parser.add_argument("--excluded_classes", "-e", type=int, nargs='+', default=[], help="Class labels to exclude from evaluation")
    parser.add_argument("--iou_threshold", "-t", type=float, default=0.1, help="IoU threshold for evaluation")
    parser.add_argument("--class_agnostic", "-ca", action="store_true", help="Evaluate in class-agnostic mode")
    parser.add_argument("--exclude_edge_cases", "-exc", action="store_true", help="Exclude boxes that touch image edges from evaluation")
    parser.add_argument("--edge_tolerance", "-et", type=int, default=1, help="Tolerance in pixels for detecting boxes at image edges (default: 1)")
    parser.add_argument("--img_width", "-iw", type=int, help="Image width for edge case detection (required if --exclude_edge_cases is used)")
    parser.add_argument("--img_height", "-ih", type=int, help="Image height for edge case detection (required if --exclude_edge_cases is used)")

    # Optional parameters
    parser.add_argument("--model_kwargs", "-k", type=str, help="Additional keyword arguments for model in format 'key1=value1,key2=value2'")
    parser.add_argument("--output_folder", "-o", type=Path, help="Directory to save results (default: img_source/../benchmark_results)")
    parser.add_argument("--run_name", "-n", type=str, help="Custom name for this benchmark run (default: auto-generated based on parameters)")

    return parser.parse_args()


def main():
    """Run hyperparameter optimization."""
    args = parse_cli_args()

    # Determine HBB directory
    hbb_dir = get_hbb_dir(args.img_source, args.hbb_dir)

    # Parameter combinations
    image_sizes = args.imgsz
    scale_factors = args.scale_factors
    # Create parameter grid
    param_grid = list(product(image_sizes, scale_factors))

    # Determine output folder
    if args.output_folder is None:
        if args.img_source.is_file():
            base_dir = args.img_source.parent.parent
        else:
            base_dir = args.img_source.parent
        output_folder = base_dir / "benchmark_results"
    else:
        output_folder = args.output_folder

    # Create output folder if it doesn't exist
    output_folder.mkdir(exist_ok=True, parents=True)

    # Generate run name or use provided one
    run_name = args.run_name if args.run_name else datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = output_folder / run_name
    run_folder.mkdir(exist_ok=True)

    # Save the run configuration
    config_file = run_folder / "config.yaml"
    config = {
        "run_name": str(run_name),
        "img_source": str(args.img_source),
        "gt_dir": str(args.gt_dir),
        "hbb_dir": str(hbb_dir),
        "sam_models": args.sam_models,
        "scale_factors": args.scale_factors,
        "imgsz": args.imgsz,
        "excluded_classes": list(args.excluded_classes),
        "iou_threshold": args.iou_threshold,
        "class_agnostic": args.class_agnostic,
        "model_kwargs": args.model_kwargs,
    }

    # Add system metadata
    config["system_metadata"] = get_system_metadata()

    # Save configuration to YAML file
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    # Print configuration
    print("Configuration:")
    print("-" * 105)
    print(f"Run Name: {run_name}")
    print(f"Image Source: {args.img_source}")
    print(f"Ground Truth Directory: {args.gt_dir}")
    print(f"HBB Directory: {hbb_dir}")
    print(f"Output Folder: {output_folder}")
    print(f"Scale Factors: {args.scale_factors}")
    print(f"Image Sizes: {args.imgsz}")
    print(f"Excluded Classes: {args.excluded_classes}")
    print(f"IoU Threshold: {args.iou_threshold}")
    print(f"Class Agnostic: {args.class_agnostic}")
    print(f"Model Kwargs: {args.model_kwargs}")
    print("-" * 105)
    print("Starting evaluation...")
    print("-" * 105)
    print(f"Total combinations to evaluate: {len(param_grid)}")
    print("-" * 105)

    results = []
    best_iou = 0
    best_params = None

    # Create temporary directory for evaluation
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        print(f"Temporary directory created: {temp_dir_path}")

        print("\nTesting parameter combinations:")
        print("-" * 105)
        print(f"{'Image Size':<12} {'Scale Factor':<15} {'Avg IoU':<17} {'Matches':<10} {'GT Total':<10} {'Pred Total':<10} {'IoU Threshold':<15} {'Time (s)':<10}")
        print("-" * 105)

        # Evaluate each combination
        for imgsz, sf in param_grid:
            # Create subdirectory for this run
            run_dir = temp_dir_path / f"imgsz_{imgsz}_ef_{sf}"
            run_dir.mkdir()

            # Run HBB2OBB conversion
            cmd = (
                ["hbb2obb", str(args.img_source), "--hbb_dir", str(hbb_dir),
                 "--obb_dir", str(run_dir), "--sam_models"]
                + args.sam_models
                + ["--scale_factors", str(sf), "--imgsz", str(imgsz), "--no_bar"]
            )

            if args.model_kwargs:
                cmd.extend(["--model_kwargs", args.model_kwargs])

            # Execute conversion and time it
            start_time = time.time()
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
            execution_time = time.time() - start_time

            # Evaluate results
            eval_results = evaluate_obb(
                gt_dir=args.gt_dir,
                pred_dir=run_dir,
                excluded_classes=args.excluded_classes,
                iou_threshold=args.iou_threshold,
                class_agnostic=args.class_agnostic,
                exclude_edge_cases=args.exclude_edge_cases,
                edge_tolerance=args.edge_tolerance,
                img_width=args.img_width,
                img_height=args.img_height,
                debug=False,
                no_bar=True
            )

            # Access the results directly
            avg_iou = eval_results["avg_iou"]
            std_iou = eval_results["std_iou"]
            total_matches = eval_results["total_matches"]
            total_gt = eval_results["total_gt"]
            total_pred = eval_results["total_pred"]

            # Store results
            param_result = {
                "imgsz": int(imgsz),
                "scale_factor": float(sf),
                "avg_iou": float(avg_iou),
                "std_iou": float(std_iou),
                "total_matches": int(total_matches),
                "total_gt": int(total_gt),
                "total_pred": int(total_pred),
                "class_agnostic": args.class_agnostic,
                "iou_threshold": args.iou_threshold,
                "excluded_classes": list(args.excluded_classes),
                "execution_time": float(execution_time)
            }
            results.append(param_result)

            # Track best parameters
            if avg_iou > best_iou:
                best_iou = avg_iou
                best_params = param_result

            # Print result line
            print(f"{imgsz:<12} {sf:^15.3f} {avg_iou:<1.4f} ± {std_iou:<1.4f}   {total_matches:<10} {total_gt:<10} {total_pred:<10} {args.iou_threshold:<15} {execution_time:<10.2f}")

    # Save detailed results
    results_file = run_folder / "results.yaml"
    output_data = {
        "best_parameters": best_params,
        "all_results": copy.deepcopy(results),
    }

    with open(results_file, 'w') as f:
        yaml.dump(output_data, f, default_flow_style=False, sort_keys=False)

    # Save a summary file that's easier to parse
    summary_file = run_folder / "summary.txt"
    with open(summary_file, 'w') as f:
        f.write(f"Benchmark Run: {run_name}\n")
        f.write(f"End Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("BEST PARAMETERS:\n")
        f.write(f"  Image Size: {best_params['imgsz']}\n")
        f.write(f"  Scale Factor: {best_params['scale_factor']:.4f}\n")
        f.write(f"  Average IoU: {best_params['avg_iou']:.4f} ± {best_params['std_iou']:.4f}\n")
        f.write(f"  Total Matches: {best_params['total_matches']}\n")
        f.write(f"  Total GT: {best_params['total_gt']}\n")
        f.write(f"  Total Pred: {best_params['total_pred']}\n")
        f.write(f"  Execution Time: {best_params['execution_time']:.2f} seconds\n")

        f.write("\nALL RESULTS (sorted by Average IoU):\n")
        # Sort results by avg_iou descending
        sorted_results = sorted(results, key=lambda x: x['avg_iou'], reverse=True)
        for i, result in enumerate(sorted_results):
            f.write(f"{i+1:4d}. ImgSz: {result['imgsz']:5d}, SF: {result['scale_factor']:7.4f}, IoU: {result['avg_iou']:7.4f} ± {result['std_iou']:7.4f}, Time: {result['execution_time']:5.2f}s\n")

    # Print best parameters
    print("\n" + "=" * 105)
    print("BEST PARAMETERS:")
    print(f"  Image Size: {best_params['imgsz']}")
    print(f"  Scale Factor: {best_params['scale_factor']:.4f}")
    print(f"  Average IoU: {best_params['avg_iou']:.4f} ± {best_params['std_iou']:.4f}")
    print(f"  Total Matches: {best_params['total_matches']}")
    print(f"  Total GT: {best_params['total_gt']}")
    print(f"  Total Pred: {best_params['total_pred']}")
    print(f"  Execution Time: {best_params['execution_time']:.2f} seconds")
    print(f"\nResults saved to: {run_folder}")
    print("=" * 105)


if __name__ == "__main__":
    main()
