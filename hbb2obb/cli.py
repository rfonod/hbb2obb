#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
Command-line interface for HBB2OBB.
"""

import argparse
from pathlib import Path

import tqdm

from hbb2obb.converter import hbb2obb, save_obb_annotations
from hbb2obb.evaluator import evaluate_obb, print_results
from hbb2obb.utils import get_image_paths, process_ultralytics_kwargs

SUPPORTED_SAM_MODELS = [
    "sam_b",
    "sam_l",
    "mobile_sam",
    "sam2_t",
    "sam2_s",
    "sam2_b",
    "sam2.1_t",
    "sam2.1_s",
    "sam2.1_b",
    "FastSAM-s",
    "FastSAM-x",
]


def main_hbb2obb():
    """
    Run the HBB to OBB conversion from command line.
    """

    parser = argparse.ArgumentParser(description="Convert HBB to OBB annotations")

    # Main arguments
    parser.add_argument("img_source", type=Path, help="Path to an image or directory containing images")
    parser.add_argument("--hbb_dir", "-hd", type=Path, help="Directory containing HBB annotations (default: img_source/../labels_hbb)")
    parser.add_argument("--obb_dir", "-od", type=Path, help="Directory to save OBB annotations (default: img_source/../labels_obb)")
    parser.add_argument("--sam_models", "-sm", type=str, default=["sam_b"], nargs='+', choices=SUPPORTED_SAM_MODELS,
        help="Name(s) of SAM model(s) to use (default: sam_b). Multiple models can be specified to average results.")
    parser.add_argument("--imgsz", type=int, default=1280, help="Image size for SAM model inference (default: 1280)")
    parser.add_argument("--scale_factors", "-sf", type=float, default=[0.05], nargs='+',
        help="Factor(s) to scale HBBs (default: 0.05). Use one value for uniform or two values for short/long sides.")
    parser.add_argument("--opening_kernel_percentage", "-okp", type=float, default=0.15,
        help="Percentage of mask's smaller dimension for morphological opening. Ignored if <= 0.")

    # Visualization control arguments
    viz_group = parser.add_argument_group('visualization options')
    viz_group.add_argument("--save_img", action="store_true", help="Save visualization images (default: False)")
    viz_group.add_argument("--viz_dir", type=Path, help="Directory to save visualization images (default: same as 'obb_dir')")
    viz_group.add_argument("--show_hbb", action="store_true", default=True, help="Show horizontal bounding boxes (default: True)")
    viz_group.add_argument("--hide_hbb", action="store_false", dest="show_hbb", help="Hide horizontal bounding boxes")
    viz_group.add_argument("--show_masks", action="store_true", default=True, help="Show segmentation masks (default: True)")
    viz_group.add_argument("--hide_masks", action="store_false", dest="show_masks", help="Hide segmentation masks")
    viz_group.add_argument("--show_segments", action="store_true", default=True, help="Show segmentation contours (default: True)")
    viz_group.add_argument("--hide_segments", action="store_false", dest="show_segments", help="Hide segmentation contours")
    viz_group.add_argument("--show_obb", action="store_true", default=True, help="Show oriented bounding boxes (default: True)")
    viz_group.add_argument("--hide_obb", action="store_false", dest="show_obb", help="Hide oriented bounding boxes")
    viz_group.add_argument("--show_labels", action="store_true", default=True, help="Show class labels (default: True)")
    viz_group.add_argument("--hide_labels", action="store_false", dest="show_labels", help="Hide class labels")

    # Miscellaneous arguments
    parser.add_argument("--model_kwargs", "-k", type=str, help="Additional keyword arguments for ultralytics model inference in format 'key1=value1,key2=value2'")
    parser.add_argument("--no_bar", "-nb", action="store_true", help="Disable tqdm progress bar display")

    args = parser.parse_args()
    model_kwargs = process_ultralytics_kwargs(args.model_kwargs)

    image_paths = get_image_paths(args.img_source)
    for img_path in tqdm.tqdm(image_paths, desc="Processing images", leave=True, disable=args.no_bar):
        obb_annotations = hbb2obb(
            img_path=img_path,
            hbb_dir=args.hbb_dir,
            sam_models=args.sam_models,
            imgsz=args.imgsz,
            scale_factors=args.scale_factors,
            opening_kernel_percentage=args.opening_kernel_percentage,
            save_img=args.save_img,
            viz_dir=args.viz_dir if args.viz_dir else args.obb_dir,
            show_hbb=args.show_hbb,
            show_masks=args.show_masks,
            show_segments=args.show_segments,
            show_obb=args.show_obb,
            show_labels=args.show_labels,
            model_kwargs=model_kwargs,
        )

        # Save OBB annotations to a text file
        save_obb_annotations(obb_annotations, args.obb_dir, img_path)


def main_hbb2obb_eval():
    """
    Run the OBB evaluation from command line.
    """

    parser = argparse.ArgumentParser(description="Evaluate OBB predictions against ground truth")

    # Main arguments
    parser.add_argument("gt_dir", type=Path, help="Directory containing ground truth OBB annotations")
    parser.add_argument("pred_dir", type=Path, help="Directory containing predicted/converted OBB annotations")
    parser.add_argument("--excluded_classes", "-e", type=int, nargs='+', default=[], help="Class labels to exclude from evaluation")
    parser.add_argument("--iou_threshold", "-t", type=float, default=0.1, help="IoU threshold for considering a match (default: 0.1)")
    parser.add_argument("--class_agnostic", "-ca", action="store_true", help="Evaluate in class-agnostic mode (match boxes regardless of class labels)")
    parser.add_argument("--map_path", "-mp", type=Path, help="Path to label map YAML file (optional)")
    parser.add_argument("--exclude_edge_cases", "-exc", action="store_true", help="Exclude boxes that touch image edges from evaluation")
    parser.add_argument("--edge_tolerance", "-et", type=int, default=1, help="Tolerance in pixels for detecting boxes at image edges (default: 1)")
    parser.add_argument("--img_width", "-iw", type=int, help="Image width for edge case detection (required if --exclude_edge_cases is used)")
    parser.add_argument("--img_height", "-ih", type=int, help="Image height for edge case detection (required if --exclude_edge_cases is used)")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug mode to print detailed matching information")
    parser.add_argument("--no_bar", "-nb", action="store_true", help="Disable tqdm progress bar display")

    args = parser.parse_args()

    results = evaluate_obb(
        gt_dir=args.gt_dir,
        pred_dir=args.pred_dir,
        excluded_classes=args.excluded_classes,
        iou_threshold=args.iou_threshold,
        class_agnostic=args.class_agnostic,
        exclude_edge_cases=args.exclude_edge_cases,
        edge_tolerance=args.edge_tolerance,
        img_width=args.img_width,
        img_height=args.img_height,
        debug=args.debug,
        no_bar=args.no_bar,
    )

    print_results(results, args.map_path)
