# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import tqdm
from shapely.geometry import Polygon

from hbb2obb.utils import load_label_map


def evaluate_obb(
    gt_dir,
    pred_dir,
    excluded_classes=None,
    iou_threshold=0.1,
    class_agnostic=False,
    exclude_edge_cases=False,
    edge_tolerance=1,
    img_width=None,
    img_height=None,
    debug=False,
    no_bar=False,
):
    """
    Evaluate OBB predictions against ground truth.

    Args:
        gt_dir (Path): Directory containing ground truth OBB annotations.
        pred_dir (Path): Directory containing predicted/converted OBB annotations.
        excluded_classes (list): List of class IDs to exclude from evaluation.
        iou_threshold (float): Minimum IoU for considering a match.
        class_agnostic (bool): Match boxes regardless of class labels if True.
        exclude_edge_cases (bool): Exclude boxes that touch image edges if True.
        edge_tolerance (int): Tolerance in pixels for detecting boxes at image edges.
        img_width (int): Image width for edge case detection.
        img_height (int): Image height for edge case detection.
        debug (bool): Print detailed matching information if True.
        no_bar (bool): Disable progress bar if True.

    Returns:
        dict: Evaluation results.
    """
    if not gt_dir.exists() or not gt_dir.is_dir():
        print(f"Error: Ground truth directory not found: {gt_dir}")
        sys.exit(1)

    if not pred_dir.exists() or not pred_dir.is_dir():
        print(f"Error: Prediction directory not found: {pred_dir}")
        sys.exit(1)

    if excluded_classes is None:
        excluded_classes = []

    if exclude_edge_cases and (img_width is None or img_height is None):
        print("Error: To exclude edge cases, image dimensions (--img_width and --img_height) must be provided.")
        sys.exit(1)

    gt_files = sorted(Path(gt_dir).glob('*.txt'))

    # Store results
    all_matches = []
    all_ious = []
    total_edge_cases = 0
    total_unmatched_gt = 0
    total_unmatched_pred = 0
    class_stats = defaultdict(lambda: {'matches': 0, 'gt_total': 0, 'pred_total': 0, 'iou_sum': 0, 'iou_values': []})

    # For class-agnostic evaluation, track matches where classes differ
    cross_class_matches = 0

    for gt_file in tqdm.tqdm(gt_files, desc="Evaluating files", leave=True, disable=no_bar):
        pred_file = Path(pred_dir) / gt_file.name

        if not pred_file.exists():
            print(f"Warning: No prediction file found for {gt_file.name}")
            continue

        gt_boxes = parse_obb_file(gt_file)
        pred_boxes = parse_obb_file(pred_file)

        # Filter out excluded classes
        gt_boxes = [box for box in gt_boxes if box['label'] not in excluded_classes]
        pred_boxes = [box for box in pred_boxes if box['label'] not in excluded_classes]

        # Filter out edge cases if requested
        if exclude_edge_cases:
            original_gt_count = len(gt_boxes)
            gt_boxes = [box for box in gt_boxes if not is_edge_box(box, img_width, img_height, edge_tolerance)]
            total_edge_cases += original_gt_count - len(gt_boxes)

        # Match boxes
        matches, unmatched_gt, unmatched_pred = match_boxes(gt_boxes, pred_boxes, iou_threshold, class_agnostic)

        # Update counts of unmatched boxes
        total_unmatched_gt += len(unmatched_gt)
        total_unmatched_pred += len(unmatched_pred)

        # Debug information
        if debug and (unmatched_gt or unmatched_pred):
            filename = gt_file.name
            print(f"\n--- File: {filename} ---")
            print(f"GT boxes: {len(gt_boxes)}, Pred boxes: {len(pred_boxes)}")
            print(f"Matched: {len(matches)}, Unmatched GT: {len(unmatched_gt)}, Unmatched Pred: {len(unmatched_pred)}")

            # Print examples of unmatched GT boxes
            if unmatched_gt:
                print("\nUnmatched GT examples:")
                for box in unmatched_gt[:3]:  # Show up to 3 examples
                    print(f"  {format_bbox(box)}")
                if len(unmatched_gt) > 3:
                    print(f"  ... and {len(unmatched_gt) - 3} more")

            # Print examples of unmatched Pred boxes
            if unmatched_pred:
                print("\nUnmatched Pred examples:")
                for box in unmatched_pred[:3]:  # Show up to 3 examples
                    print(f"  {format_bbox(box)}")
                if len(unmatched_pred) > 3:
                    print(f"  ... and {len(unmatched_pred) - 3} more")

        # Update overall statistics
        all_matches.extend(matches)

        # Collect IoU values
        for gt_box, pred_box, iou in matches:
            all_ious.append(iou)

            # Count cross-class matches in class-agnostic mode
            if class_agnostic and gt_box['label'] != pred_box['label']:
                cross_class_matches += 1

        # Update per-class statistics
        for gt_box, _, iou in matches:
            label = gt_box['label']
            class_stats[label]['matches'] += 1
            class_stats[label]['iou_sum'] += iou
            class_stats[label]['iou_values'].append(iou)

        # Count total GT and predicted/converted boxes per class
        for box in gt_boxes:
            class_stats[box['label']]['gt_total'] += 1

        for box in pred_boxes:
            class_stats[box['label']]['pred_total'] += 1

    # Calculate overall metrics
    total_matches = len(all_matches)
    total_gt = sum(stats['gt_total'] for stats in class_stats.values())
    total_pred = sum(stats['pred_total'] for stats in class_stats.values())

    # Calculate IoU statistics
    avg_iou, std_iou = (0, 0) if not all_ious else (np.mean(all_ious), np.std(all_ious))

    # Exclude edge cases from total GT count if requested
    if exclude_edge_cases:
        total_gt -= total_edge_cases

    results = {
        'total_matches': total_matches,
        'total_gt': total_gt,
        'total_pred': total_pred,
        'avg_iou': avg_iou,
        'std_iou': std_iou,
        'class_stats': class_stats,
        'total_edge_cases': total_edge_cases,
        'exclude_edge_cases': exclude_edge_cases,
        'excluded_classes': excluded_classes,
        'class_agnostic': class_agnostic,
        'cross_class_matches': cross_class_matches,
        'total_unmatched_gt': total_unmatched_gt,
        'total_unmatched_pred': total_unmatched_pred,
    }

    return results


def print_results(results: dict, map_path: Path = None) -> None:
    """
    Print evaluation results.

    Args:
        results (dict): Evaluation results dictionary.
        map_path (Path): Path to label map YAML file (optional).
    """
    label_map = load_label_map(map_path)

    print("\n=== Overall Results ===")

    if results['class_agnostic']:
        print("Evaluation Mode: Class-Agnostic (matches boxes regardless of class labels)")
        if results['cross_class_matches'] > 0:
            cross_pct = (results['cross_class_matches'] / results['total_matches']) * 100
            print(f"Cross-Class Matches: {results['cross_class_matches']} ({cross_pct:.2f}% of total matches)")
    else:
        print("Evaluation Mode: Class-Specific (matches only boxes with same class label)")

    if results['exclude_edge_cases']:
        print(f"Total Ground Truth Boxes (excluding edge cases): {results['total_gt']}")
    else:
        print(f"Total Ground Truth Boxes: {results['total_gt']}")

    print(f"Total Predicted/Converted Boxes: {results['total_pred']}")
    print(f"Total Matched Boxes: {results['total_matches']}")
    print(f"Total Unmatched GT Boxes: {results['total_unmatched_gt']}")
    print(f"Total Unmatched Pred Boxes: {results['total_unmatched_pred']}")

    if results['exclude_edge_cases']:
        print(f"Total Edge Cases Excluded: {results['total_edge_cases']}")

    print(f"Average IoU: {results['avg_iou']:.4f} ± {results['std_iou']:.4f}")

    if results['excluded_classes']:
        if label_map:
            excluded_names = [f"{label_map.get(cls, str(cls))} ({cls})" for cls in results['excluded_classes']]
            excluded_classes_str = ", ".join(excluded_names)
        else:
            excluded_classes_str = ", ".join(str(cls) for cls in results['excluded_classes'])
        print(f"Excluded Classes: {excluded_classes_str}")

    if not results['class_agnostic']:
        print("\n=== Results by Class ===")

        classes = sorted(results['class_stats'].keys())

        table_data = []
        for class_id in classes:
            stats = results['class_stats'][class_id]

            if stats['matches'] == 0:
                avg_class_iou, std_class_iou = 0, 0
            else:
                avg_class_iou = stats['iou_sum'] / stats['matches']
                std_class_iou = np.std(stats['iou_values']) if len(stats['iou_values']) > 1 else 0

            class_name = label_map.get(class_id, str(class_id)) if label_map else str(class_id)

            row = [
                class_name,
                stats['gt_total'],
                stats['pred_total'],
                stats['matches'],
                f"{avg_class_iou:.4f} ± {std_class_iou:.4f}",
            ]

            table_data.append(row)

        headers = ["Class", "GT", "Pred", "Matches", "IoU (mean ± std)"]
        df = pd.DataFrame(table_data, columns=headers)
        print(df.to_string(index=False))


def parse_obb_file(file_path):
    """
    Parse OBB annotation file.

    Args:
        file_path (Path): Path to OBB annotation file.

    Returns:
        list: List of dictionaries with 'label' and 'polygon' fields.
    """
    boxes = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) != 9:  # label + 8 coordinates
                continue

            label = int(parts[0])
            coordinates = list(map(float, parts[1:]))

            # Create polygon from coordinates
            points = [
                (coordinates[0], coordinates[1]),  # x1, y1
                (coordinates[2], coordinates[3]),  # x2, y2
                (coordinates[4], coordinates[5]),  # x3, y3
                (coordinates[6], coordinates[7]),  # x4, y4
            ]

            polygon = Polygon(points)
            boxes.append({'label': label, 'polygon': polygon, 'points': points})

    return boxes


def calculate_obb_iou(poly1, poly2):
    """
    Calculate IoU between two oriented bounding boxes represented as Shapely Polygons.

    Args:
        poly1 (Polygon): First oriented bounding box.
        poly2 (Polygon): Second oriented bounding box.

    Returns:
        float: Intersection over Union (IoU) between the two boxes.
    """
    if not poly1.is_valid or not poly2.is_valid:
        return 0.0

    try:
        intersection = poly1.intersection(poly2).area
    except (ValueError, AttributeError):
        return 0.0

    union = poly1.area + poly2.area - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def match_boxes(gt_boxes, pred_boxes, iou_threshold=0.1, class_agnostic=False):
    """
    Match predicted/converted boxes to ground truth boxes.

    Args:
        gt_boxes (list): List of ground truth boxes.
        pred_boxes (list): List of predicted/converted boxes.
        iou_threshold (float): Minimum IoU for considering a match.
        class_agnostic (bool): Match boxes regardless of class labels if True.

    Returns:
        tuple: List of matches, unmatched ground truth boxes, unmatched predicted boxes.
    """
    matches = []
    unmatched_gt = []

    # For each GT box, find the best prediction
    for gt_box in gt_boxes:
        best_iou = 0.0
        best_pred = None

        for pred_box in pred_boxes:
            # Skip different classes if class-specific evaluation
            if not class_agnostic and gt_box['label'] != pred_box['label']:
                continue

            iou = calculate_obb_iou(gt_box['polygon'], pred_box['polygon'])

            if iou > best_iou:
                best_iou = iou
                best_pred = pred_box

        if best_iou > iou_threshold and best_pred is not None:
            matches.append((gt_box, best_pred, best_iou))
        else:
            unmatched_gt.append(gt_box)

    matched_preds = [match[1] for match in matches]
    unmatched_pred = [box for box in pred_boxes if box not in matched_preds]

    return matches, unmatched_gt, unmatched_pred


def is_edge_box(box, img_width, img_height, tolerance=1):
    """
    Determine if a box touches the edge of an image (within tolerance pixels).

    Args:
        box (dict): Bounding box dictionary with 'points' field.
        img_width (int): Image width.
        img_height (int): Image height.
        tolerance (int): Tolerance in pixels for detecting boxes at image edges.

    Returns:
        bool: True if the box touches an image edge, False otherwise.
    """
    for x, y in box['points']:
        if x <= tolerance or x >= img_width - tolerance or y <= tolerance or y >= img_height - tolerance:
            return True
    return False


def format_bbox(box):
    """
    Format a bounding box for printing in debug mode.

    Args:
        box (dict): Bounding box dictionary.

    Returns:
        str: Formatted bounding box string.
    """
    points_str = ' '.join([f"({x:.1f},{y:.1f})" for x, y in box['points']])
    return f"Class {box['label']}: {points_str}"
