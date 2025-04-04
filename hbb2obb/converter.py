# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np
from ultralytics import SAM, FastSAM

from hbb2obb.utils import Annotations, get_hbb_dir


def hbb2obb(
    img_path: Path,
    hbb_dir: Path = None,
    sam_models: Union[str, List[str]] = "sam_b.pt",
    imgsz: int = 1280,
    scale_factors: Union[float, Tuple[float, float], List[float]] = 0.05,
    opening_kernel_percentage: float = 0.15,
    save_img: bool = False,
    viz_dir: Path = None,
    show_hbb: bool = True,
    show_masks: bool = True,
    show_segments: bool = True,
    show_obb: bool = True,
    show_labels: bool = True,
    model_kwargs: Dict[str, Any] = None,
) -> np.ndarray:
    """
    Convert HBB to OBB annotations using multiple SAM models and aggregating the masks by majority vote.

    Args:
        img_path: Path to the image
        hbb_dir: Directory containing HBB annotations
        sam_models: Name(s) of SAM model(s) to use. Multiple models can be specified to average results
        imgsz: Image size for SAM model inference
        scale_factors: Factor(s) to scale HBB bounding boxes.
                     If single value: same factor for both dimensions.
                     If two values: first for shorter side, second for longer side
        opening_kernel_percentage: Percentage of mask's smaller dimension for morphological opening kernel (0 to disable)
        save_img: Save visualization images
        viz_dir: Directory to save visualization images
        show_hbb: Show horizontal bounding boxes
        show_masks: Show segmentation masks
        show_segments: Show segmentation contours
        show_obb: Show oriented bounding boxes
        show_labels: Show class labels
        model_kwargs: Additional keyword arguments for the SAM model


   as a numpy array     OBB annotations as a numpy array
    """
    hbb_dir = get_hbb_dir(img_path, hbb_dir)

    # Read the image
    img = cv2.imread(str(img_path))

    # Load HBB annotations and scale them
    annotations = Annotations(hbb_dir / (img_path.stem + ".txt"), img)
    bbox_prompts = scale_bounding_boxes(annotations, scale_factors)

    # Convert single model to list for consistent processing
    if isinstance(sam_models, str):
        sam_models = [sam_models]

    if model_kwargs is None:
        model_kwargs = {}
    masks_all_models = []

    # Run each model and collect results
    for model_name in sam_models:
        model_path = Path('models') / (model_name if model_name.endswith(".pt") else f"{model_name}.pt")
        if "FastSAM" in model_name:
            model = FastSAM(model_path)
        else:
            model = SAM(model_path)

        # Run inference with the model
        results = model(
            img,
            bboxes=bbox_prompts[:, 1:],
            retina_masks=True,
            exist_ok=True,
            verbose=False,
            imgsz=imgsz,
            **model_kwargs,
        )

        result = results[0]
        if result.masks is not None:
            masks = result.masks.cpu().numpy()
            masks_all_models.append(masks.data)
        else:
            print(f"Warning: Model {model_name} produced no masks for {img_path.name}")

    # Convert segmentation masks within HBBs to OBB annotations
    obb_annotations, aggregated_masks, contours = create_obb_annotations_multi_model(
        bbox_prompts, masks_all_models, opening_kernel_percentage
    )

    # Save visualization images if enabled
    if save_img:
        visualize_obb_annotations(
            img,
            bbox_prompts,
            aggregated_masks,
            contours,
            obb_annotations,
            viz_dir,
            img_path,
            show_hbb=show_hbb,
            show_masks=show_masks,
            show_segments=show_segments,
            show_obb=show_obb,
            show_labels=show_labels,
        )

    return obb_annotations


def create_obb_annotations_multi_model(
    hbb_boxes: np.ndarray, masks_all_models: List[np.ndarray], opening_kernel_percentage: float
) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
    """
    Convert segmentation masks from multiple SAM models inside the HBBs to OBB an
    using majority voting for mask aggregation and return aggregated masks and contoursnotations
    using majority voting for mask aggregation and return aggregated masks and contours.

    Args:
        hbb_boxes: HBB annotations as numpy array
        masks_all_models: List of masks from different SAM models
        opening_kernel_percentage: Percentage of mask size for morphological opening kernel (0 to disable)

    Returns:
        Tuple containing:
        - List of OBB annotations
        - List of aggregated and HBB-cropped masks
        - List of contours
    """
    obb_annotations = []
    aggregated_masks = []
    contours = []

    for hbb_box in hbb_boxes:
        label, xmin, ymin, xmax, ymax = hbb_box

        # Convert to integers for mask indexing
        x_min, y_min = max(0, int(xmin)), max(0, int(ymin))
        x_max, y_max = int(xmax), int(ymax)

        # Find the best mask for each model
        best_hbb_masks = []

        for masks in masks_all_models:
            # Find the mask with maximum overlap with the bounding box
            best_model_mask = None
            max_overlap = 0

            for mask in masks:
                # Calculate overlap between mask and bounding box
                overlap = mask[y_min : y_max + 1, x_min : x_max + 1].sum()
                if overlap > max_overlap:
                    best_model_mask = mask.copy()
                    max_overlap = overlap

            # If a valid mask was found, add it to the list
            if best_model_mask is not None and max_overlap > 0:
                best_hbb_masks.append(best_model_mask)

        # If no valid masks were found, use the HBB as OBB
        if not best_hbb_masks:
            box_points = [x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max]
            obb_annotations.append([int(label), *box_points])
            aggregated_masks.append(None)
            contours.append(None)
            continue

        # Aggregate masks using majority voting
        aggregated_hbb_mask = aggregate_masks_by_majority_vote(best_hbb_masks)

        # Constrain mask to image dimensions
        mask_height, mask_width = aggregated_hbb_mask.shape
        x_min_c = max(0, x_min)
        y_min_c = max(0, y_min)
        x_max_c = min(mask_width - 1, x_max)
        y_max_c = min(mask_height - 1, y_max)

        # Crop mask to bounding box
        aggregated_hbb_mask_cropped = aggregated_hbb_mask.copy()
        aggregated_hbb_mask_cropped[:, : x_min_c + 1] = False
        aggregated_hbb_mask_cropped[:, x_max_c:] = False
        aggregated_hbb_mask_cropped[: y_min_c + 1, :] = False
        aggregated_hbb_mask_cropped[y_max_c:, :] = False

        # Apply morphological opening to remove small objects / thin protrusions
        aggregated_hbb_mask_final = apply_morphological_opening(aggregated_hbb_mask_cropped, opening_kernel_percentage)

        # Store the final mask
        aggregated_masks.append(aggregated_hbb_mask_final)

        # Find contours and minimum area rectangle
        hbb_contours, _ = cv2.findContours(
            aggregated_hbb_mask_final.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter valid contours based on shape heuristics
        valid_hbb_contours = [c for c in hbb_contours if is_valid_contour(c, hbb_area=(xmax - xmin) * (ymax - ymin))]

        # Fall back to original HBB if no valid contours found
        if not valid_hbb_contours:
            box_points = [x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max]
            obb_annotations.append([int(label), *box_points])
            contours.append(None)
            continue

        # Choose largest valid contour
        largest_hbb_contour = max(valid_hbb_contours, key=cv2.contourArea)
        contours.append(largest_hbb_contour)

        # Compute OBB
        rect = cv2.minAreaRect(largest_hbb_contour)
        box_points = cv2.boxPoints(rect).flatten().astype(np.int32)
        obb_annotations.append([int(label), *box_points])

    return (np.array(obb_annotations) if obb_annotations else np.array([]), aggregated_masks, contours)


def scale_bounding_boxes(
    annotations: Annotations, factors: Union[float, Tuple[float, float], List[float]]
) -> np.ndarray:
    """
    Scale bounding boxes according to specified factors while ensuring they stay within image dimensions.

    Args:
        annotations: Annotations object containing bounding boxes and image dimensions
        factors: Scale factor(s).
                If a single value, both dimensions are scaled by the same factor.
                If two values, the first applies to the shorter side and the second to the longer side.

    Returns:
        Scaled bounding boxes
    """
    boxes = annotations.hbb_xyxy
    width, height = annotations.img_shape

    # Convert factors to a tuple with two values
    if isinstance(factors, (list, tuple)):
        if len(factors) > 2:
            short_factor, long_factor = factors[0], factors[1]
            print(f"Warning: More than two scale factors provided. Using the first two: {short_factor}, {long_factor}")
        elif len(factors) == 2:
            short_factor, long_factor = factors[0], factors[1]
        elif len(factors) == 1:
            short_factor = long_factor = factors[0]
        else:
            short_factor = long_factor = 0.05
    else:
        short_factor = long_factor = float(factors)

    scaled_bounding_boxes = []
    for box in boxes:
        label, x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1

        # Determine which factor to use for each dimension
        if w < h:
            w_factor, h_factor = short_factor, long_factor
        else:
            w_factor, h_factor = long_factor, short_factor

        # Apply the appropriate factors to each side
        x1 = max(0, x1 - w * w_factor)
        y1 = max(0, y1 - h * h_factor)
        x2 = min(width - 1, x2 + w * w_factor)
        y2 = min(height - 1, y2 + h * h_factor)

        scaled_bounding_boxes.append([label, x1, y1, x2, y2])

    return np.array(scaled_bounding_boxes)


def aggregate_masks_by_majority_vote(masks: List[np.ndarray]) -> np.ndarray:
    """
    Aggregate multiple masks using majority voting.
    For each pixel, it's kept if it's confirmed by the majority of models.

    Args:
        masks: List of boolean masks to aggregate

    Returns:
        Aggregated boolean mask
    """
    # Stack masks along a new axis
    stacked_masks = np.stack(masks, axis=0)

    # Determine threshold for majority
    threshold = len(masks) // 2 + 1

    # Apply majority voting: sum across models and threshold
    aggregated_mask = np.sum(stacked_masks, axis=0) >= threshold

    return aggregated_mask


def is_valid_contour(
    contour: np.ndarray, hbb_area: float, min_extent: float = 0.2, min_area_ratio: float = 0.1
) -> bool:
    """
    Determine if a given contour is valid.

    Args:
        contour: Contour to evaluate
        hbb_area: Area of the bounding box used for the SAM prompt
        min_extent: Minimum extent (contour area / bounding rect area)
        min_area_ratio: Minimum ratio of contour area to HBB area

    Returns:
        True if the contour is valid
    """
    if contour is None or len(contour) < 4:
        return False

    contour_area = cv2.contourArea(contour)
    if contour_area < min_area_ratio * hbb_area:
        return False

    w, h = cv2.boundingRect(contour)[2:4]
    rect_area = w * h
    if rect_area == 0:
        return False

    extent = contour_area / rect_area
    if extent < min_extent:
        return False

    return True


def visualize_obb_annotations(
    img: np.ndarray,
    bbox_prompts: np.ndarray,
    aggregated_masks: List[np.ndarray],
    contours: List[np.ndarray],
    obb_annotations: np.ndarray,
    viz_dir: Path,
    img_path: Path,
    show_hbb: bool = True,
    show_masks: bool = True,
    show_segments: bool = True,
    show_obb: bool = True,
    show_labels: bool = True,
):
    """
    Visualize HBB, OBB, and segmentation masks on the image based on visualization flags.
    """
    # Draw HBBs if enabled
    if show_hbb:
        for hbb in bbox_prompts:
            _, x1, y1, x2, y2 = hbb
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

    # Draw segmentation masks if enabled
    if show_masks and aggregated_masks:
        for mask in aggregated_masks:
            if mask is not None:
                green_mask = np.zeros_like(img)
                green_mask[:, :, 1] = 255 * mask.astype(np.uint8)
                img = cv2.addWeighted(img, 1, green_mask, 0.5, 0)

    # Draw segmentation contours if enabled
    if show_segments and contours:
        for segment in contours:
            if segment is not None:
                cv2.drawContours(img, [segment], 0, (0, 0, 255), 2)

    # Draw OBBs and labels if enabled
    if show_obb or show_labels:
        for obb in obb_annotations:
            label, x1, y1, x2, y2, x3, y3, x4, y4 = obb

            # Draw OBB polygons
            if show_obb:
                cv2.polylines(
                    img, [np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], np.int32)], True, (0, 255, 255), 3
                )

            # Draw class labels
            if show_labels:
                text = str(int(label))
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]

                # Ensure text is inside the image
                text_x = max(int(x1), 5)
                text_y = max(int(y1), text_size[1] + 5)

                # Add background rectangle
                cv2.rectangle(
                    img,
                    (text_x - 2, text_y - text_size[1] - 2),
                    (text_x + text_size[0] + 2, text_y + 2),
                    (0, 255, 255),
                    -1,
                )
                cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    if viz_dir is None:
        viz_dir = img_path.parent.parent / "labels_obb"
    viz_dir.mkdir(exist_ok=True, parents=True)

    cv2.imwrite(str(viz_dir / img_path.name), img)
    print(f"Saved image with OBB annotations: {viz_dir / img_path.name}")


def save_obb_annotations(obb_annotations: np.ndarray, obb_dir: Path, img_path: Path):
    """
    Save OBB annotations to a text file.
    """
    if obb_dir is None:
        obb_dir = img_path.parent.parent / "labels_obb"
    obb_dir.mkdir(exist_ok=True, parents=True)
    save_filepath = obb_dir / (img_path.stem + ".txt")

    with open(save_filepath, "w", encoding="utf-8") as f:
        for obb in obb_annotations:
            label, x1, y1, x2, y2, x3, y3, x4, y4 = map(int, obb)
            f.write(f"{label} {x1} {y1} {x2} {y2} {x3} {y3} {x4} {y4}\n")


def apply_morphological_opening(mask: np.ndarray, kernel_percentage: float) -> np.ndarray:
    """
    Applies morphological opening to a boolean mask to remove small objects / thin protrusions.

    Args:
        mask: The input boolean mask (True for foreground) or None.
        kernel_percentage: The percentage of the mask's smaller dimension to use as kernel size.
                        If kernel_percentage <= 0 or mask is None, the original mask is returned unchanged.

    Returns:
        The processed boolean mask, or None if the input was None.
    """
    # Return immediately if opening is disabled or the mask is invalid/None
    if kernel_percentage <= 0 or mask is None or mask.size == 0 or not mask.any():
        return mask

    # Ensure mask is boolean before converting to uint8
    if mask.dtype != bool:
        mask = mask.astype(bool)

    # Calculate kernel size as a percentage of the smaller dimension of the bounding box
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask

    # Get the bounding box of the largest contour
    w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))[2:4]

    # Calculate the smaller dimension of the bounding box
    smaller_dim = min(w, h)

    # Calculate the kernel size based on the smaller dimension
    kernel_size = max(1, int(smaller_dim * kernel_percentage))

    # Ensure kernel size is odd
    kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size

    # Ensure kernel size is at least 3x3
    kernel_size = max(3, kernel_size)

    # Create the kernel
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Convert boolean mask to uint8 (0 and 255) for OpenCV function
    mask_uint8 = mask.astype(np.uint8) * 255

    # Apply morphological opening
    opened_mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)

    # Convert back to boolean mask
    processed_mask = opened_mask_uint8 > 0

    return processed_mask
