# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np
from ultralytics import SAM, FastSAM

from hbb2obb.utils import Annotations, get_hbb_dir

# Cache of loaded SAM/FastSAM model instances, keyed by the resolved weights path, so that
# repeated hbb2obb() calls (e.g. one per image in a directory) reuse an already-loaded model
# instead of reconstructing it from disk on every call.
_MODEL_CACHE: Dict[str, Any] = {}


def sam_checkpoint_path(model_name: str, models_dir: Path = Path('models')) -> Path:
    """
    Where a SAM/FastSAM checkpoint lives for a given model name: ``<models_dir>/<name>.pt``,
    unless a suffix is already given. The one place this rule is written, so provenance can
    report the exact path a run resolved rather than a second copy of the same formula.
    """
    return models_dir / (model_name if model_name.endswith(".pt") else f"{model_name}.pt")


def load_sam_model(model_name: str):
    """
    Load a SAM/FastSAM model by name, reusing a cached instance when one is already loaded.

    Args:
        model_name: Model name (e.g. "sam_b", "sam2.1_b", "FastSAM-s"), with or without a ".pt" suffix.

    Returns:
        The loaded ultralytics SAM or FastSAM model instance.
    """
    model_path = sam_checkpoint_path(model_name)
    cache_key = str(model_path)
    if cache_key not in _MODEL_CACHE:
        _MODEL_CACHE[cache_key] = FastSAM(model_path) if "FastSAM" in model_name else SAM(model_path)
    return _MODEL_CACHE[cache_key]


def clear_model_cache() -> None:
    """Clear the in-process SAM/FastSAM model cache, releasing the associated memory."""
    _MODEL_CACHE.clear()


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
    show_confidence: bool = False,
    model_kwargs: Dict[str, Any] = None,
    return_confidence: bool = False,
    confidence_source: str = "conversion",
    return_contours: bool = False,
) -> Union[np.ndarray, Tuple]:
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
        show_confidence: Print the per-OBB confidence score in the visualization
        model_kwargs: Additional keyword arguments for the SAM model
        return_confidence: If True, also return the per-OBB confidence scores
        confidence_source: Which score the returned confidences carry: 'conversion' (the
                     heuristic conversion-quality score), 'detector' (the confidence read
                     from the HBB input file), or 'combined' (their product)
        return_contours: If True, also return the per-object segmentation contours, in absolute
                     image pixel coordinates, with None where the OBB is a fallback HBB

    Returns:
        OBB annotations as a numpy array, with the requested extras appended in a tuple:

        - neither flag: obb_annotations
        - return_confidence: (obb_annotations, confidences)
        - return_contours: (obb_annotations, contours)
        - both flags: (obb_annotations, confidences, contours)

        The confidences and contours lists are always the same length as obb_annotations.
    """
    hbb_dir = get_hbb_dir(img_path, hbb_dir)

    # Read the image
    img = cv2.imread(str(img_path))

    # Load HBB annotations and scale them
    annotations = Annotations(hbb_dir / (img_path.stem + ".txt"), img)
    bbox_prompts = scale_bounding_boxes(annotations, scale_factors)

    # Nothing to convert: SAM cannot be prompted with zero boxes
    if len(bbox_prompts) == 0:
        return pack_results(np.array([]), [], [], return_confidence, return_contours)

    # Convert single model to list for consistent processing
    if isinstance(sam_models, str):
        sam_models = [sam_models]

    if model_kwargs is None:
        model_kwargs = {}
    masks_all_models = []

    # Run each model and collect results
    for model_name in sam_models:
        model = load_sam_model(model_name)

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
    obb_annotations, aggregated_masks, contours, confidences = create_obb_annotations_multi_model(
        bbox_prompts, masks_all_models, opening_kernel_percentage
    )

    # Blend in the detector confidence from the HBB file, but only when a caller actually asked
    # for it: --confidence_source is documented as a no-op unless --save_confidence or
    # --show_confidence is used, and resolving it unconditionally would also print its "missing
    # confidence column" warning for runs that never look at the confidence at all.
    if return_confidence or show_confidence:
        confidences = resolve_confidences(confidences, annotations.hbb_scores, confidence_source, img_path)

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
            confidences=confidences,
            show_confidence=show_confidence,
        )

    return pack_results(obb_annotations, confidences, contours, return_confidence, return_contours)


def pack_results(
    obb_annotations: np.ndarray,
    confidences: List[float],
    contours: List[np.ndarray],
    return_confidence: bool,
    return_contours: bool,
) -> Union[np.ndarray, Tuple]:
    """
    Append the optionally requested extras to the OBB annotations, in a fixed order
    (confidences before contours). Pair with `unpack_results()`, which mirrors this same
    order, rather than reconstructing it by hand at the call site.
    """
    if not return_confidence and not return_contours:
        return obb_annotations

    extras = []
    if return_confidence:
        extras.append(confidences)
    if return_contours:
        extras.append(contours)

    return (obb_annotations, *extras)


def unpack_results(
    result: Union[np.ndarray, Tuple], return_confidence: bool, return_contours: bool
) -> Tuple[np.ndarray, Union[List[float], None], Union[List[np.ndarray], None]]:
    """
    Inverse of `pack_results()`: recover (obb_annotations, confidences, contours) from a
    `hbb2obb()` return value, given the same `return_confidence`/`return_contours` flags that
    were passed to produce it. `confidences`/`contours` are None for extras that were not
    requested, rather than the caller having to know pack_results()'s append order.
    """
    values = list(result) if isinstance(result, tuple) else [result]
    obb_annotations = values.pop(0)
    confidences = values.pop(0) if return_confidence else None
    contours = values.pop(0) if return_contours else None
    return obb_annotations, confidences, contours


def resolve_confidences(
    conversion_scores: List[float],
    detector_scores: np.ndarray,
    confidence_source: str = "conversion",
    img_path: Path = None,
) -> List[float]:
    """
    Pick the per-OBB confidence score to report, given both available sources.

    Args:
        conversion_scores: Heuristic conversion-quality scores from create_obb_annotations_multi_model()
        detector_scores: Per-HBB detector confidences parsed from the input file, same length as
                         conversion_scores; nan where the input line carried no confidence column
        confidence_source: 'conversion' (default), 'detector', or 'combined' (the product of both)
        img_path: Image being processed, used only to name the file in the fallback warning

    Raises:
        ValueError: If confidence_source is unsupported, or if detector_scores and
                    conversion_scores have different lengths (they must be 1:1, one per HBB)

    Returns:
        List of confidence scores in [0, 1], one per OBB. Where a detector score was requested
        but is missing (nan), the conversion score is used instead and a single warning is printed.
    """
    if confidence_source == "conversion":
        return conversion_scores

    if confidence_source not in ("detector", "combined"):
        raise ValueError(f"Unsupported confidence_source: {confidence_source}")

    if len(detector_scores) != len(conversion_scores):
        raise ValueError(
            f"conversion_scores and detector_scores must be the same length (one per HBB), got "
            f"{len(conversion_scores)} and {len(detector_scores)}"
        )

    resolved = []
    missing = 0
    for i, conversion in enumerate(conversion_scores):
        detector = float(detector_scores[i])
        if np.isnan(detector):
            # No confidence column on that input line: fall back to the conversion score
            missing += 1
            resolved.append(conversion)
        elif confidence_source == "detector":
            resolved.append(detector)
        else:
            resolved.append(detector * conversion)

    if missing:
        name = img_path.name if img_path is not None else "input"
        print(
            f"Warning: {missing}/{len(conversion_scores)} HBBs for {name} carry no confidence column; "
            f"using the conversion score for those boxes."
        )

    return resolved


def create_obb_annotations_multi_model(
    hbb_boxes: np.ndarray, masks_all_models: List[np.ndarray], opening_kernel_percentage: float
) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray], List[float]]:
    """
    Convert segmentation masks from multiple SAM models inside the HBBs to OBB annotations
    using majority voting for mask aggregation, and return the aggregated masks, contours,
    and a per-OBB confidence score.

    The confidence is a heuristic quality score in [0, 1]: fallback boxes (where no usable
    mask/contour was found and the original HBB is emitted) score 0.0, while genuine OBBs
    score ``rectangularity * consensus`` where ``rectangularity`` measures how tightly the
    fitted rotated box wraps the segmented mask and ``consensus`` measures agreement across
    the SAM model ensemble (1.0 for a single model).

    Args:
        hbb_boxes: HBB annotations as numpy array
        masks_all_models: List of masks from different SAM models
        opening_kernel_percentage: Percentage of mask size for morphological opening kernel (0 to disable)

    Returns:
        Tuple containing:
        - List of OBB annotations
        - List of aggregated and HBB-cropped masks
        - List of contours
        - List of per-OBB confidence scores in [0, 1]
    """
    obb_annotations = []
    aggregated_masks = []
    contours = []
    confidences = []

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

        # If no valid masks were found, use the HBB as OBB (fallback -> zero confidence)
        if not best_hbb_masks:
            box_points = [x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max]
            obb_annotations.append([int(label), *box_points])
            aggregated_masks.append(None)
            contours.append(None)
            confidences.append(0.0)
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

        # Fall back to original HBB if no valid contours found (fallback -> zero confidence)
        if not valid_hbb_contours:
            box_points = [x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max]
            obb_annotations.append([int(label), *box_points])
            contours.append(None)
            confidences.append(0.0)
            continue

        # Choose largest valid contour
        largest_hbb_contour = max(valid_hbb_contours, key=cv2.contourArea)
        contours.append(largest_hbb_contour)

        # Compute OBB
        rect = cv2.minAreaRect(largest_hbb_contour)
        box_points = cv2.boxPoints(rect).flatten().astype(np.int32)
        obb_annotations.append([int(label), *box_points])

        # Confidence = rectangularity (fit of the rotated box to the mask) * ensemble consensus.
        # Rectangularity: contour area over the min-area rotated rect area, clipped to [0, 1].
        rect_area = rect[1][0] * rect[1][1]
        rectangularity = min(cv2.contourArea(largest_hbb_contour) / rect_area, 1.0) if rect_area > 0 else 0.0
        # Consensus: fraction of the per-model union that survived the majority vote (1.0 for one model).
        union_mask = np.logical_or.reduce(best_hbb_masks)
        union_sum = union_mask.sum()
        consensus = aggregated_hbb_mask.sum() / union_sum if union_sum > 0 else 0.0
        confidences.append(float(rectangularity * consensus))

    return (np.array(obb_annotations) if obb_annotations else np.array([]), aggregated_masks, contours, confidences)


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
    confidences: List[float] = None,
    show_confidence: bool = False,
):
    """
    Visualize HBB, OBB, and segmentation masks on the image based on visualization flags.

    When confidences are provided, OBB polygons are colored on a green->red gradient by
    score (fallback boxes score 0.0 and appear red); otherwise the default cyan is used.
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

    # Draw OBBs, class labels, and/or confidence scores if enabled
    if show_obb or show_labels or show_confidence:
        for i, obb in enumerate(obb_annotations):
            label, x1, y1, x2, y2, x3, y3, x4, y4 = obb

            # Color OBBs by confidence (green=high -> red=low/fallback) when scores are available,
            # otherwise use the default cyan.
            if confidences is not None and i < len(confidences):
                c = max(0.0, min(1.0, confidences[i]))
                obb_color = (0, int(255 * c), int(255 * (1 - c)))
            else:
                obb_color = (0, 255, 255)

            # Draw OBB polygons
            if show_obb:
                cv2.polylines(img, [np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], np.int32)], True, obb_color, 3)

            # Compose the text overlay from the class label and/or the confidence score. The two are
            # independent: show_labels controls only the class id, show_confidence only the score.
            text_parts = []
            if show_labels:
                text_parts.append(str(int(label)))
            if show_confidence and confidences is not None and i < len(confidences):
                text_parts.append(f"{confidences[i]:.2f}")

            if text_parts:
                text = " ".join(text_parts)
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]

                # Ensure text is inside the image
                text_x = max(int(x1), 5)
                text_y = max(int(y1), text_size[1] + 5)

                # Add background rectangle
                cv2.rectangle(
                    img,
                    (text_x - 2, text_y - text_size[1] - 2),
                    (text_x + text_size[0] + 2, text_y + 2),
                    obb_color,
                    -1,
                )
                cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    viz_dir = resolve_output_dir(viz_dir, img_path, "labels_obb")

    cv2.imwrite(str(viz_dir / img_path.name), img)
    print(f"Saved image with OBB annotations: {viz_dir / img_path.name}")


def resolve_output_dir(output_dir: Path, img_path: Path, default_subdir: str) -> Path:
    """
    Resolve the directory a per-image annotation/visualization file should be saved to, creating
    it if needed. Defaults to ``img_path.parent.parent / default_subdir`` when ``output_dir`` is
    None, the convention shared by all of hbb2obb's output writers.
    """
    if output_dir is None:
        output_dir = img_path.parent.parent / default_subdir
    output_dir.mkdir(exist_ok=True, parents=True)
    return output_dir


def format_annotation_line(fields: str, confidences: List[float], i: int) -> str:
    """
    Append the per-object confidence score to an annotation line when ``confidences`` is
    provided, as the trailing field written by both ``save_obb_annotations`` and
    ``save_polygon_annotations``.
    """
    if confidences is not None:
        fields += f" {confidences[i]:.4f}"
    return fields


def save_obb_annotations(obb_annotations: np.ndarray, obb_dir: Path, img_path: Path, confidences: List[float] = None):
    """
    Save OBB annotations to a text file.

    Each line is ``class x1 y1 x2 y2 x3 y3 x4 y4``. When ``confidences`` is provided, a 10th
    field with the per-OBB confidence score is appended (``... x4 y4 conf``).
    """
    obb_dir = resolve_output_dir(obb_dir, img_path, "labels_obb")
    save_filepath = obb_dir / (img_path.stem + ".txt")

    with open(save_filepath, "w", encoding="utf-8") as f:
        for i, obb in enumerate(obb_annotations):
            label, x1, y1, x2, y2, x3, y3, x4, y4 = map(int, obb)
            line = f"{label} {x1} {y1} {x2} {y2} {x3} {y3} {x4} {y4}"
            f.write(format_annotation_line(line, confidences, i) + "\n")


def save_polygon_annotations(
    contours: List[np.ndarray],
    obb_annotations: np.ndarray,
    polygon_dir: Path,
    img_path: Path,
    confidences: List[float] = None,
    simplify_epsilon: float = 0.0,
) -> None:
    """
    Save the segmentation polygons to a text file, as a tighter alternative to the OBBs.

    Each line is ``class x1 y1 x2 y2 ... xN yN``, the corners in absolute pixel coordinates,
    the same convention ``save_obb_annotations`` uses. When ``confidences`` is provided, the
    per-object score is appended as a final field, exactly as the OBB writer does.

    Row alignment is guaranteed: the file holds exactly one line per OBB, in the same order,
    so line i here and line i of the OBB file for the same image describe the same object.
    Objects that fell back to the HBB (no usable mask, contour ``None``) are written as the
    four corners of their OBB row rather than skipped, which makes a fallback recognizable
    as a four-point polygon identical to its OBB line.

    Args:
        contours: Per-object contours from hbb2obb(..., return_contours=True), None for fallbacks
        obb_annotations: Matching OBB annotations, supplying the class label and the fallback corners
        polygon_dir: Directory to save the polygon annotations
        img_path: Image the annotations belong to, used to derive the output file name
        confidences: Per-object confidence scores, written as a trailing field
        simplify_epsilon: If > 0, simplify each contour with cv2.approxPolyDP using an epsilon of
                     this fraction of the contour perimeter (cv2.arcLength), so one value behaves
                     consistently across object sizes; 0.005 to 0.02 are typical. The default 0
                     writes the raw contour. Fallback rectangles are never simplified.
    """
    if contours is None or len(contours) != len(obb_annotations):
        raise ValueError(
            f"Expected one contour per OBB, got {len(contours) if contours is not None else None} "
            f"contours for {len(obb_annotations)} OBBs"
        )

    polygon_dir = resolve_output_dir(polygon_dir, img_path, "labels_polygon")
    save_filepath = polygon_dir / (img_path.stem + ".txt")

    with open(save_filepath, "w", encoding="utf-8") as f:
        for i, obb in enumerate(obb_annotations):
            if contours[i] is None:
                # Fallback: reuse the OBB corners so both files stay in agreement
                points = np.array(obb[1:], dtype=np.int32).reshape(-1, 2)
            else:
                points = simplify_contour(contours[i], simplify_epsilon)
            line = f"{int(obb[0])} " + " ".join(f"{int(x)} {int(y)}" for x, y in points)
            f.write(format_annotation_line(line, confidences, i) + "\n")


def simplify_contour(contour: np.ndarray, epsilon: float) -> np.ndarray:
    """
    Reduce a contour to fewer vertices using the Douglas-Peucker algorithm.

    Args:
        contour: Contour as returned by cv2.findContours
        epsilon: Approximation accuracy as a fraction of the contour perimeter (cv2.arcLength),
                 not an absolute pixel distance. Values <= 0 leave the contour untouched.

    Returns:
        The contour points as an (N, 2) array, left unsimplified if the simplification would
        leave fewer than three points.
    """
    points = contour.reshape(-1, 2)
    if epsilon <= 0:
        return points

    simplified = cv2.approxPolyDP(contour, epsilon * cv2.arcLength(contour, True), True).reshape(-1, 2)
    return simplified if len(simplified) >= 3 else points


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
