#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
Convert bounding box annotations from JSON files to YOLO text format (HBB or OBB).

Supports optional coordinate normalization and precision control.

Usage:
    python scripts/json2yolo.py <json_dir> [--map_path MAP_PATH]
                                [--txt_dir TXT_DIR]
                                [--input_format {coco,labelme}]
                                [--normalize]
                                [--precision PRECISION]

Arguments:
    json_dir           Path to the directory containing JSON annotation files (default/labelme mode),
                       or path to a COCO instance JSON file (when using --input_format coco)
    --map_path, -mp    Path to the label map YAML file that maps label IDs to names (default: None)
    --txt_dir, -td     Optional path to output directory for YOLO .txt files (default: json_dir in
                       default/labelme mode; <json_dir_parent>/labels_yolo in COCO mode)
    --input_format, -if
                       Input annotation format: 'coco' for COCO instance JSON (single file, HBB only),
                       'labelme' for LabelMe JSON directory (HBB and OBB). Default: None (same as 'labelme')
    --normalize, -n    Normalize coordinates to [0, 1] using image dimensions
    --precision, -p    Decimal precision for output (default: 0 for unnormalized, 10 for normalized)

Input JSON format (default/labelme mode):
    - Must contain 'imageHeight', 'imageWidth', and a list of 'shapes'
    - Each shape must have a 'label' and a list of 2 (for HBB) or 4 (for OBB) points
    - Bounding box type (HBB or OBB) is automatically determined from the number of points

Examples:
    1) Convert annotations to YOLO format, output in same folder:
       python scripts/json2yolo.py data/labels -mp data/classes.yaml

    2) Convert annotations to normalized YOLO format, output to custom folder:
       python scripts/json2yolo.py data/labels -mp data/classes.yaml --normalize --txt_dir data/labels_yolo

    3) Convert COCO instance JSON annotations to YOLO format:
       python scripts/json2yolo.py data/instances.json --input_format coco --txt_dir data/labels_yolo

Output format:
    - HBB: <label> <xc> <yc> <w> <h>
    - OBB: <label> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4>
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from hbb2obb.utils import load_label_map


def parse_cli_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Convert JSON annotations to YOLO text files")
    parser.add_argument("json_dir", type=Path, help="Directory with JSON annotation files (default/labelme mode), or path to a COCO instance JSON file (when using --input_format coco)")  # noqa: E501
    parser.add_argument("--map_path", "-mp", type=Path, help="Path to label map YAML file that maps label IDs to names (default: None)")  # noqa: E501
    parser.add_argument("--txt_dir", "-td", type=Path, help="Directory for YOLO text files (default: same as JSON dir in default/labelme mode; <json_dir_parent>/labels_yolo in COCO mode)")
    parser.add_argument("--input_format", "-if", type=str, default=None, choices=["coco", "labelme"], help="Input annotation format: 'coco' for COCO instance JSON (single file, HBB only), 'labelme' for LabelMe JSON directory (HBB and OBB). Default: None (same as 'labelme').")  # noqa: E501
    parser.add_argument("--normalize", "-n", action="store_true", help="Normalize coordinates")
    parser.add_argument("--precision", "-p", type=int, help="Decimal precision (default: 0 for unnormalized, 10 for normalized)")  # noqa: E501
    return parser.parse_args()


def process_files(
    json_dir: Path,
    map_path: Path | None = None,
    txt_dir: Path | None = None,
    normalize: bool = False,
    precision: int | None = None,
    input_format: str | None = None,
) -> None:
    """
    Process annotation files and convert them to YOLO text files.
    """
    label_map = load_label_map(map_path, reverse=True) if map_path else None

    if input_format == "coco":
        if txt_dir is None:
            txt_dir = json_dir.parent / "labels_yolo"
        coco2txt(json_dir, txt_dir, label_map, normalize, precision)
        return

    if txt_dir is None:
        txt_dir = json_dir
    txt_dir.mkdir(exist_ok=True, parents=True)
    for json_file in json_dir.glob("*.json"):
        txt_file = txt_dir / (json_file.stem + ".txt")
        json2txt(json_file, txt_file, label_map, normalize, precision)


def detect_coordinate_format(shapes: list, img_width: int, img_height: int) -> str:
    """
    Detect if the coordinates are normalized or unnormalized based on the first few shapes.
    """
    if not shapes:
        return 'unnormalized'
    for shape in shapes[:5]:
        for x, y in shape['points']:
            if x > 1.1 or y > 1.1 or x < 0 or y < 0:
                return 'unnormalized'
            if img_width > 0 and img_height > 0 and 0 < x < 1.0 and 0 < y < 1.0:
                if x * img_width < 1.0 and y * img_height < 1.0:
                    continue
    for shape in shapes[:5]:
        for x, y in shape['points']:
            if 0 < x < 0.01 or 0 < y < 0.01:
                return 'normalized'
    return 'unnormalized'


def detect_bbox_type(points: list) -> str:
    """
    Determine if the bounding box is HBB or OBB based on number of points.
    """
    if len(points) == 2:
        return "hbb"
    elif len(points) == 4:
        return "obb"
    else:
        return None


def json2txt(
    json_path: Path,
    txt_path: Path,
    label_map: dict = None,
    normalize: bool = False,
    precision: int = None,
) -> None:
    """
    Convert a JSON annotation file to YOLO text format.
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"Error loading/parsing JSON file: {json_path}")
        return

    try:
        img_height, img_width = data["imageHeight"], data["imageWidth"]
    except KeyError:
        if normalize:
            print(f"Warning: Image dimensions not found in {json_path}, cannot normalize")
        normalize = False
        img_height = img_width = 1

    input_format = detect_coordinate_format(data.get("shapes", []), img_width, img_height)
    if precision is None:
        precision = 10 if normalize else 0
    fmt = f"{{:.{precision}f}}"

    # Check for consistency in bbox types
    shapes = data.get("shapes", [])
    if not shapes:
        print(f"Warning: No shapes found in {json_path}")
        return

    # Get type of first valid shape
    first_bb_type = None
    for shape in shapes:
        bb_type = detect_bbox_type(shape["points"])
        if bb_type:
            first_bb_type = bb_type
            break

    if not first_bb_type:
        print(f"Error: No valid bounding box shapes found in {json_path}")
        return

    # Check if all shapes have the same type
    inconsistent_shapes = []
    for i, shape in enumerate(shapes):
        bb_type = detect_bbox_type(shape["points"])
        if bb_type is None:
            print(
                f"Warning: Shape at index {i} with label '{shape['label']}' has "
                f"{len(shape['points'])} points, expected 2 (HBB) or 4 (OBB)."
            )
            inconsistent_shapes.append(i)
        elif bb_type != first_bb_type:
            print(
                f"Error: Inconsistent bounding box types in {json_path}. "
                f"First valid shape is {first_bb_type} but shape at index {i} is {bb_type}."
            )
            inconsistent_shapes.append(i)

    if inconsistent_shapes:
        print(
            f"Error: Cannot save {txt_path} due to inconsistent bounding box types. "
            f"All shapes must be of the same type ({first_bb_type})."
        )
        return

    if label_map is None:
        print(
            "\033[93mWarning: No label map provided or found. Numerical labels "
            "will be assigned based on the order of class appearance.\033[0m"
        )
        label_map = {}
    try:
        with open(txt_path, "w", encoding="utf-8") as f:
            for obj in shapes:
                label = obj["label"]
                if label not in label_map:
                    label_map[label] = len(label_map)
                label = label_map[label]
                bbox = obj["points"]

                if first_bb_type == "hbb":
                    x1, y1 = bbox[0]
                    x2, y2 = bbox[1]
                    if input_format == 'normalized':
                        x1, y1, x2, y2 = x1 * img_width, y1 * img_height, x2 * img_width, y2 * img_height
                    xc, yc, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
                    if normalize:
                        xc, yc, w, h = xc / img_width, yc / img_height, w / img_width, h / img_height
                    f.write(f"{label} {fmt.format(xc)} {fmt.format(yc)} {fmt.format(w)} {fmt.format(h)}\n")

                elif first_bb_type == "obb":
                    points = []
                    for point_x, point_y in bbox:
                        processed_x, processed_y = point_x, point_y
                        if input_format == 'normalized' and not normalize:
                            processed_x, processed_y = point_x * img_width, point_y * img_height
                        elif input_format == 'unnormalized' and normalize:
                            processed_x, processed_y = point_x / img_width, point_y / img_height
                        points.append(fmt.format(processed_x))
                        points.append(fmt.format(processed_y))
                    f.write(f"{label} {' '.join(points)}\n")
    except (OSError, IOError) as e:
        print(f"Error saving YOLO text file: {txt_path}\n{e}")
    else:
        print(f"Saved YOLO text file: {txt_path} (detected input format: {input_format}, bbox type: {first_bb_type})")


def coco2txt(
    coco_json_path: Path,
    txt_dir: Path,
    label_map: dict = None,
    normalize: bool = False,
    precision: int = None,
) -> None:
    """
    Convert COCO JSON annotations to YOLO HBB text files.
    """
    try:
        with open(coco_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"Error loading/parsing COCO JSON file: {coco_json_path}")
        return

    txt_dir.mkdir(exist_ok=True, parents=True)
    if precision is None:
        precision = 10 if normalize else 0
    fmt = f"{{:.{precision}f}}"

    images = {img["id"]: img for img in data.get("images", []) if "id" in img}
    categories = {cat["id"]: cat.get("name", str(cat["id"])) for cat in data.get("categories", []) if "id" in cat}
    annotations_by_image = defaultdict(list)
    for ann in data.get("annotations", []):
        if "image_id" in ann:
            annotations_by_image[ann["image_id"]].append(ann)

    if label_map is None:
        category_ids = sorted(categories.keys())
        label_map_by_category_id = {cat_id: i for i, cat_id in enumerate(category_ids)}
    else:
        label_map_by_category_id = {}
        next_id = max(label_map.values(), default=-1) + 1
        for cat_id, cat_name in categories.items():
            if cat_name in label_map:
                label_map_by_category_id[cat_id] = label_map[cat_name]
            else:
                label_map_by_category_id[cat_id] = next_id
                label_map[cat_name] = next_id
                next_id += 1

    for image_id, image_data in images.items():
        width = image_data.get("width")
        height = image_data.get("height")
        file_name = image_data.get("file_name")
        if not file_name:
            continue

        txt_file = txt_dir / (Path(file_name).stem + ".txt")
        image_annotations = annotations_by_image.get(image_id, [])
        with open(txt_file, "w", encoding="utf-8") as f:
            for ann in image_annotations:
                bbox = ann.get("bbox")
                category_id = ann.get("category_id")
                if not bbox or len(bbox) < 4 or category_id not in label_map_by_category_id:
                    continue
                # COCO bbox format is [x_min, y_min, width, height]
                x, y, w, h = map(float, bbox[:4])
                if w <= 0 or h <= 0:
                    continue

                xc = x + w / 2
                yc = y + h / 2
                if normalize:
                    if not width or not height:
                        continue
                    xc, yc, w, h = xc / width, yc / height, w / width, h / height

                class_id = label_map_by_category_id[category_id]
                f.write(
                    f"{class_id} {fmt.format(xc)} {fmt.format(yc)} {fmt.format(w)} {fmt.format(h)}\n"
                )
        print(f"Saved YOLO text file: {txt_file} (input format: coco, bbox type: hbb)")


if __name__ == "__main__":
    args = parse_cli_args()
    process_files(**vars(args))
