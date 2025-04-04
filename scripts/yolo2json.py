#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
Convert YOLO text format (HBB or OBB) bounding box annotations to JSON files compatible with LabelMe.

Supports optional coordinate normalization and precision control.

Usage:
    python scripts/yolo2json.py <txt_dir> <map_path>
                             [--json_dir JSON_DIR]
                             [--img_dir IMG_DIR]
                             [--normalize]
                             [--precision PRECISION]

Arguments:
    txt_dir             Path to the directory containing YOLO .txt annotation files
    map_path            Path to label map YAML file that maps label IDs to names
    --json_dir, -jd     Optional path to output directory for JSON files (default: txt_dir)
    --img_dir, -id      Directory containing images (defaults to txt_dir/../images)
    --normalize, -n     Keep coordinates normalized [0-1] (default: convert to pixel values)
    --precision, -p     Decimal precision for output (default: 0 for pixel coordinates, 10 for normalized)

Input format:
    - HBB: <label> <xc> <yc> <w> <h>
    - OBB: <label> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4>

Examples:
    1) Convert YOLO annotations to JSON format with dimensions from images (default denormalized):
       python scripts/yolo2json.py data/labels_yolo data/classes.yaml --img_dir data/images

    2) Convert YOLO annotations to JSON format keeping normalized coordinates:
       python scripts/yolo2json.py data/labels_yolo data/classes.yaml --json_dir data/labels_json --img_dir data/images --normalize

Output JSON format:
    - Contains 'imageHeight', 'imageWidth', and a list of 'shapes'
    - Each shape has a 'label' and a list of points
    - Points format depends on input: 2 points [top-left, bottom-right] for HBB or 4 points for OBB
"""

import argparse
import json
from pathlib import Path

from PIL import Image

from hbb2obb.utils import load_label_map


def parse_cli_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Convert YOLO text files to JSON annotations")
    parser.add_argument("txt_dir", type=Path, help="Directory with YOLO text annotations")
    parser.add_argument("map_path", type=Path, help="Path to label map YAML file that maps label IDs to names")
    parser.add_argument("--json_dir", "-jd", type=Path, help="Directory for JSON files (defaults to txt_dir)")
    parser.add_argument("--img_dir", "-id", type=Path, help="Directory containing images (default: txt_dir/../images)")
    parser.add_argument("--normalize", "-n", action="store_true", help="Keep coordinates normalized [0-1] (default: convert to pixel values)")
    parser.add_argument("--precision", "-p", type=int, help="Decimal precision (default: 0 for pixel values, 10 for normalized)")
    return parser.parse_args()


def process_files(
    txt_dir: Path,
    map_path: Path,
    json_dir: Path = None,
    img_dir: Path = None,
    normalize: bool = False,
    precision: int = None,
) -> None:
    """
    Process all YOLO text files in a directory and convert them to JSON files.
    """
    label_map = load_label_map(map_path)
    if json_dir is None:
        json_dir = txt_dir
    json_dir.mkdir(exist_ok=True, parents=True)

    for txt_file in txt_dir.glob("*.txt"):
        json_file = json_dir / (txt_file.stem + ".json")
        yolo2json(txt_file, json_file, label_map, img_dir, normalize, precision)


def yolo2json(
    txt_path: Path,
    json_path: Path,
    label_map: dict = None,
    img_dir: Path = None,
    normalize: bool = False,
    precision: int = None,
) -> None:
    """
    Convert a YOLO text file to JSON annotation format.
    """
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except (FileNotFoundError, IOError) as e:
        print(f"Error reading YOLO text file: {txt_path}\n{e}")
        return

    if not lines:
        print(f"Warning: No annotations found in {txt_path}")
        return

    # Detect bbox type from first valid line
    bbox_type = None
    for line in lines:
        if line.strip():
            bbox_type = detect_bbox_type(line)
            if bbox_type:
                break

    if not bbox_type:
        print(f"Error: No valid bounding box format found in {txt_path}")
        return

    # Check for consistency in bbox types
    inconsistent_lines = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        line_type = detect_bbox_type(line)
        if line_type is None:
            print(f"Warning: Line {i + 1} has invalid format")
            inconsistent_lines.append(i)
        elif line_type != bbox_type:
            print(f"Error: Inconsistent bounding box types in {txt_path}. First valid line is {bbox_type} but line {i + 1} is {line_type}.")
            inconsistent_lines.append(i)

    if inconsistent_lines:
        print(f"Error: Cannot convert {txt_path} due to inconsistent bounding box types. All lines must be of the same type ({bbox_type}).")
        return

    if img_dir is None:
        img_dir = txt_path.parent.parent / "images"

    # Detect coordinate format
    is_normalized = detect_normalization(lines)

    # Get image dimensions
    img_width, img_height = get_image_dimensions(txt_path, img_dir)
    if is_normalized and not normalize and (img_width is None or img_height is None):
        print(f"Error: Cannot find image for {txt_path.stem}. Image dimensions are required for denormalization.")
        return
    if not is_normalized and normalize and (img_width is None or img_height is None):
        print(f"Error: Cannot find image for {txt_path.stem}. Image dimensions are required for normalization.")
        return
    if img_width is None or img_height is None:
        print(f"Warning: Image dimensions not found for {txt_path.stem}. Using default dimensions (1x1).")
        img_width, img_height = 1, 1

    # Set default precision
    if precision is None:
        precision = 10 if normalize else 0

    # Format string for precision control
    fmt = f"{{:.{precision}f}}"

    # Create JSON structure
    image_file = None
    if img_dir:
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
            potential_img = Path(img_dir) / f"{txt_path.stem}{ext}"
            if potential_img.exists():
                image_file =  str(Path('..') / "images" / potential_img.name)
                break

    if image_file is None:
        image_file = str(Path('..') / "images" / f"{txt_path.stem}.jpg") # Default to .jpg if no image found

    json_data = {
        "version": "5.2.1",
        "flags": {},
        "shapes": [],
        "imagePath": image_file,
        "imageData": None,
        "imageHeight": img_height,
        "imageWidth": img_width,
    }

    if label_map is None:
        print("\033[93mWarning: No label map provided. Numerical labels will be used.\033[0m")
        label_map = {}

    for line in lines:
        if not line.strip():
            continue

        values = line.strip().split()
        label_id = int(values[0])
        label = label_map.get(label_id, str(label_id))
        coords = [float(v) for v in values[1:]]

        if bbox_type == "hbb":
            xc, yc, w, h = coords
            if not normalize and is_normalized:
                # Convert from normalized to pixel coordinates
                xc, yc = xc * img_width, yc * img_height
                w, h = w * img_width, h * img_height
            elif normalize and not is_normalized:
                # Convert from pixel to normalized coordinates
                xc, yc = xc / img_width, yc / img_height
                w, h = w / img_width, h / img_height

            # Convert to top-left and bottom-right points
            points = hbb_to_points(xc, yc, w, h)
            # Apply precision formatting or rounding
            if precision == 0:
                points = [[round(x), round(y)] for x, y in points]
            else:
                points = [[float(fmt.format(x)), float(fmt.format(y))] for x, y in points]
        else:  # obb
            if not normalize and is_normalized:
                # Convert from normalized to pixel coordinates
                for i in range(0, len(coords), 2):
                    coords[i] *= img_width  # x coordinates
                    coords[i + 1] *= img_height  # y coordinates
            elif normalize and not is_normalized:
                # Convert from pixel to normalized coordinates
                for i in range(0, len(coords), 2):
                    coords[i] /= img_width
                    coords[i + 1] /= img_height
            # Convert to 4 points
            points = obb_to_points(coords)
            # Apply precision formatting or rounding
            if precision == 0:
                points = [[round(x), round(y)] for x, y in points]
            else:
                points = [[float(fmt.format(x)), float(fmt.format(y))] for x, y in points]

        shape = {
            "label": label,
            "points": points,
            "group_id": None,
            "shape_type": "rectangle" if bbox_type == "hbb" else "polygon",
            "flags": {},
        }
        json_data["shapes"].append(shape)

    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2)
    except (OSError, IOError) as e:
        print(f"Error saving JSON file: {json_path}\n{e}")
    else:
        print(f"Saved JSON file: {json_path} (detected bbox type: {bbox_type})")


def detect_normalization(lines: list) -> bool:
    """
    Detect if the coordinates are normalized or unnormalized based on the first few lines.
    """
    for line in lines[:5]:
        if line.strip():
            values = line.strip().split()
            if len(values) == 5:  # HBB
                xc, yc, w, h = map(float, values[1:])
                if 0 <= xc <= 1 and 0 <= yc <= 1 and 0 < w <= 1 and 0 < h <= 1:
                    return True
            elif len(values) == 9:  # OBB
                coords = list(map(float, values[1:]))
                if all(0 <= coord <= 1 for coord in coords):
                    return True
    return False


def detect_bbox_type(line: str) -> str:
    """
    Determine if the line contains HBB or OBB based on number of values.
    """
    values = line.strip().split()
    if len(values) == 5:  # label, xc, yc, w, h
        return "hbb"
    elif len(values) == 9:  # label, x1, y1, x2, y2, x3, y3, x4, y4
        return "obb"
    else:
        return None


def hbb_to_points(xc: float, yc: float, w: float, h: float) -> list:
    """
    Convert YOLO HBB format (center, width, height) to two points (top-left, bottom-right).
    """
    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2
    return [[x1, y1], [x2, y2]]


def obb_to_points(coords: list) -> list:
    """
    Convert YOLO OBB format (8 values) to 4 points.
    """
    return [[coords[i], coords[i + 1]] for i in range(0, 8, 2)]


def get_image_dimensions(txt_path: Path, img_dir: Path = None) -> tuple:
    """
    Get the dimensions of the image corresponding to the txt file.
    Searches for image files with common extensions.

    Returns (width, height) tuple or (None, None) if image not found.
    """

    img_name = txt_path.stem
    extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']

    for ext in extensions:
        img_path = img_dir / f"{img_name}{ext}"
        if img_path.exists():
            try:
                with Image.open(img_path) as img:
                    return img.width, img.height
            except Exception as e:
                print(f"Error reading image dimensions from {img_path}: {e}")
                break

    # Try case-insensitive extensions
    for ext in extensions:
        for file in img_dir.glob(f"{img_name}.*"):
            if file.suffix.lower() == ext:
                try:
                    with Image.open(file) as img:
                        return img.width, img.height
                except Exception as e:
                    print(f"Error reading image dimensions from {file}: {e}")
                    return None, None

    return None, None


if __name__ == "__main__":
    args = parse_cli_args()
    process_files(**vars(args))
