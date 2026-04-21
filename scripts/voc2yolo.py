#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert Pascal VOC XML annotations to YOLO HBB text format.
"""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

from hbb2obb.utils import load_label_map


def parse_cli_args():
    parser = argparse.ArgumentParser(description="Convert Pascal VOC XML annotations to YOLO text files")
    parser.add_argument("xml_dir", type=Path, help="Directory with Pascal VOC XML files")
    parser.add_argument("--map_path", "-mp", type=Path, help="Path to label map YAML file")
    parser.add_argument("--txt_dir", "-td", type=Path, help="Directory for YOLO text files (default: same as XML dir)")
    parser.add_argument("--normalize", "-n", action="store_true", help="Normalize coordinates")
    parser.add_argument(
        "--precision",
        "-p",
        type=int,
        help="Decimal precision (default: 0 for unnormalized, 10 for normalized)",
    )
    return parser.parse_args()


def parse_pascal_voc(xml_path: Path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    width = int(float(size.findtext("width", "0"))) if size is not None else 0
    height = int(float(size.findtext("height", "0"))) if size is not None else 0

    objects = []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        bndbox = obj.find("bndbox")
        if not name or bndbox is None:
            continue
        try:
            xmin = float(bndbox.findtext("xmin"))
            ymin = float(bndbox.findtext("ymin"))
            xmax = float(bndbox.findtext("xmax"))
            ymax = float(bndbox.findtext("ymax"))
        except (TypeError, ValueError):
            continue
        if xmax <= xmin or ymax <= ymin:
            continue
        objects.append({"label": name, "bbox": [xmin, ymin, xmax, ymax]})

    return width, height, objects


def voc2txt(
    xml_path: Path,
    txt_path: Path,
    label_map: dict = None,
    normalize: bool = False,
    precision: int = None,
):
    if precision is None:
        precision = 10 if normalize else 0
    fmt = f"{{:.{precision}f}}"

    try:
        width, height, objects = parse_pascal_voc(xml_path)
    except (ET.ParseError, OSError) as e:
        print(f"Error loading/parsing XML file: {xml_path}\n{e}")
        return

    if label_map is None:
        label_map = {}

    with open(txt_path, "w", encoding="utf-8") as f:
        for obj in objects:
            label = obj["label"]
            if label not in label_map:
                label_map[label] = len(label_map)
            class_id = label_map[label]

            xmin, ymin, xmax, ymax = obj["bbox"]
            xc = (xmin + xmax) / 2
            yc = (ymin + ymax) / 2
            w = xmax - xmin
            h = ymax - ymin

            if normalize:
                if width <= 0 or height <= 0:
                    print(
                        f"Warning: Missing/invalid image size in {xml_path}, "
                        "skipping normalization and writing unnormalized coordinates"
                    )
                else:
                    xc, yc, w, h = xc / width, yc / height, w / width, h / height

            f.write(f"{class_id} {fmt.format(xc)} {fmt.format(yc)} {fmt.format(w)} {fmt.format(h)}\n")

    print(f"Saved YOLO text file: {txt_path} (input format: pascal_voc, bbox type: hbb)")


def process_files(
    xml_dir: Path,
    map_path: Path = None,
    txt_dir: Path = None,
    normalize: bool = False,
    precision: int = None,
):
    label_map = load_label_map(map_path, reverse=True) if map_path else None
    if txt_dir is None:
        txt_dir = xml_dir
    txt_dir.mkdir(exist_ok=True, parents=True)

    for xml_file in xml_dir.glob("*.xml"):
        txt_file = txt_dir / f"{xml_file.stem}.txt"
        voc2txt(xml_file, txt_file, label_map, normalize, precision)


if __name__ == "__main__":
    args = parse_cli_args()
    process_files(**vars(args))
