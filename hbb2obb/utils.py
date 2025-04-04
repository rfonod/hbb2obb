# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
Utility functions for HBB to OBB conversion and evaluation
"""

import platform
import sys
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from typing import Union

import numpy as np
import psutil
import torch
import yaml


class Annotations:
    def __init__(self, filepath: list, img: np.ndarray, input_format: str = "xywh"):
        self.hbb_filepath = filepath
        self.input_format = input_format
        self.img_shape = (img.shape[1], img.shape[0])

        self.hbb_xyxy = self.load_hbb_annotations()
        self.hbb_xywh = self.convert_to_xywh()

        self.segments = None
        self.masks = None

    def load_hbb_annotations(self) -> np.ndarray:
        """
        Load HBB annotations from a text file
        """
        if not self.hbb_filepath.exists():
            print(f"Annotation file not found: {self.hbb_filepath}")
            sys.exit(1)
        with open(self.hbb_filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()

        hbb_xyxy = []
        first_line = lines[0].strip().split()
        xc, yc = map(float, first_line[1:3])
        self.normalized = 0 <= xc <= 1 and 0 <= yc <= 1

        for line in lines:
            line_parts = line.strip().split()
            label = int(line_parts[0])
            if self.input_format == "xywh":
                xc, yc, w, h = map(float, line_parts[1:])
                if self.normalized:
                    xc *= self.img_shape[0]
                    yc *= self.img_shape[1]
                    w *= self.img_shape[0]
                    h *= self.img_shape[1]
                x1 = xc - w / 2
                y1 = yc - h / 2
                x2 = xc + w / 2
                y2 = yc + h / 2
            elif self.input_format == "xyxy":
                x1, y1, x2, y2 = map(float, line_parts[1:])
                if self.normalized:
                    x1 *= self.img_shape[0]
                    y1 *= self.img_shape[1]
                    x2 *= self.img_shape[0]
                    y2 *= self.img_shape[1]
            else:
                raise ValueError(f"Unsupported format: {self.input_format}")
            hbb_xyxy.append([label, x1, y1, x2, y2])

        return np.array(hbb_xyxy)

    def convert_to_xywh(self) -> np.ndarray:
        """
        Convert HBB xyxy to xywh format
        """
        hbb_xywh = []
        for box in self.hbb_xyxy:
            label, x1, y1, x2, y2 = box
            xc = (x1 + x2) / 2
            yc = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1
            hbb_xywh.append([label, xc, yc, w, h])
        return np.array(hbb_xywh)


def get_image_paths(input_path: Path) -> list:
    """
    Get image paths from a directory or a single image file
    """
    if input_path.is_dir():
        jpg_files = list(input_path.glob("*.[jJ][pP][gG]"))
        jpeg_files = list(input_path.glob("*.[jJ][pP][eE][gG]"))
        png_files = list(input_path.glob("*.[pP][nN][gG]"))
        bmp_files = list(input_path.glob("*.[bB][mM][pP]"))
        image_paths = jpg_files + jpeg_files + png_files + bmp_files
        if not image_paths:
            print(f"No image files found in directory: {input_path}")
    elif input_path.is_file():
        image_paths = [input_path]
    else:
        print(f"Invalid input path: {input_path}")
        sys.exit(1)
    return image_paths


def get_hbb_dir(img_path: Path, hbb_dir: Path = None) -> Path:
    """
    Get the directory containing HBB annotations.
    If not provided, it defaults to the parent directory of the image path.
    """
    if hbb_dir is None:
        if img_path.is_dir():
            hbb_dir = img_path.parent / "labels_hbb"
        else:
            hbb_dir = img_path.parent.parent / "labels_hbb"
    if not hbb_dir.is_dir():
        print(f"Error: HBB directory not found: {hbb_dir}")
        sys.exit(1)
    return hbb_dir


def process_ultralytics_kwargs(kwargs_string: str) -> dict:
    """
    Parse and process additional keyword arguments for Ultralytics model inference.
    """
    if not kwargs_string:
        return {}

    def parse_value(value: str):
        """Helper function to parse individual values."""
        if value.lower() == 'true':
            return True
        elif value.lower() == 'false':
            return False
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value

    try:
        kwargs = {}
        for kv_pair in kwargs_string.split(','):
            if '=' not in kv_pair:
                raise ValueError(f"Invalid key-value pair: '{kv_pair}'")
            k, v = kv_pair.split('=', 1)
            kwargs[k.strip()] = parse_value(v.strip())
        return kwargs
    except Exception as e:
        print(f"Error parsing model_kwargs: {e}. Using default model settings.")
        return {}


def load_label_map(label_map_path: Path, reverse=False) -> Union[dict, None]:
    """
    Load label map from YAML file
    """
    if label_map_path is None:
        return None
    try:
        with open(label_map_path, "r", encoding="utf-8") as f:
            label_map = yaml.safe_load(f)
    except (FileNotFoundError, yaml.YAMLError) as e:
        print(f"Error loading/parsing label map file: {e}")
        return None
    else:
        if reverse:
            label_map = {value: key for key, value in label_map.items()}
        return label_map


def get_system_metadata() -> dict:
    """
    Get system metadata including OS, Python version, and hardware specifications.
    """
    # Create a dictionary to store system metadata
    system_metadata = {
        'start_date': datetime.now().strftime("%Y-%m-%d"),
        'start_time': datetime.now().strftime("%H:%M:%S.%f")[:-3],
        'os': {
            'system': platform.system(),
            'platform': platform.platform(),
            'release': platform.release(),
            'version': platform.version(),
        },
        'software': {
            'python': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            'python_full': sys.version,
            'hbb2obb': version("hbb2obb"),
            'ultralytics': version("ultralytics"),
            'opencv': version("opencv-python"),
            'numpy': version("numpy"),
            'torch': version("torch"),
        },
        'hardware': {
            'architecture': platform.machine(),
            'processor': platform.processor(),
            'cpu_cores': f"{psutil.cpu_count(logical=False)} core(s)",
            'ram': f"{psutil.virtual_memory().total / 1024**3:.2f} GB",
        },
    }

    # Add GPU information if available
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        system_metadata['hardware'].update(
            {
                'gpu': f"{gpu_count} device(s)",
                'gpu_name': [torch.cuda.get_device_name(i) for i in range(gpu_count)],
                'gpu_memory': [
                    f"{torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB" for i in range(gpu_count)
                ],
            }
        )
    else:
        system_metadata['hardware'].update({'gpu': "0 devices", 'gpu_name': [], 'gpu_memory': []})

    return system_metadata
