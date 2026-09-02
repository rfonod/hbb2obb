# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
Read and write bounding box annotations in the formats HBB2OBB understands.

One in-memory representation, readers and writers around it. A box is always four corners in
absolute pixel coordinates, so horizontal and oriented boxes share a single model and an HBB is
simply a ``Box`` whose quad is axis-aligned.

Supported formats:

===========  ====  ====  ====================================================================
name         HBB   OBB   shape
===========  ====  ====  ====================================================================
``yolo``     yes   yes   one .txt per frame; ``cls xc yc w h`` or ``cls x1 y1 ... x4 y4``,
                         with an optional trailing confidence column
``dota``     no    yes   one file per frame; ``x1 y1 ... x4 y4 name difficult``, integer px,
                         preceded by ``imagesource:`` and ``gsd:`` header lines
``voc``      yes   no    one Pascal VOC .xml per frame, integer px
``coco``     yes   yes   a single .json for the whole set; the quad goes in ``segmentation``
                         and ``bbox`` holds its envelope
``labelme``  yes   yes   one LabelMe .json per frame; 2 points for an HBB, 4 for an OBB
===========  ====  ====  ====================================================================

**Rounding.** ``dota``, ``voc`` and ``coco`` are integer formats. They must be derived by rounding
the *canonical* quad, never by rounding full precision independently: rounding full precision and
rounding a value already rounded to two decimals disagree wherever the canonical lands on ``.5``,
and the envelope then stops matching the box it came from. ``canonicalize()`` is the single place
that rounding happens; every writer takes what it produces. ``verify()`` proves the outputs agree
by exact equality after rounding rather than by a tolerance.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from xml.etree import ElementTree as ET

import numpy as np

# Formats that can carry each kind of box
OBB_FORMATS = ("yolo", "dota", "coco", "labelme")
HBB_FORMATS = ("yolo", "voc", "coco", "labelme")
ALL_FORMATS = ("yolo", "dota", "voc", "coco", "labelme")

# One .json for the whole set rather than one file per frame
SET_FORMATS = ("coco",)

# Default file extension per format; DOTA is the odd one out, see DOTA_EXT below
EXTENSIONS = {"yolo": ".txt", "dota": ".txt", "voc": ".xml", "coco": ".json", "labelme": ".json"}

# Canonical DOTA uses .txt inside a labelTxt/ directory. When the DOTA files sit beside the YOLO
# ones, as in the Songdo Vision OBB release, they need a distinct extension.
DOTA_EXT = ".dota"

# Directories hbb2obb writes beside a label set that hold one .txt per frame but are not label
# sets: a score per line, and a variable-length contour per line. Anything walking a tree looking
# for annotations has to skip them, or it reads a confidence file as a malformed YOLO label.
SIDECAR_DIRS = frozenset({"labels_confidence", "labels_polygon"})

# Two decimals in absolute pixels is well under the precision any annotation actually carries, and
# it is what the integer formats round from.
DEFAULT_PRECISION = 2

DOTA_IMAGESOURCE = "drone"
DOTA_GSD = "null"

LABELME_VERSION = "5.5.0"


@dataclass
class Box:
    """A single annotation: four corners in absolute pixel coordinates."""

    cls: int
    quad: np.ndarray  # (4, 2) float
    score: Optional[float] = None
    difficult: int = 0

    def __post_init__(self):
        self.quad = np.asarray(self.quad, dtype=float).reshape(4, 2)

    @property
    def xyxy(self) -> Tuple[float, float, float, float]:
        """The axis-aligned envelope of the quad."""
        return (
            float(self.quad[:, 0].min()),
            float(self.quad[:, 1].min()),
            float(self.quad[:, 0].max()),
            float(self.quad[:, 1].max()),
        )

    @property
    def is_axis_aligned(self) -> bool:
        """True when the quad is a rectangle with sides parallel to the image axes."""
        return len(np.unique(np.round(self.quad[:, 0], 6))) <= 2 and len(np.unique(np.round(self.quad[:, 1], 6))) <= 2

    @property
    def area(self) -> float:
        """Shoelace area of the quad."""
        x, y = self.quad[:, 0], self.quad[:, 1]
        return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)


@dataclass
class FrameAnnotations:
    """Every box on one image."""

    stem: str
    width: int
    height: int
    boxes: List[Box] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.boxes)


# --------------------------------------------------------------------------------------- geometry
def rect(x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    """An axis-aligned quad, corners clockwise from the top left."""
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)


def canonicalize(frames: Iterable[FrameAnnotations], precision: int = DEFAULT_PRECISION) -> List[FrameAnnotations]:
    """
    Round every quad once, so that all output formats derive from one source.

    This is what keeps the integer formats consistent with each other and with the envelope of the
    box they came from. Rounding is monotonic, so ``min(round(x)) == round(min(x))`` holds only when
    both sides round the same values; deriving one format from full precision and another from a
    rounded canonical is what makes them disagree on boxes that land exactly on a half pixel.
    """
    out = []
    for fr in frames:
        boxes = [Box(b.cls, np.round(b.quad, precision), b.score, b.difficult) for b in fr.boxes]
        out.append(FrameAnnotations(fr.stem, fr.width, fr.height, boxes))
    return out


def _round_quad(box: Box) -> List[int]:
    """The quad as a flat list of integers, the form every integer format writes."""
    return [int(round(v)) for v in box.quad.reshape(-1)]


def _round_xyxy(box: Box) -> List[int]:
    """The envelope as integers, rounded from the same values the quad rounds from."""
    x0, y0, x1, y1 = box.xyxy
    return [int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))]


# ---------------------------------------------------------------------------------------- readers
def _looks_normalized(values: Sequence[float]) -> bool:
    """Coordinates that all sit in [0, 1] are relative; anything larger is absolute pixels."""
    return bool(values) and all(0.0 <= v <= 1.0 for v in values)


def read_yolo(path: Path, width: int, height: int) -> List[Box]:
    """
    Read a YOLO TXT file, HBB (``cls xc yc w h``) or OBB (``cls x1 y1 ... x4 y4``).

    The kind is decided by the field count, and an optional trailing confidence column is accepted
    on either. Coordinates may be relative or absolute; blank lines and an empty file are valid.
    """
    lines = [ln.split() for ln in path.read_text(encoding="utf-8").splitlines()]
    lines = [ln for ln in lines if ln]
    if not lines:
        return []

    counts = {len(ln) for ln in lines}
    if counts <= {5, 6}:
        n_coords = 4
    elif counts <= {9, 10}:
        n_coords = 8
    else:
        raise ValueError(
            f"Malformed YOLO file {path}: expected 5 or 6 fields (HBB) or 9 or 10 (OBB) on every "
            f"line, got {sorted(counts)}"
        )

    flat = [float(v) for ln in lines for v in ln[1 : 1 + n_coords]]
    normalized = _looks_normalized(flat)

    boxes = []
    for ln in lines:
        cls = int(ln[0])
        coords = [float(v) for v in ln[1 : 1 + n_coords]]
        if normalized:
            coords = [v * (width if i % 2 == 0 else height) for i, v in enumerate(coords)]
        if n_coords == 4:
            xc, yc, w, h = coords
            quad = rect(xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2)
        else:
            quad = np.array(coords, dtype=float).reshape(4, 2)
        score = float(ln[1 + n_coords]) if len(ln) > 1 + n_coords else None
        boxes.append(Box(cls, quad, score))
    return boxes


def read_dota(path: Path, names: Sequence[str]) -> List[Box]:
    """Read a DOTA file: ``x1 y1 ... x4 y4 name difficult``, after any ``key:value`` header lines."""
    boxes = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        fields = ln.split()
        if not fields or ":" in fields[0]:  # imagesource:/gsd: header
            continue
        if len(fields) < 9:
            raise ValueError(f"Malformed DOTA line in {path}: {ln}")
        quad = np.array([float(v) for v in fields[:8]], dtype=float).reshape(4, 2)
        name = fields[8]
        if name not in names:
            raise ValueError(f"Unknown class {name!r} in {path}; known classes: {list(names)}")
        difficult = int(fields[9]) if len(fields) > 9 else 0
        boxes.append(Box(names.index(name), quad, difficult=difficult))
    return boxes


def read_voc(path: Path, names: Sequence[str]) -> Tuple[List[Box], int, int]:
    """Read a Pascal VOC XML file. Returns the boxes plus the image size it declares."""
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as e:
        print(f"Warning: skipping malformed VOC file {path}: {e}")
        return [], 0, 0
    size = root.find("size")
    width = int(float(size.findtext("width", "0"))) if size is not None else 0
    height = int(float(size.findtext("height", "0"))) if size is not None else 0

    boxes = []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        bndbox = obj.find("bndbox")
        if not name or bndbox is None:
            continue
        try:
            x0, y0, x1, y1 = (float(bndbox.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        except (TypeError, ValueError):
            continue
        if x1 <= x0 or y1 <= y0:
            continue
        if name not in names:
            raise ValueError(f"Unknown class {name!r} in {path}; known classes: {list(names)}")
        boxes.append(Box(names.index(name), rect(x0, y0, x1, y1), difficult=int(obj.findtext("difficult", "0"))))
    return boxes, width, height


def read_coco(path: Path, names: Optional[Sequence[str]] = None) -> Tuple[List[FrameAnnotations], List[str]]:
    """
    Read a COCO instance file. Oriented boxes come from ``segmentation``, horizontal ones from
    ``bbox``. Returns one FrameAnnotations per image plus the class names, ordered by category id.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    categories = sorted(data.get("categories", []), key=lambda c: c["id"])
    coco_names = [c.get("name", str(c["id"])) for c in categories]
    if names is None:
        index = {c["id"]: i for i, c in enumerate(categories)}
    else:
        unknown = [n for n in coco_names if n not in names]
        if unknown:
            raise ValueError(f"Unknown class(es) {unknown} in {path}; known classes: {list(names)}")
        index = {c["id"]: names.index(c["name"]) for c in categories}

    frames: Dict[int, FrameAnnotations] = {}
    for img in data.get("images", []):
        frames[img["id"]] = FrameAnnotations(
            Path(img["file_name"]).stem, int(img.get("width", 0)), int(img.get("height", 0))
        )

    for ann in data.get("annotations", []):
        frame = frames.get(ann.get("image_id"))
        cls = index.get(ann.get("category_id"))
        if frame is None or cls is None:
            continue
        seg = ann.get("segmentation")
        if seg and isinstance(seg, list) and len(seg[0]) == 8:
            quad = np.array(seg[0], dtype=float).reshape(4, 2)
        else:
            bbox = ann.get("bbox")
            if not bbox or len(bbox) < 4:
                continue
            x, y, w, h = (float(v) for v in bbox[:4])
            quad = rect(x, y, x + w, y + h)
        score = ann.get("score")
        frame.boxes.append(Box(cls, quad, float(score) if score is not None else None))

    return list(frames.values()), (list(names) if names is not None else coco_names)


def read_labelme(
    path: Path, names: Optional[Sequence[str]] = None, discovered: Optional[List[str]] = None
) -> Tuple[List[Box], int, int, List[str]]:
    """
    Read a LabelMe JSON file. Shapes with 2 points are horizontal boxes, with 4 are oriented.

    LabelMe stores class names rather than ids. Pass ``names`` to pin the id of each class, which
    is what a round trip through this format needs; without it, ids follow first-seen order across
    the files read so far, which ``discovered`` accumulates.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    width = int(data.get("imageWidth", 0))
    height = int(data.get("imageHeight", 0))
    known = list(names) if names is not None else (discovered if discovered is not None else [])

    boxes = []
    for shape in data.get("shapes", []):
        points = shape.get("points", [])
        if len(points) not in (2, 4):
            continue
        label = shape["label"]
        if label not in known:
            if names is not None:
                raise ValueError(f"Unknown class {label!r} in {path}; known classes: {list(names)}")
            known.append(label)
        pts = np.array(points, dtype=float)
        quad = rect(pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()) if len(points) == 2 else pts
        boxes.append(Box(known.index(label), quad))
    return boxes, width, height, known


# ---------------------------------------------------------------------------------------- writers
def write_yolo(
    path: Path,
    boxes: Sequence[Box],
    width: int,
    height: int,
    obb: bool,
    normalize: bool,
    precision: Optional[int] = None,
) -> None:
    """Write a YOLO TXT file, oriented or horizontal, with the confidence column when boxes carry one."""
    if precision is None:
        precision = 10 if normalize else DEFAULT_PRECISION
    fmt = f"{{:.{precision}f}}"

    lines = []
    for box in boxes:
        if obb:
            coords = list(box.quad.reshape(-1))
        else:
            x0, y0, x1, y1 = box.xyxy
            coords = [(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0]
        if normalize:
            coords = [v / (width if i % 2 == 0 else height) for i, v in enumerate(coords)]
        line = f"{box.cls} " + " ".join(fmt.format(v) for v in coords)
        if box.score is not None:
            line += f" {box.score:.4f}"
        lines.append(line)
    path.write_text("".join(ln + "\n" for ln in lines), encoding="utf-8")


def write_dota(
    path: Path, boxes: Sequence[Box], names: Sequence[str], imagesource: str = DOTA_IMAGESOURCE, gsd: str = DOTA_GSD
) -> None:
    """Write a DOTA file: the two header lines, then one integer quad per box."""
    lines = [f"imagesource:{imagesource}", f"gsd:{gsd}"]
    for box in boxes:
        coords = " ".join(str(v) for v in _round_quad(box))
        lines.append(f"{coords} {names[box.cls]} {box.difficult}")
    path.write_text("".join(ln + "\n" for ln in lines), encoding="utf-8")


def write_voc(
    path: Path,
    boxes: Sequence[Box],
    names: Sequence[str],
    stem: str,
    width: int,
    height: int,
    image_ext: str = ".jpg",
    folder: str = "images",
    database: str = "Unknown",
    edge_tolerance: int = 1,
) -> None:
    """
    Write a Pascal VOC XML file.

    ``truncated`` is set on boxes that reach an image edge, within ``edge_tolerance`` pixels. The
    default of 1 px matches ``hbb2obb-eval --edge_tolerance``, so the two agree on which boxes are
    clipped by the frame.
    """
    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = folder
    ET.SubElement(root, "filename").text = f"{stem}{image_ext}"
    ET.SubElement(root, "path").text = f"{folder}/{stem}{image_ext}"
    ET.SubElement(ET.SubElement(root, "source"), "database").text = database
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "3"
    ET.SubElement(root, "segmented").text = "0"

    for box in boxes:
        x0, y0, x1, y1 = _round_xyxy(box)
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = names[box.cls]
        ET.SubElement(obj, "pose").text = "Unspecified"
        truncated = x0 <= edge_tolerance or y0 <= edge_tolerance
        truncated = truncated or x1 >= width - edge_tolerance or y1 >= height - edge_tolerance
        ET.SubElement(obj, "truncated").text = str(int(truncated))
        ET.SubElement(obj, "difficult").text = str(box.difficult)
        bndbox = ET.SubElement(obj, "bndbox")
        for key, value in zip(("xmin", "ymin", "xmax", "ymax"), (x0, y0, x1, y1)):
            ET.SubElement(bndbox, key).text = str(value)

    ET.indent(root, space="    ")
    path.write_text('<?xml version="1.0" ?>\n' + ET.tostring(root, encoding="unicode") + "\n", encoding="utf-8")


def write_coco(
    path: Path,
    frames: Sequence[FrameAnnotations],
    names: Sequence[str],
    obb: bool,
    image_ext: str = ".jpg",
    info: Optional[dict] = None,
    licenses: Optional[list] = None,
    supercategory: Optional[str] = None,
) -> None:
    """
    Write a single COCO instance file for the whole set.

    Oriented boxes carry the quad in ``segmentation`` and its envelope in ``bbox``; horizontal ones
    carry ``bbox`` alone. Annotation ids are assigned in frame then row order, so an OBB file and an
    HBB file written from the same boxes can be joined by id.
    """
    images, annotations = [], []
    for image_id, frame in enumerate(sorted(frames, key=lambda f: f.stem), start=1):
        images.append(
            {"id": image_id, "file_name": f"{frame.stem}{image_ext}", "width": frame.width, "height": frame.height}
        )
        for box in frame.boxes:
            x0, y0, x1, y1 = _round_xyxy(box)
            ann = {
                "id": len(annotations) + 1,
                "image_id": image_id,
                "category_id": box.cls + 1,
                "bbox": [x0, y0, x1 - x0, y1 - y0],
                "area": 0,
                "iscrowd": 0,
            }
            if obb:
                quad = _round_quad(box)
                ann["segmentation"] = [quad]
                pts = np.array(quad, dtype=float).reshape(4, 2)
                ann["area"] = int(round(Box(box.cls, pts).area))
                ann = {k: ann[k] for k in ("id", "image_id", "category_id", "segmentation", "bbox", "area", "iscrowd")}
            else:
                ann["area"] = (x1 - x0) * (y1 - y0)
            if box.score is not None:
                ann["score"] = round(box.score, 4)
            annotations.append(ann)

    data = {
        "info": info or {},
        "licenses": licenses or [],
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": i + 1, "name": name, **({"supercategory": supercategory} if supercategory else {})}
            for i, name in enumerate(names)
        ],
    }
    path.write_text(json.dumps(data, indent=1), encoding="utf-8")


def write_labelme(
    path: Path,
    boxes: Sequence[Box],
    names: Sequence[str],
    stem: str,
    width: int,
    height: int,
    obb: bool,
    image_ext: str = ".jpg",
) -> None:
    """Write a LabelMe JSON file: 4 points for an oriented box, 2 for a horizontal one."""
    shapes = []
    for box in boxes:
        if obb:
            points = [[float(x), float(y)] for x, y in box.quad]
        else:
            x0, y0, x1, y1 = box.xyxy
            points = [[x0, y0], [x1, y1]]
        shapes.append(
            {
                "label": names[box.cls],
                "points": points,
                "group_id": None,
                "description": "",
                "shape_type": "polygon" if obb else "rectangle",
                "flags": {},
            }
        )
    data = {
        "version": LABELME_VERSION,
        "flags": {},
        "shapes": shapes,
        "imagePath": f"{stem}{image_ext}",
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


# -------------------------------------------------------------------------------------- discovery
def sniff_format(path: Path) -> Optional[str]:
    """
    Guess the format of one label file, or None if it is not a label file at all.

    Extensions settle ``.xml`` and ``.dota``. ``.json`` is COCO when it has the top-level keys and
    LabelMe otherwise. ``.txt`` is shared: it is DOTA when the ninth field is a class name rather
    than a number, YOLO when every field after the class id parses as a number and the field count
    is one of the four it allows, and neither otherwise, which is what keeps a stray ``names.txt``
    or ``README.txt`` from being read as annotations.
    """
    suffix = path.suffix.lower()
    if suffix == ".xml":
        return "voc"
    if suffix == DOTA_EXT:
        return "dota"
    if suffix == ".json":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None
        if not isinstance(data, dict):
            return None
        if {"images", "annotations", "categories"} <= set(data):
            return "coco"
        return "labelme" if "shapes" in data else None
    if suffix != ".txt":
        return None

    try:
        lines = [ln.split() for ln in path.read_text(encoding="utf-8").splitlines()]
    except UnicodeDecodeError:
        return None
    lines = [ln for ln in lines if ln and ":" not in ln[0]]
    if not lines:
        return "yolo"  # an empty label file is valid YOLO input
    for fields in lines:
        if len(fields) >= 9:
            try:
                float(fields[8])
            except ValueError:
                return "dota"
        if len(fields) not in (5, 6, 9, 10):
            return None
        try:
            [float(v) for v in fields]
        except ValueError:
            return None
    return "yolo"


def detect_format(path: Path) -> str:
    """
    Guess the format of a label file or of a directory of them, raising if it cannot.

    A directory is resolved in ``ALL_FORMATS`` order rather than in file order, so that YOLO wins
    wherever a set ships its canonical files beside derived ones: the derived formats are rounded
    and cannot carry a confidence, so reading them by alphabetical accident loses information.
    """
    if path.is_dir():
        for fmt in ALL_FORMATS:
            if label_files(path, fmt):
                return fmt
        raise ValueError(f"No label files found in {path}")

    fmt = sniff_format(path)
    if fmt is None:
        raise ValueError(f"Cannot tell the format of {path}")
    return fmt


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def image_size(path: Path) -> Optional[Tuple[int, int]]:
    """
    Read an image's (width, height) from its header, without decoding it.

    Only the size is ever needed here, to denormalize relative coordinates and to fill in the
    formats that declare one, and decoding a directory of 4K frames to learn it is pure waste.
    Falls back to OpenCV for anything the three header layouts below do not cover.
    """
    try:
        with open(path, "rb") as f:
            head = f.read(32)
            if head[:8] == b"\x89PNG\r\n\x1a\n":
                return (int.from_bytes(head[16:20], "big"), int.from_bytes(head[20:24], "big"))
            if head[:2] == b"BM":
                # biHeight is negative for a top-down bitmap; the pixel height is its magnitude.
                width = int.from_bytes(head[18:22], "little", signed=True)
                height = int.from_bytes(head[22:26], "little", signed=True)
                return (abs(width), abs(height))
            if head[:2] == b"\xff\xd8":  # JPEG: walk the segments to the frame header
                f.seek(2)
                while True:
                    marker = f.read(2)
                    if len(marker) < 2 or marker[0] != 0xFF:
                        break
                    if 0xC0 <= marker[1] <= 0xCF and marker[1] not in (0xC4, 0xC8, 0xCC):
                        f.read(3)  # segment length and sample precision
                        height = int.from_bytes(f.read(2), "big")
                        width = int.from_bytes(f.read(2), "big")
                        return (width, height)
                    length = int.from_bytes(f.read(2), "big")
                    if length < 2:
                        break
                    f.seek(length - 2, 1)
    except OSError:
        return None

    import cv2

    img = cv2.imread(str(path))
    return None if img is None else (img.shape[1], img.shape[0])


def image_sizes(images_dir: Optional[Path]) -> Dict[str, Tuple[int, int]]:
    """Map each image stem in a directory to its (width, height)."""
    if images_dir is None:
        return {}
    paths = [images_dir] if images_dir.is_file() else sorted(images_dir.iterdir()) if images_dir.is_dir() else []
    sizes = {}
    for path in paths:
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        size = image_size(path)
        if size:
            sizes[path.stem] = size
    return sizes


def label_files(directory: Path, fmt: str) -> List[Path]:
    """
    Every label file of one format in a directory, sorted by stem.

    YOLO and DOTA both claim ``.txt``. Files carrying the ``.dota`` extension are unambiguous;
    otherwise a ``.txt`` file belongs to whichever of the two it actually parses as.
    """
    if fmt == "dota":
        dota = sorted(directory.glob(f"*{DOTA_EXT}"))
        if dota:
            return dota
    return [p for p in sorted(directory.glob(f"*{EXTENSIONS[fmt]}")) if sniff_format(p) == fmt]


# ----------------------------------------------------------------------------------------- verify
def verify(sets: Dict[str, List[FrameAnnotations]], reference: str = "yolo") -> List[str]:
    """
    Check that several readings of the same annotations agree, and report every disagreement.

    The reference format carries full precision and the others are one common rounding of it, so
    the comparison is exact equality after rounding rather than a tolerance. ``difficult`` is only
    compared between the formats that can express it.
    """
    problems: List[str] = []
    if reference not in sets:
        raise ValueError(f"Reference format {reference!r} is not among {sorted(sets)}")

    ref_by_stem = {f.stem: f for f in sets[reference]}
    for fmt, frames in sets.items():
        if fmt == reference:
            continue
        by_stem = {f.stem: f for f in frames}
        if set(by_stem) != set(ref_by_stem):
            missing = sorted(set(ref_by_stem) - set(by_stem))
            extra = sorted(set(by_stem) - set(ref_by_stem))
            if missing:
                problems.append(f"{fmt}: missing frames {missing[:5]}{' ...' if len(missing) > 5 else ''}")
            if extra:
                problems.append(f"{fmt}: unexpected frames {extra[:5]}{' ...' if len(extra) > 5 else ''}")

        for stem, ref in ref_by_stem.items():
            other = by_stem.get(stem)
            if other is None:
                continue
            if len(other) != len(ref):
                problems.append(f"{stem}: {fmt} has {len(other)} boxes, {reference} has {len(ref)}")
                continue
            for i, (a, b) in enumerate(zip(ref.boxes, other.boxes)):
                if a.cls != b.cls:
                    problems.append(f"{stem}#{i}: {fmt} class {b.cls} != {reference} class {a.cls}")
                if _round_quad(a) != _round_quad(b):
                    problems.append(f"{stem}#{i}: {fmt} corners disagree with {reference}")
    return problems


def count_boxes(frames: Iterable[FrameAnnotations]) -> int:
    """Total number of boxes across frames."""
    return sum(len(f) for f in frames)


def infer_kind(frames: Iterable[FrameAnnotations]) -> str:
    """'obb' if any box is rotated, otherwise 'hbb'."""
    return "obb" if any(not b.is_axis_aligned for f in frames for b in f.boxes) else "hbb"


def apply_difficult(frames: Sequence[FrameAnnotations], source: Sequence[FrameAnnotations]) -> int:
    """
    Copy the ``difficult`` flag from one reading of a set onto another, by row index.

    Only DOTA and Pascal VOC can express the flag, so a conversion out of YOLO or COCO would
    otherwise silently reset it to 0. Both readings must describe the same boxes in the same order,
    which is what the row alignment of these formats guarantees. Returns the number of flags set.
    """
    by_stem = {f.stem: f for f in source}
    n = 0
    for frame in frames:
        other = by_stem.get(frame.stem)
        if other is None:
            raise ValueError(f"No difficult flags found for frame {frame.stem}")
        if len(other) != len(frame):
            raise ValueError(
                f"Cannot copy difficult flags for {frame.stem}: {len(other)} boxes there, {len(frame)} here"
            )
        for box, ref in zip(frame.boxes, other.boxes):
            box.difficult = ref.difficult
            n += ref.difficult
    return n


def read_confidences(directory: Path) -> Dict[str, List[float]]:
    """
    Read a side-car confidence directory: one ``<stem>.txt`` per frame, one score per line.

    This is what ``hbb2obb --confidence_dir`` writes, for annotation sets whose label files have
    to stay strictly standard and so cannot carry the score as a trailing column. The scores are
    row-aligned with the labels of the same stem.
    """
    out: Dict[str, List[float]] = {}
    for path in sorted(Path(directory).glob("*.txt")):
        out[path.stem] = [float(line) for line in path.read_text(encoding="utf-8").split() if line]
    return out


def apply_difficult_from_confidence(
    frames: Sequence[FrameAnnotations], scores: Dict[str, List[float]], threshold: float
) -> int:
    """
    Set ``difficult`` on every box whose confidence is below ``threshold``. Returns how many.

    The scores stay out of the boxes on purpose: this is for writing a DOTA or Pascal VOC set
    whose flag says "do not trust this box's orientation" while the coordinates and the other
    formats beside them carry nothing extra. A fallback box scores 0.0 and is therefore always
    flagged, whatever the threshold.
    """
    n = 0
    for frame in frames:
        row = _rows_for(frame, scores)
        for box, score in zip(frame.boxes, row):
            box.difficult = int(score < threshold)
            n += box.difficult
    return n


def _rows_for(frame: FrameAnnotations, scores: Dict[str, List[float]]) -> List[float]:
    """The confidence row for one frame, checked against its box count."""
    row = scores.get(frame.stem)
    if row is None:
        raise ValueError(f"No confidence scores found for frame {frame.stem}")
    if len(row) != len(frame):
        raise ValueError(f"Cannot align confidences for {frame.stem}: {len(row)} scores, {len(frame)} boxes")
    return row


# ------------------------------------------------------------------------------------ set level
def read_set(
    source: Path,
    fmt: Optional[str] = None,
    names: Optional[Sequence[str]] = None,
    sizes: Optional[Dict[str, Tuple[int, int]]] = None,
    default_size: Optional[Tuple[int, int]] = None,
) -> Tuple[List[FrameAnnotations], List[str], str]:
    """
    Read a whole annotation set. Returns its frames, the class names, and the detected format.

    ``source`` is a directory of per-frame files, or the .json file itself for COCO. Image sizes
    come from ``sizes`` (keyed by frame stem), from ``default_size``, or from the format itself
    where it records them; they are needed to denormalize relative YOLO coordinates and to write
    the formats that declare an image size.
    """
    fmt = fmt or detect_format(source)
    sizes = sizes or {}
    known = list(names) if names is not None else None
    discovered: List[str] = []

    def size_for(stem: str) -> Tuple[int, int]:
        if stem in sizes:
            return sizes[stem]
        if default_size is not None:
            return default_size
        return (0, 0)

    if fmt == "coco":
        path = source if source.is_file() else next(iter(sorted(source.glob("*.json"))))
        frames, found = read_coco(path, known)
        for frame in frames:
            if not frame.width or not frame.height:
                frame.width, frame.height = size_for(frame.stem)
        return frames, found, fmt

    if not source.is_dir():
        raise ValueError(f"{fmt} annotations are one file per frame, so {source} must be a directory")

    frames: List[FrameAnnotations] = []
    for path in label_files(source, fmt):
        stem = path.stem
        width, height = size_for(stem)
        if fmt == "yolo":
            boxes = read_yolo(path, width, height)
        elif fmt == "dota":
            if known is None:
                raise ValueError("DOTA files name their classes, so a label map is required to read them")
            boxes = read_dota(path, known)
        elif fmt == "voc":
            if known is None:
                known = _names_from_voc(source)
            boxes, vw, vh = read_voc(path, known)
            width, height = (vw or width), (vh or height)
        elif fmt == "labelme":
            boxes, lw, lh, discovered = read_labelme(path, names, discovered)
            known = known or discovered
            width, height = (lw or width), (lh or height)
        else:
            raise ValueError(f"Unsupported format: {fmt}")
        frames.append(FrameAnnotations(stem, width, height, boxes))

    return frames, (known or discovered), fmt


def _names_from_voc(directory: Path) -> List[str]:
    """Collect the class names a directory of VOC files uses, in first-seen order."""
    names: List[str] = []
    for path in sorted(directory.glob("*.xml")):
        for obj in ET.parse(path).getroot().findall("object"):
            name = obj.findtext("name")
            if name and name not in names:
                names.append(name)
    return names


def write_set(
    frames: Sequence[FrameAnnotations],
    dest: Path,
    fmt: str,
    names: Sequence[str],
    kind: str,
    normalize: bool = False,
    precision: Optional[int] = None,
    dota_ext: str = DOTA_EXT,
    image_ext: str = ".jpg",
    coco_name: Optional[str] = None,
    **kwargs,
) -> List[Path]:
    """Write a whole annotation set in one format. Returns the paths written."""
    obb = kind == "obb"
    if obb and fmt not in OBB_FORMATS:
        raise ValueError(f"{fmt} cannot represent oriented boxes; it supports {list(HBB_FORMATS)}")
    if not obb and fmt not in HBB_FORMATS:
        raise ValueError(f"{fmt} cannot represent horizontal boxes; it supports {list(OBB_FORMATS)}")

    if fmt == "coco":
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / (coco_name or f"coco_annotations_{kind}.json")
        write_coco(path, frames, names, obb, image_ext=image_ext, **kwargs)
        return [path]

    dest.mkdir(parents=True, exist_ok=True)
    written = []
    for frame in frames:
        if fmt == "yolo":
            path = dest / f"{frame.stem}.txt"
            write_yolo(path, frame.boxes, frame.width, frame.height, obb, normalize, precision)
        elif fmt == "dota":
            path = dest / f"{frame.stem}{dota_ext}"
            write_dota(path, frame.boxes, names, **kwargs)
        elif fmt == "voc":
            path = dest / f"{frame.stem}.xml"
            write_voc(path, frame.boxes, names, frame.stem, frame.width, frame.height, image_ext, **kwargs)
        elif fmt == "labelme":
            path = dest / f"{frame.stem}.json"
            write_labelme(path, frame.boxes, names, frame.stem, frame.width, frame.height, obb, image_ext)
        else:
            raise ValueError(f"Unsupported format: {fmt}")
        written.append(path)
    return written
