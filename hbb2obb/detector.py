# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
Produce horizontal bounding boxes with an Ultralytics detector.

This is the step before the conversion for anyone who has images but no annotations: run a
detector over them, write YOLO HBB TXT, and feed that to ``hbb2obb``. The output carries the
detector confidence in the 6th column, which ``hbb2obb --confidence_source detector`` (or
``combined``) can then carry through to the OBBs.

It also merges: given a set of hand-drawn boxes, ``merge_detections`` keeps their geometry and
only attaches the confidence of the detection that backs each one, which is how an existing
manually corrected set gains confidence scores without losing the corrections.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# The ultralytics and converter imports are deferred into the functions that need them, so that
# importing this module for its registry alone (which --help does) stays fast.

# Curated detectors that are not Ultralytics' own, keyed by the name --model accepts. Anything
# not listed here is passed through: an Ultralytics model name or a path to a local .pt file,
# or a Hugging Face reference written as '<repo_id>/<file>.pt'.
HF_URL = "https://huggingface.co/{repo}/resolve/main/{filename}"


@dataclass(frozen=True)
class DetectorSpec:
    """A known detector: where its weights come from and the settings it was validated at."""

    filename: str
    url: str
    imgsz: int
    conf: float
    iou: float
    classes: Optional[Tuple[int, ...]]
    description: str


SUPPORTED_DETECTORS: Dict[str, DetectorSpec] = {
    "geotrax": DetectorSpec(
        filename="geotrax_hbb_yolov8s_1920_v1.pt",
        url=HF_URL.format(repo="rfonod/geo-trax", filename="geotrax_hbb_yolov8s_1920_v1.pt"),
        imgsz=1920,
        conf=0.25,
        iou=0.45,
        # Car, Bus, Truck, Motorcycle. The model also carries pedestrian and bicycle classes,
        # which are underrepresented in its training data and not recommended for use.
        classes=(0, 1, 2, 3),
        description="YOLOv8s vehicle detector for high-altitude drone imagery (Geo-trax, CC BY 4.0)",
    ),
}

# Settings for a detector that is not in the registry, matching the Ultralytics defaults.
DEFAULT_SPEC = DetectorSpec(
    filename="",
    url="",
    imgsz=640,
    conf=0.25,
    iou=0.7,
    classes=None,
    description="",
)

WEIGHTS_DIR = Path("models")

# Cache of loaded detectors, keyed by the resolved weights path, so that a run over a directory
# of images loads the model once rather than once per image, as the SAM cache does.
_MODEL_CACHE: Dict[str, Any] = {}


def spec_for(model: str) -> DetectorSpec:
    """The registered settings for a detector name, or the Ultralytics defaults for anything else."""
    return SUPPORTED_DETECTORS.get(model, DEFAULT_SPEC)


def resolve_weights(model: str) -> Path:
    """
    Resolve a ``--model`` value to a weights file on disk, downloading it if it is not there yet.

    Four things are accepted, in this order: a name in ``SUPPORTED_DETECTORS``, a path to an
    existing file, a Hugging Face reference ``<repo_id>/<file>.pt``, and an Ultralytics model
    name such as ``yolo11s.pt``. Everything lands in ``models/`` beside the SAM checkpoints,
    which is the convention the rest of the package uses, and the Ultralytics names are left
    for Ultralytics itself to fetch on first use.
    """
    if model in SUPPORTED_DETECTORS:
        spec = SUPPORTED_DETECTORS[model]
        return download_weights(spec.url, WEIGHTS_DIR / spec.filename)

    path = Path(model)
    if path.exists():
        return path

    if "/" in model and model.endswith(".pt"):
        repo, _, filename = model.rpartition("/")
        return download_weights(HF_URL.format(repo=repo, filename=filename), WEIGHTS_DIR / filename)

    return WEIGHTS_DIR / (model if model.endswith(".pt") else f"{model}.pt")


def download_weights(url: str, destination: Path) -> Path:
    """Fetch a checkpoint to ``destination`` unless it is already there."""
    if destination.exists():
        return destination

    from ultralytics.utils.downloads import safe_download

    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}\n  -> {destination}")
    safe_download(url, file=destination, unzip=False)
    return destination


def load_detector(model: str = "geotrax"):
    """Load a detector by name or path, reusing a cached instance when one is already loaded."""
    from ultralytics import YOLO

    weights = resolve_weights(model)
    key = str(weights)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = YOLO(weights)
    return _MODEL_CACHE[key]


def clear_detector_cache() -> None:
    """Clear the in-process detector cache, releasing the associated memory."""
    _MODEL_CACHE.clear()


def detect_hbb(
    img: np.ndarray,
    model: str = "geotrax",
    imgsz: int = None,
    conf: float = None,
    iou: float = None,
    classes: Sequence[int] = None,
    class_map: Dict[int, int] = None,
    max_det: int = None,
    device: str = None,
    model_kwargs: Dict[str, Any] = None,
) -> np.ndarray:
    """
    Detect horizontal bounding boxes in one image.

    Args:
        img: Image as read by ``cv2.imread`` (BGR)
        model: Detector name, path or Hugging Face reference, as ``resolve_weights`` accepts
        imgsz: Inference resolution (default: the detector's own, 640 if it is not registered)
        conf: Confidence threshold (default: the detector's own)
        iou: NMS IoU threshold (default: the detector's own)
        classes: Class IDs to keep, in the detector's own numbering (default: the detector's own)
        class_map: Remap the detector's class IDs onto yours; boxes of an unlisted class are
                   dropped, so it selects and renumbers in one step
        max_det: Maximum detections per image
        device: Inference device, e.g. 'cpu', '0', 'mps' (default: Ultralytics picks)
        model_kwargs: Additional keyword arguments for the Ultralytics predictor

    Returns:
        An (N, 6) array of ``class, x_center, y_center, width, height, confidence``, the
        coordinates in absolute pixels, ordered by descending confidence.
    """
    spec = spec_for(model)
    detector = load_detector(model)

    kwargs = dict(model_kwargs or {})
    kwargs.setdefault("imgsz", imgsz if imgsz is not None else spec.imgsz)
    kwargs.setdefault("conf", conf if conf is not None else spec.conf)
    kwargs.setdefault("iou", iou if iou is not None else spec.iou)
    selected = classes if classes is not None else spec.classes
    if selected is not None:
        kwargs.setdefault("classes", list(selected))
    if max_det is not None:
        kwargs.setdefault("max_det", max_det)
    if device is not None:
        kwargs.setdefault("device", device)

    result = detector(img, verbose=False, **kwargs)[0]
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return np.empty((0, 6))

    rows = np.column_stack(
        [
            boxes.cls.cpu().numpy().reshape(-1, 1),
            boxes.xywh.cpu().numpy(),
            boxes.conf.cpu().numpy().reshape(-1, 1),
        ]
    )
    return apply_class_map(rows, class_map)


def apply_class_map(rows: np.ndarray, class_map: Dict[int, int] = None) -> np.ndarray:
    """
    Renumber the class column of detection rows, dropping the classes the map does not name.

    A detector trained on COCO calls a car 2 and a bus 5; ``{2: 0, 5: 1}`` turns those into the
    0 and 1 of a four-class vehicle label map and discards everything else.
    """
    if not class_map or len(rows) == 0:
        return rows

    kept = np.array([int(cls) in class_map for cls in rows[:, 0]], dtype=bool)
    rows = rows[kept].copy()
    for i, cls in enumerate(rows[:, 0]):
        rows[i, 0] = class_map[int(cls)]
    return rows


def parse_class_map(spec: str) -> Dict[int, int]:
    """Parse a ``--class_map`` string such as ``'2=0,5=1,7=2'`` into ``{2: 0, 5: 1, 7: 2}``."""
    if not spec:
        return {}

    mapping = {}
    for pair in spec.split(","):
        source, sep, target = pair.partition("=")
        if not sep:
            raise ValueError(f"Invalid class map entry '{pair}', expected 'source=target'")
        mapping[int(source)] = int(target)
    return mapping


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert (N, 4) center-form boxes to corner form."""
    if len(boxes) == 0:
        return np.empty((0, 4))
    xy, wh = boxes[:, :2], boxes[:, 2:4] / 2
    return np.hstack([xy - wh, xy + wh])


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of every box in ``a`` against every box in ``b``, both (N, 4) in corner form."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))

    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(rb - lt, 0, None), axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


@dataclass
class MergeReport:
    """What ``merge_detections`` found, so that a caller can report it or review the leftovers."""

    scores: np.ndarray = field(default_factory=lambda: np.empty((0,)))
    matched: List[int] = field(default_factory=list)
    missed: List[int] = field(default_factory=list)
    extra: List[int] = field(default_factory=list)
    conflicts: List[Tuple[int, int]] = field(default_factory=list)


def merge_detections(manual: np.ndarray, detected: np.ndarray, iou_threshold: float = 0.5) -> MergeReport:
    """
    Attach detector confidences to hand-drawn boxes without touching their geometry.

    The hand-drawn set is authoritative: it stays as it is, in its own order, and each box only
    takes the confidence of the detection that overlaps it most. A box the detector missed gets
    1.0, because a box somebody drew by hand is not less certain than one a model proposed, it
    is certain by construction. Detections that back no hand-drawn box are reported rather than
    added, since whether they are real objects the annotator missed is a judgement call.

    Matching is greedy on IoU and class-agnostic, so a vehicle the detector called a truck and
    the annotator called a car still matches; the disagreement is reported in ``conflicts``.

    Args:
        manual: (M, 5) hand-drawn boxes as ``class, x_center, y_center, width, height``
        detected: (N, 6) detections as ``class, x_center, y_center, width, height, confidence``
        iou_threshold: Minimum IoU for a detection to be considered the same object

    Returns:
        A ``MergeReport`` whose ``scores`` has one confidence per hand-drawn box, in order.
    """
    report = MergeReport(scores=np.ones(len(manual)), extra=list(range(len(detected))))
    if len(manual) == 0 or len(detected) == 0:
        report.missed = list(range(len(manual)))
        return report

    ious = box_iou(xywh_to_xyxy(manual[:, 1:5]), xywh_to_xyxy(detected[:, 1:5]))
    taken_manual, taken_detected = set(), set()
    order = np.argsort(ious, axis=None)[::-1]
    for flat in order:
        i, j = np.unravel_index(flat, ious.shape)
        if ious[i, j] < iou_threshold:
            break
        if i in taken_manual or j in taken_detected:
            continue
        taken_manual.add(int(i))
        taken_detected.add(int(j))
        report.scores[i] = detected[j, 5]
        if int(manual[i, 0]) != int(detected[j, 0]):
            report.conflicts.append((int(i), int(j)))

    report.matched = sorted(taken_manual)
    report.missed = [i for i in range(len(manual)) if i not in taken_manual]
    report.extra = [j for j in range(len(detected)) if j not in taken_detected]
    return report


def save_hbb_annotations(
    hbb_annotations: np.ndarray,
    hbb_dir: Path,
    img_path: Path,
    scores: Union[Sequence[float], np.ndarray] = None,
    precision: int = 2,
    normalize: bool = False,
    img_shape: Tuple[int, int] = None,
) -> Path:
    """
    Save HBB annotations to a text file, the format ``hbb2obb`` reads back.

    Each line is ``class x_center y_center width height``, in absolute pixels unless
    ``normalize`` is set. When ``scores`` is provided, the confidence is appended as a 6th
    field. Trailing zeros are trimmed, so a whole pixel is written as such rather than padded
    to the full precision.

    Args:
        hbb_annotations: (N, 5) or (N, 6) array; a 6th column is ignored in favour of ``scores``
        hbb_dir: Directory to write into (default: ``<img_path>/../../labels_hbb``)
        img_path: Image the annotations belong to, which names the output file
        scores: Per-box confidence, written as the trailing field
        precision: Decimal places for the coordinates
        normalize: Write coordinates relative to the frame size instead of in pixels
        img_shape: (width, height) of the frame, required when ``normalize`` is set

    Returns:
        The path written.
    """
    from hbb2obb.converter import format_annotation_line, resolve_output_dir

    if normalize and not img_shape:
        raise ValueError("img_shape is required to write normalized coordinates")

    hbb_dir = resolve_output_dir(hbb_dir, img_path, "labels_hbb")
    save_filepath = hbb_dir / (img_path.stem + ".txt")
    scale = np.array([*img_shape, *img_shape], dtype=float) if normalize else np.ones(4)
    scores = None if scores is None else list(scores)

    with open(save_filepath, "w", encoding="utf-8") as f:
        for i, box in enumerate(hbb_annotations):
            coords = np.asarray(box[1:5], dtype=float) / scale
            fields = f"{int(box[0])} " + " ".join(_trim(value, precision) for value in coords)
            f.write(format_annotation_line(fields, scores, i) + "\n")
    return save_filepath


def _trim(value: float, precision: int) -> str:
    """Format a coordinate at the given precision without trailing zeros ('12.50' -> '12.5')."""
    text = f"{value:.{precision}f}"
    return text.rstrip("0").rstrip(".") if "." in text else text
