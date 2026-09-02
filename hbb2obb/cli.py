#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
Command-line interface for HBB2OBB.
"""

import argparse
from pathlib import Path

from hbb2obb import __version__
from hbb2obb.version_check import check_for_updates_once

SUPPORTED_SAM_MODELS = [
    "sam_b",
    "sam_l",
    "mobile_sam",
    "sam2_t",
    "sam2_s",
    "sam2_b",
    "sam2_l",
    "sam2.1_t",
    "sam2.1_s",
    "sam2.1_b",
    "sam2.1_l",
    "sam3",
    "FastSAM-s",
    "FastSAM-x",
]

# Shown at the end of `hbb2obb --help` so a new user discovers the rest of the toolkit.
ENTRY_POINTS_EPILOG = (
    "The hbb2obb toolkit ships six commands:\n"
    "  hbb2obb           convert HBB annotations to OBBs by prompting SAM (this command)\n"
    "  hbb2obb-detect    detect HBBs with an Ultralytics model, for images that have none\n"
    "  hbb2obb-convert   convert annotations between YOLO, DOTA, Pascal VOC, COCO and LabelMe\n"
    "  hbb2obb-view      inspect HBB and OBB annotations over their images, interactively\n"
    "  hbb2obb-eval      score predicted OBBs against ground truth (precision, recall, F1, IoU)\n"
    "  hbb2obb-optimize  search hyperparameters, as one sweep or a whole benchmark\n\n"
    "Run any of them with --help for its own options, e.g. hbb2obb-detect --help."
)


def provenance_path(label_dir: Path) -> Path:
    """
    Where a conversion's or detection's PROVENANCE.txt goes: beside the label directory, not in it.

    It is a ``.txt`` file, and a label directory is read with ``labels/*.txt`` by most tooling that
    is not this one, so a record left inside would be parsed as a frame. One level up it sits with
    ``images/`` and ``labels/``, describing the set rather than belonging to it. ``hbb2obb-optimize``
    is unaffected: its record goes in the output folder, which holds runs rather than labels.
    """
    return label_dir.parent / "PROVENANCE.txt"


def main_hbb2obb():
    """
    Run the HBB to OBB conversion from command line.
    """

    parser = argparse.ArgumentParser(
        description="Convert HBB to OBB annotations",
        epilog=ENTRY_POINTS_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", "-V", action="version", version=f"%(prog)s {__version__}")

    # Main arguments
    parser.add_argument("img_source", type=Path, help="Path to an image or directory containing images")
    parser.add_argument(
        "--hbb_dir", "-hd", type=Path, help="Directory containing HBB annotations (default: img_source/../labels_hbb)"
    )
    parser.add_argument(
        "--obb_dir", "-od", type=Path, help="Directory to save OBB annotations (default: img_source/../labels_obb)"
    )
    parser.add_argument(
        "--polygon_dir",
        "-pd",
        type=Path,
        help="Directory to save polygon annotations (default: img_source/../labels_polygon)",
    )
    parser.add_argument(
        "--sam_models",
        "-sm",
        type=str,
        default=["sam_b"],
        nargs='+',
        choices=SUPPORTED_SAM_MODELS,
        help="Name(s) of SAM model(s) to use (default: sam_b). Multiple models can be specified to average results.",
    )
    parser.add_argument("--imgsz", type=int, default=1280, help="Image size for SAM model inference (default: 1280)")
    parser.add_argument(
        "--scale_factors",
        "-sf",
        type=float,
        default=[0.05],
        nargs='+',
        help="Factor(s) to scale HBBs (default: 0.05). Use one value for uniform or two values for short/long sides.",
    )
    parser.add_argument(
        "--opening_kernel_percentage",
        "-okp",
        type=float,
        default=0.15,
        help="Percentage of mask's smaller dimension for morphological opening. Ignored if <= 0.",
    )

    # Visualization control arguments
    viz_group = parser.add_argument_group('visualization options')
    viz_group.add_argument("--save_img", action="store_true", help="Save visualization images (default: False)")
    viz_group.add_argument(
        "--viz_dir", type=Path, help="Directory to save visualization images (default: same as 'obb_dir')"
    )
    viz_group.add_argument(
        "--show_hbb", action="store_true", default=True, help="Show horizontal bounding boxes (default: True)"
    )
    viz_group.add_argument("--hide_hbb", action="store_false", dest="show_hbb", help="Hide horizontal bounding boxes")
    viz_group.add_argument(
        "--show_masks", action="store_true", default=True, help="Show segmentation masks (default: True)"
    )
    viz_group.add_argument("--hide_masks", action="store_false", dest="show_masks", help="Hide segmentation masks")
    viz_group.add_argument(
        "--show_segments", action="store_true", default=True, help="Show segmentation contours (default: True)"
    )
    viz_group.add_argument(
        "--hide_segments", action="store_false", dest="show_segments", help="Hide segmentation contours"
    )
    viz_group.add_argument(
        "--show_obb", action="store_true", default=True, help="Show oriented bounding boxes (default: True)"
    )
    viz_group.add_argument("--hide_obb", action="store_false", dest="show_obb", help="Hide oriented bounding boxes")
    viz_group.add_argument(
        "--show_class_labels",
        "--show_labels",
        action="store_true",
        default=True,
        dest="show_labels",
        help="Show class labels (default: True). '--show_labels' is a deprecated alias.",
    )
    viz_group.add_argument(
        "--hide_class_labels",
        "--hide_labels",
        action="store_false",
        dest="show_labels",
        help="Hide class labels; does not affect --show_confidence. '--hide_labels' is a deprecated alias.",
    )
    viz_group.add_argument(
        "--show_confidence",
        action="store_true",
        default=False,
        help="Show the per-OBB confidence score in the visualization, independently of class labels (default: False)",
    )
    viz_group.add_argument(
        "--hide_confidence", action="store_false", dest="show_confidence", help="Hide the confidence score value"
    )

    # Miscellaneous arguments
    parser.add_argument(
        "--save_confidence",
        action="store_true",
        help="Append a per-OBB confidence score as a 10th column in the output TXT files",
    )
    parser.add_argument(
        "--confidence_source",
        "-cs",
        type=str,
        default="conversion",
        choices=["conversion", "detector", "combined"],
        help=(
            "Which score the reported confidence carries: 'conversion' (heuristic conversion quality, "
            "default), 'detector' (the confidence column of the HBB input), or 'combined' (their product). "
            "Only affects output when --save_confidence or --show_confidence is used."
        ),
    )
    parser.add_argument(
        "--confidence_dir",
        "-cd",
        nargs="?",
        const="",
        default=None,
        metavar="DIR",
        help=(
            "Write the per-OBB confidence scores to their own directory, one score per line, "
            "row-aligned with the label files. Give it bare for img_source/../labels_confidence. "
            "Use this rather than --save_confidence when the label files have to stay standard: "
            "Ultralytics and other YOLO OBB readers reject a 10th column"
        ),
    )
    parser.add_argument(
        "--save_polygon",
        action="store_true",
        help="Also save the segmentation polygon of each object, a tighter alternative to its OBB",
    )
    parser.add_argument(
        "--polygon_epsilon",
        "-pe",
        type=float,
        default=0.0,
        help="Simplify saved polygons with an epsilon of this fraction of the contour perimeter (0 to disable)",
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Inference device for the SAM model(s), e.g. 'cpu', '0', 'cuda:0', 'mps' (default: ultralytics picks)",
    )
    parser.add_argument(
        "--model_kwargs",
        "-k",
        type=str,
        help="Additional keyword arguments for ultralytics model inference in format 'key1=value1,key2=value2'",
    )
    parser.add_argument(
        "--save_provenance",
        action="store_true",
        help=(
            "Write a PROVENANCE.txt one level above the OBB annotations, beside the label directory "
            "rather than inside it: the command that reproduces them, the hbb2obb version and commit, "
            "the dependency versions and the SHA-256 of every checkpoint used"
        ),
    )
    parser.add_argument("--no_bar", "-nb", action="store_true", help="Disable tqdm progress bar display")

    args = parser.parse_args()

    check_for_updates_once()

    import tqdm

    from hbb2obb.converter import (
        hbb2obb,
        resolve_output_dir,
        save_confidence_annotations,
        save_obb_annotations,
        save_polygon_annotations,
        unpack_results,
    )
    from hbb2obb.utils import get_hbb_dir, get_image_paths, process_ultralytics_kwargs

    model_kwargs = process_ultralytics_kwargs(args.model_kwargs)

    # --confidence_dir writes the scores beside the labels rather than into them, so it needs the
    # scores computed without putting them in the label file. An empty string is the bare form,
    # meaning the conventional location.
    write_confidence_dir = args.confidence_dir is not None
    confidence_dir = Path(args.confidence_dir) if args.confidence_dir else None
    want_confidence = args.save_confidence or write_confidence_dir

    image_paths = get_image_paths(args.img_source)
    for img_path in tqdm.tqdm(image_paths, desc="Processing images", leave=True, disable=args.no_bar):
        result = hbb2obb(
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
            show_confidence=args.show_confidence,
            model_kwargs=model_kwargs,
            device=args.device,
            return_confidence=want_confidence,
            confidence_source=args.confidence_source,
            return_contours=args.save_polygon,
        )

        # Unpack the extras hbb2obb() appends for the flags that were requested
        obb_annotations, confidences, contours = unpack_results(
            result, return_confidence=want_confidence, return_contours=args.save_polygon
        )

        # The trailing column is written only when it was asked for by name; --confidence_dir on
        # its own leaves the label files with their standard nine fields.
        in_line = confidences if args.save_confidence else None
        save_obb_annotations(obb_annotations, args.obb_dir, img_path, confidences=in_line)
        if write_confidence_dir:
            save_confidence_annotations(confidences, confidence_dir, img_path)
        if args.save_polygon:
            save_polygon_annotations(
                contours, obb_annotations, args.polygon_dir, img_path, in_line, args.polygon_epsilon
            )

    if args.save_provenance and image_paths:
        from hbb2obb import provenance

        obb_dir = resolve_output_dir(args.obb_dir, image_paths[0], "labels_obb")
        provenance.write_conversion_provenance(
            out=provenance_path(obb_dir),
            img_source=args.img_source,
            hbb_dir=get_hbb_dir(args.img_source, args.hbb_dir),
            obb_dir=obb_dir,
            sam_models=args.sam_models,
            imgsz=args.imgsz,
            scale_factors=args.scale_factors,
            opening_kernel_percentage=args.opening_kernel_percentage,
            confidence_source=args.confidence_source,
            model_kwargs=args.model_kwargs,
            device=args.device,
        )


def main_hbb2obb_detect():
    """
    Detect horizontal bounding boxes with an Ultralytics detector, for images that have none.
    """

    from hbb2obb.detector import SUPPORTED_DETECTORS

    parser = argparse.ArgumentParser(
        description="Detect horizontal bounding boxes and write them as YOLO TXT, ready for hbb2obb",
        epilog="Registered detectors: "
        + "; ".join(f"{name} ({spec.description})" for name, spec in SUPPORTED_DETECTORS.items()),
    )
    parser.add_argument("--version", "-V", action="version", version=f"%(prog)s {__version__}")

    parser.add_argument("img_source", type=Path, help="Path to an image or directory containing images")
    parser.add_argument(
        "--hbb_dir", "-hd", type=Path, help="Directory to save HBB annotations (default: img_source/../labels_hbb)"
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="geotrax",
        help="Registered detector (default: geotrax), an Ultralytics model name or .pt path, or a "
        "Hugging Face reference written as '<user>/<repo>/<file>.pt'",
    )

    detect_group = parser.add_argument_group('detection options')
    detect_group.add_argument("--imgsz", type=int, help="Inference resolution (default: the detector's own)")
    detect_group.add_argument("--conf", type=float, help="Confidence threshold (default: the detector's own)")
    detect_group.add_argument("--iou", type=float, help="NMS IoU threshold (default: the detector's own)")
    detect_group.add_argument(
        "--classes",
        type=int,
        nargs='+',
        help="Class IDs to keep, in the detector's own numbering (default: the detector's own)",
    )
    detect_group.add_argument(
        "--class_map",
        "-cm",
        type=str,
        help="Renumber the detector's classes onto yours, as 'source=target,...' (e.g. '2=0,5=1,7=2' "
        "for a COCO-trained model). Boxes of a class the map does not name are dropped.",
    )
    detect_group.add_argument("--max_det", type=int, help="Maximum detections per image")
    detect_group.add_argument("--device", type=str, help="Inference device, e.g. 'cpu', '0', 'mps'")
    detect_group.add_argument(
        "--model_kwargs",
        "-k",
        type=str,
        help="Additional keyword arguments for ultralytics model inference in format 'key1=value1,key2=value2'",
    )

    merge_group = parser.add_argument_group('merging with existing annotations')
    merge_group.add_argument(
        "--merge_with",
        "-mw",
        type=Path,
        help="Directory of hand-drawn HBBs to score instead of writing the detections themselves. "
        "Their geometry and order are kept untouched; each box only takes the confidence of the "
        "detection covering it, and a box the detector missed gets 1.0.",
    )
    merge_group.add_argument(
        "--merge_iou", type=float, default=0.5, help="Minimum IoU for a detection to score a box (default: 0.5)"
    )
    merge_group.add_argument(
        "--extras_dir",
        type=Path,
        help="Write the detections that back no hand-drawn box here, as their own HBB set, so they "
        "can be reviewed with hbb2obb-view before being accepted or discarded",
    )

    parser.add_argument(
        "--map_path", "-mp", type=Path, help="Path to a label map YAML, used to name classes in the summary"
    )
    parser.add_argument("--precision", "-p", type=int, default=2, help="Decimal places for coordinates (default: 2)")
    parser.add_argument("--normalize", "-n", action="store_true", help="Write coordinates relative to [0, 1]")
    parser.add_argument(
        "--no_confidence", action="store_false", dest="save_confidence", help="Do not write the confidence column"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Allow writing into an output directory that already holds labels"
    )
    parser.add_argument(
        "--save_provenance",
        action="store_true",
        help=(
            "Write a PROVENANCE.txt one level above the annotations, beside the label directory "
            "rather than inside it: the command that reproduces them, the settings "
            "used and the SHA-256 of the detector checkpoint that actually ran"
        ),
    )
    parser.add_argument("--no_bar", "-nb", action="store_true", help="Disable tqdm progress bar display")

    args = parser.parse_args()

    check_for_updates_once()

    import cv2
    import numpy as np
    import tqdm

    from hbb2obb.detector import detect_hbb, merge_detections, parse_class_map, save_hbb_annotations
    from hbb2obb.utils import Annotations, get_image_paths, process_ultralytics_kwargs

    image_paths = get_image_paths(args.img_source)
    if not image_paths:
        raise SystemExit(f"No images found in {args.img_source}")

    root = args.img_source if args.img_source.is_dir() else args.img_source.parent
    hbb_dir = args.hbb_dir or root.parent / "labels_hbb"
    if not args.overwrite and hbb_dir.is_dir() and any(hbb_dir.glob("*.txt")):
        raise SystemExit(
            f"{hbb_dir} already holds annotations; pass --overwrite to replace them or --hbb_dir to write elsewhere"
        )

    try:
        class_map = parse_class_map(args.class_map)
    except ValueError as exc:
        parser.error(str(exc))

    model_kwargs = process_ultralytics_kwargs(args.model_kwargs)
    totals = {"boxes": 0, "matched": 0, "missed": 0, "extra": 0, "conflicts": 0}
    counts = {}

    for img_path in tqdm.tqdm(image_paths, desc="Detecting HBBs", leave=True, disable=args.no_bar):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: could not read {img_path}, skipping")
            continue
        img_shape = (img.shape[1], img.shape[0])

        detected = detect_hbb(
            img,
            model=args.model,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            classes=args.classes,
            class_map=class_map,
            max_det=args.max_det,
            device=args.device,
            model_kwargs=model_kwargs,
        )

        if args.merge_with:
            annotations = Annotations(args.merge_with / (img_path.stem + ".txt"), img)
            manual = annotations.hbb_xywh
            report = merge_detections(manual, detected, args.merge_iou)
            boxes, scores = manual, report.scores
            totals["matched"] += len(report.matched)
            totals["missed"] += len(report.missed)
            totals["conflicts"] += len(report.conflicts)
            if args.extras_dir is not None:
                extras = detected[report.extra] if report.extra else np.empty((0, 6))
                save_hbb_annotations(
                    extras, args.extras_dir, img_path, extras[:, 5], args.precision, args.normalize, img_shape
                )
            totals["extra"] += len(report.extra)
        else:
            boxes, scores = detected, detected[:, 5]

        totals["boxes"] += len(boxes)
        for cls in boxes[:, 0].astype(int):
            counts[cls] = counts.get(cls, 0) + 1

        save_hbb_annotations(
            boxes,
            hbb_dir,
            img_path,
            scores if args.save_confidence else None,
            args.precision,
            args.normalize,
            img_shape,
        )

    names = resolve_names(args.map_path, hbb_dir, [])
    print(f"\nWrote {totals['boxes']} boxes over {len(image_paths)} frames to {hbb_dir}")
    for cls in sorted(counts):
        label = names[cls] if cls < len(names) else str(cls)
        print(f"  {cls} {label}: {counts[cls]}")
    if args.merge_with:
        print(
            f"Scored by a detection: {totals['matched']}; kept at 1.0 (missed by the detector): {totals['missed']}; "
            f"class disagreements: {totals['conflicts']}"
        )
        print(f"Detections backing no hand-drawn box: {totals['extra']}", end="")
        print(
            f", written to {args.extras_dir} for review" if args.extras_dir else " (pass --extras_dir to review them)"
        )

    if args.save_provenance:
        from hbb2obb import provenance
        from hbb2obb.detector import resolve_weights, spec_for

        # Record what actually ran, not what was typed: an unset flag takes the registered
        # detector's own validated setting, which is the value worth pinning.
        spec = spec_for(args.model)
        provenance.write_detection_provenance(
            out=provenance_path(hbb_dir),
            img_source=args.img_source,
            hbb_dir=hbb_dir,
            model=args.model,
            weights=resolve_weights(args.model),
            imgsz=args.imgsz if args.imgsz is not None else spec.imgsz,
            conf=args.conf if args.conf is not None else spec.conf,
            iou=args.iou if args.iou is not None else spec.iou,
            classes=args.classes if args.classes is not None else spec.classes,
            merged_with=args.merge_with,
            model_kwargs=args.model_kwargs,
            device=args.device,
        )


def resolve_names(map_path: Path, source: Path, discovered: list) -> list:
    """
    Work out the class names to use, in class-id order.

    A label map YAML wins, then a ``names.txt`` beside the annotations or one directory up, then
    whatever names the source format carried, then generic ``class_0`` placeholders.
    """
    from hbb2obb.utils import load_label_map

    if map_path is not None:
        label_map = load_label_map(map_path)
        if label_map:
            return [label_map[key] for key in sorted(label_map)]

    root = source if source.is_dir() else source.parent
    for directory in (root, root.parent):
        candidate = directory / "names.txt"
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8").split()
        candidate = directory / "classes.yaml"
        if candidate.is_file():
            label_map = load_label_map(candidate)
            if label_map:
                return [label_map[key] for key in sorted(label_map)]

    return list(discovered)


def main_hbb2obb_convert():
    """
    Convert annotations between formats, or check that several formats agree.
    """

    from hbb2obb.formats import ALL_FORMATS, DOTA_EXT

    parser = argparse.ArgumentParser(
        description="Convert bounding box annotations between YOLO, DOTA, Pascal VOC, COCO and LabelMe"
    )
    parser.add_argument("--version", "-V", action="version", version=f"%(prog)s {__version__}")

    parser.add_argument("source", type=Path, help="Directory of annotation files, or the .json file itself for COCO")
    parser.add_argument(
        "--to",
        "-t",
        type=str,
        nargs='+',
        choices=ALL_FORMATS,
        help="Output format(s); several can be written in one pass",
    )
    parser.add_argument(
        "--from",
        "-f",
        dest="src_format",
        type=str,
        choices=ALL_FORMATS,
        help="Input format (default: detected from the files)",
    )
    parser.add_argument(
        "--out_dir", "-o", type=Path, help="Directory to write the converted annotations (default: source directory)"
    )
    parser.add_argument("--map_path", "-mp", type=Path, help="Path to a label map YAML mapping class IDs to names")
    parser.add_argument(
        "--images", "-i", type=Path, help="Image directory, read for the frame dimensions the formats need"
    )
    parser.add_argument("--img_width", "-iw", type=int, help="Frame width, if there is no image directory")
    parser.add_argument("--img_height", "-ih", type=int, help="Frame height, if there is no image directory")
    parser.add_argument(
        "--kind",
        "-k",
        type=str,
        default="auto",
        choices=["auto", "hbb", "obb"],
        help="Write horizontal or oriented boxes (default: auto, oriented if any box is rotated)",
    )
    parser.add_argument("--normalize", "-n", action="store_true", help="Write YOLO coordinates relative to [0, 1]")
    parser.add_argument(
        "--precision", "-p", type=int, help="Decimal places for YOLO output (default: 10 normalized, 2 absolute)"
    )
    parser.add_argument(
        "--dota_ext",
        type=str,
        default=DOTA_EXT,
        help=f"Extension for DOTA files (default: {DOTA_EXT}; use .txt for a standalone labelTxt directory)",
    )
    parser.add_argument(
        "--difficult_from",
        type=str,
        choices=["dota", "voc", "confidence"],
        help=(
            "Take the per-box 'difficult' flag from this format in the source directory. Only DOTA "
            "and Pascal VOC can carry it, so writing either one from YOLO or COCO resets it to 0 "
            "unless this is given. 'confidence' instead derives it from the per-box conversion "
            "score, flagging everything below --difficult_below; the scores come from the source "
            "labels if they carry a trailing column, otherwise from --confidence_dir."
        ),
    )
    parser.add_argument(
        "--confidence_dir",
        metavar="DIR",
        type=Path,
        help=(
            "Side-car directory of per-box confidence scores, as written by 'hbb2obb --confidence_dir'. "
            "Read only by --difficult_from confidence; the scores stay out of the written labels."
        ),
    )
    parser.add_argument(
        "--difficult_below",
        type=float,
        default=0.5,
        help="With --difficult_from confidence, flag every box scoring below this (default: 0.5). "
        "A box that fell back to its source HBB scores 0.0 and is therefore always flagged",
    )
    parser.add_argument(
        "--voc_database", type=str, default="Unknown", help="Value of the <database> field in Pascal VOC output"
    )
    parser.add_argument(
        "--coco_name",
        type=str,
        help="File name for COCO output (default: coco_annotations_<kind>.json). Use it to match a "
        "labels_<name> directory, which is how --verify pairs the two.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Check that every format found under the source encodes the same boxes, and write nothing",
    )
    parser.add_argument("--no_bar", "-nb", action="store_true", help="Disable tqdm progress bar display")

    args = parser.parse_args()

    check_for_updates_once()

    if args.verify:
        raise SystemExit(run_verify(args))
    if not args.to:
        parser.error("--to is required unless --verify is used")

    from hbb2obb import formats

    sizes = formats.image_sizes(args.images)
    default_size = (args.img_width, args.img_height) if args.img_width and args.img_height else None
    names = resolve_names(args.map_path, args.source, [])

    frames, discovered, src_format = formats.read_set(args.source, args.src_format, names or None, sizes, default_size)
    names = names or discovered
    if not names:
        names = [f"class_{i}" for i in range(1 + max((b.cls for f in frames for b in f.boxes), default=-1))]

    kind = formats.infer_kind(frames) if args.kind == "auto" else args.kind
    frames = formats.canonicalize(frames)

    out_dir = args.out_dir or (args.source if args.source.is_dir() else args.source.parent)
    print(
        f"Read {formats.count_boxes(frames)} {kind.upper()} boxes over {len(frames)} frames "
        f"from {args.source} ({src_format})"
    )

    if args.confidence_dir and args.difficult_from != "confidence":
        raise SystemExit(
            "--confidence_dir is read only by --difficult_from confidence, which is what turns the "
            "scores into a per-box flag; without it the scores would go nowhere, since no output "
            "format takes them from a side-car"
        )

    if args.difficult_from == "confidence":
        if args.confidence_dir:
            scores = formats.read_confidences(args.confidence_dir)
        elif all(b.score is not None for f in frames for b in f.boxes) and formats.count_boxes(frames):
            # Every box, not merely some: a set that mixes scored and unscored lines has no score
            # to compare for the unscored ones, and guessing one would flag or spare them silently.
            scores = {f.stem: [b.score for b in f.boxes] for f in frames}
        else:
            raise SystemExit(
                "--difficult_from confidence found no scores: not every box in the source labels "
                "carries a confidence column, so pass --confidence_dir with the side-car directory"
            )
        try:
            n = formats.apply_difficult_from_confidence(frames, scores, args.difficult_below)
        except ValueError as exc:
            raise SystemExit(f"--difficult_from confidence: {exc}") from exc
        print(f"Flagged {n} box(es) difficult, scoring below {args.difficult_below:g}")
    elif args.difficult_from:
        source = args.source if args.source.is_dir() else args.source.parent
        flagged, _, _ = formats.read_set(source, args.difficult_from, names, sizes, default_size)
        print(f"Carried {formats.apply_difficult(frames, flagged)} difficult flag(s) from {args.difficult_from}")

    for fmt in args.to:
        extra = {}
        if fmt == "voc":
            extra = {"database": args.voc_database}
        elif fmt == "coco" and args.coco_name:
            extra = {"coco_name": args.coco_name}
        written = formats.write_set(
            frames,
            out_dir,
            fmt,
            names,
            kind,
            normalize=args.normalize,
            precision=args.precision,
            dota_ext=args.dota_ext,
            **extra,
        )
        print(f"Wrote {len(written)} {fmt} file(s) to {out_dir}")


def run_verify(args) -> int:
    """
    Check that every format present under the source encodes the same boxes.

    A directory holding label files is checked on its own; a directory of label directories is
    checked one subdirectory at a time, pairing ``labels_<name>/`` with a ``coco_annotations_
    <name>.json`` beside it, which is the layout both the sample data and Songdo Vision OBB use.
    """
    from hbb2obb import formats

    root = args.source if args.source.is_dir() else args.source.parent
    targets = []
    if any(formats.label_files(root, fmt) for fmt in ("yolo", "dota", "voc")):
        targets.append((root, None))
    else:
        for sub in sorted(p for p in root.iterdir() if p.is_dir()):
            if not any(sub.iterdir()) or sub.name in formats.SIDECAR_DIRS:
                continue
            # labels_<name>/ pairs with coco_annotations_<name>.json; a directory called plainly
            # labels/, as a published images/ + labels/ release uses, pairs with
            # coco_annotations.json. Any other directory is checked on its own, with no COCO
            # counterpart, so that images/ is not paired with the labels' own file.
            if sub.name == "labels":
                coco = root / "coco_annotations.json"
            elif sub.name.startswith("labels_"):
                coco = root / f"coco_annotations_{sub.name[len('labels_') :]}.json"
            else:
                coco = None
            targets.append((sub, coco if coco is not None and coco.is_file() else None))
    if not targets:
        print(f"No annotation files found under {root}")
        return 1

    sizes = formats.image_sizes(args.images if args.images else root / "images")
    default_size = (args.img_width, args.img_height) if args.img_width and args.img_height else None

    problems, checked = [], 0
    for directory, coco in targets:
        names = resolve_names(args.map_path, directory, [])
        present = [fmt for fmt in formats.ALL_FORMATS if fmt != "coco" and formats.label_files(directory, fmt)]
        sets = {}
        for fmt in present:
            frames, discovered, _ = formats.read_set(directory, fmt, names or None, sizes, default_size)
            names = names or discovered
            sets[fmt] = formats.canonicalize(frames)
        if coco is not None:
            frames, _, _ = formats.read_set(coco, "coco", names or None, sizes, default_size)
            sets["coco"] = formats.canonicalize(frames)

        if len(sets) < 2:
            if sets:
                print(f"{directory.name}: only the {next(iter(sets))} format is present, nothing to compare")
            continue
        found = formats.verify(sets, reference="yolo" if "yolo" in sets else sorted(sets)[0])
        n = formats.count_boxes(next(iter(sets.values())))
        checked += n
        print(
            f"{directory.name}: {len(next(iter(sets.values())))} frames, {n} boxes, {len(sets)} formats "
            f"({', '.join(sorted(sets))})"
        )
        problems.extend(f"{directory.name}/{p}" for p in found)

    if problems:
        print(f"\nFAILED with {len(problems)} problems:")
        for problem in problems[:40]:
            print("  " + problem)
        if len(problems) > 40:
            print(f"  ... and {len(problems) - 40} more")
        return 1
    print(f"OK: {checked} boxes, every format encodes the same boxes")
    return 0


def main_hbb2obb_view():
    """
    Inspect HBB and OBB annotations over their images, interactively or as image files.
    """

    from hbb2obb.formats import ALL_FORMATS

    parser = argparse.ArgumentParser(
        description="View HBB and OBB annotations over their images: pan, zoom, and step through frames",
        epilog=(
            "Keys: q/Esc quit, n/p or arrows change frame, wheel or +/- zoom, f fit, 1 100%%, "
            "h HBBs, l labels, d difficult, c confidence, g polygons, x comparison, s save. "
            "Drag with the left mouse button to pan."
        ),
    )
    parser.add_argument("--version", "-V", action="version", version=f"%(prog)s {__version__}")

    parser.add_argument("img_source", type=Path, help="Path to an image or directory containing images")
    parser.add_argument(
        "--obb_dir", "-od", type=Path, help="Directory of OBB annotations (default: img_source/../labels_obb)"
    )
    parser.add_argument(
        "--hbb_dir", "-hd", type=Path, help="Directory of HBB annotations (default: img_source/../labels_hbb)"
    )
    parser.add_argument(
        "--polygon_dir",
        "-pd",
        type=Path,
        help="Directory of segmentation polygons (default: img_source/../labels_polygon)",
    )
    parser.add_argument(
        "--compare",
        "-c",
        type=Path,
        help="A second OBB set to overlay in blue, for comparing predictions against ground truth",
    )
    parser.add_argument("--map_path", "-mp", type=Path, help="Path to a label map YAML mapping class IDs to names")
    parser.add_argument("--obb_format", type=str, choices=ALL_FORMATS, help="OBB format (default: detected)")
    parser.add_argument("--hbb_format", type=str, choices=ALL_FORMATS, help="HBB format (default: detected)")
    parser.add_argument("--frame", "-f", type=str, help="Frame stem to open first, or the only one to write")
    parser.add_argument("--out_dir", "-o", type=Path, help="Write annotated images here instead of opening a window")
    parser.add_argument(
        "--crops", action="store_true", help="Write a contact sheet of the individual objects rather than whole frames"
    )
    parser.add_argument(
        "--indices",
        type=int,
        nargs='+',
        help="With --crops, restrict the sheet to these box indices, for reviewing particular objects",
    )
    parser.add_argument("--hide_hbb", action="store_false", dest="show_hbb", help="Do not draw the horizontal boxes")
    parser.add_argument(
        "--hide_class_labels", action="store_false", dest="show_labels", help="Do not draw class labels"
    )
    parser.add_argument(
        "--show_confidence", action="store_true", help="Color the OBBs by confidence and print the score"
    )
    parser.add_argument(
        "--confidence_dir",
        "-cd",
        type=Path,
        help="Side-car directory of confidence scores, for labels that carry no confidence column "
        "(default: img_source/../labels_confidence)",
    )
    parser.add_argument("--window", type=str, default="1600x900", help="Window size as WxH (default: 1600x900)")

    args = parser.parse_args()

    check_for_updates_once()

    from hbb2obb import viewer

    paths = viewer.image_paths(args.img_source)
    if not paths:
        raise SystemExit(f"No images found in {args.img_source}")
    root = args.img_source if args.img_source.is_dir() else args.img_source.parent

    # hbb2obb's own output first, then a hand-drawn reference, then the plain labels/ of an
    # images/ + labels/ release, which is what a published OBB dataset usually looks like.
    obb_dir = args.obb_dir or _first_existing(root.parent, "labels_obb", "labels_obb_gt", "labels")
    hbb_dir = args.hbb_dir or _first_existing(root.parent, "labels_hbb")
    polygon_dir = args.polygon_dir or _first_existing(root.parent, "labels_polygon")
    names = resolve_names(args.map_path, obb_dir or root, [])

    from hbb2obb.formats import image_sizes

    sizes = image_sizes(args.img_source)
    obb = viewer.load_annotations(obb_dir, args.obb_format, names, sizes)
    hbb = viewer.load_annotations(hbb_dir, args.hbb_format, names, sizes)
    compared = viewer.load_annotations(args.compare, None, names, sizes)

    # An explicitly requested format that turns up nothing is a mistake worth reporting: without
    # this the window opens and simply draws no boxes, which reads as the viewer being broken.
    requested = (("OBB", obb_dir, args.obb_format, obb), ("HBB", hbb_dir, args.hbb_format, hbb))
    for label, directory, fmt, loaded in requested:
        if fmt and not loaded:
            where = f"in {directory}" if directory is not None else f"next to {args.img_source}"
            raise SystemExit(f"No {fmt} {label} annotations {where}")
    if not obb and not hbb:
        raise SystemExit(f"No annotations found next to {args.img_source}; pass --obb_dir or --hbb_dir")
    if args.show_confidence and not any(b.score is not None for boxes in obb.values() for b in boxes):
        # A release that keeps its label files standard puts the scores in their own directory, so
        # look there before telling the reader the annotations have none.
        from hbb2obb.formats import read_confidences

        confidence_dir = args.confidence_dir or _first_existing(root.parent, "labels_confidence")
        scores = read_confidences(confidence_dir) if confidence_dir else {}
        attached = 0
        for stem, boxes in obb.items():
            row = scores.get(stem)
            if row is not None and len(row) == len(boxes):
                for box, score in zip(boxes, row):
                    box.score = score
                attached += len(boxes)
        if attached:
            print(f"Read {attached} confidence score(s) from {confidence_dir}")
        else:
            print(
                f"Warning: no confidence scores in {obb_dir}; re-run hbb2obb with --save_confidence, "
                "or point --confidence_dir at a side-car directory"
            )
    if not names:
        seen = [b.cls for boxes in obb.values() for b in boxes]
        names = [f"{i}" for i in range(1 + max(seen, default=-1))]

    frames = [
        {
            "path": path,
            "obb": obb.get(path.stem, []),
            "hbb": hbb.get(path.stem, []),
            "cmp": compared.get(path.stem, []),
            "polygons": viewer.read_polygons(polygon_dir / f"{path.stem}.txt") if polygon_dir else [],
        }
        for path in paths
    ]
    if args.frame:
        stems = [f["path"].stem for f in frames]
        if args.frame not in stems:
            raise SystemExit(f"No such frame: {args.frame}")

    if args.out_dir is None and not args.crops:
        width, _, height = args.window.partition("x")
        view = viewer.Viewer(frames, names, int(width), int(height), args.show_hbb, args.show_labels)
        view.show_confidence = args.show_confidence
        if args.frame:
            view.idx = [f["path"].stem for f in frames].index(args.frame)
            view.load()
        view.run()
        return

    todo = [f for f in frames if not args.frame or f["path"].stem == args.frame]
    out_dir = args.out_dir or Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)
    import cv2

    for frame in todo:
        if args.crops:
            img = cv2.imread(str(frame["path"]))
            pages = viewer.contact_sheet(img, frame["obb"], frame["hbb"], names, args.indices)
            for n, page in enumerate(pages, 1):
                out = out_dir / f"{frame['path'].stem}.crops{n}.jpg"
                cv2.imwrite(str(out), page, [cv2.IMWRITE_JPEG_QUALITY, 92])
                print(out)
            continue
        img = viewer.render(frame, names, args.show_hbb, args.show_labels, args.show_confidence)
        out = out_dir / f"{frame['path'].stem}.jpg"
        cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(out)


def _first_existing(parent: Path, *candidates: str):
    """The first of several sibling directories that exists, or None."""
    for name in candidates:
        path = parent / name
        if path.is_dir():
            return path
    return None


def main_hbb2obb_eval():
    """
    Run the OBB evaluation from command line.
    """

    parser = argparse.ArgumentParser(description="Evaluate OBB predictions against ground truth")
    parser.add_argument("--version", "-V", action="version", version=f"%(prog)s {__version__}")

    # Main arguments
    parser.add_argument("gt_dir", type=Path, help="Directory containing ground truth OBB annotations")
    parser.add_argument("pred_dir", type=Path, help="Directory containing predicted/converted OBB annotations")
    parser.add_argument(
        "--excluded_classes", "-e", type=int, nargs='+', default=[], help="Class labels to exclude from evaluation"
    )
    parser.add_argument(
        "--iou_threshold", "-t", type=float, default=0.1, help="IoU threshold for considering a match (default: 0.1)"
    )
    parser.add_argument(
        "--class_agnostic",
        "-ca",
        action="store_true",
        help="Evaluate in class-agnostic mode (match boxes regardless of class labels)",
    )
    parser.add_argument("--map_path", "-mp", type=Path, help="Path to label map YAML file (optional)")
    parser.add_argument(
        "--exclude_edge_cases", "-exc", action="store_true", help="Exclude boxes that touch image edges from evaluation"
    )
    parser.add_argument(
        "--edge_tolerance",
        "-et",
        type=int,
        default=1,
        help="Tolerance in pixels for detecting boxes at image edges (default: 1)",
    )
    parser.add_argument(
        "--img_width",
        "-iw",
        type=int,
        help="Image width for edge case detection (required if --exclude_edge_cases is used)",
    )
    parser.add_argument(
        "--img_height",
        "-ih",
        type=int,
        help="Image height for edge case detection (required if --exclude_edge_cases is used)",
    )
    parser.add_argument(
        "--debug", "-d", action="store_true", help="Enable debug mode to print detailed matching information"
    )
    parser.add_argument("--no_bar", "-nb", action="store_true", help="Disable tqdm progress bar display")

    args = parser.parse_args()

    check_for_updates_once()

    from hbb2obb.evaluator import evaluate_obb, print_results

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


def main_hbb2obb_optimize():
    """
    Search hyperparameters for the HBB to OBB conversion, one sweep or a whole benchmark.
    """

    parser = argparse.ArgumentParser(
        description="Find optimal hyperparameters for HBB2OBB conversion",
        epilog=(
            "Two ways to run. Give an image source and a ground truth directory for a single sweep, "
            "or -c CONFIG for a benchmark of several sweeps described by one YAML file:\n\n"
            "  img_source: data/images\n"
            "  gt_dir: data/labels_obb_gt\n"
            "  output_folder: data/benchmark_results\n"
            "  defaults:\n"
            "    imgsz: [640, 960, 1280]\n"
            "    scale_factors: [0.03, 0.05, 0.07]\n"
            "  runs:\n"
            "    - sam_models: [sam_b]\n"
            "    - sam_models: [sam_l, sam_b]\n\n"
            "A run writes into <output_folder>/<name>, where name defaults to its models joined by '-'. "
            "Paths resolve against the current working directory, like the models/ weights directory does."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", "-V", action="version", version=f"%(prog)s {__version__}")

    # Main arguments
    parser.add_argument("img_source", type=Path, nargs="?", help="Path to an image or directory containing images")
    parser.add_argument("gt_dir", type=Path, nargs="?", help="Directory containing ground truth OBB annotations")
    parser.add_argument(
        "--config", "-c", type=Path, help="Benchmark YAML describing several sweeps; replaces the two positionals"
    )
    parser.add_argument(
        "--hbb_dir", "-hd", type=Path, help="Directory containing HBB annotations (default: img_source/../labels_hbb)"
    )
    parser.add_argument(
        "--output_folder", "-o", type=Path, help="Directory to save results (default: img_source/../benchmark_results)"
    )
    parser.add_argument("--run_name", "-n", type=str, help="Name of this sweep's folder (default: its models)")

    # Hyperparameters to sweep
    grid_group = parser.add_argument_group('grid axes (single sweep; a config sets these per run)')
    grid_group.add_argument(
        "--sam_models",
        "-sm",
        type=str,
        default=["sam_b"],
        nargs='+',
        choices=SUPPORTED_SAM_MODELS,
        help="SAM model(s) to use (default: sam_b)",
    )
    grid_group.add_argument(
        "--imgsz", "-iz", type=int, nargs='+', default=None, help="Image sizes to test (default: 640 960 1280)"
    )
    grid_group.add_argument(
        "--scale_factors",
        "-sf",
        type=float,
        nargs='+',
        default=None,
        help="Scale factors to test (default: -0.01 to 0.1 in steps of 0.01)",
    )
    grid_group.add_argument(
        "--opening_kernels",
        "-ok",
        type=float,
        nargs='+',
        default=None,
        help="Opening kernel percentages to test (default: 0.15, a single value). Third grid axis: a run "
        "performs len(imgsz) x len(scale_factors) x len(opening_kernels) full SAM passes over the image "
        "set, so the defaults give 3 x 12 x 1 = 36, and three kernels instead of one triples that to 108",
    )

    # Evaluation options
    eval_group = parser.add_argument_group('evaluation options')
    eval_group.add_argument(
        "--excluded_classes", "-e", type=int, nargs='+', default=[], help="Class labels to exclude from evaluation"
    )
    eval_group.add_argument("--iou_threshold", "-t", type=float, default=0.1, help="IoU threshold for evaluation")
    eval_group.add_argument("--class_agnostic", "-ca", action="store_true", help="Evaluate in class-agnostic mode")
    eval_group.add_argument(
        "--exclude_edge_cases", "-exc", action="store_true", help="Exclude boxes that touch image edges"
    )
    eval_group.add_argument(
        "--edge_tolerance", "-et", type=int, default=1, help="Tolerance in pixels for detecting boxes at image edges"
    )
    eval_group.add_argument("--img_width", "-iw", type=int, help="Image width for edge case detection")
    eval_group.add_argument("--img_height", "-ih", type=int, help="Image height for edge case detection")

    # Benchmark control
    bench_group = parser.add_argument_group('benchmark control')
    bench_group.add_argument("--only", nargs='+', metavar="NAME", help="Run only these runs of the config")
    bench_group.add_argument(
        "--resume", action="store_true", help="Skip runs whose results.yaml already holds a full grid"
    )
    bench_group.add_argument(
        "--refresh",
        action="store_true",
        help="Re-render the plots and summary.md from the results already on disk, running no SAM passes",
    )
    bench_group.add_argument(
        "--dry_run", action="store_true", help="Print the runs, the grid size and the total cost, then stop"
    )

    parser.add_argument(
        "--device",
        type=str,
        help=(
            "Inference device for the SAM model(s), e.g. 'cpu', '0', 'cuda:0', 'mps'. Applies to "
            "every run, overriding any 'device' set in the config (default: ultralytics picks)"
        ),
    )
    parser.add_argument(
        "--model_kwargs",
        "-k",
        type=str,
        help="Additional keyword arguments for model in format 'key1=value1,key2=value2'",
    )
    parser.add_argument("--no_plot", action="store_true", help="Do not render any plot")
    parser.add_argument("--no_bar", "-nb", action="store_true", help="Disable tqdm progress bar display")

    args = parser.parse_args()

    check_for_updates_once()

    import time

    from hbb2obb import optimizer
    from hbb2obb.utils import get_hbb_dir

    plot = not args.no_plot

    if args.config:
        config = optimizer.load_config(args.config)
        specs = optimizer.expand_runs(config, SUPPORTED_SAM_MODELS, str(args.config))
        img_source = Path(config["img_source"])
        gt_dir = Path(config["gt_dir"])
        hbb_dir = args.hbb_dir or (Path(config["hbb_dir"]) if config.get("hbb_dir") else None)
        output_folder = args.output_folder or (
            Path(config["output_folder"]) if config.get("output_folder") else _default_benchmark_dir(img_source)
        )
        config_text = args.config.read_text(encoding="utf-8")
        command = f"hbb2obb-optimize -c {args.config}"
        if args.device:
            command += f" --device {args.device}"
    else:
        if args.img_source is None or args.gt_dir is None:
            parser.error("img_source and gt_dir are required unless --config is given")
        specs = [
            optimizer.RunSpec(
                name=args.run_name or "-".join(args.sam_models),
                sam_models=list(args.sam_models),
                imgsz=args.imgsz if args.imgsz is not None else list(optimizer.DEFAULT_IMGSZ),
                scale_factors=(
                    args.scale_factors if args.scale_factors is not None else list(optimizer.DEFAULT_SCALE_FACTORS)
                ),
                opening_kernels=(
                    args.opening_kernels
                    if args.opening_kernels is not None
                    else list(optimizer.DEFAULT_OPENING_KERNELS)
                ),
                excluded_classes=list(args.excluded_classes),
                iou_threshold=args.iou_threshold,
                class_agnostic=args.class_agnostic,
                exclude_edge_cases=args.exclude_edge_cases,
                edge_tolerance=args.edge_tolerance,
                img_width=args.img_width,
                img_height=args.img_height,
                model_kwargs=args.model_kwargs,
                device=args.device,
            )
        ]
        img_source = args.img_source
        gt_dir = args.gt_dir
        hbb_dir = args.hbb_dir
        output_folder = args.output_folder or _default_benchmark_dir(img_source)
        config_text = None
        spec = specs[0]
        command = " ".join(
            [
                "hbb2obb-optimize",
                str(img_source),
                str(gt_dir),
                "--sam_models",
                *spec.sam_models,
                "--imgsz",
                *[str(v) for v in spec.imgsz],
                "--scale_factors",
                *[str(v) for v in spec.scale_factors],
                "--opening_kernels",
                *[str(v) for v in spec.opening_kernels],
                "--run_name",
                spec.name,
            ]
            + (["--device", args.device] if args.device else [])
        )

    # A --device on the command line applies to every run, benchmark configs included: the same
    # sweep is run on whatever machine is free, and the config should not have to be edited for it.
    if args.device:
        for s in specs:
            s.device = args.device

    if args.only:
        known = {s.name for s in specs}
        unknown = [n for n in args.only if n not in known]
        if unknown:
            raise SystemExit(f"--only names run(s) that are not configured: {unknown}")
        specs = [s for s in specs if s.name in args.only]

    hbb_dir = get_hbb_dir(img_source, hbb_dir)

    if args.dry_run or args.refresh:
        pass
    elif not img_source.exists():
        raise SystemExit(f"image source not found: {img_source}")
    elif not gt_dir.is_dir():
        raise SystemExit(f"ground truth directory not found: {gt_dir}")

    print(f"Image source   : {img_source}")
    print(f"HBB directory  : {hbb_dir}")
    print(f"Ground truth   : {gt_dir}")
    print(f"Output folder  : {output_folder}")
    print(f"Runs           : {len(specs)}")
    total_points = sum(len(s.grid) for s in specs)
    for spec in specs:
        marker = ""
        if args.resume and optimizer.is_complete(output_folder / spec.name, spec):
            marker = "  [complete, will be skipped]"
        print(f"  {spec.name:<44} {' '.join(spec.sam_models):<40} {spec.describe_grid()}{marker}")
    print(f"Total          : {total_points} conversions of the whole image set")

    if args.dry_run:
        # A dry run says what would happen and touches nothing, so the output folder is created
        # only below. Otherwise a typo in output_folder leaves a stray empty directory behind,
        # which is exactly what someone checking their config before an unattended sweep does not
        # want to have to notice.
        return

    output_folder.mkdir(parents=True, exist_ok=True)

    started = time.time()
    swept, skipped = [], []
    if not args.refresh:
        for spec in specs:
            run_folder = output_folder / spec.name
            if args.resume and optimizer.is_complete(run_folder, spec):
                print(f"\nSkipping {spec.name}: already complete")
                skipped.append(spec.name)
                continue
            swept.append(spec.name)

            print("\n" + "=" * 116)
            print(f"Run: {spec.name}   ({spec.describe_grid()})")
            print("=" * 116)

            outcome = optimizer.sweep(spec, img_source, gt_dir, hbb_dir, no_bar=True)
            config_dict = optimizer.run_config_dict(spec, img_source, gt_dir, hbb_dir)
            optimizer.write_run(run_folder, spec, outcome, config_dict, plot=plot)
            optimizer.print_best(outcome["best_parameters"], run_folder)

            # Release the checkpoints before the next ensemble loads its own
            from hbb2obb.converter import clear_model_cache

            clear_model_cache()
    else:
        if plot:
            from hbb2obb import plotting

            for folder in sorted(p for p in output_folder.iterdir() if (p / optimizer.RESULTS_NAME).is_file()):
                print(f"Rendering {plotting.run_plot(folder)}")

    elapsed = time.time() - started

    # Summarise every run the folder holds, not only the ones this invocation touched, so a
    # resumed or partial benchmark still reports what is actually on disk.
    names = sorted(p.name for p in output_folder.iterdir() if (p / optimizer.RESULTS_NAME).is_file())
    rows = optimizer.collect_rows(output_folder, names)

    wrote_provenance = False
    # A --refresh writes no copy of its own, but must still point at the one already on disk.
    config_copy = output_folder / args.config.name if args.config else None
    if config_copy is not None and not config_copy.is_file():
        config_copy = None
    # Provenance is written only by an invocation that actually measured something. A --refresh
    # redraws and a fully skipped --resume measures nothing, and rewriting the record for either
    # would stamp today's code and today's checkpoints onto numbers produced by neither.
    if swept and rows:
        from hbb2obb import provenance

        if args.config:
            config_copy = optimizer.write_config_copy(output_folder, args.config, config_text)
        notes = []
        if skipped:
            notes = [
                f"Resumed benchmark: {len(swept)} run(s) were measured by this invocation and",
                f"{len(skipped)} were already complete on disk and were left untouched, so their",
                "numbers come from an earlier sweep and possibly from different code. The wall",
                "time above covers only the runs measured here.",
                f"  measured : {' '.join(swept)}",
                f"  kept     : {' '.join(skipped)}",
            ]

        provenance.write_benchmark_provenance(
            out=output_folder / optimizer.BENCHMARK_PROVENANCE_NAME,
            command=command,
            config_text=config_text,
            config_path=config_copy,
            runs=[{"name": s.name, "sam_models": s.sam_models} for s in specs],
            img_source=img_source,
            hbb_dir=hbb_dir,
            gt_dir=gt_dir,
            grid_description=specs[0].describe_grid() if specs else "no runs",
            elapsed_seconds=elapsed,
            notes=notes,
        )
        wrote_provenance = True

    summary = optimizer.write_summary(
        output_folder,
        rows,
        img_source,
        hbb_dir,
        gt_dir,
        command,
        # None whenever this invocation measured nothing itself: a --refresh, or a --resume that
        # found every run already complete. write_summary then falls back to summing each run's
        # own recorded sweep_seconds instead of reporting this invocation's near-zero wall time.
        elapsed_seconds=elapsed if swept else None,
        plot=plot,
        provenance=wrote_provenance or (output_folder / optimizer.BENCHMARK_PROVENANCE_NAME).is_file(),
        config_name=config_copy.name if config_copy else None,
    )
    print(f"\nWrote {summary}")


def _default_benchmark_dir(img_source: Path) -> Path:
    """Where a sweep writes when no output folder is given: beside the image directory."""
    base = img_source.parent.parent if img_source.is_file() else img_source.parent
    return base / "benchmark_results"
