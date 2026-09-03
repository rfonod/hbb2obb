# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
Interactive viewer for HBB and OBB annotations.

Draws oriented boxes, their source horizontal boxes, segmentation polygons and, when the labels
carry one, the per-box confidence score, over the images they belong to. It reads every format
``hbb2obb.formats`` understands, so the same command inspects a YOLO directory, a DOTA directory,
a Pascal VOC directory or a COCO file, and renders from any two of them can be compared.

The window pans and zooms, which is what looking at 4K aerial frames actually requires: a vehicle
is forty pixels across and its orientation is not judgeable at fit-to-screen scale.

Run it through ``hbb2obb-view``; see ``hbb2obb-view --help`` for the arguments.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from hbb2obb import formats

OBB_COLOR = (60, 230, 60)  # green
HBB_COLOR = (255, 255, 255)  # white
DIFF_COLOR = (0, 165, 255)  # orange, for difficult=1
CMP_COLOR = (255, 130, 0)  # blue, for the comparison set
POLY_COLOR = (0, 0, 255)  # red, for segmentation polygons
CASING_COLOR = (35, 35, 35)  # near-black, drawn under a light line so it survives a light background
WINDOW = "hbb2obb"

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def confidence_color(score: float) -> Tuple[int, int, int]:
    """
    Green for a confident conversion, red for a fallback.

    The same gradient ``converter.visualize_obb_annotations`` uses, so a frame looks the same
    whether it was rendered during the conversion or opened here afterwards.
    """
    c = max(0.0, min(1.0, score))
    return (0, int(255 * c), int(255 * (1 - c)))


# ---------------------------------------------------------------------------------------- loading
def read_polygons(path: Path) -> List[np.ndarray]:
    """
    Read a polygon side-car written by ``hbb2obb --save_polygon``.

    Each line is ``class x1 y1 ... xN yN [confidence]`` with a variable number of corners, so these
    do not go through the box readers.
    """
    if not path.is_file():
        return []
    polygons = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 7:
            continue
        coords = [float(v) for v in fields[1:]]
        if len(coords) % 2:  # a trailing confidence column
            coords = coords[:-1]
        polygons.append(np.array(coords, dtype=float).reshape(-1, 2))
    return polygons


def image_paths(img_source: Path) -> List[Path]:
    """Every image under a directory, or the single image given."""
    if img_source.is_file():
        return [img_source]
    if not img_source.is_dir():
        return []
    return sorted(p for p in img_source.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def load_annotations(source: Optional[Path], fmt: Optional[str], names: Optional[Sequence[str]], sizes: dict) -> dict:
    """Read one annotation set and index it by frame stem, or return {} when there is nothing to read."""
    if source is None or not source.exists():
        return {}
    try:
        frames, _, _ = formats.read_set(source, fmt, names or None, sizes)
    except (ValueError, OSError) as e:
        print(f"Warning: could not read {source}: {e}")
        return {}
    return {frame.stem: frame.boxes for frame in frames}


# ---------------------------------------------------------------------------------------- drawing
def draw(
    img: np.ndarray,
    boxes: Sequence[formats.Box],
    color: Tuple[int, int, int],
    names: Sequence[str],
    labels: bool = True,
    thickness: int = 2,
    mark_difficult: bool = True,
    by_confidence: bool = False,
    show_score: bool = False,
    scale: float = 1.0,
    offset: Tuple[float, float] = (0.0, 0.0),
    casing: bool = False,
) -> None:
    """
    Draw a set of boxes onto an image, in place.

    ``casing`` lays a near-black line under the box before drawing it. A light outline over a
    4K aerial frame is otherwise a hairline of one grey on another: asphalt, rooftops and pale
    car bodies all sit within a few values of white. The casing costs a pixel on each side and
    buys legibility everywhere, which is what a secondary overlay needs; the primary one is
    already legible from its colour.
    """
    for box in boxes:
        c = color
        if by_confidence and box.score is not None:
            c = confidence_color(box.score)
        elif box.difficult and mark_difficult:
            c = DIFF_COLOR

        pts = (box.quad - offset) * scale if scale != 1.0 or offset != (0.0, 0.0) else box.quad
        # Round rather than truncate, so the canonical files and the integer ones put a box in the
        # same place and two renders of the same annotations can be diffed.
        quad = [np.round(pts).astype(np.int32)]
        line_width = max(1, int(round(thickness)))
        if casing:
            cv2.polylines(img, quad, True, CASING_COLOR, line_width + 2, cv2.LINE_AA)
        cv2.polylines(img, quad, True, c, line_width, cv2.LINE_AA)

        text = []
        if labels and box.cls < len(names):
            text.append(names[box.cls])
        if show_score and box.score is not None:
            text.append(f"{box.score:.2f}")
        if text:
            anchor = np.round(pts[pts[:, 1].argmin()]).astype(int)
            cv2.putText(
                img,
                " ".join(text),
                (int(anchor[0]), int(anchor[1]) - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45 * max(1.0, thickness / 2),
                c,
                max(1, int(round(thickness / 2))),
                cv2.LINE_AA,
            )


def draw_polygons(
    img: np.ndarray, polygons: Sequence[np.ndarray], scale: float = 1.0, offset: Tuple[float, float] = (0.0, 0.0)
) -> None:
    """Draw the segmentation contours the OBBs were fitted to."""
    for polygon in polygons:
        pts = (polygon - offset) * scale
        cv2.polylines(img, [np.round(pts).astype(np.int32)], True, POLY_COLOR, 1, cv2.LINE_AA)


def contact_sheet(
    img: np.ndarray,
    obb: Sequence[formats.Box],
    hbb: Sequence[formats.Box],
    names: Sequence[str],
    indices: Optional[Sequence[int]] = None,
    cell: int = 260,
    cols: int = 8,
    rows: int = 5,
) -> List[np.ndarray]:
    """
    Crop every box out of a frame and tile them, so a whole frame can be reviewed object by object.

    ``indices`` restricts the sheet to particular boxes, which is how a short review list of
    specific objects is produced.
    """
    pages = []
    order = list(indices) if indices is not None else list(range(len(obb)))
    per_page = cols * rows
    for start in range(0, len(order), per_page):
        chunk = order[start : start + per_page]
        canvas = np.full((rows * (cell + 20), cols * cell, 3), 25, np.uint8)
        for n, i in enumerate(chunk):
            pts = obb[i].quad
            cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
            half = max(50, int(max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1])) * 0.85))
            x0, y0 = int(max(0, cx - half)), int(max(0, cy - half))
            x1, y1 = int(min(img.shape[1], cx + half)), int(min(img.shape[0], cy + half))
            crop = img[y0:y1, x0:x1].copy()
            if crop.size == 0:
                continue
            s = cell / max(crop.shape[:2])
            crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
            for source, color in ((hbb[i] if i < len(hbb) else None, HBB_COLOR), (obb[i], OBB_COLOR)):
                if source is not None:
                    quad = [np.round((source.quad - [x0, y0]) * s).astype(np.int32)]
                    if color == HBB_COLOR:
                        cv2.polylines(crop, quad, True, CASING_COLOR, 4, cv2.LINE_AA)
                    cv2.polylines(crop, quad, True, color, 2, cv2.LINE_AA)
            ch, cw = crop.shape[:2]
            r, c = n // cols, n % cols
            canvas[r * (cell + 20) : r * (cell + 20) + ch, c * cell : c * cell + cw] = crop
            name = names[obb[i].cls] if obb[i].cls < len(names) else str(obb[i].cls)
            cv2.putText(
                canvas,
                f"{i}: {name}",
                (c * cell + 3, r * (cell + 20) + cell + 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        pages.append(canvas)
    return pages


# ------------------------------------------------------------------------------------ interactive
class Viewer:
    """
    Pan and zoom over one frame at a time.

    Left-drag pans, the wheel zooms about the cursor, ``q`` quits. Every toggle is a single key so
    that a long review session needs no menus.
    """

    KEYS = "q quit  n/p frame  +/- zoom  f fit  o obb  h hbb  l labels  d difficult  c conf  g poly  x cmp  s save"

    def __init__(self, frames, names, win_w=1600, win_h=900, show_hbb=True, show_labels=True):
        self.frames = frames  # list of dicts: path, obb, hbb, cmp, polygons
        self.names = names
        self.win_w, self.win_h = win_w, win_h
        self.idx = 0
        self.show_hbb = show_hbb
        self.show_obb = True
        self.show_labels = show_labels
        self.show_difficult = True
        self.show_confidence = False
        self.show_polygons = True
        self.compare_mode = 0  # 0 both, 1 primary only, 2 comparison only
        self.drag = None
        self.canvas = None
        self.load()

    # ------------------------------------------------------------------ state
    def load(self):
        self.frame = self.frames[self.idx]
        self.img = cv2.imread(str(self.frame["path"]))
        if self.img is None:
            raise SystemExit(f"cannot read {self.frame['path']}")
        self.fit()

    def fit(self):
        h, w = self.img.shape[:2]
        self.zoom = min(self.win_w / w, self.win_h / h)
        self.cx, self.cy = w / 2, h / 2

    def clamp(self):
        h, w = self.img.shape[:2]
        self.zoom = float(np.clip(self.zoom, min(self.win_w / w, self.win_h / h), 16.0))
        half_w, half_h = self.win_w / (2 * self.zoom), self.win_h / (2 * self.zoom)
        self.cx = float(np.clip(self.cx, min(half_w, w / 2), max(w - half_w, w / 2)))
        self.cy = float(np.clip(self.cy, min(half_h, h / 2), max(h - half_h, h / 2)))

    def to_image(self, x, y):
        half_w, half_h = self.win_w / (2 * self.zoom), self.win_h / (2 * self.zoom)
        return (self.cx - half_w + x / self.zoom, self.cy - half_h + y / self.zoom)

    # ----------------------------------------------------------------- render
    def view(self):
        self.clamp()
        h, w = self.img.shape[:2]
        half_w, half_h = self.win_w / (2 * self.zoom), self.win_h / (2 * self.zoom)
        ix0, iy0 = max(0, int(np.floor(self.cx - half_w))), max(0, int(np.floor(self.cy - half_h)))
        ix1, iy1 = min(w, int(np.ceil(self.cx + half_w))), min(h, int(np.ceil(self.cy + half_h)))
        crop = self.img[iy0:iy1, ix0:ix1]
        interp = cv2.INTER_NEAREST if self.zoom > 2 else cv2.INTER_AREA
        vis = cv2.resize(crop, None, fx=self.zoom, fy=self.zoom, interpolation=interp)

        offset = (ix0, iy0)
        thickness = max(1, int(round(2 * min(max(1.0, self.zoom), 3.0))))

        def visible(boxes):
            return [b for b in boxes if self.show_difficult or not b.difficult]

        if self.show_polygons and self.frame["polygons"]:
            draw_polygons(vis, self.frame["polygons"], self.zoom, offset)
        if self.show_hbb:
            draw(
                vis,
                visible(self.frame["hbb"]),
                HBB_COLOR,
                self.names,
                labels=False,
                # Never thicker than the OBB, never thinner than 2: below that the casing
                # swallows the white core and the HBB reads as a dark line instead of a light one.
                thickness=max(2, thickness - 1),
                mark_difficult=False,
                scale=self.zoom,
                offset=offset,
                casing=True,
            )
        if self.show_obb and self.compare_mode != 2:
            draw(
                vis,
                visible(self.frame["obb"]),
                OBB_COLOR,
                self.names,
                labels=self.show_labels,
                thickness=thickness,
                by_confidence=self.show_confidence,
                show_score=self.show_confidence,
                scale=self.zoom,
                offset=offset,
            )
        if self.compare_mode != 1 and self.frame["cmp"]:
            draw(
                vis,
                visible(self.frame["cmp"]),
                CMP_COLOR,
                self.names,
                labels=False,
                thickness=thickness,
                mark_difficult=False,
                scale=self.zoom,
                offset=offset,
            )

        pad = np.full((self.win_h, self.win_w, 3), 20, np.uint8)
        vh, vw = min(vis.shape[0], self.win_h), min(vis.shape[1], self.win_w)
        pad[:vh, :vw] = vis[:vh, :vw]
        cv2.rectangle(pad, (0, 0), (self.win_w, 30), (20, 20, 20), -1)
        cv2.putText(pad, self.status(), (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (240, 240, 240), 1, cv2.LINE_AA)
        self.canvas = pad
        return pad

    def status(self) -> str:
        obb = self.frame["obb"]
        n_diff = sum(1 for b in obb if b.difficult)
        extra = f" ({n_diff} difficult)" if n_diff else ""
        if self.frame["cmp"]:
            extra += f" vs {len(self.frame['cmp'])} compared"
        return (
            f"[{self.idx + 1}/{len(self.frames)}] {self.frame['path'].stem}   "
            f"{len(obb)} objects{extra}   zoom {self.zoom * 100:.0f}%   {self.KEYS}"
        )

    # ------------------------------------------------------------------ input
    def on_mouse(self, event, x, y, flags, _):
        if event == cv2.EVENT_MOUSEWHEEL:
            before = self.to_image(x, y)  # keep the point under the cursor fixed
            self.zoom *= 1.25 if flags > 0 else 1 / 1.25
            self.clamp()
            after = self.to_image(x, y)
            self.cx += before[0] - after[0]
            self.cy += before[1] - after[1]
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.drag = (x, y, self.cx, self.cy)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drag = None
        elif event == cv2.EVENT_MOUSEMOVE and self.drag:
            sx, sy, ox, oy = self.drag
            self.cx = ox - (x - sx) / self.zoom
            self.cy = oy - (y - sy) / self.zoom
        # No redraw here: run()'s loop already calls cv2.imshow() every ~30ms regardless of
        # mouse activity, so this only updates state. That keeps on_mouse callable without a
        # live window, which is what lets it be unit-tested headlessly; calling cv2.imshow()
        # from here crashed hard on a display-less Linux CI runner.

    def run(self) -> None:
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW, self.on_mouse)
        while True:
            cv2.imshow(WINDOW, self.view())
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27):  # q or Esc
                break
            try:
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break  # the window was closed
            except cv2.error:
                break

            if key in (ord("n"), 83, 84):  # n, right, down
                self.idx = (self.idx + 1) % len(self.frames)
                self.load()
            elif key in (ord("p"), 81, 82):  # p, left, up
                self.idx = (self.idx - 1) % len(self.frames)
                self.load()
            elif key in (ord("+"), ord("=")):
                self.zoom *= 1.25
            elif key in (ord("-"), ord("_")):
                self.zoom /= 1.25
            elif key == ord("1"):
                self.zoom = 1.0
            elif key in (ord("f"), ord("0")):
                self.fit()
            elif key == ord("o"):
                self.show_obb = not self.show_obb
            elif key == ord("h"):
                self.show_hbb = not self.show_hbb
            elif key == ord("l"):
                self.show_labels = not self.show_labels
            elif key == ord("d"):
                self.show_difficult = not self.show_difficult
            elif key == ord("c"):
                self.show_confidence = not self.show_confidence
            elif key == ord("g"):
                self.show_polygons = not self.show_polygons
            elif key == ord("x"):
                self.compare_mode = (self.compare_mode + 1) % 3
            elif key == ord("s"):
                out = Path.cwd() / f"{self.frame['path'].stem}.view.jpg"
                cv2.imwrite(str(out), self.canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
                print(f"saved {out}")
        cv2.destroyAllWindows()
        cv2.waitKey(1)  # let macOS actually close the window


# ------------------------------------------------------------------------------------ file output
def render(
    frame: dict,
    names: Sequence[str],
    show_hbb: bool = True,
    show_labels: bool = True,
    show_confidence: bool = False,
    show_polygons: bool = True,
) -> np.ndarray:
    """Render one frame at full resolution, for writing to a file."""
    img = cv2.imread(str(frame["path"]))
    if img is None:
        raise SystemExit(f"cannot read {frame['path']}")
    if show_polygons and frame["polygons"]:
        draw_polygons(img, frame["polygons"])
    if show_hbb:
        draw(img, frame["hbb"], HBB_COLOR, names, labels=False, thickness=2, mark_difficult=False, casing=True)
    draw(
        img,
        frame["obb"],
        OBB_COLOR,
        names,
        labels=show_labels,
        by_confidence=show_confidence,
        show_score=show_confidence,
    )
    if frame["cmp"]:
        draw(img, frame["cmp"], CMP_COLOR, names, labels=False, mark_difficult=False)

    legend = "green=OBB white=HBB orange=difficult" + (" blue=compared" if frame["cmp"] else "")
    cv2.putText(
        img,
        f"{frame['path'].stem}   {len(frame['obb'])} objects   {legend}",
        (12, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return img
