"""Tests for hbb2obb.formats: readers, writers, round trips and the rounding rule."""

import json

import numpy as np
import pytest

from hbb2obb import formats
from hbb2obb.formats import Box, FrameAnnotations

NAMES = ["Car", "Bus", "Truck"]
W, H = 640, 480


def make_frame(stem="img1", boxes=None):
    return FrameAnnotations(stem, W, H, boxes if boxes is not None else [Box(0, formats.rect(10, 20, 50, 60))])


def rotated(cx, cy, w, h, angle_deg):
    """A rotated rectangle, so tests exercise real quads rather than axis-aligned ones."""
    a = np.deg2rad(angle_deg)
    r = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    corners = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
    return corners @ r.T + [cx, cy]


# ------------------------------------------------------------------------------------- basic model
def test_box_envelope_and_axis_alignment():
    box = Box(0, formats.rect(10, 20, 50, 60))
    assert box.xyxy == (10, 20, 50, 60)
    assert box.is_axis_aligned
    assert box.area == pytest.approx(40 * 40)

    turned = Box(0, rotated(100, 100, 40, 20, 30))
    assert not turned.is_axis_aligned
    assert turned.area == pytest.approx(40 * 20)


def test_infer_kind():
    assert formats.infer_kind([make_frame()]) == "hbb"
    assert formats.infer_kind([make_frame(boxes=[Box(0, rotated(100, 100, 40, 20, 15))])]) == "obb"


# ------------------------------------------------------------------------------------- YOLO reader
def test_read_yolo_hbb_absolute_and_normalized(tmp_path):
    absolute = tmp_path / "a.txt"
    absolute.write_text("0 100 200 40 60\n", encoding="utf-8")
    box = formats.read_yolo(absolute, W, H)[0]
    assert box.xyxy == (80, 170, 120, 230)

    relative = tmp_path / "b.txt"
    relative.write_text(f"0 {100 / W} {200 / H} {40 / W} {60 / H}\n", encoding="utf-8")
    box = formats.read_yolo(relative, W, H)[0]
    assert box.xyxy == pytest.approx((80, 170, 120, 230))


def test_read_yolo_obb_and_confidence_column(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("0 10 10 30 10 30 20 10 20 0.75\n1 5 5 15 5 15 9 5 9\n", encoding="utf-8")
    boxes = formats.read_yolo(path, W, H)
    assert [b.cls for b in boxes] == [0, 1]
    assert boxes[0].score == 0.75
    assert boxes[1].score is None
    assert boxes[0].xyxy == (10, 10, 30, 20)


def test_read_yolo_empty_file_is_valid(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("\n\n", encoding="utf-8")
    assert formats.read_yolo(path, W, H) == []


def test_read_yolo_rejects_ragged_lines(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("0 1 2 3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed YOLO file"):
        formats.read_yolo(path, W, H)


# ------------------------------------------------------------------------------------ round trips
@pytest.mark.parametrize("fmt", ["dota", "coco", "labelme"])
def test_obb_round_trip(tmp_path, fmt):
    boxes = [Box(i % 3, rotated(100 + 40 * i, 90 + 30 * i, 44, 21, 17 * i)) for i in range(6)]
    frames = formats.canonicalize([make_frame(boxes=boxes)])

    out = tmp_path / "out"
    formats.write_set(frames, out, fmt, NAMES, "obb")
    source = out / "coco_annotations_obb.json" if fmt == "coco" else out
    back, _, detected = formats.read_set(source, fmt, NAMES, {"img1": (W, H)})

    assert detected == fmt
    assert formats.count_boxes(back) == len(boxes)
    for original, restored in zip(frames[0].boxes, back[0].boxes):
        assert original.cls == restored.cls
        assert formats._round_quad(original) == formats._round_quad(restored)


@pytest.mark.parametrize("fmt", ["voc", "coco"])
def test_hbb_round_trip(tmp_path, fmt):
    boxes = [Box(i % 3, formats.rect(10 + 7 * i, 20 + 5 * i, 60 + 7 * i, 90 + 5 * i)) for i in range(5)]
    frames = formats.canonicalize([make_frame(boxes=boxes)])

    out = tmp_path / "out"
    formats.write_set(frames, out, fmt, NAMES, "hbb")
    source = out / "coco_annotations_hbb.json" if fmt == "coco" else out
    back, _, detected = formats.read_set(source, fmt, NAMES, {"img1": (W, H)})

    assert detected == fmt
    for original, restored in zip(frames[0].boxes, back[0].boxes):
        assert original.cls == restored.cls
        assert [round(v) for v in original.xyxy] == [round(v) for v in restored.xyxy]


def test_yolo_hbb_normalized_round_trip_preserves_the_envelope(tmp_path):
    boxes = [Box(0, formats.rect(11.495, 20.5, 61.505, 90.5))]
    frames = formats.canonicalize([make_frame(boxes=boxes)])

    out = tmp_path / "out"
    formats.write_set(frames, out, "yolo", NAMES, "hbb", normalize=True)
    back, _, _ = formats.read_set(out, "yolo", NAMES, {"img1": (W, H)})
    assert back[0].boxes[0].xyxy == pytest.approx(frames[0].boxes[0].xyxy)


# ---------------------------------------------------------------------------------- rounding rule
def test_every_integer_format_rounds_the_same_canonical_source(tmp_path):
    """
    The trap this pins down: rounding full precision and rounding the two-decimal canonical
    disagree wherever the canonical lands on a half pixel. 11.495 rounds to 11, but its canonical
    11.50 rounds to 12. Every integer format must round the canonical, so they all agree.
    """
    quad = np.array([[11.495, 20.495], [61.495, 20.495], [61.495, 90.495], [11.495, 90.495]])
    frames = formats.canonicalize([make_frame(boxes=[Box(0, quad)])])
    canonical = frames[0].boxes[0]

    assert canonical.quad[0].tolist() == [11.5, 20.5]  # rounded to two decimals
    assert formats._round_quad(canonical)[:2] == [12, 20]  # and .5 goes to even from there

    out = tmp_path / "out"
    formats.write_set(frames, out, "dota", NAMES, "obb")
    formats.write_set(frames, out, "coco", NAMES, "obb")
    formats.write_set(frames, out, "yolo", NAMES, "obb")

    dota = formats.read_dota(out / "img1.dota", NAMES)[0]
    coco = json.loads((out / "coco_annotations_obb.json").read_text())["annotations"][0]
    yolo = formats.read_yolo(out / "img1.txt", W, H)[0]

    assert formats._round_quad(dota) == formats._round_quad(canonical)
    assert coco["segmentation"][0] == formats._round_quad(canonical)
    assert formats._round_quad(yolo) == formats._round_quad(canonical)

    # and the COCO bbox is the envelope of its own segmentation, which holds only because
    # min(round(x)) == round(min(x)) when both sides round the same values
    corners = np.array(coco["segmentation"][0]).reshape(4, 2)
    x0, y0 = corners[:, 0].min(), corners[:, 1].min()
    x1, y1 = corners[:, 0].max(), corners[:, 1].max()
    assert coco["bbox"] == [x0, y0, x1 - x0, y1 - y0]


def test_hbb_written_from_an_obb_is_its_rounded_envelope(tmp_path):
    box = Box(0, rotated(120.31, 88.77, 51.3, 23.9, 23))
    frames = formats.canonicalize([make_frame(boxes=[box])])

    out = tmp_path / "out"
    formats.write_set(frames, out, "voc", NAMES, "hbb")
    voc_boxes, _, _ = formats.read_voc(out / "img1.xml", NAMES)
    assert [round(v) for v in voc_boxes[0].xyxy] == formats._round_xyxy(frames[0].boxes[0])


# ------------------------------------------------------------------------------------- difficult
def test_difficult_survives_dota_and_voc_but_not_yolo(tmp_path):
    boxes = [Box(0, rotated(100, 100, 40, 20, 10), difficult=1), Box(1, rotated(200, 200, 40, 20, 0))]
    frames = formats.canonicalize([make_frame(boxes=boxes)])

    out = tmp_path / "out"
    formats.write_set(frames, out, "dota", NAMES, "obb")
    formats.write_set(frames, out, "yolo", NAMES, "obb")

    assert [b.difficult for b in formats.read_dota(out / "img1.dota", NAMES)] == [1, 0]
    assert [b.difficult for b in formats.read_yolo(out / "img1.txt", W, H)] == [0, 0]


def test_apply_difficult_copies_flags_by_row(tmp_path):
    flagged = formats.canonicalize(
        [
            make_frame(
                boxes=[
                    Box(0, formats.rect(0, 0, 10, 10), difficult=1),
                    Box(0, formats.rect(20, 20, 30, 30), difficult=0),
                ]
            )
        ]
    )
    plain = formats.canonicalize(
        [make_frame(boxes=[Box(0, formats.rect(0, 0, 10, 10)), Box(0, formats.rect(20, 20, 30, 30))])]
    )
    assert formats.apply_difficult(plain, flagged) == 1
    assert [b.difficult for b in plain[0].boxes] == [1, 0]


def test_apply_difficult_refuses_a_mismatched_count():
    a = formats.canonicalize([make_frame(boxes=[Box(0, formats.rect(0, 0, 10, 10))])])
    b = formats.canonicalize([make_frame(boxes=[Box(0, formats.rect(0, 0, 10, 10))] * 2)])
    with pytest.raises(ValueError, match="Cannot copy difficult flags"):
        formats.apply_difficult(a, b)


def test_voc_truncated_uses_the_evaluator_edge_tolerance(tmp_path):
    """A box reaching within 1 px of a border is truncated, matching hbb2obb-eval --edge_tolerance."""
    boxes = [Box(0, formats.rect(1, 50, 40, 90)), Box(0, formats.rect(100, 50, 140, 90))]
    frames = [FrameAnnotations("img1", W, H, boxes)]
    formats.write_set(frames, tmp_path, "voc", NAMES, "hbb")
    text = (tmp_path / "img1.xml").read_text()
    assert text.count("<truncated>1</truncated>") == 1
    assert text.count("<truncated>0</truncated>") == 1


# -------------------------------------------------------------------------------------- discovery
def test_sniff_format_ignores_files_that_are_not_annotations(tmp_path):
    (tmp_path / "names.txt").write_text("Car\nBus\n", encoding="utf-8")
    (tmp_path / "labels.txt").write_text("0 10 20 30 40\n", encoding="utf-8")
    assert formats.sniff_format(tmp_path / "names.txt") is None
    assert formats.sniff_format(tmp_path / "labels.txt") == "yolo"
    assert [p.name for p in formats.label_files(tmp_path, "yolo")] == ["labels.txt"]


def test_sniff_format_tells_dota_from_yolo(tmp_path):
    yolo = tmp_path / "a.txt"
    yolo.write_text("0 10 10 30 10 30 20 10 20\n", encoding="utf-8")
    dota = tmp_path / "b.txt"
    dota.write_text("imagesource:drone\n10 10 30 10 30 20 10 20 Car 0\n", encoding="utf-8")
    assert formats.sniff_format(yolo) == "yolo"
    assert formats.sniff_format(dota) == "dota"


def test_detect_format_raises_on_an_unreadable_directory(tmp_path):
    (tmp_path / "notes.md").write_text("nothing here", encoding="utf-8")
    with pytest.raises(ValueError, match="No label files found"):
        formats.detect_format(tmp_path)


def test_image_size_reads_the_header(tmp_path):
    cv2 = pytest.importorskip("cv2")
    img = np.zeros((37, 53, 3), np.uint8)
    for name in ("a.png", "a.bmp", "a.jpg"):
        path = tmp_path / name
        cv2.imwrite(str(path), img)
        assert formats.image_size(path) == (53, 37), name
    assert formats.image_sizes(tmp_path)["a"] == (53, 37)


def test_image_size_returns_none_for_a_missing_file(tmp_path):
    assert formats.image_size(tmp_path / "nope.jpg") is None


def test_image_size_applies_the_exif_orientation(tmp_path):
    """
    cv2.imread turns a quarter-turn JPEG, so the header's own width and height are not the frame
    the boxes were fitted in. Normalizing against the unturned size divides x by the height.
    """
    cv2 = pytest.importorskip("cv2")
    Image = pytest.importorskip("PIL.Image")

    landscape = np.zeros((200, 400, 3), np.uint8)
    turned, upright = tmp_path / "turned.jpg", tmp_path / "upright.jpg"

    image = Image.fromarray(landscape)
    exif = image.getexif()
    exif[274] = 6  # rotate 90 degrees, so decoders present it as 200x400
    image.save(turned, exif=exif)
    Image.fromarray(landscape).save(upright)

    assert formats.image_size(turned) == (200, 400)
    assert formats.image_size(turned) == tuple(reversed(cv2.imread(str(turned)).shape[:2]))
    assert formats.image_size(upright) == (400, 200), "an untagged JPEG must be left alone"


# ------------------------------------------------------------------ the relative/absolute question
def test_looks_normalized_separates_the_two_conventions():
    assert formats.looks_normalized([]) is None, "no coordinates is not an answer"
    assert formats.looks_normalized([0.1, 0.9, 0.5]) is True
    assert formats.looks_normalized([10.0, 200.0]) is False


def test_a_corner_past_the_frame_stays_normalized():
    """
    A fitted OBB may extend past the frame it was fitted in, and the shipped sample already holds
    corners 17 px beyond the edge. Under a [0, 1] test one such corner turned the whole file into
    absolute pixels and every box in it collapsed into the top-left corner.
    """
    assert formats.looks_normalized([-0.005, 0.5, 1.00787]) is True


def test_read_yolo_denormalizes_a_file_that_runs_past_the_edge(tmp_path):
    path = tmp_path / "f.txt"
    path.write_text("0 0.5 0.9 0.6 0.9 0.6 1.00787 0.5 1.00787\n", encoding="utf-8")

    (box,) = formats.read_yolo(path, 3840, 2160)
    assert box.quad[0].tolist() == pytest.approx([1920.0, 1944.0])
    assert box.quad[2].tolist() == pytest.approx([2304.0, 2177.0], abs=0.5)


# ----------------------------------------------------------------------- the normalized round trip
def test_sufficient_precision_grows_with_the_frame():
    assert formats.sufficient_precision((3840, 2160)) == 6
    assert formats.sufficient_precision((640, 480)) == 5
    assert formats.sufficient_precision((10, 10)) == 4
    assert formats.sufficient_precision((3840, 2160)) <= formats.DEFAULT_NORMALIZED_PRECISION


@pytest.mark.parametrize("size", [(3840, 2160), (640, 480), (1920, 1080)])
def test_a_normalized_yolo_set_verifies_against_its_absolute_dota_sibling(tmp_path, size):
    """
    The round trip that matters for a release: labels written relative, read back, and proved to
    encode the same boxes as the absolute formats derived from them.
    """
    width, height = size
    boxes = [Box(0, rotated(width * 0.5, height * 0.5, 120, 40, 25)), Box(1, formats.rect(4, 4, 60, 30))]
    frames = formats.canonicalize([FrameAnnotations("a", width, height, boxes)])

    precision = formats.sufficient_precision(size)
    formats.write_set(frames, tmp_path, "yolo", NAMES, "obb", normalize=True, precision=precision)
    formats.write_set(frames, tmp_path, "dota", NAMES, "obb")

    sizes = {"a": size}
    yolo, _, _ = formats.read_set(tmp_path, "yolo", NAMES, sizes)
    dota, _, _ = formats.read_set(tmp_path, "dota", NAMES, sizes)
    assert formats.verify({"yolo": formats.canonicalize(yolo), "dota": formats.canonicalize(dota)}) == []


def test_too_few_decimals_move_the_boxes(tmp_path):
    """The guard earns its place: below what the frame needs and the round trip stops agreeing."""
    size = (3840, 2160)
    frames = formats.canonicalize([FrameAnnotations("a", *size, [Box(0, rotated(1000, 700, 120, 40, 25))])])

    formats.write_set(frames, tmp_path, "yolo", NAMES, "obb", normalize=True, precision=2)
    formats.write_set(frames, tmp_path, "dota", NAMES, "obb")

    sizes = {"a": size}
    yolo, _, _ = formats.read_set(tmp_path, "yolo", NAMES, sizes)
    dota, _, _ = formats.read_set(tmp_path, "dota", NAMES, sizes)
    problems = formats.verify({"yolo": formats.canonicalize(yolo), "dota": formats.canonicalize(dota)})
    assert any("corners disagree" in p for p in problems)


# ----------------------------------------------------------------------------------------- verify
def test_verify_accepts_agreeing_sets_and_names_the_disagreement():
    a = formats.canonicalize([make_frame(boxes=[Box(0, formats.rect(10, 10, 20, 20))])])
    b = formats.canonicalize([make_frame(boxes=[Box(0, formats.rect(10, 10, 20, 20))])])
    assert formats.verify({"yolo": a, "dota": b}) == []

    b[0].boxes[0] = Box(0, formats.rect(10, 10, 25, 20))
    assert any("corners disagree" in p for p in formats.verify({"yolo": a, "dota": b}))

    b[0].boxes[0] = Box(1, formats.rect(10, 10, 20, 20))
    assert any("class" in p for p in formats.verify({"yolo": a, "dota": b}))


def test_verify_reports_a_count_mismatch():
    a = formats.canonicalize([make_frame(boxes=[Box(0, formats.rect(10, 10, 20, 20))] * 2)])
    b = formats.canonicalize([make_frame(boxes=[Box(0, formats.rect(10, 10, 20, 20))])])
    assert any("boxes" in p for p in formats.verify({"yolo": a, "dota": b}))


# --------------------------------------------------------------------------------- kind guardrails
def test_a_format_refuses_a_kind_it_cannot_express(tmp_path):
    frames = formats.canonicalize([make_frame(boxes=[Box(0, rotated(100, 100, 40, 20, 10))])])
    with pytest.raises(ValueError, match="cannot represent oriented boxes"):
        formats.write_set(frames, tmp_path, "voc", NAMES, "obb")
    with pytest.raises(ValueError, match="cannot represent horizontal boxes"):
        formats.write_set(frames, tmp_path, "dota", NAMES, "hbb")


def test_dota_rejects_an_unknown_class(tmp_path):
    path = tmp_path / "a.dota"
    path.write_text("10 10 30 10 30 20 10 20 Bicycle 0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown class"):
        formats.read_dota(path, NAMES)


def test_labelme_discovers_classes_across_files(tmp_path):
    """LabelMe stores names, not ids, so ids follow first-seen order across the whole directory."""
    frames = [
        FrameAnnotations("a", W, H, [Box(0, formats.rect(0, 0, 10, 10))]),
        FrameAnnotations("b", W, H, [Box(2, formats.rect(0, 0, 10, 10))]),
    ]
    formats.write_set(frames, tmp_path, "labelme", NAMES, "hbb")
    _, discovered, _ = formats.read_set(tmp_path, "labelme")
    assert discovered == ["Car", "Truck"]

    pinned, _, _ = formats.read_set(tmp_path, "labelme", NAMES)
    assert [b.cls for f in pinned for b in f.boxes] == [0, 2]
