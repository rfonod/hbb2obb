"""Tests for hbb2obb.viewer. Headless: nothing here opens a window."""

import numpy as np
import pytest

from hbb2obb import formats, viewer
from hbb2obb.cli import main_hbb2obb_view
from hbb2obb.formats import Box, FrameAnnotations

cv2 = pytest.importorskip("cv2")

NAMES = ["Car", "Bus", "Truck"]


@pytest.fixture
def canvas():
    return np.zeros((200, 300, 3), np.uint8)


def test_confidence_color_matches_the_conversion_gradient():
    assert viewer.confidence_color(1.0) == (0, 255, 0)  # green, a clean fit
    assert viewer.confidence_color(0.0) == (0, 0, 255)  # red, a fallback
    assert viewer.confidence_color(0.5) == (0, 127, 127)
    assert viewer.confidence_color(-3) == (0, 0, 255)  # clamped
    assert viewer.confidence_color(9) == (0, 255, 0)


def test_draw_rounds_rather_than_truncates(canvas):
    """
    A box at x=9.6 must be drawn at 10, not 9.

    Truncating is what made renders of the same annotations differ between the canonical
    two-decimal files and the integer ones, which stopped the two from being diffable.
    """
    rounded = canvas.copy()
    viewer.draw(rounded, [Box(0, formats.rect(9.6, 9.6, 60.6, 60.6))], (0, 255, 0), NAMES, labels=False, thickness=1)
    reference = canvas.copy()
    viewer.draw(reference, [Box(0, formats.rect(10, 10, 61, 61))], (0, 255, 0), NAMES, labels=False, thickness=1)
    assert np.array_equal(rounded, reference)


def test_draw_marks_difficult_boxes(canvas):
    marked = canvas.copy()
    viewer.draw(marked, [Box(0, formats.rect(10, 10, 60, 60), difficult=1)], (0, 255, 0), NAMES, labels=False)
    assert (marked == np.array(viewer.DIFF_COLOR, np.uint8)).all(2).any()

    plain = canvas.copy()
    viewer.draw(
        plain,
        [Box(0, formats.rect(10, 10, 60, 60), difficult=1)],
        (0, 255, 0),
        NAMES,
        labels=False,
        mark_difficult=False,
    )
    assert not (plain == np.array(viewer.DIFF_COLOR, np.uint8)).all(2).any()


def test_draw_colors_by_confidence_when_asked(canvas):
    img = canvas.copy()
    viewer.draw(
        img, [Box(0, formats.rect(10, 10, 60, 60), score=0.0)], (0, 255, 0), NAMES, labels=False, by_confidence=True
    )
    assert (img == np.array((0, 0, 255), np.uint8)).all(2).any()  # red, not the default green


def test_draw_leaves_a_scoreless_box_the_default_color(canvas):
    img = canvas.copy()
    viewer.draw(img, [Box(0, formats.rect(10, 10, 60, 60))], (0, 255, 0), NAMES, labels=False, by_confidence=True)
    assert (img == np.array((0, 255, 0), np.uint8)).all(2).any()


def test_draw_applies_scale_and_offset(canvas):
    scaled = canvas.copy()
    viewer.draw(
        scaled,
        [Box(0, formats.rect(20, 20, 40, 40))],
        (0, 255, 0),
        NAMES,
        labels=False,
        thickness=1,
        scale=2.0,
        offset=(10, 10),
    )
    reference = canvas.copy()
    viewer.draw(reference, [Box(0, formats.rect(20, 20, 60, 60))], (0, 255, 0), NAMES, labels=False, thickness=1)
    assert np.array_equal(scaled, reference)


def test_read_polygons_handles_the_optional_confidence_column(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("0 1 2 3 4 5 6 7 8\n0 1 2 3 4 5 6 0.91\n\n", encoding="utf-8")
    polygons = viewer.read_polygons(path)
    assert [len(p) for p in polygons] == [4, 3]
    assert polygons[1][-1].tolist() == [5, 6]


def test_read_polygons_of_a_missing_file_is_empty(tmp_path):
    assert viewer.read_polygons(tmp_path / "nope.txt") == []


def test_contact_sheet_pages_and_selection(tmp_path):
    img = np.full((400, 600, 3), 128, np.uint8)
    boxes = [Box(0, formats.rect(10 + 5 * i, 10, 50 + 5 * i, 50)) for i in range(45)]
    pages = viewer.contact_sheet(img, boxes, [], NAMES, cols=4, rows=5)
    assert len(pages) == 3  # 45 objects at 20 per page

    only_three = viewer.contact_sheet(img, boxes, [], NAMES, indices=[0, 7, 44], cols=4, rows=5)
    assert len(only_three) == 1


def test_image_paths_finds_images_and_accepts_a_single_file(tmp_path):
    for name in ("b.jpg", "a.PNG", "notes.txt"):
        (tmp_path / name).write_bytes(b"x")
    assert [p.name for p in viewer.image_paths(tmp_path)] == ["a.PNG", "b.jpg"]
    assert viewer.image_paths(tmp_path / "b.jpg") == [tmp_path / "b.jpg"]


def test_load_annotations_indexes_by_stem(tmp_path):
    (tmp_path / "f1.txt").write_text("0 10 10 30 10 30 20 10 20\n", encoding="utf-8")
    (tmp_path / "f2.txt").write_text("", encoding="utf-8")
    loaded = viewer.load_annotations(tmp_path, None, NAMES, {})
    assert sorted(loaded) == ["f1", "f2"]
    assert len(loaded["f1"]) == 1 and loaded["f2"] == []


def test_load_annotations_of_a_missing_directory_is_empty(tmp_path):
    assert viewer.load_annotations(tmp_path / "nope", None, NAMES, {}) == {}
    assert viewer.load_annotations(None, None, NAMES, {}) == {}


def test_render_draws_every_layer(tmp_path):
    image = tmp_path / "f1.jpg"
    cv2.imwrite(str(image), np.zeros((200, 300, 3), np.uint8))
    frame = {
        "path": image,
        "obb": [Box(0, formats.rect(10, 10, 60, 60))],
        "hbb": [Box(0, formats.rect(8, 8, 62, 62))],
        "cmp": [Box(0, formats.rect(12, 12, 58, 58))],
        "polygons": [np.array([[20.0, 20.0], [40.0, 20.0], [40.0, 40.0]])],
    }
    img = viewer.render(frame, NAMES)
    for color in (viewer.OBB_COLOR, viewer.HBB_COLOR, viewer.CMP_COLOR, viewer.POLY_COLOR):
        assert (img == np.array(color, np.uint8)).all(2).any(), color


def _viewer(tmp_path, n_frames=2):
    frames = []
    for i in range(n_frames):
        path = tmp_path / f"f{i}.jpg"
        cv2.imwrite(str(path), np.zeros((240, 320, 3), np.uint8))
        frames.append(
            {
                "path": path,
                "obb": [Box(0, formats.rect(10, 10, 60, 60), score=0.4, difficult=i)],
                "hbb": [Box(0, formats.rect(8, 8, 62, 62))],
                "cmp": [],
                "polygons": [],
            }
        )
    return viewer.Viewer(frames, NAMES, win_w=320, win_h=240)


def test_viewer_view_produces_a_window_sized_canvas(tmp_path):
    v = _viewer(tmp_path)
    canvas = v.view()
    assert canvas.shape == (240, 320, 3)
    assert v.frame["path"].stem in v.status()
    assert "q quit" in v.status()


def test_viewer_zoom_is_clamped_to_fit_and_to_16x(tmp_path):
    v = _viewer(tmp_path)
    v.zoom = 1000
    v.clamp()
    assert v.zoom == 16.0
    v.zoom = 0.001
    v.clamp()
    assert v.zoom == pytest.approx(1.0)  # the frame exactly fills the window


def test_viewer_zoom_keeps_the_point_under_the_cursor(tmp_path):
    v = _viewer(tmp_path)
    v.zoom = 4.0
    v.clamp()
    before = v.to_image(100, 80)
    v.on_mouse(cv2.EVENT_MOUSEWHEEL, 100, 80, 1, None)
    assert v.to_image(100, 80) == pytest.approx(before, abs=1e-6)
    cv2.destroyAllWindows()


def test_viewer_hides_difficult_boxes_when_toggled_off(tmp_path):
    v = _viewer(tmp_path)
    v.idx = 1  # this frame's single box is flagged difficult
    v.load()
    v.show_difficult = False
    shown = v.view()
    v.show_difficult = True
    assert not np.array_equal(shown, v.view())


def test_viewer_compare_mode_cycles(tmp_path):
    v = _viewer(tmp_path)
    assert v.compare_mode == 0
    for expected in (1, 2, 0):
        v.compare_mode = (v.compare_mode + 1) % 3
        assert v.compare_mode == expected


def test_viewer_frame_stems_wrap(tmp_path):
    v = _viewer(tmp_path, n_frames=3)
    assert isinstance(v.frames[0]["obb"][0], Box)
    v.idx = (v.idx - 1) % len(v.frames)
    v.load()
    assert v.idx == 2


def test_viewer_status_counts_difficult_and_compared(tmp_path):
    v = _viewer(tmp_path)
    v.idx = 1
    v.load()
    assert "1 difficult" in v.status()
    v.frame["cmp"] = [Box(0, formats.rect(1, 1, 5, 5))]
    assert "1 compared" in v.status()


def test_frame_annotations_length():
    assert len(FrameAnnotations("a", 10, 10, [Box(0, formats.rect(0, 0, 1, 1))])) == 1


# ------------------------------------------------------------------------------------ the view CLI
@pytest.fixture
def dataset(tmp_path, monkeypatch):
    """One frame with YOLO OBBs and no derived formats, laid out the way hbb2obb writes it."""
    monkeypatch.setenv("HBB2OBB_DISABLE_UPDATE_CHECK", "1")
    (tmp_path / "images").mkdir()
    cv2.imwrite(str(tmp_path / "images" / "img1.jpg"), np.zeros((60, 80, 3), np.uint8))
    (tmp_path / "labels_obb").mkdir()
    (tmp_path / "labels_obb" / "img1.txt").write_text("0 10 10 30 12 28 32 8 30\n", encoding="utf-8")
    return tmp_path


def run_view(monkeypatch, *argv):
    monkeypatch.setattr("sys.argv", ["hbb2obb-view", *[str(a) for a in argv]])
    try:
        main_hbb2obb_view()
    except SystemExit as exc:
        return exc.code or 0
    return 0


def test_view_reports_a_requested_format_that_is_not_there(dataset, monkeypatch, tmp_path):
    """Without this the window opens and draws nothing, which reads as the viewer being broken."""
    result = run_view(monkeypatch, dataset / "images", "--obb_format", "dota", "-o", tmp_path / "out")
    assert isinstance(result, str) and "No dota OBB annotations" in result


def test_view_warns_when_there_are_no_confidences_to_show(dataset, monkeypatch, tmp_path, capsys):
    assert run_view(monkeypatch, dataset / "images", "--show_confidence", "-o", tmp_path / "out") == 0
    assert "no confidence scores" in capsys.readouterr().out
    assert (tmp_path / "out" / "img1.jpg").is_file()


def test_the_hbb_casing_keeps_a_light_line_readable_on_a_light_background():
    """
    A white HBB over pale asphalt or a lane marking is one grey on another. The casing lays a
    near-black line under it, so the box survives whatever it is drawn over without becoming
    the loudest thing in the frame.
    """
    pale = np.full((120, 160, 3), 235, np.uint8)

    plain = pale.copy()
    viewer.draw(plain, [Box(0, formats.rect(20, 20, 120, 90))], viewer.HBB_COLOR, NAMES, labels=False, thickness=2)

    cased = pale.copy()
    viewer.draw(
        cased,
        [Box(0, formats.rect(20, 20, 120, 90))],
        viewer.HBB_COLOR,
        NAMES,
        labels=False,
        thickness=2,
        casing=True,
    )

    # Without the casing nothing on the box edge is darker than the background it sits on
    assert plain.min() >= 235
    assert cased.min() <= 60  # the casing is there
    assert (cased == 255).any()  # and the white core survived it


def test_the_hbb_is_white_and_the_obb_stays_the_dominant_line():
    """The HBB is the supporting overlay: never thicker than the OBB, and never the saturated one."""
    assert viewer.HBB_COLOR == (255, 255, 255)
    assert sum(viewer.CASING_COLOR) < 150

    canvas_hbb = np.zeros((120, 160, 3), np.uint8)
    canvas_obb = np.zeros((120, 160, 3), np.uint8)
    box = [Box(0, formats.rect(20, 20, 120, 90))]
    viewer.draw(canvas_hbb, box, viewer.HBB_COLOR, NAMES, labels=False, thickness=2, casing=True)
    viewer.draw(canvas_obb, box, viewer.OBB_COLOR, NAMES, labels=False, thickness=2)

    assert (canvas_obb[:, :, 1] > 0).sum() <= (canvas_hbb[:, :, 1] > 0).sum()  # casing widens the HBB band


def test_viewer_toggles_each_box_layer_independently(tmp_path):
    """`o` hides the oriented boxes as `h` hides the horizontal ones, and neither touches the other."""
    v = _viewer(tmp_path)
    assert v.show_obb and v.show_hbb, "both layers start visible"

    def drawn(canvas, color):
        return (canvas == np.array(color, np.uint8)).all(2).any()

    both = v.view()
    assert drawn(both, viewer.OBB_COLOR) and drawn(both, viewer.HBB_COLOR)

    v.show_obb = False
    without_obb = v.view()
    assert not drawn(without_obb, viewer.OBB_COLOR)
    assert drawn(without_obb, viewer.HBB_COLOR), "hiding the OBBs must leave the HBBs alone"

    v.show_obb, v.show_hbb = True, False
    without_hbb = v.view()
    assert drawn(without_hbb, viewer.OBB_COLOR)
    assert not drawn(without_hbb, viewer.HBB_COLOR)


def test_the_key_legend_names_every_layer_toggle(tmp_path):
    """The status bar is the only place the keys are listed, so a new toggle has to appear there."""
    import inspect
    import re

    source = inspect.getsource(viewer)
    toggles = set(re.findall(r'key == ord\("(\w)"\):\n\s+self\.(?:show_\w+|compare_mode)', source))
    listed = set(re.findall(r"(?:^|\s)(\w)\s", viewer.Viewer.KEYS))

    assert "o" in toggles, "the OBB layer needs a key of its own"
    assert toggles <= listed, f"toggles handled but not listed: {sorted(toggles - listed)}"
    assert "o obb" in viewer.Viewer.KEYS and "h hbb" in viewer.Viewer.KEYS
