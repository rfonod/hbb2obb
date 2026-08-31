"""Tests for hbb2obb.detector and the hbb2obb-detect CLI. No weights are downloaded or run."""

import numpy as np
import pytest

from hbb2obb import detector
from hbb2obb.cli import main_hbb2obb_detect
from hbb2obb.utils import Annotations

cv2 = pytest.importorskip("cv2")


class FakeTensor:
    """The slice of the torch API that detect_hbb touches on an Ultralytics Boxes object."""

    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class FakeBoxes:
    def __init__(self, rows):
        rows = np.asarray(rows, dtype=float).reshape(-1, 6)
        self.cls = FakeTensor(rows[:, 0])
        self.xywh = FakeTensor(rows[:, 1:5])
        self.conf = FakeTensor(rows[:, 5])

    def __len__(self):
        return len(self.cls.values)


class FakeResult:
    def __init__(self, rows):
        self.boxes = FakeBoxes(rows)


class FakeModel:
    """Stands in for an Ultralytics YOLO, recording the predictor arguments it was called with."""

    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def __call__(self, img, **kwargs):
        self.calls.append(kwargs)
        return [FakeResult(self.rows)]


@pytest.fixture
def fake_detector(monkeypatch):
    """Replace the loader so nothing is downloaded; the model returns two fixed boxes."""
    model = FakeModel([[0, 100, 100, 40, 20, 0.9], [2, 300, 200, 60, 30, 0.4]])
    monkeypatch.setattr(detector, "load_detector", lambda *_args, **_kwargs: model)
    return model


# ---------------------------------------------------------------------------- weights resolution
def test_registered_detector_downloads_into_the_models_directory(monkeypatch, tmp_path):
    asked = {}

    def fake_download(url, destination):
        asked["url"], asked["destination"] = url, destination
        return destination

    monkeypatch.setattr(detector, "download_weights", fake_download)
    monkeypatch.setattr(detector, "WEIGHTS_DIR", tmp_path / "models")

    resolved = detector.resolve_weights("geotrax")
    assert resolved == tmp_path / "models" / "geotrax_hbb_yolov8s_1920_v1.pt"
    assert asked["url"].startswith("https://huggingface.co/rfonod/geo-trax/resolve/main/")


def test_an_existing_file_is_used_as_is(tmp_path):
    weights = tmp_path / "my_model.pt"
    weights.write_bytes(b"x")
    assert detector.resolve_weights(str(weights)) == weights


def test_a_hugging_face_reference_becomes_a_resolve_url(monkeypatch, tmp_path):
    asked = {}

    def fake_download(url, destination):
        asked["url"] = url
        return destination

    monkeypatch.setattr(detector, "download_weights", fake_download)
    monkeypatch.setattr(detector, "WEIGHTS_DIR", tmp_path / "models")

    resolved = detector.resolve_weights("someone/custom_hbb.pt")
    assert resolved == tmp_path / "models" / "custom_hbb.pt"
    assert asked["url"] == "https://huggingface.co/someone/resolve/main/custom_hbb.pt"


def test_an_ultralytics_name_is_left_for_ultralytics_to_fetch(monkeypatch, tmp_path):
    """No download here: passing the path to YOLO() is what triggers Ultralytics' own fetch."""
    monkeypatch.setattr(detector, "WEIGHTS_DIR", tmp_path / "models")
    monkeypatch.setattr(detector, "download_weights", lambda *_: pytest.fail("should not download"))
    assert detector.resolve_weights("yolo11s") == tmp_path / "models" / "yolo11s.pt"


def test_an_existing_checkpoint_is_not_downloaded_again(tmp_path):
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"x")
    assert detector.download_weights("https://example.invalid/weights.pt", weights) == weights


# ------------------------------------------------------------------------------------- detection
def test_detect_hbb_returns_absolute_xywh_with_confidence(fake_detector):
    rows = detector.detect_hbb(np.zeros((100, 100, 3), np.uint8), model="geotrax")
    assert rows.shape == (2, 6)
    assert rows[0].tolist() == [0, 100, 100, 40, 20, pytest.approx(0.9)]


def test_detect_hbb_applies_the_registered_defaults(fake_detector):
    detector.detect_hbb(np.zeros((100, 100, 3), np.uint8), model="geotrax")
    kwargs = fake_detector.calls[0]
    assert kwargs["imgsz"] == 1920  # the resolution geo-trax was trained and validated at
    assert kwargs["classes"] == [0, 1, 2, 3]  # its two unreliable classes are left out


def test_detect_hbb_overrides_beat_the_registered_defaults(fake_detector):
    detector.detect_hbb(np.zeros((100, 100, 3), np.uint8), model="geotrax", imgsz=960, conf=0.5, classes=[0])
    kwargs = fake_detector.calls[0]
    assert (kwargs["imgsz"], kwargs["conf"], kwargs["classes"]) == (960, 0.5, [0])


def test_an_unregistered_detector_falls_back_to_the_ultralytics_defaults(monkeypatch):
    model = FakeModel([[0, 10, 10, 4, 2, 0.5]])
    monkeypatch.setattr(detector, "load_detector", lambda *_args, **_kwargs: model)
    detector.detect_hbb(np.zeros((20, 20, 3), np.uint8), model="yolo11s.pt")
    assert model.calls[0]["imgsz"] == 640
    assert "classes" not in model.calls[0]


def test_detect_hbb_of_an_empty_result_is_an_empty_array(monkeypatch):
    monkeypatch.setattr(detector, "load_detector", lambda *_a, **_k: FakeModel(np.empty((0, 6))))
    assert detector.detect_hbb(np.zeros((20, 20, 3), np.uint8)).shape == (0, 6)


# ------------------------------------------------------------------------------------ class maps
def test_parse_class_map_reads_pairs():
    assert detector.parse_class_map("2=0,5=1,7=2") == {2: 0, 5: 1, 7: 2}
    assert detector.parse_class_map(None) == {}
    with pytest.raises(ValueError, match="Invalid class map entry"):
        detector.parse_class_map("2->0")


def test_a_class_map_renumbers_and_drops_what_it_does_not_name():
    rows = np.array([[2, 10, 10, 4, 2, 0.9], [5, 20, 20, 8, 4, 0.8], [0, 30, 30, 2, 2, 0.7]])
    mapped = detector.apply_class_map(rows, {2: 0, 5: 1})
    assert mapped[:, 0].tolist() == [0, 1]  # the COCO person is gone, car and bus renumbered
    assert mapped[1, 1:].tolist() == [20, 20, 8, 4, pytest.approx(0.8)]


# ---------------------------------------------------------------------------------------- merge
def test_merge_keeps_the_hand_drawn_geometry_and_only_takes_the_score():
    manual = np.array([[0, 100, 100, 40, 20], [1, 300, 200, 60, 30]])
    detected = np.array([[0, 102, 101, 41, 21, 0.87], [1, 299, 200, 59, 31, 0.62]])

    report = detector.merge_detections(manual, detected)
    assert report.scores == pytest.approx([0.87, 0.62])
    assert report.matched == [0, 1] and report.missed == [] and report.extra == []


def test_a_box_the_detector_missed_keeps_full_confidence():
    manual = np.array([[0, 100, 100, 40, 20], [0, 900, 900, 40, 20]])
    detected = np.array([[0, 100, 100, 40, 20, 0.51]])

    report = detector.merge_detections(manual, detected)
    assert report.scores == pytest.approx([0.51, 1.0])
    assert report.missed == [1]


def test_detections_backing_no_hand_drawn_box_are_reported_not_added():
    manual = np.array([[0, 100, 100, 40, 20]])
    detected = np.array([[0, 100, 100, 40, 20, 0.9], [0, 900, 900, 40, 20, 0.3]])

    report = detector.merge_detections(manual, detected)
    assert len(report.scores) == 1  # the merge never grows the hand-drawn set
    assert report.extra == [1]


def test_merge_matches_across_a_class_disagreement_and_reports_it():
    """The annotator's label wins; the disagreement is worth a human look, not a silent relabel."""
    manual = np.array([[0, 100, 100, 40, 20]])
    detected = np.array([[2, 100, 100, 40, 20, 0.7]])

    report = detector.merge_detections(manual, detected)
    assert report.scores == pytest.approx([0.7])
    assert report.conflicts == [(0, 0)]  # hand-drawn box 0 matched detection 0, labelled otherwise


def test_merge_is_one_to_one_and_takes_the_best_overlap_first():
    manual = np.array([[0, 100, 100, 40, 20], [0, 110, 100, 40, 20]])
    detected = np.array([[0, 110, 100, 40, 20, 0.4]])  # exactly the second box

    report = detector.merge_detections(manual, detected)
    assert report.scores == pytest.approx([1.0, 0.4])


def test_merge_below_the_threshold_does_not_match():
    manual = np.array([[0, 100, 100, 40, 20]])
    detected = np.array([[0, 130, 100, 40, 20, 0.9]])  # a quarter of the width overlaps

    assert detector.merge_detections(manual, detected, iou_threshold=0.5).scores == pytest.approx([1.0])
    assert detector.merge_detections(manual, detected, iou_threshold=0.1).scores == pytest.approx([0.9])


def test_merge_with_nothing_on_either_side():
    empty = np.empty((0, 6))
    assert len(detector.merge_detections(np.empty((0, 5)), empty).scores) == 0

    manual = np.array([[0, 100, 100, 40, 20]])
    report = detector.merge_detections(manual, empty)
    assert report.scores == pytest.approx([1.0]) and report.missed == [0]


# --------------------------------------------------------------------------------------- writing
def test_save_hbb_annotations_writes_what_hbb2obb_reads_back(tmp_path):
    boxes = np.array([[0, 1923.7, 1877.75, 63.1, 86.6], [1, 100.0, 200.0, 10.0, 20.0]])
    path = detector.save_hbb_annotations(boxes, tmp_path, tmp_path / "img1.jpg", scores=[0.9123, 1.0])

    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "0 1923.7 1877.75 63.1 86.6 0.9123"  # trailing zeros trimmed, not padded
    assert lines[1] == "1 100 200 10 20 1.0000"

    read_back = Annotations(path, np.zeros((2160, 3840, 3), np.uint8))
    assert read_back.hbb_xywh[:, 1:] == pytest.approx(boxes[:, 1:])
    assert read_back.hbb_scores == pytest.approx([0.9123, 1.0])


def test_save_hbb_annotations_can_leave_the_confidence_out(tmp_path):
    boxes = np.array([[0, 10.0, 20.0, 4.0, 2.0]])
    path = detector.save_hbb_annotations(boxes, tmp_path, tmp_path / "img1.jpg")
    assert path.read_text(encoding="utf-8").strip() == "0 10 20 4 2"


def test_save_hbb_annotations_normalizes_against_the_frame(tmp_path):
    boxes = np.array([[0, 50.0, 100.0, 10.0, 20.0]])
    path = detector.save_hbb_annotations(
        boxes, tmp_path, tmp_path / "img1.jpg", precision=4, normalize=True, img_shape=(100, 200)
    )
    assert path.read_text(encoding="utf-8").strip() == "0 0.5 0.5 0.1 0.1"

    with pytest.raises(ValueError, match="img_shape is required"):
        detector.save_hbb_annotations(boxes, tmp_path, tmp_path / "img1.jpg", normalize=True)


# ------------------------------------------------------------------------------------- the CLI
@pytest.fixture
def images(tmp_path, monkeypatch):
    monkeypatch.setenv("HBB2OBB_DISABLE_UPDATE_CHECK", "1")
    (tmp_path / "images").mkdir()
    cv2.imwrite(str(tmp_path / "images" / "img1.jpg"), np.zeros((400, 600, 3), np.uint8))
    return tmp_path


def run(monkeypatch, *argv):
    monkeypatch.setattr("sys.argv", ["hbb2obb-detect", *[str(a) for a in argv]])
    try:
        main_hbb2obb_detect()
    except SystemExit as exc:
        return exc.code or 0
    return 0


def test_detect_cli_writes_labels_beside_the_images(images, monkeypatch, fake_detector, capsys):
    assert run(monkeypatch, images / "images") == 0
    written = (images / "labels_hbb" / "img1.txt").read_text(encoding="utf-8").splitlines()
    assert written == ["0 100 100 40 20 0.9000", "2 300 200 60 30 0.4000"]
    assert "Wrote 2 boxes" in capsys.readouterr().out


def test_detect_cli_refuses_to_replace_existing_labels(images, monkeypatch, fake_detector):
    (images / "labels_hbb").mkdir()
    (images / "labels_hbb" / "img1.txt").write_text("0 1 2 3 4\n", encoding="utf-8")

    result = run(monkeypatch, images / "images")
    assert isinstance(result, str) and "--overwrite" in result
    assert (images / "labels_hbb" / "img1.txt").read_text(encoding="utf-8") == "0 1 2 3 4\n"

    assert run(monkeypatch, images / "images", "--overwrite") == 0
    assert (images / "labels_hbb" / "img1.txt").read_text(encoding="utf-8").startswith("0 100")


def test_detect_cli_merges_into_hand_drawn_boxes(images, monkeypatch, fake_detector, capsys):
    manual = images / "manual"
    manual.mkdir()
    # The first box is the one the detector found, moved a pixel; the second it never saw.
    manual.joinpath("img1.txt").write_text("0 101 100 40 20\n1 500 300 30 30\n", encoding="utf-8")

    assert run(monkeypatch, images / "images", "-mw", manual, "--extras_dir", images / "extras") == 0

    written = (images / "labels_hbb" / "img1.txt").read_text(encoding="utf-8").splitlines()
    assert written == ["0 101 100 40 20 0.9000", "1 500 300 30 30 1.0000"]  # geometry untouched
    assert (images / "extras" / "img1.txt").read_text(encoding="utf-8").strip() == "2 300 200 60 30 0.4000"

    out = capsys.readouterr().out
    assert "Scored by a detection: 1" in out and "missed by the detector): 1" in out
