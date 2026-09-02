"""Tests for the hbb2obb-convert command line, including the conversions the old scripts covered."""

import json

import pytest

from hbb2obb import formats
from hbb2obb.cli import main_hbb2obb_convert, resolve_names


@pytest.fixture(autouse=True)
def no_update_check(monkeypatch):
    monkeypatch.setenv("HBB2OBB_DISABLE_UPDATE_CHECK", "1")


def run(monkeypatch, *argv):
    """Invoke the converter CLI, returning its exit code."""
    monkeypatch.setattr("sys.argv", ["hbb2obb-convert", *[str(a) for a in argv]])
    try:
        main_hbb2obb_convert()
    except SystemExit as exc:
        return exc.code or 0
    return 0


@pytest.fixture
def coco_file(tmp_path):
    path = tmp_path / "annotations.json"
    path.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "img1.jpg", "width": 100, "height": 200}],
                "categories": [{"id": 5, "name": "Car"}],
                "annotations": [{"id": 1, "image_id": 1, "category_id": 5, "bbox": [10, 20, 30, 40]}],
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def label_map(tmp_path):
    path = tmp_path / "classes.yaml"
    path.write_text("0: Car\n1: Bus\n", encoding="utf-8")
    return path


# ------------------------------------------------------------------ what json2yolo.py used to cover
def test_coco_to_normalized_yolo(monkeypatch, tmp_path, coco_file):
    out = tmp_path / "labels"
    assert run(monkeypatch, coco_file, "--from", "coco", "--to", "yolo", "-o", out, "--normalize") == 0

    values = (out / "img1.txt").read_text(encoding="utf-8").strip().split()
    assert len(values) == 5
    assert int(values[0]) == 0
    assert float(values[1]) == pytest.approx(0.25)  # (10 + 30/2) / 100
    assert float(values[2]) == pytest.approx(0.20)  # (20 + 40/2) / 200
    assert float(values[3]) == pytest.approx(0.30)
    assert float(values[4]) == pytest.approx(0.20)


def test_coco_to_absolute_yolo(monkeypatch, tmp_path, coco_file):
    out = tmp_path / "labels"
    assert run(monkeypatch, coco_file, "--from", "coco", "--to", "yolo", "-o", out) == 0
    values = [float(v) for v in (out / "img1.txt").read_text(encoding="utf-8").split()[1:]]
    assert values == pytest.approx([25.0, 40.0, 30.0, 40.0])


# ------------------------------------------------------------------- what voc2yolo.py used to cover
def test_voc_to_yolo(monkeypatch, tmp_path, label_map):
    voc = tmp_path / "voc"
    voc.mkdir()
    (voc / "img1.xml").write_text(
        """<annotation><folder>images</folder><filename>img1.jpg</filename>
        <size><width>100</width><height>200</height><depth>3</depth></size>
        <object><name>Bus</name><difficult>0</difficult>
        <bndbox><xmin>10</xmin><ymin>20</ymin><xmax>40</xmax><ymax>60</ymax></bndbox></object>
        </annotation>""",
        encoding="utf-8",
    )
    out = tmp_path / "labels"
    assert run(monkeypatch, voc, "--to", "yolo", "-o", out, "-mp", label_map) == 0
    values = (out / "img1.txt").read_text(encoding="utf-8").split()
    assert int(values[0]) == 1  # Bus, from the label map
    assert [float(v) for v in values[1:]] == pytest.approx([25.0, 40.0, 30.0, 40.0])


# --------------------------------------------------------------------------------- multiple outputs
def test_one_call_writes_several_formats(monkeypatch, tmp_path, label_map):
    src = tmp_path / "labels_obb_gt"
    src.mkdir()
    (src / "img1.txt").write_text("0 10 10 30 12 28 32 8 30\n", encoding="utf-8")
    out = tmp_path / "out"

    assert (
        run(monkeypatch, src, "--to", "dota", "coco", "labelme", "-o", out, "-mp", label_map, "-iw", 100, "-ih", 200)
        == 0
    )
    assert (out / "img1.dota").is_file()
    assert (out / "img1.json").is_file()
    assert (out / "coco_annotations_obb.json").is_file()


def test_kind_is_forced_when_asked(monkeypatch, tmp_path, label_map):
    src = tmp_path / "labels"
    src.mkdir()
    (src / "img1.txt").write_text("0 10 10 30 12 28 32 8 30\n", encoding="utf-8")
    out = tmp_path / "out"

    assert (
        run(monkeypatch, src, "--to", "voc", "-o", out, "-mp", label_map, "-iw", 100, "-ih", 200, "--kind", "hbb") == 0
    )
    text = (out / "img1.xml").read_text(encoding="utf-8")
    # the envelope of the rotated box (10,10) (30,12) (28,32) (8,30)
    for tag in ("<xmin>8</xmin>", "<ymin>10</ymin>", "<xmax>30</xmax>", "<ymax>32</ymax>"):
        assert tag in text, tag


def test_difficult_is_carried_across(monkeypatch, tmp_path, label_map):
    src = tmp_path / "labels"
    src.mkdir()
    (src / "img1.txt").write_text("0 10 10 30 12 28 32 8 30\n", encoding="utf-8")
    (src / "img1.dota").write_text("imagesource:drone\n10 10 30 12 28 32 8 30 Car 1\n", encoding="utf-8")
    out = tmp_path / "out"

    assert (
        run(
            monkeypatch,
            src,
            "--from",
            "yolo",
            "--to",
            "dota",
            "-o",
            out,
            "-mp",
            label_map,
            "-iw",
            100,
            "-ih",
            200,
            "--difficult_from",
            "dota",
        )
        == 0
    )
    assert (out / "img1.dota").read_text(encoding="utf-8").strip().endswith("Car 1")


def test_coco_output_name_can_be_pinned(monkeypatch, tmp_path, label_map):
    src = tmp_path / "labels_obb_gt"
    src.mkdir()
    (src / "img1.txt").write_text("0 10 10 30 12 28 32 8 30\n", encoding="utf-8")
    out = tmp_path / "out"

    args = ("--to", "coco", "-o", out, "-mp", label_map, "-iw", 100, "-ih", 200)
    assert run(monkeypatch, src, *args) == 0
    assert (out / "coco_annotations_obb.json").is_file()
    assert run(monkeypatch, src, *args, "--coco_name", "coco_annotations_obb_gt.json") == 0
    assert (out / "coco_annotations_obb_gt.json").is_file()


def test_confidence_survives_a_round_trip_through_coco(monkeypatch, tmp_path, label_map):
    src = tmp_path / "labels_obb"
    src.mkdir()
    (src / "img1.txt").write_text("0 10 10 30 12 28 32 8 30 0.7972\n", encoding="utf-8")
    out = tmp_path / "out"
    assert run(monkeypatch, src, "--to", "coco", "-o", out, "-mp", label_map, "-iw", 100, "-ih", 200) == 0

    data = json.loads((out / "coco_annotations_obb.json").read_text(encoding="utf-8"))
    assert data["annotations"][0]["score"] == pytest.approx(0.7972)

    back = tmp_path / "back"
    coco = out / "coco_annotations_obb.json"
    assert run(monkeypatch, coco, "--to", "yolo", "-o", back, "-mp", label_map, "-iw", 100, "-ih", 200) == 0
    assert (back / "img1.txt").read_text(encoding="utf-8").split()[-1] == "0.7972"


def test_yolo_wins_detection_over_a_derived_file_beside_it(monkeypatch, tmp_path, label_map):
    """A set ships its canonical .txt next to derived files; reading the derived one loses precision."""
    src = tmp_path / "labels_obb"
    src.mkdir()
    (src / "img1.txt").write_text("0 10.49 10.49 30 12 28 32 8 30 0.5\n", encoding="utf-8")
    assert run(monkeypatch, src, "--to", "dota", "-mp", label_map, "-iw", 100, "-ih", 200) == 0

    assert formats.detect_format(src) == "yolo"
    out = tmp_path / "out"
    assert run(monkeypatch, src, "--to", "yolo", "-o", out, "-mp", label_map, "-iw", 100, "-ih", 200) == 0
    values = (out / "img1.txt").read_text(encoding="utf-8").split()
    assert float(values[1]) == pytest.approx(10.49)  # not the 10 the DOTA file rounded it to
    assert values[-1] == "0.5000"  # and the confidence DOTA cannot carry is still there


def test_to_is_required_without_verify(monkeypatch, tmp_path):
    src = tmp_path / "labels"
    src.mkdir()
    (src / "img1.txt").write_text("0 10 20 30 40\n", encoding="utf-8")
    assert run(monkeypatch, src) == 2  # argparse usage error


# ----------------------------------------------------------------------------------------- verify
def test_verify_passes_on_formats_written_from_one_source(monkeypatch, tmp_path, label_map, capsys):
    src = tmp_path / "labels_obb_gt"
    src.mkdir()
    (src / "img1.txt").write_text("0 10.49 10.49 30.5 12 28 32 8 30\n", encoding="utf-8")
    assert run(monkeypatch, src, "--to", "dota", "-mp", label_map, "-iw", 100, "-ih", 200) == 0
    assert run(monkeypatch, src, "--verify", "-mp", label_map, "-iw", 100, "-ih", 200) == 0
    assert "OK" in capsys.readouterr().out


def test_verify_fails_and_names_the_frame(monkeypatch, tmp_path, label_map, capsys):
    src = tmp_path / "labels_obb_gt"
    src.mkdir()
    (src / "img1.txt").write_text("0 10 10 30 12 28 32 8 30\n", encoding="utf-8")
    (src / "img1.dota").write_text("imagesource:drone\n11 10 30 12 28 32 8 30 Car 0\n", encoding="utf-8")

    assert run(monkeypatch, src, "--verify", "-mp", label_map, "-iw", 100, "-ih", 200) == 1
    out = capsys.readouterr().out
    assert "FAILED" in out and "img1#0" in out


def test_verify_walks_a_dataset_root(monkeypatch, tmp_path, label_map, capsys):
    root = tmp_path / "dataset"
    (root / "labels_hbb").mkdir(parents=True)
    (root / "labels_hbb" / "img1.txt").write_text("0 25 40 30 40\n", encoding="utf-8")
    frames = [formats.FrameAnnotations("img1", 100, 200, [formats.Box(0, formats.rect(10, 20, 40, 60))])]
    formats.write_set(frames, root, "coco", ["Car", "Bus"], "hbb")

    assert run(monkeypatch, root, "--verify", "-mp", label_map, "-iw", 100, "-ih", 200) == 0
    assert "labels_hbb" in capsys.readouterr().out


# ------------------------------------------------------------------------------------ name lookup
def test_resolve_names_prefers_the_label_map(tmp_path, label_map):
    (tmp_path / "names.txt").write_text("Ignored\n", encoding="utf-8")
    assert resolve_names(label_map, tmp_path, []) == ["Car", "Bus"]


def test_resolve_names_falls_back_to_names_txt_then_classes_yaml(tmp_path):
    labels = tmp_path / "labels"
    labels.mkdir()
    (tmp_path / "classes.yaml").write_text("0: Car\n1: Bus\n", encoding="utf-8")
    assert resolve_names(None, labels, []) == ["Car", "Bus"]

    (tmp_path / "names.txt").write_text("car bus truck\n", encoding="utf-8")
    assert resolve_names(None, labels, []) == ["car", "bus", "truck"]


def test_resolve_names_falls_back_to_what_was_discovered(tmp_path):
    assert resolve_names(None, tmp_path, ["a", "b"]) == ["a", "b"]


# ---------------------------------------------------------------- difficult from a confidence side-car
@pytest.fixture
def obb_with_sidecar(tmp_path):
    """A two-box YOLO OBB frame plus the confidence side-car hbb2obb --confidence_dir writes."""
    labels = tmp_path / "labels_obb"
    labels.mkdir()
    # Rotated on purpose: an axis-aligned quad would be read back as an HBB.
    (labels / "img1.txt").write_text("0 30 10 50 30 30 50 10 30\n1 70 110 90 130 70 150 50 130\n", encoding="utf-8")
    scores = tmp_path / "labels_confidence"
    scores.mkdir()
    (scores / "img1.txt").write_text("0.9100\n0.0000\n", encoding="utf-8")
    return labels, scores


def test_difficult_from_a_confidence_sidecar(monkeypatch, tmp_path, obb_with_sidecar, label_map):
    labels, scores = obb_with_sidecar
    out = tmp_path / "out"
    code = run(
        monkeypatch,
        labels,
        "--from",
        "yolo",
        "--to",
        "dota",
        "-o",
        out,
        "-iw",
        100,
        "-ih",
        200,
        "-mp",
        label_map,
        "--difficult_from",
        "confidence",
        "--confidence_dir",
        scores,
        "--difficult_below",
        0.5,
    )
    assert code == 0

    lines = [ln.split() for ln in (out / "img1.dota").read_text(encoding="utf-8").splitlines() if ":" not in ln]
    # The fallback box scores 0.0 and is flagged; the confident one is not.
    assert [ln[-1] for ln in lines] == ["0", "1"]


def test_a_confidence_sidecar_never_leaks_into_the_coordinates(monkeypatch, tmp_path, obb_with_sidecar, label_map):
    """The flag is derived from the scores; the scores themselves stay out of the written labels."""
    labels, scores = obb_with_sidecar
    out = tmp_path / "out"
    assert (
        run(
            monkeypatch,
            labels,
            "--from",
            "yolo",
            "--to",
            "yolo",
            "-o",
            out,
            "-iw",
            100,
            "-ih",
            200,
            "-mp",
            label_map,
            "--difficult_from",
            "confidence",
            "--confidence_dir",
            scores,
        )
        == 0
    )
    for line in (out / "img1.txt").read_text(encoding="utf-8").splitlines():
        assert len(line.split()) == 9, "a YOLO OBB line must keep its nine standard fields"


def test_difficult_from_confidence_without_scores_says_so(monkeypatch, tmp_path, obb_with_sidecar, label_map):
    labels, _ = obb_with_sidecar
    code = run(
        monkeypatch,
        labels,
        "--from",
        "yolo",
        "--to",
        "dota",
        "-o",
        tmp_path / "out",
        "-iw",
        100,
        "-ih",
        200,
        "-mp",
        label_map,
        "--difficult_from",
        "confidence",
    )
    assert "confidence_dir" in str(code)


def test_a_partly_scored_source_is_refused(monkeypatch, tmp_path, label_map):
    """One frame with a confidence column and one without has no score to compare for half the boxes."""
    labels = tmp_path / "labels_obb"
    labels.mkdir()
    (labels / "img1.txt").write_text("0 30 10 50 30 30 50 10 30 0.91\n", encoding="utf-8")
    (labels / "img2.txt").write_text("0 30 10 50 30 30 50 10 30\n", encoding="utf-8")
    code = run(
        monkeypatch,
        labels,
        "--from",
        "yolo",
        "--to",
        "dota",
        "-o",
        tmp_path / "out",
        "-iw",
        100,
        "-ih",
        200,
        "-mp",
        label_map,
        "--difficult_from",
        "confidence",
    )
    assert "not every box" in str(code)


def test_a_confidence_dir_that_feeds_nothing_is_refused(monkeypatch, tmp_path, obb_with_sidecar, label_map):
    """--confidence_dir is an input to --difficult_from confidence; alone it would be read by nothing."""
    labels, scores = obb_with_sidecar
    code = run(
        monkeypatch,
        labels,
        "--from",
        "yolo",
        "--to",
        "coco",
        "-o",
        tmp_path / "out",
        "-iw",
        100,
        "-ih",
        200,
        "-mp",
        label_map,
        "--confidence_dir",
        scores,
    )
    assert "difficult_from confidence" in str(code)


def test_a_mismatched_sidecar_is_refused(monkeypatch, tmp_path, obb_with_sidecar, label_map):
    """A side-car that does not line up with the labels stops the run, and says why, not a traceback."""
    labels, scores = obb_with_sidecar
    (scores / "img1.txt").write_text("0.9\n", encoding="utf-8")  # one score, two boxes
    code = run(
        monkeypatch,
        labels,
        "--from",
        "yolo",
        "--to",
        "dota",
        "-o",
        tmp_path / "out",
        "-iw",
        100,
        "-ih",
        200,
        "-mp",
        label_map,
        "--difficult_from",
        "confidence",
        "--confidence_dir",
        scores,
    )
    assert "align confidences" in str(code)


# ------------------------------------------------- verifying a published images/ + labels/ release
@pytest.fixture
def release_subset(tmp_path):
    """The layout a released subset has: labels/ beside its side-cars and one COCO file."""
    subset = tmp_path / "train"
    (subset / "labels").mkdir(parents=True)
    (subset / "labels" / "img1.txt").write_text("0 30 10 50 30 30 50 10 30\n", encoding="utf-8")
    (subset / "labels" / "img1.dota").write_text(
        "imagesource:drone\ngsd:null\n30 10 50 30 30 50 10 30 Car 0\n", encoding="utf-8"
    )
    (subset / "labels_confidence").mkdir()
    (subset / "labels_confidence" / "img1.txt").write_text("0.9100\n", encoding="utf-8")
    (subset / "labels_polygon").mkdir()
    (subset / "labels_polygon" / "img1.txt").write_text("0 30 10 40 20 50 30 40 40 30 50\n", encoding="utf-8")
    (subset / "coco_annotations.json").write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "img1.jpg", "width": 100, "height": 200}],
                "categories": [{"id": 1, "name": "Car"}],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "segmentation": [[30, 10, 50, 30, 30, 50, 10, 30]],
                        "bbox": [10, 10, 40, 40],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (subset / "classes.yaml").write_text("0: Car\n", encoding="utf-8")
    return subset


def test_verify_pairs_a_plain_labels_directory_with_its_coco_file(monkeypatch, capsys, release_subset):
    # labels_<name>/ pairs with coco_annotations_<name>.json, but a released subset calls the
    # directory plainly labels/ and the file coco_annotations.json. Without the pairing the COCO
    # file, the one most likely to drift, would never be compared against anything.
    assert run(monkeypatch, release_subset, "--verify", "-iw", 100, "-ih", 200) == 0
    out = capsys.readouterr().out
    assert "3 formats (coco, dota, yolo)" in out


def test_verify_skips_the_sidecar_directories(monkeypatch, capsys, release_subset):
    # labels_confidence/ holds one float per line and labels_polygon/ a variable-length contour;
    # read as YOLO labels either one raises, so walking every subdirectory has to skip them.
    assert run(monkeypatch, release_subset, "--verify", "-iw", 100, "-ih", 200) == 0
    out = capsys.readouterr().out
    assert "labels_confidence" not in out
    assert "labels_polygon" not in out
