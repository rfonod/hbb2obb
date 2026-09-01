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
