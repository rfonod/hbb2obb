"""The credit for detector weights comes from the Hugging Face Hub's own metadata, and nothing is guessed or kept."""

from pathlib import Path

import pytest

from hbb2obb import detector, hub, provenance

HF = "https://huggingface.co/someone/detector"
GEOTRAX_LIKE = {
    "id": "someone/detector",
    "sha": "bfb79e7a29334fc7cec7cfc2eb08fca0cd91d9d5",
    "tags": ["ultralytics", "arxiv:2411.02136", "doi:10.57967/hf/9296", "license:cc-by-4.0", "region:us"],
    "cardData": {"license": "cc-by-4.0", "github": "https://github.com/someone/detector"},
}


def test_the_record_reads_only_what_the_hub_maintains():
    record = hub.parse_hub_record("someone/detector", GEOTRAX_LIKE)
    assert record.licence == "cc-by-4.0"
    assert record.dois == ("10.57967/hf/9296",)
    assert record.arxiv == ("2411.02136",)
    assert record.revision == GEOTRAX_LIKE["sha"]
    assert record.page == "https://huggingface.co/someone/detector"


def test_undeclared_fields_are_empty():
    record = hub.parse_hub_record("someone/bare", {"id": "someone/bare", "tags": ["pytorch"]})
    assert (record.licence, record.dois, record.arxiv, record.revision) == (None, (), (), None)


def test_a_custom_licence_is_named_and_linked():
    card = {"license_name": "acme-1.0", "license_link": "https://example.org/acme"}
    assert hub.licence_from(["license:other"], card) == "other: acme-1.0 (https://example.org/acme)"
    assert hub.licence_from(["license:other"], {}) == "other (no name or link declared)"


def test_nothing_is_written_to_disk(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(hub, "fetch_json", lambda url, timeout=None: GEOTRAX_LIKE)
    hub.load_hub_record("someone/detector")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("model", "repo", "revision", "path"),
    [
        ("geotrax", "rfonod/geo-trax", "main", "geotrax_hbb_yolov8s_1920_v1.pt"),
        ("someone/detector/best.pt", "someone/detector", "main", "best.pt"),
        ("someone/detector/weights/v2/best.pt", "someone/detector", "main", "weights/v2/best.pt"),
        (
            "https://huggingface.co/someone/detector/resolve/abc123/weights/best.pt",
            "someone/detector",
            "abc123",
            "weights/best.pt",
        ),
        (
            "https://huggingface.co/someone/detector/blob/main/best.pt?download=true",
            "someone/detector",
            "main",
            "best.pt",
        ),
    ],
)
def test_hub_files_are_recognized_however_they_are_written(model, repo, revision, path):
    assert detector.hub_file(model) == detector.HubFile(repo, revision, path)
    assert detector.hf_repo(model) == repo


@pytest.mark.parametrize("model", ["https://example.org/models/best.pt", "yolo11s.pt", "./runs/detect/best.pt"])
def test_anything_else_is_not_a_hub_file(model):
    assert detector.hf_repo(model) is None


def test_a_local_file_is_never_taken_for_a_hub_reference(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a" / "b").mkdir(parents=True)
    (tmp_path / "a" / "b" / "best.pt").write_bytes(b"x")
    assert detector.hf_repo("a/b/best.pt") is None
    assert detector.resolve_weights("a/b/best.pt") == Path("a/b/best.pt")


def test_links_download_under_the_files_own_name(monkeypatch, tmp_path):
    asked = []
    monkeypatch.setattr(detector, "download_weights", lambda url, destination: asked.append(url) or destination)
    hub_link = "https://huggingface.co/someone/detector/blob/v1/weights/best.pt"
    assert detector.resolve_weights(hub_link, tmp_path) == tmp_path / "best.pt"
    assert detector.resolve_weights("https://example.org/files/other.pt?x=1", tmp_path) == tmp_path / "other.pt"
    assert asked == [
        "https://huggingface.co/someone/detector/resolve/v1/weights/best.pt",
        "https://example.org/files/other.pt?x=1",
    ]


def test_a_path_that_exists_nowhere_is_an_error_not_a_guess(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="not an existing file"):
        detector.resolve_weights("weights/best.pt", tmp_path)


def write(tmp_path, model):
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"detector")
    out = tmp_path / "PROVENANCE_hbb.txt"
    status = provenance.write_detection_provenance(
        out=out, img_source=None, hbb_dir=None, model=model, weights=weights, imgsz=1920, conf=0.25, iou=0.45
    )
    return status, out.read_text(encoding="utf-8")


def test_provenance_records_the_hub_metadata(tmp_path, monkeypatch):
    asked = []
    monkeypatch.setattr(hub, "fetch_json", lambda url, timeout=None: asked.append(url) or GEOTRAX_LIKE)
    status, text = write(tmp_path, "geotrax")
    assert status == 0
    assert asked == ["https://huggingface.co/api/models/rfonod/geo-trax"]
    assert "Model page     : https://huggingface.co/rfonod/geo-trax" in text
    assert "Licence        : cc-by-4.0" in text
    assert "DOI            : https://doi.org/10.57967/hf/9296" in text
    assert "arXiv          : https://arxiv.org/abs/2411.02136" in text
    assert f"Hub revision   : {GEOTRAX_LIKE['sha']}" in text
    assert "github" not in text  # a free-form card field, not Hub metadata


def test_undeclared_metadata_is_said_to_be_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(hub, "fetch_json", lambda url, timeout=None: {"tags": []})
    _, text = write(tmp_path, "someone/detector/best.pt")
    assert "Licence        : not declared on the Hub; check the model page" in text
    assert "no DOI or arXiv paper declared on the Hub" in text


def test_an_unreachable_hub_still_writes_the_provenance(tmp_path, capsys):
    status, text = write(tmp_path, "geotrax")  # the suite has no network
    assert status == 0
    assert "could not be read (OSError" in text and "Check the model page" in text
    assert "could not read the Hub metadata" in capsys.readouterr().out


@pytest.mark.parametrize("model", ["yolo11s.pt", "https://example.org/files/best.pt"])
def test_weights_off_the_hub_are_left_to_be_credited_by_hand(tmp_path, model):
    _, text = write(tmp_path, model)
    assert "not a Hugging Face Hub file" in text
