"""
Tests for hbb2obb.provenance.

The point of a provenance file is that somebody can check it years later, so what matters is
that the hashes are of the right bytes and that a missing checkpoint is reported rather than
quietly left out.
"""

import hashlib

from hbb2obb import provenance


def recorded_command(path):
    """The reproducing command, which is the line under the section that promises it."""
    lines = path.read_text().splitlines()
    heading = next(i for i, ln in enumerate(lines) if ln.startswith("Command that reproduces"))
    return lines[heading + 2]


def test_sha256_matches_hashlib(tmp_path):
    path = tmp_path / "weights.pt"
    path.write_bytes(b"not really a checkpoint")
    assert provenance.sha256(path) == hashlib.sha256(b"not really a checkpoint").hexdigest()


def test_sha256_of_set_changes_with_the_content(tmp_path):
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "a.txt").write_text("0 1 2 3 4\n")
    (labels / "b.txt").write_text("1 5 6 7 8\n")
    first = provenance.sha256_of_set(labels)

    (labels / "b.txt").write_text("1 5 6 7 9\n")
    assert provenance.sha256_of_set(labels) != first
    assert "(2 files)" in provenance.sha256_of_set(labels)


def test_sha256_of_set_changes_when_a_frame_is_renamed(tmp_path):
    """
    Names are hashed as well as contents.

    A set that lost a frame and gained an identical one under another name is a different set,
    and a benchmark measured against it is measuring something else.
    """
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "00001.txt").write_text("0 1 2 3 4\n")
    before = provenance.sha256_of_set(labels)

    (labels / "00001.txt").rename(labels / "00002.txt")
    assert provenance.sha256_of_set(labels) != before


def test_sha256_of_set_reports_an_absent_directory(tmp_path):
    assert provenance.sha256_of_set(tmp_path / "nowhere") == "directory not found"
    empty = tmp_path / "empty"
    empty.mkdir()
    assert provenance.sha256_of_set(empty) == "no files matched"


def test_source_digest_follows_the_bytes_not_the_history(tmp_path):
    """
    The digest is over the source itself, which is the point.

    A commit hash stops identifying anything once the branch holding it is squashed, rebased
    or deleted, and an uncommitted tree never had one. The digest keeps working in both cases,
    so it is what a reader years later can actually check.
    """
    package = tmp_path / "hbb2obb"
    (package / "sub").mkdir(parents=True)
    (package / "converter.py").write_text("x = 1\n")
    (package / "sub" / "helper.py").write_text("y = 2\n")
    (package / "notes.md").write_text("not source\n")
    first = provenance.source_digest(package)
    assert "(2 files)" in first  # the markdown file is not code and does not count

    (package / "notes.md").write_text("still not source\n")
    assert provenance.source_digest(package) == first

    (package / "sub" / "helper.py").write_text("y = 3\n")
    assert provenance.source_digest(package) != first


def test_header_names_the_source_digest_when_the_tree_is_dirty(tmp_path, monkeypatch):
    monkeypatch.setattr(
        provenance, "git_state", lambda: {"commit": "a" * 40, "describe": "v1.4.0-4-gaaaaaaa-dirty", "dirty": 7}
    )
    text = "\n".join(provenance.header("HBB2OBB test provenance"))
    assert "git commit     : " + "a" * 40 in text
    assert "git describe   : v1.4.0-4-gaaaaaaa-dirty" in text
    assert "source state   : MODIFIED, 7 path(s) under the package differ" in text
    assert "source sha256  : " in text


def test_header_says_the_commit_is_checkoutable_when_it_is(monkeypatch):
    monkeypatch.setattr(provenance, "git_state", lambda: {"commit": "b" * 40, "describe": "v1.5.0", "dirty": 0})
    text = "\n".join(provenance.header("HBB2OBB test provenance"))
    assert "source state   : matches that commit" in text
    assert "MODIFIED" not in text


def test_dirtiness_is_measured_over_the_package_not_the_whole_repository(tmp_path):
    """
    A sweep writing results into the repository must not mark its own record as untrustworthy.

    The line answers "can a reader check out this commit and get the code that ran", and files
    the run itself produced elsewhere in the repository cannot change that answer.
    """
    import subprocess

    repo = tmp_path / "repo"
    package = repo / "hbb2obb"
    package.mkdir(parents=True)
    (package / "converter.py").write_text("x = 1\n")
    (repo / "results.txt").write_text("committed\n")
    for args in (["init", "-q"], ["add", "-A"], ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "i"]):
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)

    assert provenance.git_state(repo, package)["dirty"] == 0

    (repo / "results.txt").write_text("a sweep rewrote this\n")
    assert provenance.git_state(repo, package)["dirty"] == 0  # output, not code

    (package / "converter.py").write_text("x = 2\n")
    assert provenance.git_state(repo, package)["dirty"] == 1


def test_header_survives_an_installed_wheel(monkeypatch):
    """A PyPI install has no checkout, and that is not an error: the digest still identifies it."""
    monkeypatch.setattr(provenance, "git_state", lambda: {"commit": None, "describe": None, "dirty": None})
    text = "\n".join(provenance.header("HBB2OBB test provenance"))
    assert "git commit     : not a git checkout" in text
    assert "source sha256  : " in text


def test_conversion_provenance_records_the_command_and_the_checkpoints(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    (models / "sam_b.pt").write_bytes(b"weights")
    hbb = tmp_path / "labels_hbb"
    hbb.mkdir()
    (hbb / "00001.txt").write_text("0 10 10 20 20 0.9\n")

    out = tmp_path / "PROVENANCE.txt"
    status = provenance.write_conversion_provenance(
        out=out,
        img_source=tmp_path / "images",
        hbb_dir=hbb,
        obb_dir=tmp_path / "labels_obb",
        sam_models=["sam_b"],
        imgsz=1280,
        scale_factors=[0.05],
        opening_kernel_percentage=0.15,
        confidence_source="combined",
        models_dir=models,
    )

    assert status == 0
    text = out.read_text()
    assert "hbb2obb " + str(tmp_path / "images") in text
    assert "--sam_models sam_b --imgsz 1280 --scale_factors 0.05" in text
    assert "--confidence_source combined" in text
    assert hashlib.sha256(b"weights").hexdigest() in text
    assert "HBB sha256" in text


def test_conversion_provenance_records_the_inference_device(tmp_path):
    out = tmp_path / "PROVENANCE.txt"
    provenance.write_conversion_provenance(
        out=out,
        img_source=None,
        hbb_dir=None,
        obb_dir=None,
        sam_models=["sam_b"],
        imgsz=1280,
        scale_factors=[0.05],
        opening_kernel_percentage=0.15,
        device="cuda:0",
        models_dir=tmp_path / "models",
    )
    text = out.read_text()
    assert "--device cuda:0" in text
    assert "Inference device         : cuda:0" in text


def test_conversion_provenance_reports_a_missing_checkpoint(tmp_path):
    out = tmp_path / "PROVENANCE.txt"
    status = provenance.write_conversion_provenance(
        out=out,
        img_source=None,
        hbb_dir=None,
        obb_dir=None,
        sam_models=["sam_b"],
        imgsz=1280,
        scale_factors=[0.05],
        opening_kernel_percentage=0.15,
        models_dir=tmp_path / "models",
    )

    assert status == 1  # non-zero, so a release script cannot ship an incomplete record
    assert "MISSING" in out.read_text()


def test_an_even_ensemble_is_flagged_as_stricter(tmp_path):
    out = tmp_path / "PROVENANCE.txt"
    provenance.write_conversion_provenance(
        out=out,
        img_source=None,
        hbb_dir=None,
        obb_dir=None,
        sam_models=["sam_b", "sam_l"],
        imgsz=1280,
        scale_factors=[0.05],
        opening_kernel_percentage=0.15,
        models_dir=tmp_path,
    )
    assert "even-sized ensemble is stricter" in out.read_text()


def test_detection_provenance_hashes_the_detector(tmp_path):
    weights = tmp_path / "geotrax.pt"
    weights.write_bytes(b"detector")

    out = tmp_path / "PROVENANCE.txt"
    status = provenance.write_detection_provenance(
        out=out,
        img_source=tmp_path / "images",
        hbb_dir=tmp_path / "labels_hbb",
        model="geotrax",
        weights=weights,
        imgsz=1920,
        conf=0.25,
        iou=0.45,
        classes=(0, 1, 2, 3),
    )

    assert status == 0
    text = out.read_text()
    assert hashlib.sha256(b"detector").hexdigest() in text
    assert "--imgsz 1920" in text
    assert "Classes kept             : 0 1 2 3" in text


def test_benchmark_provenance_embeds_the_config_and_hashes_both_inputs(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    (models / "sam_b.pt").write_bytes(b"weights")
    for name in ("labels_hbb", "labels_obb_gt"):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "00001.txt").write_text("0 1 2 3 4\n")

    out = tmp_path / "PROVENANCE.txt"
    provenance.write_benchmark_provenance(
        out=out,
        command="hbb2obb-optimize -c data/benchmark.yaml",
        config_text="img_source: data/images\nruns:\n  - sam_models: [sam_b]\n",
        config_path=tmp_path / "benchmark_results" / "benchmark.yaml",
        runs=[{"name": "sam_b", "sam_models": ["sam_b"]}],
        img_source=tmp_path / "images",
        hbb_dir=tmp_path / "labels_hbb",
        gt_dir=tmp_path / "labels_obb_gt",
        grid_description="1 x 1 x 1 = 1 point",
        elapsed_seconds=7200,
        models_dir=models,
    )

    text = out.read_text()
    assert "hbb2obb-optimize -c data/benchmark.yaml" in text
    assert "runs:\n  - sam_models: [sam_b]" in text
    assert "Total wall time          : 2.00 h" in text
    assert "HBB sha256" in text and "GT sha256" in text
    assert provenance.sha256_of_set(tmp_path / "labels_hbb") in text
    assert "hbb2obb-optimize -c " + str(tmp_path / "benchmark_results" / "benchmark.yaml") in text


def test_benchmark_provenance_lists_each_checkpoint_once(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    for name in ("sam_b.pt", "sam_l.pt"):
        (models / name).write_bytes(name.encode())

    out = tmp_path / "PROVENANCE.txt"
    provenance.write_benchmark_provenance(
        out=out,
        command="hbb2obb-optimize -c b.yaml",
        config_text=None,
        config_path=None,
        runs=[
            {"name": "sam_b", "sam_models": ["sam_b"]},
            {"name": "sam_l", "sam_models": ["sam_l"]},
            {"name": "sam_l-sam_b", "sam_models": ["sam_l", "sam_b"]},
        ],
        img_source=None,
        hbb_dir=None,
        gt_dir=None,
        grid_description="1 point",
        elapsed_seconds=60,
        models_dir=models,
    )

    text = out.read_text()
    assert text.count(hashlib.sha256(b"sam_b.pt").hexdigest()) == 1
    assert text.count(hashlib.sha256(b"sam_l.pt").hexdigest()) == 1


def test_conversion_provenance_records_the_coordinate_convention(tmp_path):
    """
    A normalized set rewritten at another precision is a different set, so the record has to
    carry both or the command it prints does not reproduce the labels it sits beside.
    """
    out = tmp_path / "PROVENANCE.txt"
    provenance.write_conversion_provenance(
        out=out,
        img_source=tmp_path / "images",
        hbb_dir=None,
        obb_dir=None,
        sam_models=["sam_b"],
        imgsz=1280,
        scale_factors=[0.05],
        opening_kernel_percentage=0.15,
        models_dir=tmp_path / "models",
        normalize=True,
        precision=6,
    )

    text = out.read_text()
    assert "--normalize --precision 6" in text
    assert "Coordinates              : normalized to [0, 1], 6 decimals" in text


def test_absolute_conversion_output_claims_no_decimals(tmp_path):
    """The conversion writes whole pixels when it is not normalizing, so there are none to name."""
    out = tmp_path / "PROVENANCE.txt"
    provenance.write_conversion_provenance(
        out=out,
        img_source=tmp_path / "images",
        hbb_dir=None,
        obb_dir=None,
        sam_models=["sam_b"],
        imgsz=1280,
        scale_factors=[0.05],
        opening_kernel_percentage=0.15,
        models_dir=tmp_path / "models",
    )

    text = out.read_text()
    assert "Coordinates              : absolute pixels\n" in text
    assert "--normalize" not in text
    assert "--precision" not in text


def test_detection_provenance_records_the_coordinate_convention(tmp_path):
    """
    hbb2obb-detect writes decimals in either convention, and used to record neither, so its
    command reproduced absolute whole pixels whatever the run had actually written.
    """
    out = tmp_path / "PROVENANCE_hbb.txt"
    provenance.write_detection_provenance(
        out=out,
        img_source=tmp_path / "images",
        hbb_dir=None,
        model="geotrax",
        weights=None,
        imgsz=1920,
        conf=0.25,
        iou=0.45,
        normalize=True,
        precision=8,
    )

    text = out.read_text()
    assert "--normalize --precision 8" in text
    assert "Coordinates              : normalized to [0, 1], 8 decimals" in text


def test_detection_provenance_records_absolute_decimals(tmp_path):
    """Absolute detector output is not integral: its precision changes the bytes too."""
    out = tmp_path / "PROVENANCE_hbb.txt"
    provenance.write_detection_provenance(
        out=out,
        img_source=tmp_path / "images",
        hbb_dir=None,
        model="geotrax",
        weights=None,
        imgsz=1920,
        conf=0.25,
        iou=0.45,
        precision=2,
    )

    text = out.read_text()
    assert "--precision 2" in text
    assert "Coordinates              : absolute pixels, 2 decimals" in text


def test_conversion_provenance_records_where_the_confidence_went(tmp_path):
    """
    The header promises a command that reproduces the annotations, and --save_confidence is what
    puts the 10th column in them. Naming the confidence source while omitting the flag that wrote
    it described labels the command would not produce.
    """
    out = tmp_path / "PROVENANCE.txt"
    common = dict(
        img_source=tmp_path / "images",
        hbb_dir=None,
        obb_dir=None,
        sam_models=["sam_b"],
        imgsz=1280,
        scale_factors=[0.05],
        opening_kernel_percentage=0.15,
        models_dir=tmp_path / "models",
    )

    provenance.write_conversion_provenance(out=out, save_confidence=True, **common)
    text = out.read_text()
    assert "--save_confidence" in text
    assert "Confidence written       : 10th column" in text

    provenance.write_conversion_provenance(out=out, **common)
    assert "Confidence written       : not written" in out.read_text()


def test_a_bare_confidence_dir_is_recorded_as_given(tmp_path):
    """`--confidence_dir` takes an optional value; bare means the conventional location."""
    out = tmp_path / "PROVENANCE.txt"
    common = dict(
        img_source=tmp_path / "images",
        hbb_dir=None,
        obb_dir=None,
        sam_models=["sam_b"],
        imgsz=1280,
        scale_factors=[0.05],
        opening_kernel_percentage=0.15,
        models_dir=tmp_path / "models",
    )

    provenance.write_conversion_provenance(out=out, confidence_dir="", **common)
    assert recorded_command(out).endswith("--confidence_dir"), "the bare form takes no value"
    assert "side-car (default location)" in out.read_text()

    provenance.write_conversion_provenance(out=out, save_confidence=True, confidence_dir="scores", **common)
    text = out.read_text()
    assert "--save_confidence --confidence_dir scores" in text
    assert "Confidence written       : 10th column and side-car scores" in text
