"""
Tests for the `hbb2obb` conversion command line: what a run reports about itself.

A box that SAM produced no usable mask for is written out as its own HBB. That is the right
output for one box, and a run that does it for a whole frame has gone wrong, so the run has to
say both: which frames fell back, and how many boxes over the whole run, in the summary and in
the record a dataset release cites.

Nothing here loads a checkpoint or runs SAM. `converter.load_sam_model` is swapped for a fake
whose masks depend only on the prompts it is handed, so a frame converts to the same bytes
wherever it sits in a directory unless the per-frame release is missing, which is the failure
these tests exist to pin.
"""

import cv2
import numpy as np
import pytest

import hbb2obb.converter as converter
from hbb2obb.cli import main_hbb2obb


@pytest.fixture(autouse=True)
def no_update_check(monkeypatch):
    monkeypatch.setenv("HBB2OBB_DISABLE_UPDATE_CHECK", "1")


@pytest.fixture
def metal(monkeypatch):
    """Count the device-cache trims, and keep the real Metal call out of the suite."""
    import torch

    calls = []
    monkeypatch.setattr(torch.mps, "empty_cache", lambda: calls.append(1))
    return calls


def run(monkeypatch, *argv):
    """Invoke the conversion CLI, returning its exit code."""
    monkeypatch.setattr("sys.argv", ["hbb2obb", *[str(a) for a in argv]])
    try:
        main_hbb2obb()
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1
    return 0


class FakeDevice:
    def __init__(self, kind):
        self.type = kind


class FakePredictor:
    """What ultralytics leaves behind after a call: the Results it yielded, on the device."""

    def __init__(self, kind):
        self.results = None
        self.device = FakeDevice(kind)


class FakeMasks:
    def __init__(self, data):
        self.data = data

    def cpu(self):
        return self

    def numpy(self):
        return self


class FakeResult:
    def __init__(self, masks):
        self.masks = masks


class FakeSAM:
    """
    A SAM stand-in that goes wrong the way the reported failure did.

    Given a frame's prompt boxes it segments a rectangle inside each one, so its answer depends
    on the frame and on nothing else. But it answers that way only while nothing of the previous
    frame is still held: a caller that leaves `predictor.results` in place is pinning the last
    frame's masks on the device, and this model then returns a zero-length mask set, which is
    exactly what ultralytics hands back when its own confidence filter drops everything. Every
    box in the frame falls back, and `result.masks` is not None, so nothing says why.
    """

    def __init__(self, device="mps", drop_after=None):
        self.predictor = FakePredictor(device)
        self.calls = []
        self.drop_after = drop_after

    def __call__(self, img, bboxes=None, **kwargs):
        self.calls.append(kwargs)
        height, width = img.shape[:2]
        stale = self.predictor.results is not None
        boxes = [] if stale else list(bboxes)
        if self.drop_after is not None:
            boxes = boxes[: self.drop_after]

        data = np.zeros((len(boxes), height, width), dtype=bool)
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            inset_x, inset_y = (x2 - x1) * 0.25, (y2 - y1) * 0.25
            data[i, int(y1 + inset_y) : int(y2 - inset_y), int(x1 + inset_x) : int(x2 - inset_x)] = True

        result = FakeResult(FakeMasks(data))
        # What ultralytics does: keep a reference to what it just handed back
        self.predictor.results = [result]
        return [result]


def make_frames(root, boxes_per_frame):
    """Write one 200x200 JPEG and one HBB label file per entry, and return the images directory."""
    images, labels = root / "images", root / "labels_hbb"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    for index, boxes in enumerate(boxes_per_frame):
        stem = f"frame{index:02d}"
        frame = np.zeros((200, 200, 3), np.uint8)
        cv2.imwrite(str(images / f"{stem}.jpg"), frame)
        (labels / f"{stem}.txt").write_text("".join(f"0 {xc} {yc} {w} {h}\n" for xc, yc, w, h in boxes))
    return images


FOUR_BOXES = [(50, 50, 40, 30), (150, 50, 40, 30), (50, 150, 40, 30), (150, 150, 40, 30)]


@pytest.fixture
def sam(monkeypatch):
    model = FakeSAM()
    monkeypatch.setattr(converter, "load_sam_model", lambda *_args, **_kwargs: model)
    return model


def test_a_frame_converts_the_same_alone_and_in_a_batch(tmp_path, monkeypatch, sam, metal):
    """
    The regression: a frame's labels must not depend on where it sits in the directory.

    The frame that failed produced 75 fallbacks of 76 boxes as the tenth of a run and none at
    all on its own, which is what a model reusing or tripping over the previous frame's state
    looks like from the outside.
    """
    alone = make_frames(tmp_path / "alone", [FOUR_BOXES])
    batch = make_frames(tmp_path / "batch", [FOUR_BOXES] * 5)

    assert run(monkeypatch, alone) == 0
    assert run(monkeypatch, batch) == 0

    single = (tmp_path / "alone" / "labels_obb" / "frame00.txt").read_bytes()
    for index in range(5):
        assert (tmp_path / "batch" / "labels_obb" / f"frame{index:02d}.txt").read_bytes() == single
    assert single.strip(), "the fake segmented every box, so nothing should have fallen back"


def test_nothing_falls_back_when_every_prompt_is_answered(tmp_path, monkeypatch, sam, metal, capsys):
    images = make_frames(tmp_path, [FOUR_BOXES] * 3)
    assert run(monkeypatch, images) == 0
    out = capsys.readouterr()
    assert "Wrote 12 boxes over 3 frames" in out.out
    assert "Fallback to HBB: 0 boxes (0.00%)" in out.out
    assert "Warning" not in out.err


def test_a_model_returning_fewer_masks_than_prompts_warns(tmp_path, monkeypatch, metal, capsys):
    """
    The silent case: ultralytics drops a mask whose predicted IoU is under `conf`, so the model
    answers four prompts with one mask and `result.masks` is not None. Nothing used to say so.
    """
    model = FakeSAM(drop_after=1)
    monkeypatch.setattr(converter, "load_sam_model", lambda *_args, **_kwargs: model)
    images = make_frames(tmp_path, [FOUR_BOXES])

    assert run(monkeypatch, images) == 0
    err = capsys.readouterr().err
    assert "returned 1 mask(s) for 4 prompt(s) in frame00.jpg" in err


def test_an_image_that_falls_back_entirely_is_named_on_stderr(tmp_path, monkeypatch, metal, capsys):
    model = FakeSAM(drop_after=0)
    monkeypatch.setattr(converter, "load_sam_model", lambda *_args, **_kwargs: model)
    images = make_frames(tmp_path, [FOUR_BOXES])

    assert run(monkeypatch, images) == 0
    captured = capsys.readouterr()
    assert "4 of 4 boxes in frame00.jpg fell back to their HBB (100.0%)" in captured.err
    assert "Fallback to HBB: 4 boxes (100.00%)" in captured.out
    assert "Fell back entirely: 1 frame(s), frame00.jpg" in captured.out


def test_the_fallback_warning_respects_its_threshold(tmp_path, monkeypatch, metal, capsys):
    model = FakeSAM(drop_after=3)  # one box of four falls back, a share of 0.25
    monkeypatch.setattr(converter, "load_sam_model", lambda *_args, **_kwargs: model)
    images = make_frames(tmp_path, [FOUR_BOXES])

    assert run(monkeypatch, images, "--fallback_warn_share", "0.5") == 0
    assert "fell back" not in capsys.readouterr().err

    assert run(monkeypatch, images, "--fallback_warn_share", "0.1") == 0
    assert "1 of 4 boxes in frame00.jpg fell back" in capsys.readouterr().err


def test_a_threshold_outside_the_unit_range_is_refused(tmp_path, monkeypatch, sam, metal):
    images = make_frames(tmp_path, [FOUR_BOXES])
    assert run(monkeypatch, images, "--fail_on_fallback_share", "5") == 2
    assert run(monkeypatch, images, "--fallback_warn_share", "-1") == 2


def test_the_run_summary_counts_every_fallback(tmp_path, monkeypatch, metal, capsys):
    model = FakeSAM(drop_after=2)  # two of the four boxes in each frame
    monkeypatch.setattr(converter, "load_sam_model", lambda *_args, **_kwargs: model)
    images = make_frames(tmp_path, [FOUR_BOXES] * 3)

    assert run(monkeypatch, images) == 0
    out = capsys.readouterr().out
    assert "Wrote 12 boxes over 3 frames" in out
    assert "Fallback to HBB: 6 boxes (50.00%)" in out
    assert "Fell back entirely" not in out


def test_fail_on_fallback_share_exits_non_zero_after_writing_everything(tmp_path, monkeypatch, metal):
    model = FakeSAM(drop_after=0)
    monkeypatch.setattr(converter, "load_sam_model", lambda *_args, **_kwargs: model)
    images = make_frames(tmp_path, [FOUR_BOXES])

    assert run(monkeypatch, images, "--save_provenance", "--fail_on_fallback_share", "0.1") == 1
    assert (tmp_path / "labels_obb" / "frame00.txt").read_text().strip(), "the labels are written anyway"
    assert (tmp_path / "PROVENANCE_obb.txt").exists(), "the record is written anyway"


def test_a_run_under_the_failure_threshold_still_exits_zero(tmp_path, monkeypatch, metal):
    model = FakeSAM(drop_after=3)  # a share of 0.25
    monkeypatch.setattr(converter, "load_sam_model", lambda *_args, **_kwargs: model)
    images = make_frames(tmp_path, [FOUR_BOXES])
    assert run(monkeypatch, images, "--fail_on_fallback_share", "0.5") == 0


def test_the_provenance_records_the_fallback_count(tmp_path, monkeypatch, metal):
    model = FakeSAM(drop_after=2)
    monkeypatch.setattr(converter, "load_sam_model", lambda *_args, **_kwargs: model)
    images = make_frames(tmp_path, [FOUR_BOXES] * 2)

    run(monkeypatch, images, "--save_provenance")
    text = (tmp_path / "PROVENANCE_obb.txt").read_text()
    assert "OBB boxes      : 8" in text
    assert "HBB fallbacks  : 4 (50.00%)" in text


def test_the_reporting_flags_are_not_in_the_reproducing_command(tmp_path, monkeypatch, sam, metal):
    """They change what a run says, not what it writes, so replaying without them is the same run."""
    images = make_frames(tmp_path, [FOUR_BOXES])
    run(monkeypatch, images, "--save_provenance", "--fallback_warn_share", "0.9")
    text = (tmp_path / "PROVENANCE_obb.txt").read_text()
    assert "--fallback_warn_share" not in text
    assert "--fail_on_fallback_share" not in text


def test_an_empty_label_file_is_not_a_fallback(tmp_path, monkeypatch, sam, metal, capsys):
    """A frame with nothing to convert never prompts SAM, and must not divide by zero either."""
    images = make_frames(tmp_path, [[], FOUR_BOXES])
    assert run(monkeypatch, images) == 0
    out = capsys.readouterr().out
    assert "Wrote 4 boxes over 2 frames" in out
    assert "Fallback to HBB: 0 boxes (0.00%)" in out


def test_the_device_cache_is_trimmed_once_per_frame_on_metal(tmp_path, monkeypatch, sam, metal):
    images = make_frames(tmp_path, [FOUR_BOXES] * 3)
    assert run(monkeypatch, images) == 0
    assert len(metal) == 3


def test_a_cuda_run_is_left_alone(tmp_path, monkeypatch, metal):
    """A per-frame empty_cache() on CUDA synchronizes the stream, moving every measured time."""
    model = FakeSAM(device="cuda")
    monkeypatch.setattr(converter, "load_sam_model", lambda *_args, **_kwargs: model)
    images = make_frames(tmp_path, [FOUR_BOXES] * 3)
    assert run(monkeypatch, images) == 0
    assert metal == []
