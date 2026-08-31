# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
Provenance records for annotations and benchmarks.

A ``PROVENANCE.txt`` says exactly how a set of annotations, or a set of numbers, was produced:
the command that reproduces it, the versions of hbb2obb and everything it depends on, and the
SHA-256 of every checkpoint that ran. The conversion is deterministic given the same inputs,
the same checkpoints and the same library versions, so this file is what turns "these are the
annotations" into "these are the annotations and here is how to get them again".

Checkpoint hashes matter more than they look. Ultralytics resolves a bare model name by
downloading it on first use, and the file behind a given name can change between asset
releases, so recording the name alone does not pin the model that actually ran.

The same argument applies to the code. A commit hash is a *history* identifier, and history
is mutable: a branch can be squashed, rebased or deleted long after a run, and a dirty tree
never had a commit describing it in the first place. So the code is pinned three ways here:
the released version (``pip install hbb2obb==X.Y.Z``), the commit and ``git describe`` when
there is a checkout, and a SHA-256 over the package source, which is computed from the bytes
that ran and therefore survives everything done to the history afterwards.

The settings recorded here come from the run that happened, not from arguments repeated by
hand, which is why this is a module the commands call rather than a script run afterwards:
``hbb2obb --save_provenance``, ``hbb2obb-detect --save_provenance``, and every
``hbb2obb-optimize`` sweep.
"""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from hbb2obb.__version__ import __version__

DEPENDENCIES = ("ultralytics", "torch", "torchvision", "opencv-python", "numpy", "shapely", "matplotlib", "PyYAML")
IMAGE_PATTERNS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.JPEG", "*.PNG")
RULE = "-" * 78
PACKAGE_ROOT = Path(__file__).resolve().parent


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    """SHA-256 of a file, read in chunks so a 1 GB checkpoint does not land in memory."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def sha256_of_set(directory: Optional[Path], pattern: str = "*.txt") -> str:
    """
    One hash over a whole annotation set: the sorted file names and their contents.

    A benchmark number means nothing without the labels it was measured against, and those
    labels are a directory rather than a file. Hashing the names as well as the contents means
    a renamed or removed frame changes the hash, not just an edited box.
    """
    if directory is None or not directory.is_dir():
        return "directory not found"
    h = hashlib.sha256()
    files = sorted(directory.glob(pattern), key=lambda p: p.name)
    if not files:
        return "no files matched"
    for path in files:
        h.update(path.name.encode("utf-8"))
        h.update(path.read_bytes())
    return f"{h.hexdigest()}  ({len(files)} files)"


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def source_digest(package_root: Path = PACKAGE_ROOT) -> str:
    """
    One SHA-256 over the package's own source files, names included.

    This is the code identifier that survives what a commit hash does not. A branch can be
    squashed, rebased or deleted after the run it describes, and an uncommitted tree has no
    commit at all, but the bytes that ran are still the bytes that ran. Anyone holding a copy
    of hbb2obb can recompute this and see whether they have the same code.
    """
    files = sorted(package_root.rglob("*.py"), key=lambda p: str(p.relative_to(package_root)))
    if not files:
        return "no source files found"
    h = hashlib.sha256()
    for path in files:
        h.update(str(path.relative_to(package_root)).encode("utf-8"))
        h.update(path.read_bytes())
    return f"{h.hexdigest()}  ({len(files)} files)"


def install_source(package_root: Path = PACKAGE_ROOT) -> str:
    """Whether this ran from a working checkout or from an installed distribution."""
    parent = package_root.parent
    if (parent / ".git").exists():
        return f"source checkout at {parent}"
    if "site-packages" in package_root.parts:
        return f"installed distribution at {package_root}"
    return str(package_root)


def git_state(repo: Optional[Path] = None, package_root: Path = PACKAGE_ROOT) -> dict:
    """
    Commit, ``git describe``, and whether the package source differs from that commit.

    A wheel installed from PyPI has no checkout, and that is not an error.

    ``dirty`` counts modified paths **under the package directory only**, not across the whole
    repository. The question this line answers is "can a reader check out this commit and get
    the code that ran", and output written by the run itself cannot change that answer: a sweep
    writing its results into the repository would otherwise mark its own record as untrustworthy
    while the code behind it sat clean and committed.
    """
    repo = Path(repo) if repo is not None else package_root.parent

    def git(*args: str) -> Optional[str]:
        try:
            done = subprocess.run(
                ["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=10, check=False
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return done.stdout.strip() if done.returncode == 0 else None

    commit = git("rev-parse", "HEAD")
    if commit is None:
        return {"commit": None, "describe": None, "dirty": None}
    status = git("status", "--porcelain", "--", str(package_root))
    dirty = None if status is None else len([line for line in status.splitlines() if line.strip()])
    # No --dirty here: that flag is repository-wide and would contradict the package-scoped
    # answer above every time a sweep wrote its results back into the repository.
    return {"commit": commit, "describe": git("describe", "--tags", "--always"), "dirty": dirty}


def resolve_model(name: str, models_dir: Path) -> Path:
    """Mirror converter.load_sam_model: <models_dir>/<name>.pt unless a suffix is given."""
    return models_dir / (name if name.endswith(".pt") else f"{name}.pt")


def count(directory: Optional[Path], pattern: str) -> str:
    if directory is None or not directory.is_dir():
        return "directory not found"
    return str(len(list(directory.glob(pattern))))


def count_images(directory: Optional[Path]) -> Optional[int]:
    if directory is None or not directory.is_dir():
        return None
    return sum(len(list(directory.glob(p))) for p in IMAGE_PATTERNS)


def checkpoint_section(
    names: Sequence[str], models_dir: Path, title: str = "SAM checkpoints"
) -> Tuple[List[str], bool]:
    """List each checkpoint with its size and hash. Returns the lines and whether any was missing."""
    lines = ["", title, RULE]
    missing = False
    for name in names:
        path = resolve_model(name, models_dir)
        lines.append(f"{name}")
        if path.is_file():
            lines.append(f"    path   : {path}")
            lines.append(f"    size   : {path.stat().st_size} bytes")
            lines.append(f"    sha256 : {sha256(path)}")
        else:
            missing = True
            lines.append(f"    MISSING: {path} does not exist, so no hash could be recorded")
    return lines, missing


def environment_section() -> List[str]:
    lines = [
        "",
        "Environment",
        RULE,
        f"python         : {sys.version.split()[0]}",
        f"platform       : {platform.platform()}",
        f"machine        : {platform.machine()}",
    ]
    for dep in DEPENDENCIES:
        lines.append(f"{dep:<15}: {package_version(dep)}")
    return lines


def header(title: str) -> List[str]:
    """Title plus the code state: the four lines that say which hbb2obb produced this file."""
    lines = [
        title,
        "=" * 78,
        "",
        f"Written        : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"hbb2obb        : {__version__}",
        f"source sha256  : {source_digest()}",
        f"installed from : {install_source()}",
    ]
    git = git_state()
    if git["commit"] is None:
        lines.append("git commit     : not a git checkout")
        return lines
    lines.append(f"git commit     : {git['commit']}")
    if git["describe"]:
        lines.append(f"git describe   : {git['describe']}")
    if git["dirty"]:
        lines.append(f"source state   : MODIFIED, {git['dirty']} path(s) under the package differ")
        lines.append("                 from that commit, so checking it out does not give the code")
        lines.append("                 that ran; only the source sha256 identifies it")
    elif git["dirty"] == 0:
        lines.append("source state   : matches that commit, which can be checked out directly")
    return lines


def _write(out: Path, lines: Iterable[str], missing: bool) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")
    if missing:
        print("Warning: one or more checkpoints were not found; their hashes are absent.", file=sys.stderr)
        return 1
    return 0


REPRODUCIBILITY = [
    "",
    "Reproducibility",
    RULE,
    "Code. Install the version this ran from with",
    f"    pip install hbb2obb=={__version__}",
    "and compare its source sha256 with the one above. That digest is computed from the",
    "package source, so it still matches after a branch has been squashed, rebased or",
    "deleted, which a commit hash does not; use the commit to find the change, the digest",
    "to prove you have the same code.",
    "",
    "Checkpoints. Verify each one against the SHA-256 above before concluding that a",
    "mismatch is a bug in the conversion: ultralytics downloads by name, and the file",
    "behind a name can change between asset releases.",
    "",
    "Determinism. The conversion is deterministic given the same images, the same HBB",
    "inputs, the same checkpoints and the same library versions. SAM inference is not",
    "bit-reproducible across torch builds and hardware, so expect a box or two to move by",
    "a pixel on a different machine.",
]


def write_conversion_provenance(
    out: Path,
    img_source: Optional[Path],
    hbb_dir: Optional[Path],
    obb_dir: Optional[Path],
    sam_models: Sequence[str],
    imgsz: int,
    scale_factors: Sequence[float],
    opening_kernel_percentage: float,
    confidence_source: str = "conversion",
    model_kwargs: Optional[str] = None,
    models_dir: Path = Path("models"),
    notes: Sequence[str] = (),
) -> int:
    """Record the settings an `hbb2obb` conversion actually ran with."""
    command = ["hbb2obb", str(img_source) if img_source else "<img_source>"]
    if hbb_dir:
        command += ["--hbb_dir", str(hbb_dir)]
    if obb_dir:
        command += ["--obb_dir", str(obb_dir)]
    command += ["--sam_models", *sam_models]
    command += ["--imgsz", str(imgsz)]
    command += ["--scale_factors", *[str(s) for s in scale_factors]]
    command += ["--opening_kernel_percentage", str(opening_kernel_percentage)]
    if confidence_source != "conversion":
        command += ["--confidence_source", confidence_source]
    if model_kwargs:
        command += ["--model_kwargs", model_kwargs]

    lines = header("HBB2OBB conversion provenance")
    lines += [
        "",
        "Command that reproduces these annotations",
        RULE,
        " ".join(command),
        "",
        "Conversion settings",
        RULE,
        f"SAM ensemble             : {' '.join(sam_models)} ({len(sam_models)} model(s))",
        f"Inference image size     : {imgsz}",
        f"Scale factor(s)          : {' '.join(str(s) for s in scale_factors)}",
        f"Opening kernel           : {opening_kernel_percentage}",
        f"Confidence source        : {confidence_source}",
    ]
    if model_kwargs:
        lines.append(f"Extra model kwargs       : {model_kwargs}")

    if len(sam_models) % 2 == 0:
        lines += [
            "",
            "Note: majority voting needs floor(M/2)+1 of M models to agree, so this",
            "even-sized ensemble is stricter than the next larger odd one.",
        ]

    checkpoints, missing = checkpoint_section(sam_models, models_dir)
    lines += checkpoints
    lines += environment_section()

    lines += ["", "Inputs and outputs", RULE, f"images         : {img_source if img_source else 'not recorded'}"]
    n_images = count_images(img_source if img_source and img_source.is_dir() else None)
    if n_images is not None:
        lines.append(f"image count    : {n_images}")
    lines.append(f"HBB labels     : {hbb_dir if hbb_dir else 'not recorded'}")
    if hbb_dir:
        lines.append(f"HBB sha256     : {sha256_of_set(hbb_dir)}")
    lines.append(f"OBB labels     : {obb_dir if obb_dir else 'not recorded'}")
    if obb_dir:
        lines.append(f"OBB count      : {count(obb_dir, '*.txt')} .txt files")

    if notes:
        lines += ["", "Notes", RULE] + list(notes)
    lines += REPRODUCIBILITY
    return _write(out, lines, missing)


def write_detection_provenance(
    out: Path,
    img_source: Optional[Path],
    hbb_dir: Optional[Path],
    model: str,
    weights: Optional[Path],
    imgsz: int,
    conf: float,
    iou: float,
    classes: Optional[Sequence[int]] = None,
    merged_with: Optional[Path] = None,
    model_kwargs: Optional[str] = None,
    notes: Sequence[str] = (),
) -> int:
    """Record the detector that drew a set of horizontal boxes."""
    command = ["hbb2obb-detect", str(img_source) if img_source else "<img_source>"]
    if hbb_dir:
        command += ["--hbb_dir", str(hbb_dir)]
    command += ["--model", model, "--imgsz", str(imgsz), "--conf", str(conf), "--iou", str(iou)]
    if classes:
        command += ["--classes", *[str(c) for c in classes]]
    if merged_with:
        command += ["--merge_with", str(merged_with)]
    if model_kwargs:
        command += ["--model_kwargs", model_kwargs]

    lines = header("HBB2OBB detection provenance")
    lines += [
        "",
        "Command that reproduces these annotations",
        RULE,
        " ".join(command),
        "",
        "Detection settings",
        RULE,
        f"Model                    : {model}",
        f"Inference image size     : {imgsz}",
        f"Confidence threshold     : {conf}",
        f"NMS IoU threshold        : {iou}",
        f"Classes kept             : {' '.join(str(c) for c in classes) if classes else 'all'}",
    ]
    if model_kwargs:
        lines.append(f"Extra model kwargs       : {model_kwargs}")
    if merged_with:
        lines += [
            f"Merged with              : {merged_with}",
            "  (the hand-drawn geometry was kept untouched; only confidences were attached)",
        ]

    missing = False
    lines += ["", "Detector checkpoint", RULE, model]
    if weights is not None and Path(weights).is_file():
        weights = Path(weights)
        lines.append(f"    path   : {weights}")
        lines.append(f"    size   : {weights.stat().st_size} bytes")
        lines.append(f"    sha256 : {sha256(weights)}")
    else:
        missing = True
        lines.append(f"    MISSING: {weights} does not exist, so no hash could be recorded")

    lines += environment_section()
    lines += ["", "Inputs and outputs", RULE, f"images         : {img_source if img_source else 'not recorded'}"]
    n_images = count_images(img_source if img_source and img_source.is_dir() else None)
    if n_images is not None:
        lines.append(f"image count    : {n_images}")
    lines.append(f"HBB labels     : {hbb_dir if hbb_dir else 'not recorded'}")
    if hbb_dir:
        lines.append(f"HBB sha256     : {sha256_of_set(hbb_dir)}")

    if notes:
        lines += ["", "Notes", RULE] + list(notes)
    lines += REPRODUCIBILITY
    return _write(out, lines, missing)


def write_benchmark_provenance(
    out: Path,
    command: str,
    config_text: Optional[str],
    config_path: Optional[Path],
    runs: Sequence[dict],
    img_source: Optional[Path],
    hbb_dir: Optional[Path],
    gt_dir: Optional[Path],
    grid_description: str,
    elapsed_seconds: float,
    models_dir: Path = Path("models"),
    notes: Sequence[str] = (),
) -> int:
    """
    Record a whole benchmark: what ran, over which labels, with which checkpoints.

    The label hashes are the part that makes the numbers meaningful later. An IoU is a
    statement about a specific set of boxes, and a set of boxes is easy to regenerate
    slightly differently without noticing.
    """
    lines = header("HBB2OBB benchmark provenance")
    lines += [
        "",
        "Command that reproduces this benchmark",
        RULE,
        command,
    ]
    if config_path is not None:
        lines += [
            "",
            "A copy of the configuration is written beside these results, so this folder",
            "re-runs on its own even when the repository is not at hand:",
            f"    hbb2obb-optimize -c {config_path}",
            "Paths inside it resolve against the working directory; adjust them if the folder",
            "has moved.",
        ]
    lines += [
        "",
        "Sweep",
        RULE,
        f"Runs                     : {len(runs)}",
        f"Grid per run             : {grid_description}",
        f"Total wall time          : {elapsed_seconds / 3600:.2f} h ({elapsed_seconds:.0f} s)",
        "",
    ]
    for run in runs:
        lines.append(f"  {run['name']:<44} {' '.join(run['sam_models'])}")

    lines += ["", "Inputs", RULE, f"images         : {img_source if img_source else 'not recorded'}"]
    n_images = count_images(img_source if img_source and img_source.is_dir() else None)
    if n_images is not None:
        lines.append(f"image count    : {n_images}")
    lines.append(f"HBB labels     : {hbb_dir if hbb_dir else 'not recorded'}")
    lines.append(f"HBB sha256     : {sha256_of_set(hbb_dir)}")
    lines.append(f"GT labels      : {gt_dir if gt_dir else 'not recorded'}")
    lines.append(f"GT sha256      : {sha256_of_set(gt_dir)}")

    used = []
    for run in runs:
        for name in run["sam_models"]:
            if name not in used:
                used.append(name)
    checkpoints, missing = checkpoint_section(used, models_dir, "SAM checkpoints used across all runs")
    lines += checkpoints
    lines += environment_section()

    if config_text is not None:
        lines += ["", "Benchmark configuration, verbatim", RULE] + config_text.rstrip("\n").splitlines()

    if notes:
        lines += ["", "Notes", RULE] + list(notes)
    lines += REPRODUCIBILITY
    return _write(out, lines, missing)
