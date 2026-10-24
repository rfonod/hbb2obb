# Provenance

A provenance file records what a run did, so a released annotation set can be regenerated and checked rather than taken on trust. Three commands write one:

| Command | Flag | File | Where |
| :--- | :--- | :--- | :--- |
| `hbb2obb` | `--save_provenance` | `PROVENANCE_obb.txt` | one level above the OBB directory: `--obb_dir train/labels` leaves `train/PROVENANCE_obb.txt` |
| `hbb2obb-detect` | `--save_provenance` | `PROVENANCE_hbb.txt` | one level above the HBB directory |
| `hbb2obb-optimize` | always | `PROVENANCE.txt` | in the output folder |

## What gets recorded

A conversion record, abridged:

```text
HBB2OBB conversion provenance
==============================================================================

Written        : 2026-10-24 14:33:54 UTC
hbb2obb        : 1.11.2
source sha256  : 3905d6cdaf8674419ab5263ac199d5facd4518428519464f29065a5867261356  (16 files)
git commit     : ed34ed7e1a00b3626307494708457867bf592780
git describe   : v1.11.2

Command that reproduces these annotations
------------------------------------------------------------------------------
hbb2obb data/images --hbb_dir data/labels_hbb --obb_dir data/labels_obb --sam_models sam_b --imgsz 1280 --scale_factors 0.05 --opening_kernel_percentage 0.15 --fragment_ratio 0.1 --save_confidence

SAM checkpoints
------------------------------------------------------------------------------
sam_b
    path   : models/sam_b.pt
    size   : 375042383 bytes
    sha256 : ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912

Inputs and outputs
------------------------------------------------------------------------------
HBB labels     : data/labels_hbb
HBB sha256     : 90611ed4cf0b8e4b5ad0819547c21ba8240189a77e254bf5da02750d0d94807d  (3 files)
OBB boxes      : 201
HBB fallbacks  : 0 (0.00%)
```

Every record holds:

- **The command** that reproduces the files, with every option that changes them, followed by the settings the run used.
- **The code**: the hbb2obb version, a SHA-256 over the package source, and the git commit when there is a checkout (see below).
- **The checkpoints**: the path, size and SHA-256 of every model used.
- **The environment**: Python, the platform, and the versions of Ultralytics, PyTorch, torchvision, OpenCV, NumPy, Shapely, matplotlib and PyYAML.

And, depending on the command:

| Record | Also holds |
| :--- | :--- |
| conversion | the SHA-256 of the HBB input set, the number of OBBs written and how many fell back to their HBB |
| detection | the SHA-256 of the HBB set written; for a detector on the Hugging Face Hub, what the Hub declares about its repository at that moment: licence, DOI and arXiv references, and the revision read |
| benchmark | the config, the SHA-256 of the HBB and ground-truth sets the numbers were measured against, and, after a partial `--resume`, which runs this invocation measured |

A Hub field the repository does not declare, or a Hub that cannot be reached, is recorded as such with a link to the model page, and the run still succeeds. Weights from anywhere else are noted as needing credit by hand.

A checkpoint that cannot be found is marked `MISSING` in the record, and the command warns on stderr.

## How the code is pinned

The **source SHA-256** is a digest of the hbb2obb package's own `.py` files, the bytes that ran. It identifies the code whatever happened to the git history since, and works for a PyPI install with no checkout at all: install the recorded version and compare.

The **git commit** and `git describe` are recorded when hbb2obb runs from a checkout that tracks its source. The record states whether the package directory was modified relative to that commit; when it was, only the digest identifies the code.

Use the commit to find the change, and the digest to prove you have the same code.

## When a benchmark writes one

Only an invocation that converts something writes `PROVENANCE.txt`. `--refresh`, and a `--resume` that skipped every run, leave the record of the runs that measured the numbers in place.

## Checking a release

1. Install the recorded version, `pip install hbb2obb==<version>`, and compare its source SHA-256 with the recorded one.
2. Hash your checkpoints, e.g. `shasum -a 256 models/sam_b.pt`, and compare.
3. Re-run the recorded command and compare the labels. SAM is not bit-reproducible across PyTorch builds and hardware, so expect a box or two to move by a pixel.
