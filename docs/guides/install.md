# Installation

HBB2OBB needs Python 3.9 to 3.13 and installs from [PyPI](https://pypi.org/project/hbb2obb/). Install it into a virtual environment:

=== "venv"

    ```bash
    python3.11 -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    pip install hbb2obb
    ```

=== "uv"

    ```bash
    uv venv --python 3.11
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    uv pip install hbb2obb
    ```

    See the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/) to get `uv` itself.

=== "conda"

    ```bash
    conda create -n hbb2obb python=3.11 -y
    conda activate hbb2obb
    pip install hbb2obb
    ```

    See the [Miniconda installation guide](https://www.anaconda.com/docs/getting-started/miniconda/install) to get `conda` itself.

Check the install:

```bash
hbb2obb --version
```

This installs seven commands, listed in the [CLI reference](../reference/cli/index.md), and the `hbb2obb` Python package described in the [API reference](../reference/api/index.md).

## GPU

HBB2OBB runs SAM and the detectors through [Ultralytics](https://docs.ultralytics.com/) and PyTorch. Without `--device`, Ultralytics uses the first CUDA GPU when PyTorch sees one and the CPU otherwise. On Apple silicon pass `--device mps`.

If `pip` installed a CPU-only PyTorch on a machine with an NVIDIA GPU, reinstall `torch` and `torchvision` with the command the [PyTorch selector](https://pytorch.org/get-started/locally/) gives for your CUDA version, then check:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## From source

```bash
git clone https://github.com/rfonod/hbb2obb.git
cd hbb2obb && pip install -e '.[dev]'
```

`-e` makes local changes take effect without reinstalling. The quotes keep zsh from expanding the brackets. The extras are:

| Extra | Adds | Used for |
| :--- | :--- | :--- |
| `dev` | `pytest`, `ruff` | `pytest` runs the test suite, `ruff check .` and `ruff format .` lint and format |
| `docs` | MkDocs, Material for MkDocs, mkdocstrings | `mkdocs serve` previews this site, `mkdocs build --strict` checks it |

A clone also holds the [sample data](https://github.com/rfonod/hbb2obb/tree/main/data) that the [Quick start](quickstart.md) runs on.

## Model weights

[Ultralytics](https://docs.ultralytics.com/models/sam/) downloads each SAM checkpoint on first use. HBB2OBB reads and writes all checkpoints, SAM and detector alike, in one models directory, resolved in this order:

1. `--models_dir`, on `hbb2obb`, `hbb2obb-detect` and `hbb2obb-optimize`;
2. the `HBB2OBB_MODELS_DIR` environment variable;
3. `models/` in the **current working directory**.

Without either setting, every directory you run from gets its own `models/`. To keep one copy, set the variable once, for example in your shell profile:

```bash
export HBB2OBB_MODELS_DIR=~/.cache/hbb2obb
```

The SAM checkpoints range from about 40 MB (`mobile_sam`) to 1.2 GB (`sam_l`).

### SAM 3

SAM 3 weights (`sam3.pt`) are **not** downloaded automatically. To use `--sam_models sam3`:

1. Request access on the [SAM 3 model page on Hugging Face](https://huggingface.co/facebook/sam3).
2. Once approved, download [`sam3.pt`](https://huggingface.co/facebook/sam3/resolve/main/sam3.pt?download=true).
3. Place it in the models directory above.

See the [Ultralytics SAM 3 documentation](https://docs.ultralytics.com/models/sam-3/) for more.

## Environment variables

| Variable | Effect |
| :--- | :--- |
| `HBB2OBB_MODELS_DIR` | the models directory, when `--models_dir` is not given |
| `HBB2OBB_DISABLE_UPDATE_CHECK=1` | turns off the update check below |

## Update check

Every command checks PyPI once a day, in the background, for a newer release and prints a one-line notice if there is one. The check never blocks or fails a run, and is silent when offline.

To upgrade:

```bash
pip install --upgrade hbb2obb
```
