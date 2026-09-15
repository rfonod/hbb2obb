# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
Where checkpoints live, for every command that loads a model.

Detector weights and SAM/FastSAM checkpoints share one directory. It is ``models/`` relative to
the working directory unless ``--models_dir`` or the ``HBB2OBB_MODELS_DIR`` environment variable
says otherwise, in that order. The environment variable exists so that a user who runs hbb2obb
from many working directories, or from another project's scripts, downloads each checkpoint once
rather than once per directory.

Kept free of heavy imports, since ``detector`` is imported for its registry by ``--help``.
"""

import os
from pathlib import Path
from typing import Optional, Union

MODELS_DIR_ENV = "HBB2OBB_MODELS_DIR"
DEFAULT_MODELS_DIR = Path("models")


def resolve_models_dir(models_dir: Optional[Union[str, Path]] = None) -> Path:
    """The directory checkpoints are read from and downloaded to: the argument, the environment, or ``models``."""
    if models_dir:
        return Path(models_dir).expanduser()
    from_environment = os.environ.get(MODELS_DIR_ENV)
    if from_environment:
        return Path(from_environment).expanduser()
    return DEFAULT_MODELS_DIR
