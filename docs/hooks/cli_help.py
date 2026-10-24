"""
MkDocs hook that renders the hbb2obb command-line help into the CLI reference at build time.

A page holds a placeholder such as `<!-- cli-help: hbb2obb-eval -->`, which is replaced by the output of
`hbb2obb-eval --help`, produced by calling the console script's entry point from `pyproject.toml`, so the
reference is generated from the real parsers and cannot drift from them. COLUMNS is pinned while rendering
so argparse wraps the text identically on every machine. A console script with no placeholder anywhere
fails the build.
"""

import contextlib
import importlib
import io
import os
import re
import sys
from pathlib import Path
from unittest import mock

import tomllib
from mkdocs.exceptions import PluginError

PLACEHOLDER = re.compile(r"<!-- cli-help: (?P<command>[\w.-]+) -->")
HELP_COLUMNS = "100"
PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"

_rendered = set()


def console_scripts() -> dict:
    """
    Return the `[project.scripts]` table of pyproject.toml: command name -> 'module:function'.
    """
    with PYPROJECT.open("rb") as f:
        return tomllib.load(f)["project"]["scripts"]


def render_help(command: str) -> str:
    """
    Return the --help text of the console script `command`.
    """
    scripts = console_scripts()
    if command not in scripts:
        raise PluginError(f"'{command}' is not a console script in pyproject.toml")
    module, function = scripts[command].split(":")
    entry_point = getattr(importlib.import_module(module), function)

    buffer = io.StringIO()
    environment = {"COLUMNS": HELP_COLUMNS, "HBB2OBB_DISABLE_UPDATE_CHECK": "1"}
    with (
        mock.patch.dict(os.environ, environment),
        mock.patch.object(sys, "argv", [command, "--help"]),
        contextlib.redirect_stdout(buffer),
        contextlib.suppress(SystemExit),
    ):
        entry_point()
    text = buffer.getvalue().rstrip()
    if not text:
        raise PluginError(f"'{command} --help' produced no output")
    _rendered.add(command)
    return text


def on_pre_build(config):
    _rendered.clear()


def on_page_markdown(markdown, page, config, files):
    """
    Replace every CLI help placeholder on the page with the corresponding help text.
    """
    return PLACEHOLDER.sub(lambda m: f"```text\n{render_help(m.group('command'))}\n```", markdown)


def on_post_build(config):
    missing = sorted(set(console_scripts()) - _rendered)
    if missing:
        raise PluginError(f"console scripts with no CLI reference page: {', '.join(missing)}")
