# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
Read what the Hugging Face Hub itself records about a model repository, to credit its weights.

Detection provenance names the licence of the weights that drew a label set. hbb2obb states none of
it: the licence and references come from the Hub's model API at the moment the record is written,
and nothing is copied out of the model card's prose or stored locally. Only fields the Hub
maintains are read:

- ``tags`` ``license:<id>``, the licence the repository declares, with ``cardData.license_name``
  and ``cardData.license_link`` when that id is ``other``;
- ``tags`` ``doi:<doi>`` and ``arxiv:<id>``, present when the repository has a DOI or links a paper;
- ``sha``, the revision the metadata was read at.

A repository may declare none of the optional ones, and the Hub may be unreachable; either is
written into the provenance as such, with the model page to check, and never fails the run.
"""

import json
import urllib.request
from dataclasses import dataclass
from typing import Optional, Tuple

HF_API_URL = "https://huggingface.co/api/models/{repo}"
HF_PAGE_URL = "https://huggingface.co/{repo}"
FETCH_TIMEOUT = 10.0


@dataclass(frozen=True)
class HubRecord:
    """The Hub's metadata for one model repository; ``None`` or empty for whatever it does not declare."""

    repo: str
    revision: Optional[str] = None
    licence: Optional[str] = None
    dois: Tuple[str, ...] = ()
    arxiv: Tuple[str, ...] = ()
    error: Optional[str] = None

    @property
    def page(self) -> str:
        return HF_PAGE_URL.format(repo=self.repo)


def fetch_json(url: str, timeout: float = FETCH_TIMEOUT) -> dict:
    """Download and decode a JSON document. Kept separate so tests can take the network away."""
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def load_hub_record(repo: str) -> HubRecord:
    """The Hub's metadata for ``repo``, or a record carrying the reason it could not be read."""
    try:
        data = fetch_json(HF_API_URL.format(repo=repo))
    except Exception as exc:  # offline, private, renamed, rate-limited: the provenance is still written
        return HubRecord(repo=repo, error=f"{type(exc).__name__}: {exc}")
    if not isinstance(data, dict):
        return HubRecord(repo=repo, error="the Hub returned no model record")
    return parse_hub_record(repo, data)


def parse_hub_record(repo: str, data: dict) -> HubRecord:
    """Pull the licence, DOIs, arXiv ids and revision out of a ``/api/models/<repo>`` response."""
    tags = [tag for tag in data.get("tags") or [] if isinstance(tag, str)]
    return HubRecord(
        repo=repo,
        revision=_string(data.get("sha")),
        licence=licence_from(tags, data.get("cardData")),
        dois=tuple(_tag_values(tags, "doi")),
        arxiv=tuple(_tag_values(tags, "arxiv")),
    )


def licence_from(tags, card_data=None) -> Optional[str]:
    """The declared licence id; for ``other``, the name and link the repository gives for it."""
    declared = next(iter(_tag_values(tags, "license")), None)
    card_data = card_data if isinstance(card_data, dict) else {}
    if declared is None:
        declared = _string(card_data.get("license"))
    if declared is None:
        return None
    if declared.lower() != "other":
        return declared
    name, link = _string(card_data.get("license_name")), _string(card_data.get("license_link"))
    described = " ".join(part for part in (name, f"({link})" if link else None) if part)
    return f"other: {described}" if described else "other (no name or link declared)"


def _tag_values(tags, prefix: str):
    return [tag.split(":", 1)[1] for tag in tags if tag.lower().startswith(f"{prefix}:") and tag.split(":", 1)[1]]


def _string(value) -> Optional[str]:
    if not isinstance(value, str):
        return None
    return value.strip() or None
