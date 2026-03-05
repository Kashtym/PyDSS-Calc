"""Helpers for resilient equipment ID matching."""

from __future__ import annotations

import re


def _id_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _id_key_loose(value: str) -> str:
    base = str(value).lower().replace("x", "").replace("х", "")
    return re.sub(r"[^a-z0-9]", "", base)


def resolve_id(requested_id: str, available_ids: list[str], kind: str) -> str:
    """Resolve ID with strict, normalized and loose matching."""
    requested = str(requested_id).strip()
    if requested in available_ids:
        return requested

    req_key = _id_key(requested)
    matches = [item for item in available_ids if _id_key(item) == req_key]
    if len(matches) == 1:
        return matches[0]

    req_key_loose = _id_key_loose(requested)
    loose_matches = [item for item in available_ids if _id_key_loose(item) == req_key_loose]
    if len(loose_matches) == 1:
        return loose_matches[0]

    raise KeyError(f"ID '{requested_id}' not found in {kind}. Available examples: {available_ids[:8]}")
