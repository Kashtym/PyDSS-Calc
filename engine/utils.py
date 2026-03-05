"""Small reusable helpers for normalization and parsing."""

from __future__ import annotations

import os
import re
from typing import Any

import numpy as np


def safe_token(name: str) -> str:
    """Normalize arbitrary text into OpenDSS-safe token."""
    token = re.sub(r"[^A-Za-z0-9_]", "_", str(name).strip())
    token = re.sub(r"_+", "_", token).strip("_")
    return token or "N"


def bus_tokens(name: str) -> set[str]:
    """Return strict and relaxed tokens for bus matching."""
    token = safe_token(name).lower()
    tokens = {token}
    tokens.add(re.sub(r"_\d+$", "", token))
    return {t for t in tokens if t}


def to_abs_curve_path(curve_value: str) -> str:
    """Normalize relative curve path from ODS into project path."""
    curve = str(curve_value).replace("\\", os.sep).strip()
    if not curve or curve.lower() in {"nan", "none", ""}:
        return ""
    return os.path.join("data", curve) if not os.path.isabs(curve) else curve


def read_numeric(
    row: dict[str, Any],
    candidates: list[str],
    default: float | None = None,
    fallback_to_any: bool = True,
) -> float:
    """Read first finite numeric value from candidate keys or row values."""
    for key in candidates:
        if key in row:
            try:
                return float(row[key])
            except (TypeError, ValueError):
                continue
    if fallback_to_any:
        for value in row.values():
            try:
                num = float(value)
                if np.isfinite(num):
                    return num
            except (TypeError, ValueError):
                continue
    if default is not None:
        return float(default)
    raise ValueError(f"Cannot parse numeric value from row for keys: {candidates}")
