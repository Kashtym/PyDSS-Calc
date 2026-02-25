from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _safe_slug(name: str) -> str:
    clean = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(name))
    clean = "_".join(part for part in clean.split("_") if part)
    return clean or "tcc_plot"


def _interp_loglog(x_src: np.ndarray, y_src: np.ndarray, x_dst: np.ndarray) -> np.ndarray:
    x = np.asarray(x_src, dtype=float)
    y = np.asarray(y_src, dtype=float)
    xd = np.asarray(x_dst, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return np.full_like(xd, np.nan, dtype=float)

    order = np.argsort(x)
    lx = np.log10(x[order])
    ly = np.log10(y[order])
    lxd = np.log10(xd)
    lyd = np.interp(lxd, lx, ly, left=ly[0], right=ly[-1])
    return np.power(10.0, lyd)


def plot_tcc_curves(
    devices_data: list[dict[str, Any]],
    fault_currents: list[dict[str, float | str]] | None = None,
    project_name: str = "TCC Plot",
) -> str:
    """Plot min/max TCC envelopes for multiple devices on one log-log chart.

    Args:
        devices_data: Items with keys: ``label``, ``In`` and ``tcc`` (UniversalTCC instance).
        fault_currents: Optional items with keys: ``label``, ``current``.
        project_name: Plot/report name. If it contains a path, parent is used as output directory.

    Returns:
        Absolute path to generated PNG file.
    """

    fig, ax = plt.subplots(figsize=(11, 7))

    palette = plt.get_cmap("tab10")
    rendered: list[dict[str, Any]] = []

    for idx, device in enumerate(devices_data):
        label = str(device.get("label", f"Device {idx + 1}"))
        tcc = device.get("tcc")
        if tcc is None:
            continue

        x_min, y_min = tcc.get_plot_points(mode="min", i_min=1.0, i_max=10000.0)
        x_max, y_max = tcc.get_plot_points(mode="max", i_min=1.0, i_max=10000.0)

        x_min_arr = np.asarray(x_min, dtype=float)
        y_min_arr = np.asarray(y_min, dtype=float)
        x_max_arr = np.asarray(x_max, dtype=float)
        y_max_arr = np.asarray(y_max, dtype=float)

        x_fill = np.logspace(np.log10(1.0), np.log10(10000.0), 700)
        y_min_fill = _interp_loglog(x_min_arr, y_min_arr, x_fill)
        y_max_fill = _interp_loglog(x_max_arr, y_max_arr, x_fill)

        color = palette(idx % 10)
        ax.fill_between(x_fill, y_min_fill, y_max_fill, color=color, alpha=0.3)
        ax.step(x_min_arr, y_min_arr, where="post", color=color, linewidth=1.6, label=f"{label} min")
        ax.step(x_max_arr, y_max_arr, where="post", color=color, linewidth=1.6, linestyle="--", label=f"{label} max")

        rendered.append(
            {
                "label": label,
                "tcc": tcc,
                "x_max": x_max_arr,
                "y_max": y_max_arr,
                "color": color,
            }
        )

    for fault in fault_currents or []:
        current = float(fault.get("current", 0.0))
        if current <= 0:
            continue
        flabel = str(fault.get("label", "Fault"))
        ax.axvline(current, color="red", linestyle="--", linewidth=1.1, alpha=0.9)

        for item in rendered:
            trip_time = float(item["tcc"].calculate_time(current, mode="max"))
            if not np.isfinite(trip_time):
                continue
            ax.plot(current, trip_time, marker="o", markersize=4, color=item["color"])
            ax.annotate(
                f"{flabel}: {current:.0f}A / {trip_time:.3g}s",
                xy=(current, trip_time),
                xytext=(6, 4),
                textcoords="offset points",
                fontsize=8,
                color=item["color"],
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, 10000)
    ax.set_ylim(0.001, 1000)
    ax.set_xlabel("Current (A)")
    ax.set_ylabel("Trip Time (s)")
    ax.set_title(str(project_name))
    ax.grid(True, which="both", ls="-", alpha=0.5)
    ax.legend(loc="best", fontsize=8)

    project_path = Path(project_name)
    if project_path.parent != Path("."):
        out_dir = project_path.parent
        out_name = f"{_safe_slug(project_path.stem)}_tcc.png"
    else:
        out_dir = Path.cwd()
        out_name = f"{_safe_slug(project_name)}_tcc.png"

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = (out_dir / out_name).resolve()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    return str(out_path)
