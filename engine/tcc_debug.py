from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from engine.app.run_project import (
    _build_breaker_tcc_defaults,
    _build_fuse_tcc_defaults,
    _clean_protection_overrides,
)
from engine.protection_core import UniversalTCC
from engine.schema_normalizer import init_equipment_manager
from engine.visualizer import plot_tcc_curves


def _crossings_for_time(x: list[float], y: list[float], target_t: float) -> list[float]:
    if target_t <= 0:
        return []
    points: list[float] = []
    for i in range(len(x) - 1):
        x1, x2 = float(x[i]), float(x[i + 1])
        y1, y2 = float(y[i]), float(y[i + 1])
        if not (np.isfinite(y1) and np.isfinite(y2)):
            continue
        if x1 <= 0 or x2 <= 0 or y1 <= 0 or y2 <= 0:
            continue
        lo, hi = (y1, y2) if y1 <= y2 else (y2, y1)
        if not (lo <= target_t <= hi):
            continue
        if abs(y2 - y1) < 1e-12:
            points.append(min(x1, x2))
            continue
        lx1, lx2 = np.log10(x1), np.log10(x2)
        ly1, ly2 = np.log10(y1), np.log10(y2)
        lt = np.log10(target_t)
        frac = (lt - ly1) / (ly2 - ly1)
        lx = lx1 + frac * (lx2 - lx1)
        points.append(float(10**lx))
    uniq = sorted({round(v, 6) for v in points})
    if len(uniq) <= 2:
        return uniq
    return [uniq[0], uniq[-1]]


def run_tcc_debug(config_path: str) -> str:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    project_name = str(cfg.get("project_name", "tcc_debug"))
    devices_cfg = cfg.get("devices", [])
    if not isinstance(devices_cfg, list) or not devices_cfg:
        raise ValueError("'devices' list is required")

    current_lines = [float(v) for v in cfg.get("current_lines_a", [])]
    time_lines = [float(v) for v in cfg.get("time_lines_s", [])]
    i_min = float(cfg.get("plot_i_min_a", 0.1))
    i_max = float(cfg.get("plot_i_max_a", 100000.0))

    manager = init_equipment_manager()
    devices_for_plot: list[dict[str, Any]] = []
    report_rows: list[str] = []

    for item in devices_cfg:
        if not isinstance(item, dict):
            continue
        p = _clean_protection_overrides(item.get("protection_settings", {}) if isinstance(item.get("protection_settings"), dict) else {})
        dtype = str(p.get("type", item.get("type", ""))).strip().lower()
        label = str(item.get("name", p.get("name", item.get("breaker_id", item.get("fuse_id", "Device")))))

        if dtype == "breaker":
            breaker_id = str(item.get("breaker_id", p.get("breaker_id", ""))).strip()
            br = manager.get_raw_breaker(breaker_id)
            trip_id = str(br.get("TripUnit_ID", "")).strip()
            trip = manager.get_raw_trip_unit(trip_id) if trip_id else {}
            defaults = _build_breaker_tcc_defaults(br, trip)
            in_a = float(br.get("In", defaults.get("In")))
            tcc = UniversalTCC.from_config(defaults, p, nominal_current_in=in_a)
        elif dtype == "fuse":
            fuse_id = str(item.get("fuse_id", p.get("fuse_id", ""))).strip()
            fu = manager.get_raw_fuse(fuse_id)
            defaults = _build_fuse_tcc_defaults(fu)
            in_a = float(fu.get("In", defaults.get("In")))
            tcc = UniversalTCC.from_config(defaults, p, nominal_current_in=in_a)
        elif dtype == "relay":
            l_stage = p.get("L_stage") if isinstance(p.get("L_stage"), dict) else {}
            in_a = float(l_stage.get("I1_A", p.get("In", 1.0)))
            defaults = {"type": "Relay", "In": in_a, "L_stage": {"active": False}, "S_stage": {"active": False}, "I_stage": {"active": False}}
            tcc = UniversalTCC.from_config(defaults, p, nominal_current_in=in_a)
        else:
            raise ValueError(f"Unsupported device type: {dtype}")

        devices_for_plot.append({"label": label, "In": in_a, "tcc": tcc})
        x_min, y_min = tcc.get_plot_points(mode="min", i_min=i_min, i_max=i_max)
        x_max, y_max = tcc.get_plot_points(mode="max", i_min=i_min, i_max=i_max)

        report_rows.append(f"\nDevice: {label} ({dtype})")
        for i_line in current_lines:
            t_min = tcc.calculate_time(i_line, mode="min")
            t_max = tcc.calculate_time(i_line, mode="max")
            report_rows.append(f"I={i_line:.3f} A -> t_min={t_min:.6f} s, t_max={t_max:.6f} s")

        for t_line in time_lines:
            i_cross_min = _crossings_for_time(x_min, y_min, t_line)
            i_cross_max = _crossings_for_time(x_max, y_max, t_line)
            report_rows.append(
                f"t={t_line:.6f} s -> I_cross_min={i_cross_min if i_cross_min else '-'}, I_cross_max={i_cross_max if i_cross_max else '-'}"
            )

    fault_points = [{"label": f"I={v:.0f}A", "current": float(v)} for v in current_lines]
    out_base = Path(config_path).parent / project_name
    plot_tcc_curves(devices_for_plot, fault_points, str(out_base))

    report_path = Path(config_path).parent / f"{project_name}_report.txt"
    lines = [
        f"Project: {project_name}",
        "TCC Debug Report",
        "================",
        f"Current lines: {current_lines}",
        f"Time lines: {time_lines}",
    ]
    lines.extend(report_rows)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(report_path)
