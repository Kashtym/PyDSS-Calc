from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch
from math import sqrt

import numpy as np
import pandas as pd
import yaml

import traceback

from engine.db_manager import EquipmentManager
from engine.models import BatteryModel, LineModel, LoadModel, SourceModel, TransformerModel
from engine.protection_core import UniversalTCC
from engine.solver import DSSSolver
from engine.visualizer import plot_tcc_curves
from engine.report import SubstationReport


def _init_equipment_manager() -> EquipmentManager:
    """Initialize EquipmentManager with a fallback for missing breaker sheet."""
    try:
        manager = EquipmentManager()
    except ValueError as exc:
        if "Missing required sheet(s):" not in str(exc):
            raise
        sheets = pd.read_excel(EquipmentManager._default_db_path(), sheet_name=None, engine="odf", skiprows=1)
        if "CircuitBreakers" not in sheets:
            sheets["CircuitBreakers"] = pd.DataFrame({"ID": pd.Series(dtype=str)})
        if "Fuses" not in sheets:
            sheets["Fuses"] = pd.DataFrame({"ID": pd.Series(dtype=str)})
        with patch("pandas.read_excel", return_value=sheets):
            manager = EquipmentManager()

    _normalize_schema_for_known_ods_layout(manager)
    return manager


def _normalize_schema_for_known_ods_layout(manager: EquipmentManager) -> None:
    """Normalize current ODS column names if a localized header row leaked in."""
    if "ID" not in manager.cables_df.columns:
        if len(manager.cables_df.columns) >= 12:
            cable_cols = [
                "ID",
                "Name",
                "R20",
                "X1",
                "R0",
                "X0",
                "C1",
                "I_adm",
                "I_sc_1s",
                "cond_material",
                "insul_material",
                "Diameter",
            ]
        else:
            cable_cols = [
                "ID",
                "R20",
                "X1",
                "R0",
                "X0",
                "C1",
                "I_adm",
                "I_sc_1s",
                "cond_material",
                "insul_material",
                "Diameter",
            ]
        n = min(len(cable_cols), len(manager.cables_df.columns))
        manager.cables_df.columns = cable_cols[:n] + [f"extra_{i}" for i in range(len(manager.cables_df.columns) - n)]

    if "ID" not in manager.batteries_df.columns:
        battery_cols = ["ID", "Name", "Capacity", "U_nom", "Ri_cell", "I_sc_cell"]
        n = min(len(battery_cols), len(manager.batteries_df.columns))
        manager.batteries_df.columns = battery_cols[:n] + [f"extra_{i}" for i in range(len(manager.batteries_df.columns) - n)]

    if hasattr(manager, "breakers_df") and "ID" not in manager.breakers_df.columns:
        breaker_cols = [
            "ID",
            "Name",
            "Manufacturer",
            "Series",
            "In",
            "Poles",
            "P_loss_W",
            "Icu_kA",
            "curve_min",
            "curve_max",
        ]
        n = min(len(breaker_cols), len(manager.breakers_df.columns))
        manager.breakers_df.columns = breaker_cols[:n] + [f"extra_{i}" for i in range(len(manager.breakers_df.columns) - n)]

    if hasattr(manager, "fuses_df") and "ID" not in manager.fuses_df.columns:
        fuse_cols = [
            "ID",
            "Name",
            "Manufacturer",
            "Series",
            "In",
            "Poles",
            "P_loss_W",
            "Icu_kA",
            "curve",
        ]
        n = min(len(fuse_cols), len(manager.fuses_df.columns))
        manager.fuses_df.columns = fuse_cols[:n] + [f"extra_{i}" for i in range(len(manager.fuses_df.columns) - n)]

    manager.cables_df["ID"] = manager.cables_df["ID"].astype(str).str.strip()
    manager.batteries_df["ID"] = manager.batteries_df["ID"].astype(str).str.strip()
    if hasattr(manager, "breakers_df") and "ID" in manager.breakers_df.columns:
        manager.breakers_df["ID"] = manager.breakers_df["ID"].astype(str).str.strip()
    if hasattr(manager, "fuses_df") and "ID" in manager.fuses_df.columns:
        manager.fuses_df["ID"] = manager.fuses_df["ID"].astype(str).str.strip()


def _load_project(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("YAML root must be a dictionary")
    return data


def _safe_token(name: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_]", "_", str(name).strip())
    token = re.sub(r"_+", "_", token).strip("_")
    return token or "N"


def _id_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _id_key_loose(value: str) -> str:
    base = str(value).lower().replace("x", "").replace("х", "")
    return re.sub(r"[^a-z0-9]", "", base)


def _resolve_id(requested_id: str, available_ids: list[str], kind: str) -> str:
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


def _bus_tokens(name: str) -> set[str]:
    token = _safe_token(name).lower()
    tokens = {token}
    tokens.add(re.sub(r"_\d+$", "", token))
    return {t for t in tokens if t}


def _to_abs_curve_path(curve_value: str) -> str:
    curve = str(curve_value).replace("\\", os.sep).strip()
    # Защита от pandas NaN и пустых значений
    if not curve or curve.lower() in {"nan", "none", ""}:
        return ""
    return os.path.join("data", curve) if not os.path.isabs(curve) else curve


def _build_breaker_tcc_defaults(breaker_row: dict[str, Any]) -> dict[str, Any]:
    row = {k.lower(): v for k, v in breaker_row.items()}

    defaults: dict[str, Any] = {
        "type": "MCB",
        "In": _read_numeric(row, ["in", "i_n", "in_a", "i_nom_a"]),
        "L_stage": {"active": True},
        "S_stage": {"active": False},
        "I_stage": {"active": False},
    }

    curve_min = _to_abs_curve_path(str(row.get("curve_min", "")))
    curve_max = _to_abs_curve_path(str(row.get("curve_max", "")))
    curve_single = _to_abs_curve_path(str(row.get("curve", "")))

    if curve_min and curve_max:
        defaults["L_stage"].update({
            "source_type": "2csv",
            "curve_csv_min": curve_min,
            "curve_csv_max": curve_max,
        })
    elif curve_single:
        defaults["L_stage"].update({"source_type": "csv", "curve_csv": curve_single})

    i_inst_mult = float(row.get("i_inst", 0.0) or 0.0)
    t_inst = float(row.get("t_inst", 0.02) or 0.02)
    curr_tol_inst_pct = float(row.get("curr_tol_inst_pct", 0.0) or 0.0)

    if i_inst_mult > 0:
        defaults["I_stage"] = {
            "active": True,
            "source_type": "constant",
            "Ii": i_inst_mult,
            "t_instant": t_inst,
            "t_instant_min": 0.002,
            "curr_tol_inst_pct": curr_tol_inst_pct,  # <-- передаём допуск
        }

    return defaults

def _build_fuse_tcc_defaults(fuse_row: dict[str, Any]) -> dict[str, Any]:
    row = {k.lower(): v for k, v in fuse_row.items()}

    curve_single = _to_abs_curve_path(str(row.get("curve", row.get("curve_min", ""))))

    return {
        "type": "Fuse",
        "In": _read_numeric(row, ["in", "i_n", "in_a", "i_nom_a"]),
        "L_stage": {
            "active": True,
            "source_type": "csv",
            "curve_csv": curve_single,
        },
        "S_stage": {"active": False},
        "I_stage": {"active": False},
    }

def _read_numeric(row: dict[str, Any], candidates: list[str], default: float | None = None) -> float:
    for key in candidates:
        if key in row:
            try:
                return float(row[key])
            except (TypeError, ValueError):
                continue
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

def run(project_path: str) -> str:
    project = _load_project(project_path)
    manager = _init_equipment_manager()

    project_name = str(project.get("project_name", "unnamed_project"))
    mode = str(project.get("mode", project.get("calculation_mode", "DC"))).upper()
    frequency = float(project.get("frequency", 50.0))
    default_voltage_kv = float(project.get("voltage_kv", 0.22))
    default_temperature = float(project.get("temperature", 20.0))
    topology = project.get("topology", [])
    if not isinstance(topology, list):
        raise ValueError("'topology' must be a list")
    load_cfg = project.get("load")


    load_items: list[dict[str, Any]] = []
    if isinstance(load_cfg, dict):
        load_items = [load_cfg]
    elif isinstance(load_cfg, list):
        load_items = [item for item in load_cfg if isinstance(item, dict)]


    solver = DSSSolver()
    dss = solver._get_dss()
    battery_cfg: dict[str, Any] = {}
    battery_params: dict[str, Any] = {}
    source_cfg: dict[str, Any] = {}

    if mode == "AC":
        source_cfg = dict(project.get("source", {}))
        r_neutral_raw = source_cfg.get("r_neutral_ohm")
        r_neutral_ohm = float(r_neutral_raw) if r_neutral_raw is not None else None
        source_pu = float(source_cfg.get("pu", 1.0))
        source_x_r = float(source_cfg.get("x_r_ratio", source_cfg.get("X1R1", 2.0)))
        source_phases = int(source_cfg.get("phases", 3))
        source_phase = int(source_cfg.get("phase", 1))
        source_isc3 = float(source_cfg["isc3"]) if source_cfg.get("isc3") is not None else None
        source_isc1 = float(source_cfg["isc1"]) if source_cfg.get("isc1") is not None else None
        source_mvasc3 = float(source_cfg["mvasc3"]) if source_cfg.get("mvasc3") is not None else None
        source_mvasc1 = float(source_cfg["mvasc1"]) if source_cfg.get("mvasc1") is not None else None
        ng_mode = str(source_cfg.get("neutral_grounding", "grounded")).strip().lower()
        if source_isc1 is None and source_mvasc1 is None and ng_mode != "isolated":
            source_mvasc1 = source_mvasc3
        source_base_kv = float(source_cfg.get("basekv", default_voltage_kv if default_voltage_kv > 0 else 0.4))
        default_voltage_kv = source_base_kv
        default_source_bus_raw = str(topology[0].get("from_bus", "SourceBus")) if topology else "SourceBus"
        source_bus_raw = str(source_cfg.get("bus", default_source_bus_raw))
        solver.setup_simulation(
            mode=mode,
            frequency=frequency,
            source_base_kv=source_base_kv,
            voltage_bases_kv=[source_base_kv, 0.4, 0.23],
        )
        source_model = SourceModel(
            basekv=source_base_kv,
            mvasc3=source_mvasc3 if source_isc3 is None else None,
            mvasc1=source_mvasc1 if source_isc1 is None else None,
            isc3_a=source_isc3,
            isc1_a=source_isc1,
            pu=source_pu,
            x_r_ratio=source_x_r,
            neutral_grounding=str(source_cfg.get("neutral_grounding", "grounded")),
            r_neutral_ohm=r_neutral_ohm,
            phases=source_phases,
            phase=source_phase,
        )
        solver.add_element(
            source_model,
            name="Source",
            bus1=_safe_token(source_bus_raw),
            phases=source_phases,
            phase=source_phase,
        )
        

        source_bus = _safe_token(source_bus_raw)
        source_kv = source_base_kv
        nominal_v = source_base_kv * 1000.0 / (1.732 if source_phases == 3 else 1.0)
    else:
        solver.setup_simulation(mode=mode, frequency=frequency)
        battery_cfg = project.get("battery", {})
        battery_id = str(battery_cfg.get("id", battery_cfg.get("model_id")))
        if battery_id in {"None", ""}:
            raise ValueError("battery.id or battery.model_id is required")
        battery_id = _resolve_id(battery_id, manager.get_all_battery_ids(), "Batteries")
        n_cells = int(battery_cfg["n_cells"])
        jumpers_mohm = float(battery_cfg.get("jumpers_mohm", 0.5))
        battery_bus_raw = str(battery_cfg.get("bus", "SourceBus"))
        battery_name = _safe_token(str(battery_cfg.get("name", "BAT1")))
        source_bus = _safe_token(battery_bus_raw)

        raw_cell = manager.get_raw_cell(battery_id)
        battery_model = BatteryModel(raw_cell)
        battery_params = battery_model.get_params(n_cells=n_cells, jumpers_mohm=jumpers_mohm)
        source_kv = float(battery_cfg.get("voltage_kv", float(battery_params["U_total_V"]) / 1000.0))
        nominal_v = float(battery_params.get("U_total_V", 0.0))
        solver.add_element(
            battery_model,
            name=battery_name,
            bus1=source_bus,
            n_cells=n_cells,
            jumpers_mohm=jumpers_mohm,
            phases=1,
            base_kv=source_kv,
        )

    line_rows: list[dict[str, Any]] = []
    protection_plan: list[dict[str, Any]] = []
    known_line_buses: list[tuple[str, str]] = []

    for i, item in enumerate(topology, start=1):

        bus_nominal_kv: dict[str, float] = {}
        item_type = str(item.get("type", "line")).lower()

        if item_type == "transformer":
            tr_id = _resolve_id(
                str(item["transformer_id"]),
                manager.get_all_transformer_ids(),
                "Transformers",
            )
            from_bus = _safe_token(str(item["from_bus"]))
            to_bus = _safe_token(str(item["to_bus"]))
            tr_name = _safe_token(str(item.get("name", f"T{i}")))

            raw_tr = manager.get_raw_transformer(tr_id)
            tr_model = TransformerModel(raw_tr)
            tr_params = tr_model.get_params()

            solver.add_element(
                tr_model,
                name=tr_name,
                bus_hv=from_bus,
                bus_lv=to_bus,
            )

            # Добавляем в line_rows для отчёта
            line_rows.append({
                "name": tr_name,
                "from_bus": str(item["from_bus"]),
                "to_bus": str(item["to_bus"]),
                "R_ohm": tr_params["R_ohm_lv"],
                "X_ohm": tr_params["X_ohm_lv"],
                "C_nf": 0.0,
            })
            known_line_buses.append((to_bus, str(item["to_bus"])))

            bus_nominal_kv[from_bus] = float(tr_params["un_hv_kv"])
            bus_nominal_kv[to_bus] = float(tr_params["un_lv_kv"])
            continue

        cable_id = _resolve_id(str(item["cable_id"]), manager.get_all_cable_ids(), "Cables")
        from_bus_raw = str(item["from_bus"])
        to_bus_raw = str(item["to_bus"])
        from_bus = _safe_token(from_bus_raw)
        to_bus = _safe_token(to_bus_raw)
        line_name = _safe_token(str(item.get("name", f"L{i}")))
        length_km = float(item.get("length_km", 0.0))
        temperature = float(item.get("temperature", default_temperature))
        line_phases = int(item.get("phases", 1 if mode == "DC" else 3))
        if "phase" in item:
            line_phase = int(item.get("phase", 1))
        elif mode == "AC" and line_phases == 1:
            # If single-phase line has no explicit phase, inherit from connected single-phase load.
            to_tokens = _bus_tokens(to_bus_raw)
            inferred_phase = None
            for li in load_items:
                if int(li.get("phases", 3)) != 1:
                    continue
                bus_tokens = _bus_tokens(str(li.get("bus", "")))
                if to_tokens & bus_tokens:
                    inferred_phase = int(li.get("phase", 1))
                    break
            line_phase = inferred_phase if inferred_phase is not None else 1
        else:
            line_phase = 1

        raw_cable = manager.get_raw_cable(cable_id)
        line_model = LineModel(raw_cable)
        line_params = line_model.get_params(mode=mode, length_km=length_km, temperature=temperature)

        if mode == "DC":
            # DC: ток идёт по + и - — двойная длина
            cable_r = float(line_params["R_ohm"]) * 2.0
            cable_x = 0.0
            cable_r0 = 0.0
            cable_x0 = 0.0
        elif line_phases == 1:
            # AC однофазная линия: аналогично x2
            cable_r = float(line_params["R_ohm"]) * 2.0
            cable_x = float(line_params["X_ohm"]) * 2.0
            cable_r0 = float(line_params["R0_ohm"])
            cable_x0 = float(line_params["X0_ohm"])
        else:
            # AC трёхфазная: x1
            cable_r = float(line_params["R_ohm"])
            cable_x = float(line_params["X_ohm"])
            cable_r0 = float(line_params["R0_ohm"])
            cable_x0 = float(line_params["X0_ohm"])

        total_r_ohm = cable_r

        breaker_id_raw = item.get("breaker_id")
        if breaker_id_raw:
            breaker_id_resolved = _resolve_id(str(breaker_id_raw), manager.get_all_breaker_ids(), "CircuitBreakers")
            breaker_row = manager.get_raw_breaker(breaker_id_resolved)

            p_loss_w = _read_numeric(breaker_row, ["P_loss_W", "P_loss", "Ploss_W"])
            in_a = _read_numeric(breaker_row, ["In", "I_n", "In_A", "I_nom_A"])
            poles = _read_numeric(breaker_row, ["Poles", "N_poles", "PoleCount"], default=1.0)
            r_one_pole = p_loss_w / (in_a ** 2)

            # DC и однофазный AC: ток через все полюса последовательно
            # Трёхфазный AC: ток только через один полюс (фазный)
            if mode == "DC" or line_phases == 1:
                r_breaker = r_one_pole * poles
            else:
                r_breaker = r_one_pole

            total_r_ohm += float(r_breaker)
            protection_plan.append(
                {
                    "device_type": "Breaker",
                    "device_id": breaker_id_resolved,
                    "to_bus": to_bus,
                    "to_bus_raw": to_bus_raw,
                    "r_device_ohm": float(r_breaker),
                }
            )

        fuse_id_raw = item.get("fuse_id")
        if fuse_id_raw:
            fuse_id_resolved = _resolve_id(str(fuse_id_raw), manager.get_all_fuse_ids(), "Fuses")
            fuse_row = manager.get_raw_fuse(fuse_id_resolved)
            p_loss = _read_numeric(fuse_row, ["P_loss_W", "P_loss", "Ploss_W"])
            in_a = _read_numeric(fuse_row, ["In", "I_n", "In_A", "I_nom_A"])
            poles = _read_numeric(fuse_row, ["Poles", "N_poles", "PoleCount"], default=1.0)
            r_one_pole = p_loss / (in_a ** 2)

            if mode == "DC" or line_phases == 1:
                r_fuse = r_one_pole * poles
            else:
                r_fuse = r_one_pole

            total_r_ohm += r_fuse
            protection_plan.append(
                {
                    "device_type": "Fuse",
                    "device_id": fuse_id_resolved,
                    "to_bus": to_bus,
                    "to_bus_raw": to_bus_raw,
                    "r_device_ohm": float(r_fuse),
                }
            )


        try:
            solver.add_element(
                line_model,
                name=line_name,
                bus1=from_bus,
                bus2=to_bus,
                length_km=length_km,
                temperature=temperature,
                phases=line_phases,
                phase=line_phase,
            )

        except Exception as e:

            traceback.print_exc()


        if mode == "DC":
            solver.set_line_resistance(line_name, total_r_ohm)
        elif line_phases == 1:
            solver.set_line_resistance(
                line_name, total_r_ohm,
                x1_ohm=cable_x,
                r0_ohm=cable_r0,
                x0_ohm=cable_x0,
            )
        else:
            solver.set_line_resistance(
                line_name, total_r_ohm,
                x1_ohm=cable_x,
                r0_ohm=cable_r0,
                x0_ohm=cable_x0,
            )

        line_rows.append(
            {
                "name": line_name,
                "from_bus": from_bus_raw,
                "to_bus": to_bus_raw,
                "R_ohm": total_r_ohm,
                "X_ohm": float(line_params["X_ohm"]),
                "C_nf": float(line_params["C_nf"]),
            }
        )
        known_line_buses.append((to_bus, to_bus_raw))
        bus_nominal_kv[from_bus] = default_voltage_kv
        bus_nominal_kv[to_bus] = default_voltage_kv

    for idx, load_item in enumerate(load_items, start=1):


        load_name = _safe_token(str(load_item.get("name", f"LD{idx}")))
        load_bus_raw = str(load_item.get("bus", line_rows[-1]["to_bus"] if line_rows else "LoadBus"))
        load_bus = _safe_token(load_bus_raw)

        if mode == "AC" and known_line_buses:
            known_norm = {item[0] for item in known_line_buses}
            if load_bus not in known_norm:
                candidates = [
                    item for item in known_line_buses if item[0].startswith(load_bus) or load_bus.startswith(item[0])
                ]
                if len(candidates) == 1:
                    load_bus = candidates[0][0]
                    load_bus_raw = candidates[0][1]

        load_phases = int(load_item.get("phases", 1 if mode == "DC" else 3))

        if "voltage_kv" in load_item:
            load_voltage_kv = float(load_item["voltage_kv"])
        elif mode == "DC":
            load_voltage_kv = default_voltage_kv if default_voltage_kv > 0 else source_kv
        else:
            load_voltage_kv = bus_nominal_kv.get(load_bus, source_base_kv)
            if load_phases == 1:
                load_voltage_kv = load_voltage_kv / sqrt(3.0)

        load_model = LoadModel(
            power_kw=float(load_item.get("power_kw", 0.0)),
            voltage_kv=load_voltage_kv,
            pf=float(load_item.get("pf", 1.0)),
        )



        load_phase = int(load_item.get("phase", 1))
        solver.add_element(
            load_model,
            name=load_name,
            bus1=load_bus,
            phases=load_phases,
            phase=load_phase,
        )



    if load_items:
        load_bus = _safe_token(str(load_items[0].get("bus", "LoadBus")))
    else:
        load_bus = _safe_token(str(line_rows[-1]["to_bus"] if line_rows else project.get("fault_load_bus", "LoadBus")))



    fault_results = solver.run_fault_study(source_bus=source_bus, load_bus=load_bus)


    power_flow_results = solver.run_power_flow(nominal_voltage_v=nominal_v)



    protection_rows: list[dict[str, Any]] = []
    tcc_devices: list[dict[str, Any]] = []
    for pp in protection_plan:
        bus_key = _safe_token(pp["to_bus"]).lower()
        if mode == "AC":
            i_fault = float(fault_results.get("buses", {}).get(bus_key, {}).get("Isc1", 0.0))
        else:
            i_fault = float(fault_results.get(bus_key, 0.0))

        if i_fault <= 0:
            protection_rows.append(
                {
                    "device_type": str(pp["device_type"]),
                    "device_id": str(pp["device_id"]),
                    "bus": str(pp["to_bus_raw"]),
                    "fault_current_a": 0.0,
                    "trip_time_min_s": float("inf"),
                    "trip_time_max_s": float("inf"),
                    "r_device_ohm": float(pp["r_device_ohm"]),
                }
            )
            continue

        if pp["device_type"] == "Breaker":
            br_row = manager.get_raw_breaker(pp["device_id"])
            tcc_defaults = _build_breaker_tcc_defaults(br_row)
            breaker_in = _read_numeric(br_row, ["In", "I_n", "In_A", "I_nom_A"])
            tcc = UniversalTCC.from_config(tcc_defaults, protection_settings=None, nominal_current_in=breaker_in)
            tcc_devices.append({"label": str(pp["device_id"]), "In": breaker_in, "tcc": tcc})
            t_min = float(tcc.calculate_time(i_fault, mode="min"))
            t_max = float(tcc.calculate_time(i_fault, mode="max"))
        else:
            fuse_row = manager.get_raw_fuse(pp["device_id"])
            tcc_defaults = _build_fuse_tcc_defaults(fuse_row)
            fuse_in = _read_numeric(fuse_row, ["In", "I_n", "In_A", "I_nom_A"])
            tcc = UniversalTCC.from_config(tcc_defaults, protection_settings=None, nominal_current_in=fuse_in)
            tcc_devices.append({"label": str(pp["device_id"]), "In": fuse_in, "tcc": tcc})
            t_min = float(tcc.calculate_time(i_fault, mode="min"))
            t_max = float(tcc.calculate_time(i_fault, mode="max"))

        protection_rows.append(
            {
                "device_type": str(pp["device_type"]),
                "device_id": str(pp["device_id"]),
                "bus": str(pp["to_bus_raw"]),
                "fault_current_a": i_fault,
                "trip_time_min_s": float(t_min),
                "trip_time_max_s": float(t_max),
                "r_device_ohm": float(pp["r_device_ohm"]),
            }
        )

    if tcc_devices and protection_rows:
        devices_for_plot: list[dict[str, Any]] = []
        seen_labels: set[str] = set()
        for item in tcc_devices:
            label = str(item.get("label", ""))
            if label in seen_labels:
                continue
            seen_labels.add(label)
            devices_for_plot.append(item)

        fault_points = [
            {"label": str(row["bus"]), "current": float(row["fault_current_a"])}
            for row in protection_rows
            if float(row.get("fault_current_a", 0.0)) > 0
        ]
        try:
            plot_tcc_curves(
                devices_data=devices_for_plot,
                fault_currents=fault_points,
                project_name=str(Path(project_path).parent / project_name),
            )
        except Exception:
            # Do not fail project calculation if plot generation has issues.
            pass

    from engine.report import SubstationReport
    report = SubstationReport(
        project_name=project_name,
        mode=mode,
        source_cfg=source_cfg,
        battery_cfg=battery_cfg,
        battery_params=battery_params,
        line_rows=line_rows,
        fault=fault_results,
        power_flow=power_flow_results,
        source_bus=source_bus,
        load_bus=load_bus,
        protection_rows=protection_rows,
    )
    report_path = str(Path(project_path).parent / f"{project_name}_report.txt")
    return report.write(report_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run AC/DC substation calculations from YAML project file")
    parser.add_argument("project", help="Path to project YAML file")
    args = parser.parse_args()


    try:
        report_path = run(args.project)
        print(f"Report generated: {report_path}")
        return 0
    except Exception:
        # Это выведет подробную карту (стек) вызовов
        traceback.print_exc() 
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
