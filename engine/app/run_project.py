from __future__ import annotations

from math import sqrt
from pathlib import Path
import re
from typing import Any

import yaml

from engine.constants import DEFAULT_MINIMUM_TRIP_TIME_S
from engine.id_resolver import resolve_id
from engine.models import BatteryModel, LineModel, LoadModel, SourceModel, TransformerModel
from engine.protection_core import UniversalTCC
from engine.report import SubstationReport
from engine.schema_normalizer import init_equipment_manager
from engine.solver import DSSSolver
from engine.utils import bus_tokens, read_numeric, safe_token, to_abs_curve_path
from engine.visualizer import plot_tcc_curves


def _load_project(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("YAML root must be a dictionary")
    return data


def _parse_in_from_device_id(device_id: str) -> float | None:
    match = re.search(r"_(\d+(?:[\.,]\d+)?)A_", str(device_id), flags=re.IGNORECASE)
    if not match:
        return None
    return float(match.group(1).replace(",", "."))


def _parse_poles_from_device_id(device_id: str) -> float | None:
    match = re.search(r"_(\d+)P(?:_|$)", str(device_id), flags=re.IGNORECASE)
    if not match:
        return None
    return float(match.group(1))


def _coerce_device_in(row: dict[str, Any], device_id: str) -> float:
    try:
        return read_numeric(row, ["In", "I_n", "In_A", "I_nom_A"], fallback_to_any=False)
    except ValueError:
        parsed = _parse_in_from_device_id(device_id)
        if parsed is not None:
            return parsed
        return read_numeric(row, ["In", "I_n", "In_A", "I_nom_A"])


def _coerce_device_poles(row: dict[str, Any], device_id: str) -> float:
    try:
        return read_numeric(row, ["Poles", "N_poles", "PoleCount"], default=1.0, fallback_to_any=False)
    except ValueError:
        parsed = _parse_poles_from_device_id(device_id)
        if parsed is not None:
            return parsed
        return read_numeric(row, ["Poles", "N_poles", "PoleCount"], default=1.0)


def _pick_curve_paths(row: dict[str, Any]) -> tuple[str, str, str]:
    row_lower = {str(k).lower(): v for k, v in row.items()}

    def _norm(value: Any) -> str:
        text = to_abs_curve_path(str(value))
        if not text:
            return ""
        return text if text.lower().endswith(".csv") else ""

    curve_min = _norm(row_lower.get("curve_min", ""))
    curve_max = _norm(row_lower.get("curve_max", ""))
    curve_single = _norm(row_lower.get("curve", ""))

    if curve_min and curve_max:
        return curve_min, curve_max, curve_single

    csv_values = [_norm(value) for value in row.values()]
    csv_values = [value for value in csv_values if value]
    unique_csv = list(dict.fromkeys(csv_values))

    if not curve_single and unique_csv:
        curve_single = unique_csv[0]
    if not curve_min and len(unique_csv) >= 1:
        curve_min = unique_csv[0]
    if not curve_max and len(unique_csv) >= 2:
        curve_max = unique_csv[1]

    return curve_min, curve_max, curve_single


def _extra_numeric_values(row: dict[str, Any]) -> list[float]:
    values: list[tuple[int, float]] = []
    for key, raw_value in row.items():
        key_text = str(key).lower()
        if not key_text.startswith("extra_"):
            continue
        try:
            idx = int(key_text.split("_", 1)[1])
        except (TypeError, ValueError):
            continue
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        values.append((idx, value))
    values.sort(key=lambda item: item[0])
    return [value for _, value in values]


def _read_stage_numeric(row: dict[str, Any], keys: list[str]) -> float | None:
    row_lower = {str(k).lower(): v for k, v in row.items()}
    for key in keys:
        if key in row_lower:
            try:
                return float(row_lower[key])
            except (TypeError, ValueError):
                continue
    return None


def _parse_number_list(raw: Any) -> list[float]:
    if raw is None:
        return []
    if isinstance(raw, (int, float)):
        return [float(raw)]
    text = str(raw).strip()
    if not text or text.lower() in {"nan", "none", "off"}:
        return []
    matches = re.findall(r"-?\d+(?:\.\d+)?", text)
    return [float(item) for item in matches]


def _pick_trip_setting_value(raw: Any, requested: float | None) -> float | None:
    values = _parse_number_list(raw)
    if not values:
        return None
    if requested is None:
        return float(values[0])
    nearest = min(values, key=lambda val: abs(float(val) - float(requested)))
    return float(nearest)


def _parse_tsd_map(raw: Any) -> dict[str, list[float]]:
    if raw is None:
        return {}
    text = str(raw).strip()
    if not text.startswith("{"):
        return {}
    try:
        parsed = yaml.safe_load(text)
    except Exception:
        return {}
    if not isinstance(parsed, dict):
        return {}
    result: dict[str, list[float]] = {}
    for key, value in parsed.items():
        nums = _parse_number_list(value)
        if nums:
            result[str(key)] = [float(item) for item in nums]
    return result


def _find_key_ci(mapping: dict[str, Any], expected: str) -> str | None:
    expected_norm = str(expected).strip().lower()
    for key in mapping.keys():
        if str(key).strip().lower() == expected_norm:
            return str(key)
    return None


def _extract_device_ref(item: dict[str, Any], prefix: str) -> str:
    direct = item.get(f"{prefix}_id")
    if direct not in {None, "", "None"}:
        return str(direct)

    prot = item.get("protection_settings")
    if isinstance(prot, dict):
        key = _find_key_ci(prot, f"{prefix}_id")
        if key is not None and prot.get(key) not in {None, "", "None"}:
            return str(prot.get(key))
        for raw_key, value in prot.items():
            norm = re.sub(r"[^a-z0-9]", "", str(raw_key).lower())
            expected = re.sub(r"[^a-z0-9]", "", f"{prefix}_id")
            if norm.startswith(expected) and value not in {None, "", "None"}:
                return str(value)
    return ""


def _build_breaker_tcc_defaults(breaker_row: dict[str, Any], trip_row: dict[str, Any] | None = None) -> dict[str, Any]:
    breaker_id = str(breaker_row.get("ID", breaker_row.get("id", "")))
    trip = trip_row or {}
    l_type = str(trip.get("L_zone_type", "")).strip().lower()

    defaults: dict[str, Any] = {
        "type": "Breaker",
        "In": _coerce_device_in(breaker_row, breaker_id),
        "L_stage": {"active": l_type not in {"", "none", "off", "nan"}},
        "S_stage": {"active": False},
        "I_stage": {"active": False},
    }

    if l_type == "table":
        ir_default = _pick_trip_setting_value(trip.get("L_Ir_range"), None)
        curve_min = to_abs_curve_path(str(trip.get("L_curve_min", "")))
        curve_max = to_abs_curve_path(str(trip.get("L_curve_max", "")))
        if curve_min and curve_max:
            defaults["L_stage"].update(
                {
                    "mode": "table",
                    "source_type": "2csv",
                    "curve_csv_min": curve_min,
                    "curve_csv_max": curve_max,
                    "Ir": float(ir_default) if ir_default is not None else 1.0,
                }
            )
    elif l_type == "formula":
        defaults["L_stage"].update(
            {
                "mode": "standard",
                "source_type": "formula",
                "Kl": float(_pick_trip_setting_value(trip.get("L_Kl"), None) or 1.5625),
                "Ir": float(_pick_trip_setting_value(trip.get("L_Ir_range"), None) or 1.0),
                "tr": float(_pick_trip_setting_value(trip.get("L_tr_range_s"), None) or 16.0),
            }
        )
        l_accur = _parse_number_list(trip.get("L_time_accur_pct"))
        if l_accur:
            defaults["L_stage"]["time_accuracy_pct"] = l_accur

    l_i_accur = _parse_number_list(trip.get("L_I_accur"))
    if l_i_accur:
        defaults["L_stage"]["current_accuracy_factors"] = l_i_accur

    s_mode = str(trip.get("S_i2t_mode", "")).strip().lower()
    if s_mode not in {"", "none", "off", "nan"}:
        tsd_map = _parse_tsd_map(trip.get("S_tsd_range_s"))
        defaults["S_stage"] = {
            "active": True,
            "mode": "i2t_on" if s_mode == "on" else "flat",
            "Isd": float(_pick_trip_setting_value(trip.get("S_Isd_range"), None) or 1.5),
            "tsd": float(_pick_trip_setting_value(trip.get("S_tsd_range_s"), None) or 0.1),
            "tsd_map": tsd_map,
            "isd_relative_to_ir": True,
        }
        s_accur = _parse_number_list(trip.get("S_time_accur_pct"))
        if s_accur:
            defaults["S_stage"]["time_accuracy_pct"] = s_accur

        s_i_accur = _parse_number_list(trip.get("S_I_accur_pct", trip.get("S_I_accur")))
        if not s_i_accur:
            # Backward-compatible fallback: if dedicated S current accuracy
            # column is absent, reuse S accuracy band for pickup spread.
            s_i_accur = _parse_number_list(trip.get("S_time_accur_pct"))
        if s_i_accur:
            defaults["S_stage"]["current_accuracy_pct"] = s_i_accur

    i_range = _parse_number_list(trip.get("I_Ii_range"))
    i_acc = _parse_number_list(trip.get("I_accur_pct"))
    i_time = _parse_number_list(trip.get("I_t_range_s"))
    if i_range:
        t_min = float(i_time[0]) if len(i_time) >= 1 else max(DEFAULT_MINIMUM_TRIP_TIME_S * 2.0, 0.002)
        t_max = float(i_time[1]) if len(i_time) >= 2 else float(i_time[0]) if i_time else 0.01
        defaults["I_stage"] = {
            "active": True,
            "mode": "threshold",
            "source_type": "constant",
            "Ii": float(i_range[0]),
            "curr_tol_inst_pct": float(max(abs(val) for val in i_acc)) if i_acc else 0.0,
            "t_instant": t_max,
            "t_instant_min": t_min,
            "t_instant_max": t_max,
        }
    return defaults


def _build_fuse_tcc_defaults(fuse_row: dict[str, Any]) -> dict[str, Any]:
    row = {k.lower(): v for k, v in fuse_row.items()}
    fuse_id = str(fuse_row.get("ID", fuse_row.get("id", "")))

    _, _, curve_single = _pick_curve_paths(fuse_row)
    if not curve_single:
        curve_single = to_abs_curve_path(str(row.get("curve", row.get("curve_min", ""))))

    return {
        "type": "Fuse",
        "In": _coerce_device_in(fuse_row, fuse_id),
        "L_stage": {
            "active": True,
            "source_type": "csv",
            "curve_csv": curve_single,
        },
        "S_stage": {"active": False},
        "I_stage": {"active": False},
    }


def _is_validation_mode(project: dict[str, Any]) -> bool:
    return bool(
        project.get("validation_mode")
        or project.get("ac_validation_mode")
        or project.get("debug_validation")
    )


def _collect_stage_trace(settings: dict[str, Any], stage_key: str) -> dict[str, Any]:
    stage = settings.get(stage_key, {})
    if not isinstance(stage, dict):
        return {"active": False}
    if stage_key == "L_stage":
        pickup = stage.get("I1_A", stage.get("pickup_a", stage.get("Ir")))
        time_value = stage.get("t1_s", stage.get("tr", stage.get("time")))
    elif stage_key == "S_stage":
        pickup = stage.get("I2_A", stage.get("pickup_a", stage.get("Isd")))
        time_value = stage.get("t2_s", stage.get("tsd", stage.get("time")))
    else:
        pickup = stage.get("Ii_A", stage.get("pickup_a", stage.get("Ii")))
        time_value = stage.get("t_instant", stage.get("time"))

    return {
        "active": bool(stage.get("active", False)),
        "mode": str(stage.get("mode", stage.get("source_type", ""))),
        "pickup": pickup,
        "pickup_a": stage.get("pickup_a") or stage.get("I1_A") or stage.get("I2_A") or stage.get("Ii_A"),
        "time": time_value,
    }


def _relay_nominal_current(relay_cfg: dict[str, Any]) -> float:
    l_raw = relay_cfg.get("L_stage")
    s_raw = relay_cfg.get("S_stage")
    i_raw = relay_cfg.get("I_stage")
    l_stage: dict[str, Any] = l_raw if isinstance(l_raw, dict) else {}
    s_stage: dict[str, Any] = s_raw if isinstance(s_raw, dict) else {}
    i_stage: dict[str, Any] = i_raw if isinstance(i_raw, dict) else {}
    for value in (
        l_stage.get("I1_A"),
        s_stage.get("I2_A"),
        i_stage.get("Ii_A"),
        relay_cfg.get("In"),
    ):
        if value is None:
            continue
        try:
            current = float(value)
        except (TypeError, ValueError):
            continue
        if current > 0:
            return current
    return 1.0


def _clean_protection_overrides(raw_cfg: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in raw_cfg.items():
        if value is None:
            continue
        if isinstance(value, dict):
            nested = _clean_protection_overrides(value)
            if nested:
                cleaned[key] = nested
            continue
        cleaned[key] = value
    return cleaned


def _build_ac_validation_data(
    source_cfg: dict[str, Any],
    source_base_kv: float,
    bus_nominal_kv: dict[str, float],
    power_flow_results: dict[str, Any],
    fault_results: dict[str, Any],
) -> dict[str, Any]:
    xr = float(source_cfg.get("x_r_ratio", source_cfg.get("X1R1", 2.0)))
    isc3_cfg = source_cfg.get("isc3")
    mvasc3_cfg = source_cfg.get("mvasc3")
    if isc3_cfg is not None:
        isc3_a = float(isc3_cfg)
    elif mvasc3_cfg is not None:
        isc3_a = float(mvasc3_cfg) * 1e6 / (sqrt(3.0) * source_base_kv * 1000.0)
    else:
        isc3_a = 0.0

    if isc3_a > 0:
        z1 = (source_base_kv * 1000.0) / (sqrt(3.0) * isc3_a)
        r1 = z1 / sqrt(1.0 + xr * xr)
        x1 = r1 * xr
    else:
        z1 = r1 = x1 = 0.0

    phase_voltages = power_flow_results.get("bus_phase_voltages_v", {})
    bus_validation: list[dict[str, Any]] = []
    for bus in sorted(phase_voltages.keys()):
        phase_v = phase_voltages.get(bus, {})
        v1 = float(phase_v.get("V1N_V", 0.0))
        v2 = float(phase_v.get("V2N_V", 0.0))
        v3 = float(phase_v.get("V3N_V", 0.0))
        vmax = max(v1, v2, v3)
        bus_key = safe_token(bus)
        vnom_ln = float(bus_nominal_kv.get(bus_key, source_base_kv)) * 1000.0 / sqrt(3.0)
        drop_pct = ((vnom_ln - vmax) / vnom_ln * 100.0) if vnom_ln > 0 else 0.0
        bus_validation.append(
            {
                "bus": bus,
                "vmax_ln_v": vmax,
                "vnom_ln_v": vnom_ln,
                "drop_pct": drop_pct,
            }
        )

    fault_validation: list[dict[str, Any]] = []
    for bus_key, value in sorted(fault_results.get("buses", {}).items()):
        if not isinstance(value, dict):
            continue
        fault_validation.append(
            {
                "bus": bus_key,
                "isc3_a": float(value.get("Isc3", 0.0)),
                "isc1_a": float(value.get("Isc1", 0.0)),
            }
        )

    return {
        "source_equivalent": {
            "basekv": source_base_kv,
            "isc3_a": isc3_a,
            "x_r_ratio": xr,
            "z1_ohm": z1,
            "r1_ohm": r1,
            "x1_ohm": x1,
        },
        "bus_voltage_checks": bus_validation,
        "fault_current_checks": fault_validation,
    }


def run(project_path: str) -> str:
    project = _load_project(project_path)
    manager = init_equipment_manager()

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
    battery_cfg: dict[str, Any] = {}
    battery_params: dict[str, Any] = {}
    source_cfg: dict[str, Any] = {}
    source_base_kv = default_voltage_kv if default_voltage_kv > 0 else 0.4
    bus_nominal_kv: dict[str, float] = {}

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
            bus1=safe_token(source_bus_raw),
            phases=source_phases,
            phase=source_phase,
        )

        source_bus = safe_token(source_bus_raw)
        source_kv = source_base_kv
        nominal_v = source_base_kv * 1000.0 / (1.732 if source_phases == 3 else 1.0)
    else:
        solver.setup_simulation(mode=mode, frequency=frequency)
        battery_cfg = project.get("battery", {})
        battery_id = str(battery_cfg.get("id", battery_cfg.get("model_id")))
        if battery_id in {"None", ""}:
            raise ValueError("battery.id or battery.model_id is required")
        battery_id = resolve_id(battery_id, manager.get_all_battery_ids(), "Batteries")
        n_cells = int(battery_cfg["n_cells"])
        jumpers_mohm = float(battery_cfg.get("jumpers_mohm", 0.5))
        battery_bus_raw = str(battery_cfg.get("bus", "SourceBus"))
        battery_name = safe_token(str(battery_cfg.get("name", "BAT1")))
        source_bus = safe_token(battery_bus_raw)

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
        item_type = str(item.get("type", "line")).lower()

        if item_type == "transformer":
            tr_id = resolve_id(
                str(item["transformer_id"]),
                manager.get_all_transformer_ids(),
                "Transformers",
            )
            from_bus = safe_token(str(item["from_bus"]))
            to_bus = safe_token(str(item["to_bus"]))
            tr_name = safe_token(str(item.get("name", f"T{i}")))

            raw_tr = manager.get_raw_transformer(tr_id)
            tr_model = TransformerModel(raw_tr)
            tr_params = tr_model.get_params()

            solver.add_element(
                tr_model,
                name=tr_name,
                bus_hv=from_bus,
                bus_lv=to_bus,
            )

            line_rows.append(
                {
                    "name": tr_name,
                    "from_bus": str(item["from_bus"]),
                    "to_bus": str(item["to_bus"]),
                    "R_ohm": tr_params["R_ohm_lv"],
                    "X_ohm": tr_params["X_ohm_lv"],
                    "C_nf": 0.0,
                }
            )
            known_line_buses.append((to_bus, str(item["to_bus"])))

            bus_nominal_kv[from_bus] = float(tr_params["un_hv_kv"])
            bus_nominal_kv[to_bus] = float(tr_params["un_lv_kv"])
            continue

        cable_id = resolve_id(str(item["cable_id"]), manager.get_all_cable_ids(), "Cables")
        from_bus_raw = str(item["from_bus"])
        to_bus_raw = str(item["to_bus"])
        from_bus = safe_token(from_bus_raw)
        to_bus = safe_token(to_bus_raw)
        line_name = safe_token(str(item.get("name", f"L{i}")))
        length_km = float(item.get("length_km", 0.0))
        temperature = float(item.get("temperature", default_temperature))
        line_phases = int(item.get("phases", 1 if mode == "DC" else 3))
        if "phase" in item:
            line_phase = int(item.get("phase", 1))
        elif mode == "AC" and line_phases == 1:
            to_tokens = bus_tokens(to_bus_raw)
            inferred_phase = None
            for li in load_items:
                if int(li.get("phases", 3)) != 1:
                    continue
                load_bus_tokens = bus_tokens(str(li.get("bus", "")))
                if to_tokens & load_bus_tokens:
                    inferred_phase = int(li.get("phase", 1))
                    break
            line_phase = inferred_phase if inferred_phase is not None else 1
        else:
            line_phase = 1

        raw_cable = manager.get_raw_cable(cable_id)
        line_model = LineModel(raw_cable)
        line_params = line_model.get_params(mode=mode, length_km=length_km, temperature=temperature)

        if mode == "DC":
            cable_r = float(line_params["R_ohm"]) * 2.0
            cable_x = 0.0
            cable_r0 = 0.0
            cable_x0 = 0.0
        elif line_phases == 1:
            cable_r = float(line_params["R_ohm"]) * 2.0
            cable_x = float(line_params["X_ohm"]) * 2.0
            cable_r0 = float(line_params["R0_ohm"])
            cable_x0 = float(line_params["X0_ohm"])
        else:
            cable_r = float(line_params["R_ohm"])
            cable_x = float(line_params["X_ohm"])
            cable_r0 = float(line_params["R0_ohm"])
            cable_x0 = float(line_params["X0_ohm"])

        total_r_ohm = cable_r

        protection_cfg = item.get("protection_settings")
        protection_cfg = _clean_protection_overrides(protection_cfg) if isinstance(protection_cfg, dict) else {}

        breaker_id_raw = _extract_device_ref(item, "breaker")
        if breaker_id_raw:
            breaker_id_resolved = resolve_id(str(breaker_id_raw), manager.get_all_breaker_ids(), "CircuitBreakers")
            breaker_row = manager.get_raw_breaker_with_trip_unit(breaker_id_resolved)

            p_loss_w = read_numeric(breaker_row, ["P_loss_W", "P_loss", "Ploss_W"])
            in_a = _coerce_device_in(breaker_row, breaker_id_resolved)
            poles = _coerce_device_poles(breaker_row, breaker_id_resolved)
            r_one_pole = p_loss_w / (in_a**2)

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
                    "protection_settings": protection_cfg,
                }
            )

        fuse_id_raw = _extract_device_ref(item, "fuse")
        if fuse_id_raw:
            fuse_id_resolved = resolve_id(str(fuse_id_raw), manager.get_all_fuse_ids(), "Fuses")
            fuse_row = manager.get_raw_fuse(fuse_id_resolved)
            p_loss = read_numeric(fuse_row, ["P_loss_W", "P_loss", "Ploss_W"])
            in_a = _coerce_device_in(fuse_row, fuse_id_resolved)
            poles = _coerce_device_poles(fuse_row, fuse_id_resolved)
            r_one_pole = p_loss / (in_a**2)

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
                    "protection_settings": protection_cfg,
                }
            )

        prot_type = str(protection_cfg.get("type", "")).strip().lower()
        if prot_type == "relay":
            protection_plan.append(
                {
                    "device_type": "Relay",
                    "device_id": str(protection_cfg.get("name", line_name)),
                    "to_bus": to_bus,
                    "to_bus_raw": to_bus_raw,
                    "r_device_ohm": 0.0,
                    "protection_settings": protection_cfg,
                }
            )

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

        if mode == "DC":
            solver.set_line_resistance(line_name, total_r_ohm)
        else:
            solver.set_line_resistance(
                line_name,
                total_r_ohm,
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
        segment_base_kv = float(bus_nominal_kv.get(from_bus, default_voltage_kv))
        bus_nominal_kv.setdefault(from_bus, segment_base_kv)
        bus_nominal_kv.setdefault(to_bus, segment_base_kv)

    for idx, load_item in enumerate(load_items, start=1):
        load_name = safe_token(str(load_item.get("name", f"LD{idx}")))
        load_bus_raw = str(load_item.get("bus", line_rows[-1]["to_bus"] if line_rows else "LoadBus"))
        load_bus = safe_token(load_bus_raw)

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
        load_bus = safe_token(str(load_items[0].get("bus", "LoadBus")))
    else:
        load_bus = safe_token(str(line_rows[-1]["to_bus"] if line_rows else project.get("fault_load_bus", "LoadBus")))

    fault_results = solver.run_fault_study(source_bus=source_bus, load_bus=load_bus)
    power_flow_results = solver.run_power_flow(nominal_voltage_v=nominal_v)

    protection_rows: list[dict[str, Any]] = []
    protection_trace_rows: list[dict[str, Any]] = []
    tcc_devices: list[dict[str, Any]] = []
    for pp in protection_plan:
        bus_key = safe_token(pp["to_bus"]).lower()
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
            trip_id = str(br_row.get("TripUnit_ID", "")).strip()
            try:
                trip_row = manager.get_raw_trip_unit(trip_id) if trip_id else {}
            except (KeyError, ValueError):
                trip_row = {}
            tcc_defaults = _build_breaker_tcc_defaults(br_row, trip_row)
            breaker_in = _coerce_device_in(br_row, str(pp["device_id"]))
            tcc = UniversalTCC.from_config(
                tcc_defaults,
                protection_settings=pp.get("protection_settings"),
                nominal_current_in=breaker_in,
            )
            tcc_devices.append({"label": str(pp["device_id"]), "In": breaker_in, "tcc": tcc})
            t_min = float(tcc.calculate_time(i_fault, mode="min"))
            t_max = float(tcc.calculate_time(i_fault, mode="max"))
            protection_trace_rows.append(
                {
                    "device": str(pp["device_id"]),
                    "device_type": "Breaker",
                    "bus": str(pp["to_bus_raw"]),
                    "In": breaker_in,
                    "L": _collect_stage_trace(tcc.settings, "L_stage"),
                    "S": _collect_stage_trace(tcc.settings, "S_stage"),
                    "I": _collect_stage_trace(tcc.settings, "I_stage"),
                }
            )
        elif pp["device_type"] == "Fuse":
            fuse_row = manager.get_raw_fuse(pp["device_id"])
            tcc_defaults = _build_fuse_tcc_defaults(fuse_row)
            fuse_in = _coerce_device_in(fuse_row, str(pp["device_id"]))
            tcc = UniversalTCC.from_config(
                tcc_defaults,
                protection_settings=pp.get("protection_settings"),
                nominal_current_in=fuse_in,
            )
            tcc_devices.append({"label": str(pp["device_id"]), "In": fuse_in, "tcc": tcc})
            t_min = float(tcc.calculate_time(i_fault, mode="min"))
            t_max = float(tcc.calculate_time(i_fault, mode="max"))
            protection_trace_rows.append(
                {
                    "device": str(pp["device_id"]),
                    "device_type": "Fuse",
                    "bus": str(pp["to_bus_raw"]),
                    "In": fuse_in,
                    "L": _collect_stage_trace(tcc.settings, "L_stage"),
                    "S": _collect_stage_trace(tcc.settings, "S_stage"),
                    "I": _collect_stage_trace(tcc.settings, "I_stage"),
                }
            )
        else:
            relay_cfg = dict(pp.get("protection_settings") or {})
            relay_in = _relay_nominal_current(relay_cfg)
            relay_defaults = {
                "type": "Relay",
                "In": relay_in,
                "L_stage": {"active": False},
                "S_stage": {"active": False},
                "I_stage": {"active": False},
            }
            tcc = UniversalTCC.from_config(
                relay_defaults,
                protection_settings=relay_cfg,
                nominal_current_in=relay_in,
            )
            tcc_devices.append({"label": str(pp["device_id"]), "In": relay_in, "tcc": tcc})
            t_min = float(tcc.calculate_time(i_fault, mode="min"))
            t_max = float(tcc.calculate_time(i_fault, mode="max"))
            protection_trace_rows.append(
                {
                    "device": str(pp["device_id"]),
                    "device_type": "Relay",
                    "bus": str(pp["to_bus_raw"]),
                    "In": relay_in,
                    "L": _collect_stage_trace(tcc.settings, "L_stage"),
                    "S": _collect_stage_trace(tcc.settings, "S_stage"),
                    "I": _collect_stage_trace(tcc.settings, "I_stage"),
                }
            )

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
            pass

    validation_data: dict[str, Any] | None = None
    if mode == "AC" and _is_validation_mode(project):
        validation_data = _build_ac_validation_data(
            source_cfg=source_cfg,
            source_base_kv=source_base_kv,
            bus_nominal_kv=bus_nominal_kv,
            power_flow_results=power_flow_results,
            fault_results=fault_results,
        )

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
        protection_trace_rows=protection_trace_rows,
        validation_data=validation_data,
    )
    report_path = str(Path(project_path).parent / f"{project_name}_report.txt")
    return report.write(report_path)
