"""Universal time-current characteristic (TCC) calculation core.

This module supports a hybrid data model:
- static defaults from equipment database rows (library.ods),
- dynamic runtime settings from project YAML.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import warnings
import re

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from engine.constants import DEFAULT_MINIMUM_TRIP_TIME_S


_IEC_CURVES = {
    "standard_inverse": (0.02, 0.14),
    "standard": (0.02, 0.14),
    "very_inverse": (1.0, 13.5),
    "very": (1.0, 13.5),
    "extremely_inverse": (2.0, 80.0),
    "extremely": (2.0, 80.0),
}


class UniversalTCC:
    """Universal protection curve evaluator for Fuse/MCB/Relay/LSI.

    Physics summary:
    - L-stage: overload/time-delayed region (inverse or tabular).
    - S-stage: short-time region (constant delay or I2t slope).
    - I-stage: instantaneous trip region.
    """

    def __init__(self, settings_dict: dict[str, Any], nominal_current_In: float) -> None:
        self.settings: dict[str, Any] = deepcopy(settings_dict)
        self.device_type = str(self.settings.get("type", "Device"))
        self.in_a = float(nominal_current_In)
        self.min_time_s = float(self.settings.get("minimum_time_s", DEFAULT_MINIMUM_TRIP_TIME_S))
        self.base_dir = Path(self.settings.get("base_dir") or Path(__file__).resolve().parents[1])

        self._interp: dict[str, dict[str, Any]] = {}
        self._init_stage_interpolators()

    @classmethod
    def from_config(
        cls,
        db_defaults: dict[str, Any],
        protection_settings: dict[str, Any] | None = None,
        nominal_current_in: float | None = None,
    ) -> UniversalTCC:
        """Create a model from static DB defaults + dynamic YAML settings.

        Priority: YAML settings > ODS defaults.
        """

        merged = deepcopy(db_defaults or {})
        dynamic = protection_settings or {}
        merged = cls._deep_merge(merged, dynamic)

        if "L_stage" not in merged:
            merged["L_stage"] = {}

        l_stage = merged.get("L_stage", {})
        if "curve_csv" not in l_stage and "curve" in merged:
            l_stage["curve_csv"] = merged.get("curve")
            l_stage.setdefault("source_type", "csv")
        if "curve_csv_min" not in l_stage and "curve_min" in merged:
            l_stage["curve_csv_min"] = merged.get("curve_min")
        if "curve_csv_max" not in l_stage and "curve_max" in merged:
            l_stage["curve_csv_max"] = merged.get("curve_max")
        if "curve_csv_min" in l_stage and "curve_csv_max" in l_stage:
            l_stage.setdefault("source_type", "2csv")
        merged["L_stage"] = l_stage

        in_val = nominal_current_in
        if in_val is None:
            in_val = dynamic.get("In")
        if in_val is None:
            in_val = merged.get("In", merged.get("I_n", merged.get("I_nom_A")))
        if in_val is None:
            raise ValueError("Nominal current In is required for UniversalTCC")

        return cls(merged, float(in_val))

    def calculate_time(self, current_amps: float, mode: str = "avg") -> float:
        current = float(current_amps)
        if current <= 0:
            return float("inf")

        mode_key = str(mode).strip().lower()
        if mode_key not in {"avg", "min", "max"}:
            raise ValueError("mode must be one of: avg, min, max")

        eval_current = self._apply_current_tolerance_for_fuse(current, mode_key)

        stage_times: list[float] = []
        for stage_key in ("L_stage", "S_stage", "I_stage"):
            t = self._evaluate_stage(stage_key, eval_current, mode_key)
            if np.isfinite(t):
                stage_times.append(float(t))

        if not stage_times:
            return float("inf")
        return max(min(stage_times), self.min_time_s)


    def get_plot_points(
        self,
        mode: str = "avg",
        i_min: float | None = None,
        i_max: float | None = None,
    ) -> tuple[list[float], list[float]]:
        i_low = float(i_min) if i_min is not None else max(0.1 * self.in_a, 0.001)
        i_high = float(i_max) if i_max is not None else max(100.0 * self.in_a, i_low * 10.0)
        if i_low <= 0 or i_high <= i_low:
            raise ValueError("Invalid plotting range")

        mode_key = str(mode).strip().lower()
        if mode_key not in {"avg", "min", "max"}:
            raise ValueError("mode must be one of: avg, min, max")

        currents = np.logspace(np.log10(i_low), np.log10(i_high), 600)
        x: list[float] = currents.tolist()
        y: list[float] = [self._composite_time(float(i), mode_key) for i in currents]

        finite_y = [val for val in y if np.isfinite(val)]
        top_y = (max(finite_y) * 1.5) if finite_y else 1e4
        top_y = max(top_y, self.min_time_s * 10.0, 10000.0)

        self._insert_vertical_transition(x, y, self._stage_pickup_current_for_mode("L_stage", mode_key), i_low, i_high, mode_key, top_y)
        self._insert_vertical_transition(x, y, self._stage_pickup_current_for_mode("S_stage", mode_key), i_low, i_high, mode_key, top_y)
        self._insert_vertical_transition(x, y, self._stage_pickup_current_for_mode("I_stage", mode_key), i_low, i_high, mode_key, top_y)

        return x, y

    def _composite_time(self, current_a: float, mode: str) -> float:
        eval_current = self._apply_current_tolerance_for_fuse(float(current_a), mode)

        if self._is_stage_active("I_stage"):
            i_pickup = self._stage_pickup_current_for_mode("I_stage", mode)
            if eval_current >= i_pickup:
                return max(self._instantaneous_trip_time(mode), self.min_time_s)

        stage_times: list[float] = []
        for stage_key in ("L_stage", "S_stage"):
            t = self._evaluate_stage(stage_key, eval_current, mode)
            if np.isfinite(t):
                stage_times.append(float(t))
        if not stage_times:
            return float("inf")
        return max(min(stage_times), self.min_time_s)

    def _insert_vertical_transition(
        self,
        x: list[float],
        y: list[float],
        i_pickup: float,
        i_low: float,
        i_high: float,
        mode: str,
        top_y: float,
    ) -> None:
        if not np.isfinite(i_pickup) or i_pickup <= i_low or i_pickup >= i_high:
            return
        eps = max(i_pickup * 1e-6, 1e-9)
        t_before = self._composite_time(i_pickup - eps, mode)
        t_after = self._composite_time(i_pickup + eps, mode)
        y_top = top_y if not np.isfinite(t_before) else min(top_y, max(t_before, t_after, self.min_time_s))

        idx = int(np.searchsorted(np.asarray(x, dtype=float), i_pickup))
        x.insert(idx, float(i_pickup))
        y.insert(idx, float(y_top))
        x.insert(idx + 1, float(i_pickup))
        y.insert(idx + 1, float(max(t_after, self.min_time_s) if np.isfinite(t_after) else y_top))

    def get_points(
        self,
        i_min: float | None = None,
        i_max: float | None = None,
    ) -> tuple[tuple[list[float], list[float]], tuple[list[float], list[float]], tuple[list[float], list[float]]]:
        """Return nominal/min/max curve points for plotting APIs."""
        return (
            self.get_plot_points(mode="avg", i_min=i_min, i_max=i_max),
            self.get_plot_points(mode="min", i_min=i_min, i_max=i_max),
            self.get_plot_points(mode="max", i_min=i_min, i_max=i_max),
        )
    

    def get_instantaneous_band(self) -> tuple[float, float] | None:
        """Возвращает (I_min, I_max) порога I-stage с учётом curr_tol_inst_pct.
        
        None если I-stage не активен.
        """
        if not self._is_stage_active("I_stage"):
            return None
        i_base = self._stage_pickup_current("I_stage")
        tol = float(self._stage("I_stage").get("curr_tol_inst_pct", 0.0))
        return (
            i_base * (1.0 - tol / 100.0),
            i_base * (1.0 + tol / 100.0),
        )

    def _init_stage_interpolators(self) -> None:
        for stage_key in ("L_stage", "S_stage", "I_stage"):
            stage = self._stage(stage_key)
            if not stage or not stage.get("active", True):
                continue
            source_type = str(stage.get("source_type", "")).lower()
            if source_type == "csv":
                curve_csv = stage.get("curve_csv")
                if curve_csv:
                    self._interp[stage_key] = {"avg": self._build_log_interp(curve_csv)}
            elif source_type == "2csv":
                cmin = stage.get("curve_csv_min")
                cmax = stage.get("curve_csv_max")
                if cmin and cmax:
                    self._interp[stage_key] = {
                        "min": self._build_log_interp(cmin),
                        "max": self._build_log_interp(cmax),
                    }

    def _evaluate_stage(self, stage_key: str, current_a: float, mode: str) -> float:
        stage = self._stage(stage_key)
        if not stage or not stage.get("active", True):
            return float("inf")

        stage_mode = self._stage_mode(stage_key)
        source_type = str(stage.get("source_type", "")).lower()
        i_pickup = self._stage_pickup_current_for_mode(stage_key, mode)
        if current_a < i_pickup:
            return float("inf")

        if stage_key == "L_stage" and stage_mode in {"standard", "formula"}:
            time_s = self._time_from_l_formula(current_a)
        elif stage_key == "L_stage" and stage_mode in {
            "iec_inverse",
            "standard_inverse",
            "very_inverse",
            "extremely_inverse",
        }:
            time_s = self._time_from_iec_formula(stage_key, current_a)
        elif stage_key == "S_stage" and stage_mode in {"flat", "i2t_off", "i2toff"}:
            time_s = self._time_from_s_flat(mode)
        elif stage_key == "S_stage" and stage_mode in {"i2t_on", "i2t", "i2t_slope"}:
            time_s = self._time_from_s_i2t(current_a, mode)
        elif source_type in {"csv", "2csv"} or stage_mode == "table":
            time_s = self._time_from_csv(stage_key, current_a, mode)
        elif source_type == "iec_formula":
            time_s = self._time_from_iec_formula(stage_key, current_a)
        elif source_type == "constant":
            if stage_key == "I_stage":
                time_s = self._instantaneous_trip_time(mode)
            else:
                time_s = self._time_from_constant(stage_key)
        elif source_type == "i2t_slope":
            time_s = self._time_from_i2t(stage_key, current_a)
        else:
            return float("inf")

        if not np.isfinite(time_s):
            return float("inf")
        return max(self._apply_time_tolerance(stage_key, float(time_s), mode), self.min_time_s)

    def _time_from_csv(self, stage_key: str, current_a: float, mode: str) -> float:
        stage = self._stage(stage_key)
        ratio = self._current_ratio(current_a, stage_key)
        if ratio <= 0:
            return float("inf")

        interps = self._interp.get(stage_key, {})
        if not interps:
            points = stage.get("points")
            interp_points = self._build_log_interp_from_points(points)
            if interp_points is None:
                warnings.warn("No data in interps")
                return float("inf")
            interps = {"avg": interp_points}

        if "avg" in interps:
            log_t = float(interps["avg"](np.log10(ratio)))
            return float(10**log_t)

        if mode == "min":
            key = "min"
        elif mode == "max":
            key = "max"
        else:
            log_t_min = float(interps["min"](np.log10(ratio)))
            log_t_max = float(interps["max"](np.log10(ratio)))
            return float(10 ** ((log_t_min + log_t_max) / 2.0))

        if key not in interps:
            return float("inf")
        log_t = float(interps[key](np.log10(ratio)))
        return float(10**log_t)

    def _build_log_interp_from_points(self, points: Any) -> Any | None:
        if not isinstance(points, list):
            return None
        parsed: list[tuple[float, float]] = []
        for item in points:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                try:
                    i_val = float(item[0])
                    t_val = float(item[1])
                except (TypeError, ValueError):
                    continue
                if i_val > 0 and t_val > 0:
                    parsed.append((i_val, t_val))
        if len(parsed) < 2:
            return None
        arr = np.asarray(parsed, dtype=float)
        order = np.argsort(arr[:, 0])
        x = np.log10(arr[order, 0])
        y = np.log10(arr[order, 1])
        return interp1d(x, y, kind="linear", bounds_error=False, fill_value="extrapolate")

    def _time_from_iec_formula(self, stage_key: str, current_a: float) -> float:
        stage = self._stage(stage_key)
        pickup_a = stage.get("pickup_a", stage.get("I1_A"))
        if pickup_a is not None:
            pickup = float(pickup_a)
            if pickup <= 0:
                return float("inf")
            m = float(current_a) / pickup
        else:
            ratio = self._current_ratio(current_a, stage_key)
            is_set = float(stage.get("Is_set", stage.get("Isd", 1.0)))
            if is_set <= 0:
                return float("inf")
            m = ratio / is_set
        if m <= 1.0:
            return float("inf")

        curve_name = str(stage.get("curve_type", "standard_inverse")).lower()
        alpha, beta = _IEC_CURVES.get(curve_name, _IEC_CURVES["standard_inverse"])
        k_raw = stage.get("k_multiplier")
        if k_raw is None and stage.get("t1_s") is not None:
            t1 = float(stage.get("t1_s"))
            calib_multiple = float(stage.get("calibration_multiple", 10.0))
            denom_ref = (calib_multiple**alpha) - 1.0
            k = t1 * denom_ref / beta if denom_ref > 0 else 1.0
        else:
            k = float(k_raw if k_raw is not None else 1.0)
        denom = (m**alpha) - 1.0
        if denom <= 0:
            return float("inf")
        return (k * beta) / denom

    def _time_from_constant(self, stage_key: str) -> float:
        stage = self._stage(stage_key)
        if stage_key == "I_stage":
            return float(stage.get("t_instant", stage.get("time", 0.02)))
        return float(stage.get("tsd", stage.get("time", 0.1)))

    def _time_from_i2t(self, stage_key: str, current_a: float) -> float:
        stage = self._stage(stage_key)
        m = self._current_ratio(current_a, stage_key)
        isd = float(stage.get("Isd", stage.get("Is_set", 1.0)))
        tsd = float(stage.get("tsd", 0.2))
        if m <= 0 or isd <= 0:
            return float("inf")
        return (tsd * (isd**2)) / (m**2)

    def _time_from_l_formula(self, current_a: float) -> float:
        stage = self._stage("L_stage")
        ir = self._stage_pickup_current("L_stage")
        if ir <= 0 or current_a <= 0:
            return float("inf")
        kl = float(stage.get("Kl", stage.get("k_multiplier", 1.5625)))
        tr = float(stage.get("tr", stage.get("time", 16.0)))
        return kl * tr * (6.0 * ir / float(current_a)) ** 2

    def _time_from_s_flat(self, mode: str) -> float:
        stage = self._stage("S_stage")
        if stage.get("tsd") is None and stage.get("t2_s") is not None:
            stage = {**stage, "tsd": stage.get("t2_s")}
        tsd = self._pick_tsd_for_mode(stage, mode)
        return float(tsd)

    def _time_from_s_i2t(self, current_a: float, mode: str) -> float:
        stage = self._stage("S_stage")
        if stage.get("tsd") is None and stage.get("t2_s") is not None:
            stage = {**stage, "tsd": stage.get("t2_s")}
        ir = self._stage_pickup_current("L_stage")
        if ir <= 0 or current_a <= 0:
            return float("inf")
        tsd = self._pick_tsd_for_mode(stage, mode)
        return float(tsd) * (10.0 * ir / float(current_a)) ** 2

    def _pick_tsd_for_mode(self, stage: dict[str, Any], mode: str) -> float:
        mode_key = str(mode).strip().lower()
        tsd_map = stage.get("tsd_map")
        tsd_setting = stage.get("tsd", 0.1)
        if isinstance(tsd_map, dict):
            key = str(tsd_setting)
            raw = tsd_map.get(key)
            if isinstance(raw, list) and raw:
                if mode_key == "min":
                    return float(raw[0])
                if mode_key == "max":
                    return float(raw[-1])
                return float(sum(float(x) for x in raw) / len(raw))
        return float(tsd_setting)

    def _current_ratio(self, current_a: float, stage_key: str) -> float:
        """Возвращает отношение тока к базовому для интерполяции.
        Fuse: CSV дан в амперах — ratio = I / 1 (абсолютный ток).
         MCB/Relay: CSV дан в о.е. от In — ratio = I / In.
         """
        if str(self.device_type).lower() == "fuse":
              # Ток в CSV абсолютный — нормировка на 1А (без деления на In)
            pickup_mult = self._stage_pickup_multiple(stage_key)
            return float(current_a) / max(pickup_mult, 1e-12)
        pickup_mult = self._stage_pickup_multiple(stage_key)
        return float(current_a) / max(self.in_a * pickup_mult, 1e-12)

    def _stage_pickup_multiple(self, stage_key: str) -> float:
        stage = self._stage(stage_key)
        abs_pickup = stage.get("pickup_a")
        if abs_pickup is not None:
            return float(abs_pickup) / max(self.in_a, 1e-12)
        if stage_key == "L_stage" and stage.get("I1_A") is not None:
            return float(stage.get("I1_A")) / max(self.in_a, 1e-12)
        if stage_key == "S_stage" and stage.get("I2_A") is not None:
            return float(stage.get("I2_A")) / max(self.in_a, 1e-12)
        if stage_key == "I_stage" and stage.get("Ii_A") is not None:
            return float(stage.get("Ii_A")) / max(self.in_a, 1e-12)
        if stage_key == "L_stage":
            return float(stage.get("Ir", stage.get("Ir_adjust", stage.get("Is_set", 1.0))))
        if stage_key == "S_stage":
            return float(stage.get("Isd", stage.get("Is_set", 1.0)))
        return float(stage.get("Ii", stage.get("Is_set", 1.0)))

    def _stage_pickup_current(self, stage_key: str) -> float:
        """Возвращает ток срабатывания в амперах. 
           Fuse: значения в CSV уже в амперах — pickup = множитель напрямую.
           MCB/Relay: значения в CSV в о.е. от In — pickup = In * множитель.
        """
        if str(self.device_type).lower() == "fuse":
            # Для предохранителя pickup multiple это уже ток в амперах
            return self._stage_pickup_multiple(stage_key)
        if stage_key == "S_stage" and bool(self._stage("S_stage").get("isd_relative_to_ir", False)):
            return self._stage_pickup_current("L_stage") * float(
                self._stage("S_stage").get("Isd", self._stage("S_stage").get("Is_set", 1.0))
            )
        return self.in_a * self._stage_pickup_multiple(stage_key)

    def _stage_pickup_current_for_mode(self, stage_key: str, mode: str) -> float:
        mode_key = str(mode).strip().lower()
        stage = self._stage(stage_key)
        if not stage:
            return float("inf")

        if stage_key == "I_stage":
            return self._instantaneous_pickup_current(mode_key)

        if stage_key == "L_stage":
            base_current = self._stage_pickup_current(stage_key)
            factors = stage.get("current_accuracy_factors")
            if isinstance(factors, (list, tuple)) and len(factors) >= 2:
                values = [float(v) for v in factors]
                if mode_key == "min":
                    return base_current * min(values)
                if mode_key == "max":
                    return base_current * max(values)
                return base_current * (sum(values) / len(values))

        if stage_key == "S_stage":
            if stage.get("pickup_a") is not None or stage.get("I2_A") is not None:
                base_abs = self._stage_pickup_current(stage_key)
                tol_pct_abs = self._stage_current_tolerance_pct(stage)
                if mode_key == "min":
                    return base_abs * (1.0 - tol_pct_abs / 100.0)
                if mode_key == "max":
                    return base_abs * (1.0 + tol_pct_abs / 100.0)
                return base_abs
            if bool(stage.get("isd_relative_to_ir", False)):
                # Isd is specified in multiples of L-stage setting Ir (not L accuracy band).
                l_current_nominal = self._stage_pickup_current("L_stage")
                base = l_current_nominal * float(stage.get("Isd", stage.get("Is_set", 1.0)))
            else:
                base = self.in_a * float(stage.get("Isd", stage.get("Is_set", 1.0)))

            tol_pct = self._stage_current_tolerance_pct(stage)
            if mode_key == "min":
                return base * (1.0 - tol_pct / 100.0)
            if mode_key == "max":
                return base * (1.0 + tol_pct / 100.0)
            return base

        base_current = self._stage_pickup_current(stage_key)
        tol = self._stage_current_tolerance_pct(stage)
        if mode_key == "min":
            return base_current * (1.0 - tol / 100.0)
        if mode_key == "max":
            return base_current * (1.0 + tol / 100.0)
        return base_current

    def _stage_current_tolerance_pct(self, stage: dict[str, Any]) -> float:
        for key in ("current_accuracy_pct", "curr_tol_pct", "curr_accur_pct"):
            value = stage.get(key)
            if value is None:
                continue
            if isinstance(value, (list, tuple)):
                nums = [float(v) for v in value if isinstance(v, (int, float))]
                if nums:
                    return float(max(abs(v) for v in nums))
            try:
                return abs(float(value))
            except (TypeError, ValueError):
                continue
        return 0.0

    def _instantaneous_pickup_current(self, mode: str) -> float:
        stage = self._stage("I_stage")
        base_pickup = self._stage_pickup_current("I_stage")
        mode_key = str(mode).strip().lower()
        tol = float(stage.get("curr_tol_inst_pct", 0.0))
        if mode_key == "min":
            return base_pickup * (1.0 - tol / 100.0)
        if mode_key == "max":
            return base_pickup * (1.0 + tol / 100.0)
        return base_pickup

    def _instantaneous_trip_time(self, mode: str) -> float:
        stage = self._stage("I_stage")
        base_time = float(stage.get("t_instant", stage.get("time", 0.02)))
        mode_key = str(mode).strip().lower()
        if mode_key == "min":
            return float(stage.get("t_instant_min", base_time))
        if mode_key == "max":
            return float(stage.get("t_instant_max", base_time))
        return base_time

    def _apply_current_tolerance_for_fuse(self, current_a: float, mode: str) -> float:
        if str(self.device_type).lower() != "fuse":
            return current_a
        if mode == "min": # меньше время при увеличении тока
            return current_a * 1.1
        if mode == "max": # больше время при уменьшении тока
            return current_a * 0.9
        return current_a

    def _apply_time_tolerance(self, stage_key: str, time_s: float, mode: str) -> float:
        if mode == "avg" or str(self.device_type).lower() == "fuse":
            return time_s
        stage = self._stage(stage_key)
        if stage_key == "S_stage" and isinstance(stage.get("tsd_map"), dict) and stage.get("tsd_map"):
            # For S-stage with explicit min/max tsd map, time spread is already
            # represented by the selected map interval.
            return time_s
        acc = stage.get("time_accuracy_pct", stage.get("time_accur_pct"))
        if isinstance(acc, (list, tuple)) and len(acc) >= 2:
            neg = float(acc[0])
            pos = float(acc[1])
            if mode == "min":
                return time_s * max(0.0, (1.0 + neg / 100.0))
            return time_s * (1.0 + pos / 100.0)
        if isinstance(acc, str):
            nums = re.findall(r"-?\d+(?:\.\d+)?", acc)
            if len(nums) >= 2:
                neg = float(nums[0])
                pos = float(nums[1])
                if mode == "min":
                    return time_s * max(0.0, (1.0 + neg / 100.0))
                return time_s * (1.0 + pos / 100.0)
        tol = float(stage.get("time_tolerance_pct", self.settings.get("time_tolerance_pct", 0.0)))
        if mode == "min":
            return time_s * max(0.0, (1.0 - tol / 100.0))
        return time_s * (1.0 + tol / 100.0)

    def _stage_mode(self, stage_key: str) -> str:
        stage = self._stage(stage_key)
        mode = str(stage.get("mode", stage.get("source_type", ""))).strip().lower()
        if mode in {"", "none", "off"}:
            return "off"
        return mode

    def _build_log_interp(self, csv_path: str) -> Any:
        path = Path(csv_path)
        if not path.is_absolute():
            path = self.base_dir / path

        ratio, time = self._read_curve_columns(path)

        valid = np.isfinite(ratio.to_numpy()) & np.isfinite(time.to_numpy())
        r = ratio.to_numpy(dtype=float)[valid]
        t = time.to_numpy(dtype=float)[valid]
        mask = (r > 0) & (t > 0)
        r = r[mask]
        t = t[mask]
        if r.size < 2:
            raise ValueError(f"CSV '{path}' does not contain enough valid points")

        order = np.argsort(r)
        x = np.log10(r[order])
        y = np.log10(t[order])
        return interp1d(x, y, kind="linear", bounds_error=False, fill_value="extrapolate")

    def _read_curve_columns(self, path: Path) -> tuple[pd.Series, pd.Series]:
        df = pd.read_csv(path)
        if "ratio" in df.columns and "time" in df.columns:
            return pd.to_numeric(df["ratio"], errors="coerce"), pd.to_numeric(df["time"], errors="coerce")

        raw = pd.read_csv(path, header=None)
        if raw.shape[1] >= 2:
            ratio = pd.to_numeric(raw.iloc[:, 0], errors="coerce")
            time = pd.to_numeric(raw.iloc[:, 1], errors="coerce")
            if np.isfinite(ratio.to_numpy()).sum() >= 2 and np.isfinite(time.to_numpy()).sum() >= 2:
                return ratio, time

        raw_local = pd.read_csv(path, header=None, sep=";", decimal=",")
        if raw_local.shape[1] < 2:
            raise ValueError(f"CSV '{path}' must contain at least two columns")
        return pd.to_numeric(raw_local.iloc[:, 0], errors="coerce"), pd.to_numeric(
            raw_local.iloc[:, 1], errors="coerce"
        )

    def _stage(self, stage_key: str) -> dict[str, Any]:
        value = self.settings.get(stage_key, {})
        return value if isinstance(value, dict) else {}

    def _is_stage_active(self, stage_key: str) -> bool:
        stage = self._stage(stage_key)
        if not stage:
            return False
        if not bool(stage.get("active", True)):
            return False
        return self._stage_mode(stage_key) != "off"

    @staticmethod
    def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        result = deepcopy(base)
        for key, value in (override or {}).items():
            if isinstance(value, dict) and isinstance(result.get(key), dict):
                result[key] = UniversalTCC._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
