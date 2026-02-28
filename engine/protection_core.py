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

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


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
        self.min_time_s = float(self.settings.get("minimum_time_s", 0.001))
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

        # Если I-stage активен и ток выше его порога — сразу возвращаем t_instant
        if self._is_stage_active("I_stage"):
            i_base = self._stage_pickup_current("I_stage")
            tol = float(self._stage("I_stage").get("curr_tol_inst_pct", 0.0))
            t_instant = float(self._stage("I_stage").get("t_instant", 0.02))
            # Для расчёта времени используем базовый порог (без допуска по току)
            if eval_current >= i_base * (1.0 - tol / 100.0):
                return max(t_instant, self.min_time_s)

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

        # Порог I-stage с учётом допуска по току
        i_clip = i_high
        t_instant = self.min_time_s
        if self._is_stage_active("I_stage"):
            i_base = self._stage_pickup_current("I_stage")
            tol = float(self._stage("I_stage").get("curr_tol_inst_pct", 0.0))
            t_instant = float(self._stage("I_stage").get("t_instant", 0.02))
            t_instant_min = float(self._stage("I_stage").get("t_instant_min", t_instant))
            if mode == "min":
                i_clip = i_base * (1.0 - tol / 100.0)
                t_instant = t_instant_min   # <-- 0.002 для min-кривой
            elif mode == "max":
                i_clip = i_base * (1.0 + tol / 100.0)
            else:
                i_clip = i_base

        currents = np.logspace(np.log10(i_low), np.log10(i_high), 600)
        times = []
        for i in currents:
            fi = float(i)
            if self._is_stage_active("I_stage") and fi > i_clip:
                times.append(max(t_instant, self.min_time_s))
            elif self._is_stage_active("I_stage"):
                # До порога — считаем только L и S stage, I-stage игнорируем
                stage_times = []
                for stage_key in ("L_stage", "S_stage"):
                    t = self._evaluate_stage(stage_key, fi, mode)
                    if np.isfinite(t):
                        stage_times.append(float(t))
                if stage_times:
                    times.append(max(min(stage_times), self.min_time_s))
                else:
                    times.append(float("inf"))
            else:
                times.append(self.calculate_time(fi, mode=mode))
        times = np.array(times, dtype=float)

        x: list[float] = currents.tolist()
        y: list[float] = times.tolist()

        # Вертикальный переход L→I точно на i_clip
        if self._is_stage_active("I_stage") and i_low < i_clip < i_high:
            t_l = self._evaluate_stage("L_stage", i_clip, mode)
            if not np.isfinite(t_l):
                t_l = self.calculate_time(i_clip * 0.999, mode=mode)
            t_i = max(t_instant, self.min_time_s)

            insert_idx = int(np.searchsorted(np.array(x), i_clip))
            x.insert(insert_idx, float(i_clip))
            y.insert(insert_idx, max(float(t_l), self.min_time_s))
            x.insert(insert_idx + 1, float(i_clip))
            y.insert(insert_idx + 1, t_i)

        return x, y
    

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

        source_type = str(stage.get("source_type", "")).lower()
        i_pickup = self._stage_pickup_current(stage_key)
        if current_a < i_pickup:
            return float("inf")

        if source_type in {"csv", "2csv"}:
            time_s = self._time_from_csv(stage_key, current_a, mode)
        elif source_type == "iec_formula":
            time_s = self._time_from_iec_formula(stage_key, current_a)
        elif source_type == "constant":
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
            warnings.warn("No data in interps")
            return float("inf")

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

    def _time_from_iec_formula(self, stage_key: str, current_a: float) -> float:
        stage = self._stage(stage_key)
        ratio = self._current_ratio(current_a, stage_key)
        is_set = float(stage.get("Is_set", stage.get("Isd", 1.0)))
        if is_set <= 0:
            return float("inf")
        m = ratio / is_set
        if m <= 1.0:
            return float("inf")

        curve_name = str(stage.get("curve_type", "standard_inverse")).lower()
        alpha, beta = _IEC_CURVES.get(curve_name, _IEC_CURVES["standard_inverse"])
        k = float(stage.get("k_multiplier", 1.0))
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
        if stage_key == "L_stage":
            return float(stage.get("Ir_adjust", stage.get("Is_set", 1.0)))
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
        return self.in_a * self._stage_pickup_multiple(stage_key)

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
        tol = float(stage.get("time_tolerance_pct", self.settings.get("time_tolerance_pct", 0.0)))
        if mode == "min":
            return time_s * max(0.0, (1.0 - tol / 100.0))
        return time_s * (1.0 + tol / 100.0)

    def _build_log_interp(self, csv_path: str) -> Any:
        path = Path(csv_path)
        if not path.is_absolute():
            path = self.base_dir / path

        df = pd.read_csv(path)
        if "ratio" in df.columns and "time" in df.columns:
            ratio = pd.to_numeric(df["ratio"], errors="coerce")
            time = pd.to_numeric(df["time"], errors="coerce")
        else:
            raw = pd.read_csv(path, header=None)
            if raw.shape[1] < 2:
                raise ValueError(f"CSV '{path}' must contain at least two columns")
            ratio = pd.to_numeric(raw.iloc[:, 0], errors="coerce")
            time = pd.to_numeric(raw.iloc[:, 1], errors="coerce")

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

    def _stage(self, stage_key: str) -> dict[str, Any]:
        value = self.settings.get(stage_key, {})
        return value if isinstance(value, dict) else {}

    def _is_stage_active(self, stage_key: str) -> bool:
        stage = self._stage(stage_key)
        if not stage:
            return False
        return bool(stage.get("active", True))

    @staticmethod
    def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        result = deepcopy(base)
        for key, value in (override or {}).items():
            if isinstance(value, dict) and isinstance(result.get(key), dict):
                result[key] = UniversalTCC._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
