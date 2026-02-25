"""Protection and selectivity calculations based on relative TCC curves."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from engine.db_manager import EquipmentManager


@dataclass
class TCCEvaluator:
    """Evaluate time-current characteristics in relative (I/In) form.

    The input curve CSV must contain:
    - ``ratio``: current multiple ``I / I_n``
    - ``time``: trip time in seconds
    """

    csv_path: str
    equipment_manager: EquipmentManager | None = None

    def __post_init__(self) -> None:
        self._manager = self.equipment_manager or EquipmentManager()

        df = pd.read_csv(self.csv_path)
        if "ratio" not in df.columns or "time" not in df.columns:
            df_raw = pd.read_csv(self.csv_path, header=None)
            if df_raw.shape[1] < 2:
                raise ValueError("TCC CSV must contain at least two columns: ratio, time")
            df = pd.DataFrame({"ratio": df_raw.iloc[:, 0], "time": df_raw.iloc[:, 1]})

        clean = df[["ratio", "time"]].dropna().copy()
        clean["ratio"] = pd.to_numeric(clean["ratio"], errors="coerce")
        clean["time"] = pd.to_numeric(clean["time"], errors="coerce")
        clean = clean.replace([np.inf, -np.inf], np.nan).dropna()
        clean = pd.DataFrame(clean[(clean["ratio"] > 0) & (clean["time"] > 0)])
        clean = clean.sort_values("ratio")

        if clean.empty:
            raise ValueError("TCC CSV has no valid positive points for interpolation")

        self._ratio_min = float(clean["ratio"].iloc[0])
        self._ratio_max = float(clean["ratio"].iloc[-1])

        x = np.log10(clean["ratio"].to_numpy(dtype=float))
        y = np.log10(clean["time"].to_numpy(dtype=float))
        self._log_interp = interp1d(x, y, kind="linear", bounds_error=False, fill_value="extrapolate")

    def evaluate_by_ratio(self, ratio: float) -> float:
        """Return trip time [s] for a relative current multiple ``M = I/In``."""

        m = float(ratio)
        if m <= 0:
            raise ValueError("Current ratio must be > 0")
        # Keep behavior stable near limits while still allowing extrapolation.
        m_eval = min(max(m, self._ratio_min), self._ratio_max)
        log_t = float(self._log_interp(np.log10(m_eval)))
        return float(10 ** log_t)

    def calculate_trip_time(self, breaker_id: str, fault_current: float) -> dict[str, float | str]:
        """Calculate breaker trip time from fault current using its nominal current.

        Steps:
        1. Read ``In`` from breaker database row.
        2. Compute ``M = fault_current / In``.
        3. Evaluate trip time from relative TCC curve at ``M``.
        """

        row = self._manager.get_raw_breaker(breaker_id)
        in_a = self._get_numeric(row, ["In", "I_n", "In_A", "I_nom_A"])
        i_fault = float(fault_current)
        if i_fault <= 0:
            raise ValueError("fault_current must be > 0")

        ratio = i_fault / in_a
        trip_time_s = self.evaluate_by_ratio(ratio)
        return {
            "breaker_id": breaker_id,
            "fault_current_a": i_fault,
            "in_a": in_a,
            "ratio": ratio,
            "trip_time_s": trip_time_s,
        }

    def calculate_breaker_resistance(self, breaker_id: str) -> dict[str, float | str]:
        """Calculate breaker internal resistance for network impedance integration.

        Formula:
            ``R_breaker = (P_loss_W / In^2) * Poles``
        """

        row = self._manager.get_raw_breaker(breaker_id)
        p_loss_w = self._get_numeric(row, ["P_loss_W", "P_loss", "Ploss_W"])
        in_a = self._get_numeric(row, ["In", "I_n", "In_A", "I_nom_A"])
        poles = self._get_numeric(row, ["Poles", "N_poles", "PoleCount"])

        if in_a <= 0:
            raise ValueError(f"Invalid In for breaker '{breaker_id}': {in_a}")
        if poles <= 0:
            raise ValueError(f"Invalid Poles for breaker '{breaker_id}': {poles}")

        r_breaker = (p_loss_w / (in_a**2)) * poles
        return {
            "breaker_id": breaker_id,
            "p_loss_w": p_loss_w,
            "in_a": in_a,
            "poles": poles,
            "r_breaker_ohm": float(r_breaker),
        }

    def check_coordination(
        self,
        downstream_max_curve: TCCEvaluator,
        upstream_min_curve: TCCEvaluator,
        downstream_breaker_id: str,
        upstream_breaker_id: str,
        fault_current: float,
    ) -> dict[str, float | bool | str]:
        """Check selectivity at one fault current using max/min relative curves.

        Compares ``Downstream_Max_Curve(M_down)`` with
        ``Upstream_Min_Curve(M_up)`` at the same absolute fault current.
        """

        down = downstream_max_curve.calculate_trip_time(downstream_breaker_id, fault_current)
        up = upstream_min_curve.calculate_trip_time(upstream_breaker_id, fault_current)

        t_down = float(down["trip_time_s"])
        t_up = float(up["trip_time_s"])
        is_selective = t_up > t_down

        return {
            "fault_current_a": float(fault_current),
            "downstream_breaker_id": downstream_breaker_id,
            "upstream_breaker_id": upstream_breaker_id,
            "m_down": float(down["ratio"]),
            "m_up": float(up["ratio"]),
            "t_downstream_max_s": t_down,
            "t_upstream_min_s": t_up,
            "is_selective": is_selective,
            "coordination_margin_s": t_up - t_down,
        }

    @staticmethod
    def _get_numeric(row: dict[str, Any], candidates: list[str]) -> float:
        for key in candidates:
            if key in row and row[key] is not None:
                try:
                    return float(row[key])
                except (TypeError, ValueError):
                    continue
        raise KeyError(f"None of expected columns found: {candidates}")


@dataclass
class FuseEvaluator:
    """Evaluate fuse operation using a base relative TCC with +/-10% tolerance."""

    csv_path: str
    equipment_manager: EquipmentManager | None = None

    def __post_init__(self) -> None:
        self._manager = self.equipment_manager or EquipmentManager()
        self._curve = TCCEvaluator(self.csv_path, equipment_manager=self._manager)

    def get_fuse_trip_range(self, fuse_id: str, fault_current: float) -> tuple[float, float]:
        """Return (time_min, time_max) with +/-10% current tolerance.

        - ``M = I_fault / I_nominal``
        - ``M_min = M / 0.9`` (faster operation)
        - ``M_max = M / 1.1`` (slower operation)
        """

        row = self._manager.get_raw_fuse(fuse_id)
        #? in_a = TCCEvaluator._get_numeric(row, ["In", "I_n", "In_A", "I_nom_A"])  номинальный ток для пересчета кривой если задана в относительных единицах
        i_fault = float(fault_current)
        if i_fault <= 0:
            raise ValueError("fault_current must be > 0")

        #? m = i_fault / in_a если кривая в относительных единицах
        m = i_fault
        m_min = m / 0.9
        m_max = m / 1.1

        time_min = self._curve.evaluate_by_ratio(m_min)
        time_max = self._curve.evaluate_by_ratio(m_max)
        return (float(time_min), float(time_max))

    def calculate_fuse_resistance(self, fuse_id: str) -> dict[str, float | str]:
        """Calculate fuse resistance: ``R_fuse = P_loss_W / In^2``."""

        row = self._manager.get_raw_fuse(fuse_id)
        p_loss_w = TCCEvaluator._get_numeric(row, ["P_loss_W", "P_loss", "Ploss_W"])
        in_a = TCCEvaluator._get_numeric(row, ["In", "I_n", "In_A", "I_nom_A"])
        if in_a <= 0:
            raise ValueError(f"Invalid In for fuse '{fuse_id}': {in_a}")

        r_fuse = p_loss_w / (in_a**2)
        return {
            "fuse_id": fuse_id,
            "p_loss_w": p_loss_w,
            "in_a": in_a,
            "r_fuse_ohm": float(r_fuse),
        }
