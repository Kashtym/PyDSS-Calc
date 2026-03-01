"""Physical component models used between database and OpenDSS solver."""

from __future__ import annotations

from dataclasses import dataclass
from math import acos, sqrt, tan
from typing import Any
import math


_TEMP_COEFF_BY_MATERIAL = {
    "cu": 0.004,
    "copper": 0.004,
    "al": 0.00403,
    "aluminum": 0.00403,
    "aluminium": 0.00403,
}


def _normalize_mode(mode: str) -> str:
    normalized = str(mode).strip().upper()
    if normalized not in {"DC", "AC"}:
        raise ValueError("mode must be either 'DC' or 'AC'")
    return normalized


@dataclass
class LineModel:
    """Electrical model of a cable segment.

    Args:
        raw_cable: Raw cable row from ``EquipmentManager.get_raw_cable``.
    """

    raw_cable: dict[str, Any]

    def get_params(self, mode: str, length_km: float, temperature: float = 20.0) -> dict[str, float | str]:
        """Return mode-specific line parameters for OpenDSS.

        Resistance correction:
            ``R_t = R20 * (1 + alpha * (temperature - 20))``

        - For DC: ``X=0`` and ``C=0``.
        - For AC: uses ``X1`` and ``C1`` and also returns ``R0`` and ``X0``.

        All resistance/reactance values are returned in Ohms.
        """

        calc_mode = _normalize_mode(mode)
        length = float(length_km)
        r20 = float(self.raw_cable["R20"])
        x1 = float(self.raw_cable.get("X1", 0.0))
        
        # Якщо R0 або X0 не задані (0.0), використовуємо коефіцієнт 3 для стабільності
        r0_raw = float(self.raw_cable.get("R0", 0.0))
        r0 = r0_raw if r0_raw > 0 else r20 * 3.0
        
        x0_raw = float(self.raw_cable.get("X0", 0.0))
        x0 = x0_raw if x0_raw > 0 else x1 * 3.0
        
        # Гарантуємо, що значення не нульові (мінімальний поріг для чисельної стійкості)
        x1 = max(x1, 1e-6)
        x0 = max(x0, 1e-6)


        c1 = float(self.raw_cable.get("C1", 0.0))

        material = str(self.raw_cable.get("cond_material", "Cu")).strip().lower()
        alpha = _TEMP_COEFF_BY_MATERIAL.get(material, 0.004)

        r_t_per_km = r20 * (1.0 + alpha * (float(temperature) - 20.0))
        r0_t_per_km = r0 * (1.0 + alpha * (float(temperature) - 20.0))

        if calc_mode == "DC":
            return {
                "mode": "DC",
                "R_ohm": r_t_per_km * length,
                "X_ohm": 0.0,
                "C_nf": 0.0,
                "R0_ohm": 0.0,
                "X0_ohm": 0.0,
            }

        return {
            "mode": "AC",
            "R_ohm": r_t_per_km * length,
            "X_ohm": x1 * length,
            "C_nf": c1 * length,
            "R0_ohm": r0_t_per_km * length,
            "X0_ohm": x0 * length,
        }


@dataclass
class BatteryModel:
    """Electrical model of a battery stack assembled from cells.

    Args:
        raw_cell: Raw battery row from ``EquipmentManager.get_raw_cell``.
    """

    raw_cell: dict[str, Any]

    def get_params(self, n_cells: int, jumpers_mohm: float = 0.5) -> dict[str, Any]:
        """Return equivalent battery stack parameters.

        ``U_total = n_cells * U_nom``
        ``R_total = (n_cells * Ri_cell + jumpers_mohm) / 1000``

        Returns values suitable for DC source and AC equivalent source
        impedance representation.
        """

        cells = int(n_cells)
        u_nom = float(self.raw_cell["U_nom"])
        ri_cell_mohm = float(self.raw_cell["Ri_cell"])
        capacity = float(self.raw_cell.get("Capacity", self.raw_cell.get("Capacity_Ah", 0.0)))

        u_total = cells * u_nom
        r_total_ohm = (cells * ri_cell_mohm + float(jumpers_mohm)) / 1000.0

        return {
            "U_total_V": u_total,
            "R_total_ohm": r_total_ohm,
            "X_total_ohm": 0.0,
            "Capacity_Ah": capacity,
            "Z_ac_eq": {"R_ohm": r_total_ohm, "X_ohm": 0.0},
        }


@dataclass
class LoadModel:
    """Load model helper for DC/AC input normalization."""

    power_kw: float
    voltage_kv: float
    pf: float = 1.0

    def get_params(self, mode: str) -> dict[str, float | str]:
        """Return mode-specific load parameters for OpenDSS model building.

        For AC mode, includes power factor and derived reactive power.
        """

        calc_mode = _normalize_mode(mode)
        p_kw = float(self.power_kw)
        v_kv = float(self.voltage_kv)

        if calc_mode == "DC":
            current_a = (p_kw * 1000.0) / (v_kv * 1000.0) if v_kv > 0 else 0.0
            return {
                "mode": "DC",
                "P_kw": p_kw,
                "V_kv": v_kv,
                "I_A": current_a,
            }

        pf = float(self.pf)
        if not 0 < pf <= 1:
            raise ValueError("PF must be in the range (0, 1]")

        q_kvar = p_kw * tan(acos(pf))
        s_kva = p_kw / pf
        i_a_3ph = (s_kva * 1000.0) / (sqrt(3.0) * v_kv * 1000.0) if v_kv > 0 else 0.0

        return {
            "mode": "AC",
            "P_kw": p_kw,
            "Q_kvar": q_kvar,
            "S_kva": s_kva,
            "V_kv": v_kv,
            "PF": pf,
            "I_A_3ph": i_a_3ph,
        }


@dataclass
class SourceModel:
    """AC source model with short-circuit levels and neutral grounding settings."""

    basekv: float
    mvasc3: float | None = None
    mvasc1: float | None = None
    isc3_a: float | None = None
    isc1_a: float | None = None
    pu: float = 1.0
    x_r_ratio: float = 2.0
    neutral_grounding: str = "grounded"
    r_neutral_ohm: float | None = None
    phases: int = 3
    phase: int = 1

    def get_params(self) -> dict[str, Any]:
        grounding = str(self.neutral_grounding).strip().lower()
        if grounding not in {"grounded", "isolated", "resistor_grounded"}:
            raise ValueError("neutral_grounding must be grounded, isolated, or resistor_grounded")

        if grounding == "grounded":
            rneut = -1.0
            connection = "wye"
        elif grounding == "isolated":
            rneut = 1e6
            connection = "wye"
        else:
            if self.r_neutral_ohm is None:
                raise ValueError("r_neutral_ohm is required for resistor_grounded mode")
            rneut = float(self.r_neutral_ohm)
            connection = "wye"

        if self.isc3_a is not None:
            isc3 = float(self.isc3_a)
        elif self.mvasc3 is not None:
            isc3 = float(self.mvasc3) * 1_000_000.0 / (sqrt(3.0) * float(self.basekv) * 1000.0)
        else:
            raise ValueError("Either isc3_a or mvasc3 must be provided for AC source")

        if self.isc1_a is not None:
            isc1: float | None = float(self.isc1_a)
        elif self.mvasc1 is not None:
            isc1 = float(self.mvasc1) * 1_000_000.0 / (sqrt(3.0) * float(self.basekv) * 1000.0)
        else:
            isc1 = None

        return {
            "basekv": float(self.basekv),
            "isc3_a": float(isc3),
            "isc1_a": isc1,
            "pu": float(self.pu),
            "x_r_ratio": float(self.x_r_ratio),
            "neutral_grounding": grounding,
            "connection": connection,
            "rneut_ohm": float(rneut),
            "xneut_ohm": 0.0,
            "phases": int(self.phases),
            "phase": int(self.phase),
        }

@dataclass
class TransformerModel:
    """Two-winding transformer model for AC studies.

    Args:
        raw_transformer: Raw row from ``EquipmentManager.get_raw_transformer``.
    """

    raw_transformer: dict[str, Any]

    def get_params(self) -> dict[str, Any]:
        """Return transformer parameters for OpenDSS and report.

        Returns impedances in Ohms referred to LV side,
        plus OpenDSS-ready percent values.
        """

        raw = self.raw_transformer
        sn_kva = float(raw["Sn"])
        un_hv = float(raw["Un_HV"])
        un_lv = float(raw["Un_LV"])
        ukz_pct = float(raw["Ukz"])
        pkz_kw = float(raw.get("Pkz", 0.0))
        pnh_kw = float(raw.get("Pnh", 0.0))
        i0_pct = float(raw.get("I0_pct", 0.0) or 0.0)
        conn_hv = str(raw.get("Conn_HV", "D")).strip().upper()
        conn_lv = str(raw.get("Conn_LV", "Y")).strip().upper()
        windings = int(raw.get("Wind", 2))
        i0_pct = raw.get("I0_pct", 0.0)
        try:
            i0_pct = float(i0_pct)
            if not math.isfinite(i0_pct):
                i0_pct = 0.0
        except (TypeError, ValueError):
            i0_pct = 0.0


        # Базовые величины со стороны НН
        i_base_lv = (sn_kva * 1000.0) / (sqrt(3.0) * un_lv * 1000.0)
        z_base_lv = (un_lv * 1000.0) / (sqrt(3.0) * i_base_lv)

        # Полное сопротивление КЗ
        z_t = (ukz_pct / 100.0) * z_base_lv

        # Активное сопротивление из потерь КЗ
        r_t = (pkz_kw * 1000.0) / (3.0 * i_base_lv ** 2)

        # Реактивное сопротивление
        x_t = sqrt(max(z_t ** 2 - r_t ** 2, 0.0))

        # %R и %X для OpenDSS (от мощности трансформатора)
        pct_r = (r_t / z_base_lv) * 100.0
        pct_x = (x_t / z_base_lv) * 100.0

        # Схема соединения для OpenDSS: Y→wye, D→delta
        conn_map = {"Y": "wye", "YN": "wye", "D": "delta"}
        conn_hv_dss = conn_map.get(conn_hv, "wye")
        conn_lv_dss = conn_map.get(conn_lv, "wye")
        conn_hv = str(self.raw_transformer.get("Conn_HV", "Y")).strip() 
        conn_lv = str(self.raw_transformer.get("Conn_LV", "Yn")).strip() 

        return {
            "windings": windings,
            "sn_kva": sn_kva,
            "un_hv_kv": un_hv,
            "un_lv_kv": un_lv,
            "conn_hv": conn_hv_dss,
            "conn_lv": conn_lv_dss,
            "conn_hv_raw": conn_hv, 
            "conn_lv_raw": conn_lv,  
            "ukz_pct": ukz_pct,
            "pct_r": pct_r,
            "pct_x": pct_x,
            "pct_noloadloss": (pnh_kw / sn_kva) * 100.0,
            "pct_imag": i0_pct,
            # Сопротивления в омах (сторона НН) — для отчёта
            "R_ohm_lv": r_t,
            "X_ohm_lv": x_t,
            "Z_ohm_lv": z_t,
        }