"""Report generation for AC/DC substation calculations."""

from __future__ import annotations

import datetime as dt
import os
import re
from pathlib import Path
from typing import Any
from math import sqrt


def _safe_token(name: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_]", "_", str(name).strip())
    token = re.sub(r"_+", "_", token).strip("_")
    return token or "N"


class SubstationReport:
    """Generate text reports for AC/DC substation calculations."""

    NOMINAL_DC_V = 220.0

    def __init__(
        self,
        project_name: str,
        mode: str,
        source_cfg: dict[str, Any],
        battery_cfg: dict[str, Any],
        battery_params: dict[str, Any],
        line_rows: list[dict[str, Any]],
        fault: dict[str, Any],
        power_flow: dict[str, Any],
        source_bus: str,
        load_bus: str,
        protection_rows: list[dict[str, Any]],
    ) -> None:
        self.project_name = project_name
        self.mode = mode.upper()
        self.source_cfg = source_cfg
        self.battery_cfg = battery_cfg
        self.battery_params = battery_params
        self.line_rows = line_rows
        self.fault = fault
        self.power_flow = power_flow
        self.source_bus = source_bus
        self.load_bus = load_bus
        self.protection_rows = protection_rows

    def write(self, output_path: str) -> str:
        """Write report to file and return resolved path."""
        output_dir = Path(output_path).parent
        if not output_dir.exists():
            raise ValueError(f"Output directory does not exist: {output_dir}")
        if not os.access(output_dir, os.W_OK):
            raise PermissionError(f"Output directory is not writable: {output_dir}")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(self._render())

        return str(Path(output_path).resolve())

    def _render(self) -> str:
        sections = []
        date_str = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sections.append(f"Project Name: {self.project_name}")
        sections.append(f"Date: {date_str}\n")

        if self.mode == "DC":
            sections += self._section_battery()
        else:
            sections += self._section_ac_source()

        sections += self._section_lines()
        sections += self._section_currents()
        sections += self._section_voltages()
        sections += self._section_fault_currents()

        if self.protection_rows:
            sections += self._section_protection()

        return "\n".join(sections) + "\n"

    # ------------------------------------------------------------------
    # Source sections
    # ------------------------------------------------------------------

    def _section_battery(self) -> list[str]:
        cfg = self.battery_cfg
        params = self.battery_params
        return [
            "Battery Configuration",
            "---------------------",
            f"ID: {cfg.get('id', cfg.get('model_id'))}",
            f"Cells: {cfg.get('n_cells')}",
            f"Jumpers (mOhm): {cfg.get('jumpers_mohm', 0.5)}",
            f"Total Voltage (V): {float(params.get('U_total_V', 0.0)):.3f}",
            f"Total Resistance (Ohm): {float(params.get('R_total_ohm', 0.0)):.6f}",
            "",
        ]


    def _section_ac_source(self) -> list[str]:
        cfg = self.source_cfg
        xr = cfg.get("x_r_ratio", cfg.get("X1R1", 2.0))
        
        # Расчёт Z1 источника в омах
        basekv = float(cfg.get('basekv', 0.4))
        isc3 = cfg.get("isc3")
        mvasc3 = cfg.get("mvasc3")
        if isc3 is not None:
            isc3_a = float(isc3)
        elif mvasc3 is not None:
            isc3_a = float(mvasc3) * 1e6 / (sqrt(3.0) * basekv * 1000.0)
        else:
            isc3_a = 0.0
        
        if isc3_a > 0:
            z1 = (basekv * 1000.0) / (sqrt(3.0) * isc3_a)
            r1 = z1 / sqrt(1.0 + float(xr) ** 2)
            x1 = r1 * float(xr)
        else:
            r1 = x1 = 0.0

        rows = [
            "AC Source Configuration",
            "-----------------------",
            f"BasekV: {basekv:.3f}",
        ]
        ...
        rows += [
            f"Source pu: {float(cfg.get('pu', 1.0)):.3f}",
            f"X/R ratio: {float(xr):.3f}",
            f"R1 source (Ohm): {r1:.6f}",
            f"X1 source (Ohm): {x1:.6f}",
            f"Neutral Grounding: {cfg.get('neutral_grounding', 'grounded')}",
        ]
        if str(cfg.get("neutral_grounding", "")).lower() == "resistor_grounded":
            rows.append(f"Rneutral (Ohm): {float(cfg.get('r_neutral_ohm', 0.0)):.3f}")
        rows.append("")
        return rows  # ← эта строка должна быть

    # ------------------------------------------------------------------
    # Lines
    # ------------------------------------------------------------------

    def _section_lines(self) -> list[str]:
        rows = ["Lines", "-----"]
        for row in self.line_rows:
            rows.append(
                f"{row['name']}: {row['from_bus']} -> {row['to_bus']}, "
                f"R={row['R_ohm']:.6f} Ohm, X={row['X_ohm']:.6f} Ohm"
            )
        rows.append("")
        return rows

    # ------------------------------------------------------------------
    # Currents
    # ------------------------------------------------------------------

    def _section_currents(self) -> list[str]:
        line_phase_currents: dict[str, dict[str, float]] = self.power_flow.get("line_phase_currents_a", {})
        lut = {str(k).lower(): v for k, v in line_phase_currents.items()}

        rows = ["Normal Mode Currents", "--------------------"]

        if self.mode == "DC":
            rows.append("Line | I (A)")
            for row in self.line_rows:
                name = str(row["name"])
                phase_i = line_phase_currents.get(name, lut.get(name.lower(), {}))
                i = float(phase_i.get("I1_A", 0.0))
                rows.append(f"{name} | {i:.3f}")
        else:
            rows.append("Line | I1 (A) | I2 (A) | I3 (A) | Imax (A)")
            for row in self.line_rows:
                name = str(row["name"])
                phase_i = line_phase_currents.get(name, lut.get(name.lower(), {}))
                i1 = float(phase_i.get("I1_A", 0.0))
                i2 = float(phase_i.get("I2_A", 0.0))
                i3 = float(phase_i.get("I3_A", 0.0))
                imax = max(i1, i2, i3)
                rows.append(f"{name} | {i1:.3f} | {i2:.3f} | {i3:.3f} | {imax:.3f}")

        rows.append("")
        return rows

    # ------------------------------------------------------------------
    # Voltages
    # ------------------------------------------------------------------

    def _section_voltages(self) -> list[str]:
        bus_voltages: dict[str, float] = self.power_flow.get("bus_voltages_v", {})
        bus_phase_voltages: dict[str, dict[str, float]] = self.power_flow.get("bus_phase_voltages_v", {})
        nominal_v = float(self.power_flow.get("nominal_voltage_v", 0.0) or 0.0)

        rows = ["Voltage Profile", "---------------"]

        if self.mode == "DC":
            rows.append("Bus Name | Voltage (V) | Voltage (%) | Voltage Drop (V) | Voltage Drop (%)")
            for bus, voltage in sorted(bus_voltages.items()):
                voltage_pct = voltage / self.NOMINAL_DC_V * 100.0
                drop_v = nominal_v - voltage
                drop_pct = (drop_v / nominal_v * 100.0) if nominal_v > 0 else 0.0
                rows.append(f"{bus} | {voltage:.3f} | {voltage_pct:.1f} | {drop_v:.3f} | {drop_pct:.3f}")
        else:
            rows.append("Bus Name | V1N (V) | V2N (V) | V3N (V) | Vnom (V) | Max Drop (%)")
            for bus in sorted(bus_voltages.keys()):
                phase_v = bus_phase_voltages.get(bus, {})
                v1 = float(phase_v.get("V1N_V", 0.0))
                v2 = float(phase_v.get("V2N_V", 0.0))
                v3 = float(phase_v.get("V3N_V", 0.0))
                vmax = max(v1, v2, v3)

                if vmax > 1000:
                    vnom = 6000.0 / sqrt(3.0)
                elif vmax > 100:
                    vnom = 400.0 / sqrt(3.0)
                else:
                    vnom = vmax or 1.0

                drop = ((vnom - vmax) / vnom * 100.0) if vnom > 0 else 0.0
                rows.append(f"{bus} | {v1:.3f} | {v2:.3f} | {v3:.3f} | {vnom:.1f} | {drop:.3f}")

        rows.append("")
        return rows

    # ------------------------------------------------------------------
    # Fault currents
    # ------------------------------------------------------------------

    def _section_fault_currents(self) -> list[str]:
        bus_voltages: dict[str, float] = self.power_flow.get("bus_voltages_v", {})
        rows = ["Fault Currents", "--------------"]

        if self.mode == "AC" and isinstance(self.fault.get("buses"), dict):
            buses_data = self.fault["buses"]
            rows.append("Bus Name | Isc3 (A) | Isc1 (A)")
            for bus in sorted(bus_voltages.keys()):
                norm_bus = _safe_token(bus).lower()
                data = buses_data.get(norm_bus, {})
                isc3 = float(data.get("Isc3", 0.0))
                isc1 = float(data.get("Isc1", 0.0))
                rows.append(f"{bus} | {isc3:.3f} | {isc1:.3f}")
        else:
            fault_bus_values: dict[str, float] = {}
            for key, value in self.fault.items():
                if key in {"source_fault_current_a", "load_fault_current_a", "buses"}:
                    continue
                if isinstance(value, (int, float)):
                    fault_bus_values[str(key)] = float(value)

            rows.append("Bus Name | I_sc (A)")
            for bus in sorted(bus_voltages.keys()):
                norm_bus = _safe_token(bus).lower()
                current = fault_bus_values.get(norm_bus, 0.0)
                rows.append(f"{bus} | {current:.3f}")

        rows.append("")
        return rows

    # ------------------------------------------------------------------
    # Protection
    # ------------------------------------------------------------------

    def _section_protection(self) -> list[str]:
        rows = [
            "Protection Summary",
            "------------------",
            "Type | Device ID | Bus | I_sc (A) | t_min (s) | t_max (s)",
        ]
        for row in self.protection_rows:
            rows.append(
                f"{row['device_type']} | {row['device_id']} | {row['bus']} | "
                f"{row['fault_current_a']:.3f} | "
                f"{row['trip_time_min_s']:.6f} | {row['trip_time_max_s']:.6f}"
            )
        return rows
