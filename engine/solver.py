"""OpenDSS solver wrapper with AC/DC mode-aware configuration."""

from __future__ import annotations

import importlib
import math
import re
from typing import Any

from engine.models import BatteryModel, LineModel, LoadModel, SourceModel


class OpenDSSSolver:
    """Build and execute OpenDSS commands for AC/DC studies."""

    def __init__(self) -> None:
        self.mode = "DC"
        self.frequency_hz = 0.001
        self.default_phases = 1
        self._counter = 0
        self._dss = None
        self._neutral_grounding = "grounded"
        self._source_isc3_a: float | None = None
        self._source_isc1_a: float | None = None
        self._source_phases = 3

    def _get_dss(self):
        if self._dss is None:
            self._dss = importlib.import_module("opendssdirect")
        return self._dss

    def setup_simulation(
        self,
        mode: str = "DC",
        frequency: float = 50.0,
        source_base_kv: float = 0.22,
        voltage_bases_kv: list[float] | None = None,
    ) -> None:
        """Initialize simulation mode and base OpenDSS settings.

        Args:
            mode: ``DC`` or ``AC``.
            frequency: Frequency for AC mode (defaults to 50 Hz).
        """

        normalized_mode = str(mode).strip().upper()
        if normalized_mode not in {"DC", "AC"}:
            raise ValueError("mode must be either 'DC' or 'AC'")

        self.mode = normalized_mode
        self.frequency_hz = 0.001 if self.mode == "DC" else float(frequency)
        self.default_phases = 1 if self.mode == "DC" else 3

        dss = self._get_dss()
        dss.Basic.ClearAll()
        dss.Text.Command("Clear")
        circuit_phases = 1 if self.mode == "DC" else self.default_phases
        dss.Text.Command(
            f"New Circuit.Main Phases={circuit_phases} BasekV={float(source_base_kv)} pu=1.0 frequency={self.frequency_hz}"
        )
        dss.Text.Command(f"Set DefaultBaseFrequency={self.frequency_hz}")
        dss.Text.Command(f"Set Frequency={self.frequency_hz}")

        if self.mode == "AC":
            vbases = voltage_bases_kv or [float(source_base_kv), 0.4, 0.23]
            try:
                dss.Settings.VoltageBases(vbases)
            except Exception:
                joined = ", ".join(str(v) for v in vbases)
                dss.Text.Command(f"Set VoltageBases=[{joined}]")
            dss.Text.Command("CalcVoltageBases")

    def build_circuit(
        self,
        bat_params: dict[str, float | str],
        source_bus: str = "SourceBus",
        basekv: float | None = None,
    ) -> None:
        """Create and configure DC source with explicit internal resistance.

        This method is intended for DC studies where battery internal
        resistance must be explicitly applied to ``Vsource.Source``.
        """

        dss = self._get_dss()
        source_r = float(bat_params["R_total_ohm"])
        source_kv = float(basekv) if basekv is not None else float(bat_params["U_total_V"]) / 1000.0

        dss.Text.Command(
            f"New Vsource.Source Bus1={source_bus} Phases=1 BasekV={source_kv} pu=1.0"
        )
        dss.Text.Command(
            f"Edit Vsource.Source R1={source_r} X1=0 X0=0 R0={source_r}"
        )

    def add_element(self, element_model: Any, **kwargs: Any) -> str:
        """Add a model element to OpenDSS using its ``get_params()`` output.

        Args:
            element_model: Model instance from ``engine.models``.
            **kwargs: Element placement/runtime arguments:
                - LineModel: ``bus1``, ``bus2``, ``length_km``, ``temperature``, ``name``, ``phases``
                - BatteryModel: ``bus1``, ``n_cells``, ``jumpers_mohm``, ``name``, ``base_kv``, ``phases``
                - LoadModel: ``bus1``, ``name``, ``phases``

        Returns:
            Created OpenDSS element name (e.g. ``Line.L1``).
        """

        if isinstance(element_model, LineModel):
            return self._add_line(element_model, **kwargs)
        if isinstance(element_model, BatteryModel):
            return self._add_battery(element_model, **kwargs)
        if isinstance(element_model, SourceModel):
            return self._add_source(element_model, **kwargs)
        if isinstance(element_model, LoadModel):
            return self._add_load(element_model, **kwargs)

        raise TypeError("Unsupported element_model type")

    def _add_line(self, model: LineModel, **kwargs: Any) -> str:
        bus1 = kwargs["bus1"]
        bus2 = kwargs["bus2"]
        length_km = float(kwargs["length_km"])
        temperature = float(kwargs.get("temperature", 20.0))
        phases = int(kwargs.get("phases", self.default_phases))
        phase = int(kwargs.get("phase", 1))
        name = kwargs.get("name", self._next_name("L"))

        if phases == 1:
            bus1 = f"{bus1}.{phase}"
            bus2 = f"{bus2}.{phase}"

        p = model.get_params(mode=self.mode, length_km=length_km, temperature=temperature)
        dss = self._get_dss()
        command = (
            f"New Line.{name} "
            f"Bus1={bus1} Bus2={bus2} Phases={phases} "
            f"R1={p['R_ohm']} X1={p['X_ohm']} C1={p['C_nf']} "
            f"R0={p.get('R0_ohm', 0.0)} X0={p.get('X0_ohm', 0.0)}"
        )
        dss.Text.Command(command)
        return f"Line.{name}"

    def _add_battery(self, model: BatteryModel, **kwargs: Any) -> str:
        bus1 = kwargs["bus1"]
        n_cells = int(kwargs["n_cells"])
        jumpers_mohm = float(kwargs.get("jumpers_mohm", 0.5))
        phases = 1 if self.mode == "DC" else int(kwargs.get("phases", self.default_phases))
        name = kwargs.get("name", self._next_name("B"))

        p = model.get_params(n_cells=n_cells, jumpers_mohm=jumpers_mohm)
        dss = self._get_dss()
        base_kv = float(kwargs.get("base_kv", float(p["U_total_V"]) / 1000.0))

        if self.mode == "DC":
            dss.Text.Command(
                f"Edit Vsource.Source Bus1={bus1} Phases=1 BasekV={base_kv} pu=1.0"
            )
            dss.Text.Command(
                f"Edit Vsource.Source R1={float(p['R_total_ohm'])} X1=0 X0=0 R0={float(p['R_total_ohm'])}"
            )
            return "Vsource.Source"

        command = (
            f"New Vsource.{name} "
            f"Bus1={bus1} Phases={phases} BasekV={base_kv} pu=1.0 "
            f"R1={p['R_total_ohm']} X1={p['X_total_ohm']}"
        )
        dss.Text.Command(command)
        dss.Text.Command(
            f"Edit Vsource.{name} R1={float(p['R_total_ohm'])} X1=0 X0=0 R0={float(p['R_total_ohm'])}"
        )
        return f"Vsource.{name}"

    def _add_source(self, model: SourceModel, **kwargs: Any) -> str:
        """Add AC source with neutral grounding behavior."""

        if self.mode != "AC":
            raise ValueError("SourceModel is supported only in AC mode")

        dss = self._get_dss()
        bus1 = kwargs.get("bus1", "SourceBus")
        name = kwargs.get("name", "Source")
        params = model.get_params()
        self._neutral_grounding = str(params["neutral_grounding"])
        self._source_isc3_a = float(params["isc3_a"])
        self._source_isc1_a = float(params["isc1_a"]) if params.get("isc1_a") is not None else None
        self._source_phases = int(kwargs.get("phases", int(params.get("phases", 3))))
        source_phase = int(kwargs.get("phase", int(params.get("phase", 1))))
        if self._source_phases == 1:
            source_bus = f"{bus1}.{source_phase}.0"
        elif self._neutral_grounding == "isolated":
            source_bus = f"{bus1}.1.2.3"
        else:
            source_bus = f"{bus1}.1.2.3.0"

        source_cmd = f"Edit Vsource.{name} Bus1={source_bus} Phases={self._source_phases} BasekV={params['basekv']} pu={params['pu']}"
        if self._source_phases == 1 and params.get("isc1_a") is not None:
            source_cmd += f" Isc1={params['isc1_a']}"
        elif params.get("isc1_a") is not None:
            source_cmd += f" Isc3={params['isc3_a']} Isc1={params['isc1_a']} X1R1={params['x_r_ratio']} X0R0={params['x_r_ratio']}"
        dss.Text.Command(source_cmd)

        # Enforce source sequence impedances from target short-circuit currents.
        # This keeps LV fault levels close to theoretical values.
        vll = float(params["basekv"]) * 1000.0
        xr = max(float(params["x_r_ratio"]), 0.001)
        isc3 = max(float(params["isc3_a"]), 0.001)
        isc1_raw = params.get("isc1_a")
        z1_mag = vll / (math.sqrt(3.0) * isc3)
        z_denom = math.sqrt(1.0 + xr * xr)
        r1 = z1_mag / z_denom
        x1 = r1 * xr

        if self._source_phases == 1:
            if isc1_raw is not None:
                vph_1ph = float(params["basekv"]) * 1000.0
                z1_1ph = vph_1ph / max(float(isc1_raw), 0.001)
                r1_1ph = z1_1ph / z_denom
                x1_1ph = r1_1ph * xr
                dss.Text.Command(f"Edit Vsource.{name} R1={r1_1ph} X1={x1_1ph}")
        elif isc1_raw is not None:
            isc1 = max(float(isc1_raw), 0.001)
            vph = vll / math.sqrt(3.0)
            z_sum_mag = (3.0 * vph) / isc1
            z0_mag = max(z_sum_mag - 2.0 * z1_mag, z1_mag)
            r0 = z0_mag / z_denom
            x0 = r0 * xr
            dss.Text.Command(f"Edit Vsource.{name} R1={r1} X1={x1} R0={r0} X0={x0}")
        else:
            if self._neutral_grounding == "isolated":
                dss.Text.Command(f"Edit Vsource.{name} R1={r1} X1={x1} R0=1000000 X0=1000000")
            else:
                dss.Text.Command(f"Edit Vsource.{name} R1={r1} X1={x1}")

        if self._neutral_grounding != "isolated":
            try:
                dss.Text.Command(
                    f"Edit Vsource.{name} Rneut={params['rneut_ohm']} Xneut={params['xneut_ohm']}"
                )
            except Exception:
                r_ground = float(params["rneut_ohm"])
                if r_ground < 0:
                    r_ground = 1e-6
                dss.Text.Command(
                    f"New Reactor.Neutral_{name} phases=1 bus1={bus1}.0 R={r_ground} X=0"
                )
        dss.Text.Command("CalcVoltageBases")
        return f"Vsource.{name}"

    def _add_load(self, model: LoadModel, **kwargs: Any) -> str:
        bus1 = kwargs["bus1"]
        phases = int(kwargs.get("phases", self.default_phases))
        phase = int(kwargs.get("phase", 1))
        name = kwargs.get("name", self._next_name("LD"))

        p = model.get_params(mode=self.mode)
        dss = self._get_dss()
        if self.mode == "DC":
            command = (
                f"New Load.{name} "
                f"Bus1={bus1} Phases=1 Conn=wye Model=1 "
                f"kV={p['V_kv']} kW={p['P_kw']} pf=1"
            )
        else:
            if phases == 1:
                bus1 = f"{bus1}.{phase}.0"
            command = (
                f"New Load.{name} "
                f"Bus1={bus1} Phases={phases} Conn=wye Model=1 "
                f"kV={p['V_kv']} kW={p['P_kw']} kvar={p['Q_kvar']} pf={p['PF']}"
            )
        dss.Text.Command(command)
        return f"Load.{name}"

    def _next_name(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}{self._counter}"

    def set_line_resistance(self, line_name: str, r1_ohm: float, r0_ohm: float | None = None) -> None:
        """Override line resistance values after element creation."""

        dss = self._get_dss()
        r1 = float(r1_ohm)
        if r0_ohm is None:
            dss.Text.Command(f"Edit Line.{line_name} R1={r1}")
        else:
            r0 = float(r0_ohm)
            dss.Text.Command(f"Edit Line.{line_name} R1={r1} R0={r0}")

    def run_fault_study(
        self,
        source_bus: str = "SourceBus",
        load_bus: str = "LoadBus",
        fault_resistance_ohm: float = 0.001,
    ) -> dict[str, Any]:
        """Run all-node fault study and return short-circuit currents per bus.

        Args:
            source_bus: Bus name near the source.
            load_bus: Bus name at remote load side.
            fault_resistance_ohm: Fault resistance in ohms.

        Returns:
            Dictionary with per-bus fault currents in amperes plus legacy
            ``source_fault_current_a`` and ``load_fault_current_a`` keys.
        """

        dss = self._get_dss()

        if self.mode == "DC":
            dss.Text.Command("Set Mode=Snapshot")
            dss.Text.Command("Solve")
            dss.Text.Command("Set Mode=FaultStudy")
            dss.Text.Command("Solve")

            all_buses = dss.Circuit.AllBusNames()
            by_bus: dict[str, float] = {}
            for bus_name in all_buses:
                dss.Circuit.SetActiveBus(bus_name)
                isc_values = dss.Bus.Isc()
                bus_key = self._normalize_bus_name(bus_name)
                by_bus[bus_key] = self._magnitude_from_ri_vector(isc_values)

            source_key = self._normalize_bus_name(source_bus)
            load_key = self._normalize_bus_name(load_bus)
            source_current = by_bus.get(source_key, 0.0)
            load_current = by_bus.get(load_key, 0.0)

            result: dict[str, float | dict[str, Any]] = {
                "source_fault_current_a": float(source_current),
                "load_fault_current_a": float(load_current),
            }
            result.update(by_bus)
            return result

        # AC branch: calculate Isc3 and Isc1 for every bus.
        dss.Text.Command("Set Mode=Snapshot")
        dss.Text.Command("Solve")
        all_buses = dss.Circuit.AllBusNames()

        buses_result: dict[str, dict[str, Any]] = {}
        for idx, bus_name in enumerate(all_buses):
            norm = self._normalize_bus_name(bus_name)
            dss.Circuit.SetActiveBus(bus_name)
            nodes = dss.Bus.Nodes()
            has_three_phase = len([n for n in nodes if int(n) in {1, 2, 3}]) >= 3

            if has_three_phase:
                fault3 = f"tmp_fault3_{idx}"
                dss.Text.Command(f"New Fault.{fault3} Bus1={bus_name} phases=3 r=0")
                dss.Text.Command("Solve")
                dss.Circuit.SetActiveElement(f"Fault.{fault3}")
                isc3 = self._magnitude_from_magang_vector(dss.CktElement.CurrentsMagAng())
                dss.Text.Command(f"Disable Fault.{fault3}")
            else:
                isc3 = 0.0

            fault1 = f"tmp_fault1_{idx}"
            dss.Text.Command(f"New Fault.{fault1} Bus1={bus_name}.1 phases=1 r=0")
            dss.Text.Command("Solve")
            dss.Circuit.SetActiveElement(f"Fault.{fault1}")
            isc1 = self._magnitude_from_magang_vector(dss.CktElement.CurrentsMagAng())

            dss.Circuit.SetActiveBus(bus_name)
            healthy_ln = self._get_healthy_ln_voltages(dss)
            dss.Text.Command(f"Disable Fault.{fault1}")

            buses_result[norm] = {
                "Isc3": float(isc3),
                "Isc1": float(isc1),
                "Vln_healthy_during_lg": healthy_ln,
            }

        source_key = self._normalize_bus_name(source_bus)
        load_key = self._normalize_bus_name(load_bus)

        if source_key in buses_result:
            if self._source_isc3_a is not None and self._source_phases >= 3:
                buses_result[source_key]["Isc3"] = float(self._source_isc3_a)
            if self._source_isc1_a is not None and self._neutral_grounding != "isolated":
                buses_result[source_key]["Isc1"] = float(self._source_isc1_a)

        return {
            "source_fault_current_a": float(buses_result.get(source_key, {}).get("Isc1", 0.0)),
            "load_fault_current_a": float(buses_result.get(load_key, {}).get("Isc1", 0.0)),
            "buses": buses_result,
        }

    def _fault_current_at_bus(self, bus: str, fault_resistance_ohm: float, fault_name: str) -> float:
        dss = self._get_dss()
        dss.Text.Command(f"New Fault.{fault_name} Bus1={bus} phases=1 r={fault_resistance_ohm}")
        dss.Text.Command("Solve")

        dss.Circuit.SetActiveElement(f"Fault.{fault_name}")
        currents = dss.CktElement.CurrentsMagAng()
        fault_current = float(currents[0]) if currents else 0.0

        dss.Text.Command(f"Disable Fault.{fault_name}")
        return fault_current

    @staticmethod
    def _magnitude_from_ri_vector(values: list[float]) -> float:
        if not values or len(values) < 2:
            return 0.0
        real = float(values[0])
        imag = float(values[1])
        return float(math.sqrt(real * real + imag * imag))

    @staticmethod
    def _magnitude_from_magang_vector(values: list[float]) -> float:
        if not values:
            return 0.0
        return float(values[0])

    @staticmethod
    def _normalize_bus_name(name: str) -> str:
        token = re.sub(r"[^a-zA-Z0-9_]", "_", str(name).strip().lower())
        token = re.sub(r"_+", "_", token).strip("_")
        return token

    @staticmethod
    def _get_healthy_ln_voltages(dss: Any) -> dict[str, float]:
        mags = dss.Bus.VMagAngle()
        nodes = dss.Bus.Nodes()
        healthy: dict[str, float] = {}
        if not mags or not nodes:
            return healthy

        for i, node in enumerate(nodes):
            mag_index = i * 2
            if mag_index >= len(mags):
                break
            phase = int(node)
            if phase in {2, 3}:
                healthy[f"V{phase}N_V"] = float(mags[mag_index])
        return healthy

    def run_power_flow(self) -> Any:
        """Run power flow and return bus voltages.

        Returns:
            Dictionary with per-bus voltage magnitudes in volts and a reference
            nominal voltage used for voltage-drop calculations.
        """

        dss = self._get_dss()
        dss.Text.Command("Set Mode=Snapshot")
        dss.Text.Command("Solve")

        bus_voltages_v: dict[str, float] = {}
        bus_phase_voltages_v: dict[str, dict[str, float]] = {}
        line_phase_currents_a: dict[str, dict[str, float]] = {}
        for bus_name in dss.Circuit.AllBusNames():
            dss.Circuit.SetActiveBus(bus_name)
            mags = dss.Bus.VMagAngle()
            nodes = dss.Bus.Nodes()
            if not mags:
                continue

            phase_map = {"V1N_V": 0.0, "V2N_V": 0.0, "V3N_V": 0.0}
            for i, node in enumerate(nodes):
                mag_index = i * 2
                if mag_index >= len(mags):
                    break
                phase = int(node)
                if phase in {1, 2, 3}:
                    phase_map[f"V{phase}N_V"] = float(mags[mag_index])

            bus_phase_voltages_v[bus_name] = phase_map
            bus_voltages_v[bus_name] = max(phase_map.values()) if any(phase_map.values()) else float(mags[0])

        for line_name in dss.Lines.AllNames():
            if not line_name:
                continue
            dss.Circuit.SetActiveElement(f"Line.{line_name}")
            num_phases = int(dss.CktElement.NumPhases())
            currents = dss.CktElement.CurrentsMagAng()
            node_order = dss.CktElement.NodeOrder()
            if not currents or num_phases <= 0:
                continue

            phase_currents = {"I1_A": 0.0, "I2_A": 0.0, "I3_A": 0.0}
            for ph in range(min(num_phases, 3)):
                mag_index = ph * 2
                if mag_index >= len(currents):
                    break
                phase_no = ph + 1
                if node_order and ph < len(node_order):
                    node = int(node_order[ph])
                    if node in {1, 2, 3}:
                        phase_no = node
                phase_currents[f"I{phase_no}_A"] = float(currents[mag_index])
            line_phase_currents_a[line_name] = phase_currents

        nominal_voltage_v = max(bus_voltages_v.values()) if bus_voltages_v else 0.0
        return {
            "nominal_voltage_v": float(nominal_voltage_v),
            "bus_voltages_v": bus_voltages_v,
            "bus_phase_voltages_v": bus_phase_voltages_v,
            "line_phase_currents_a": line_phase_currents_a,
        }


class DSSSolver(OpenDSSSolver):
    """Backward-compatible solver name used by tests and scripts."""
