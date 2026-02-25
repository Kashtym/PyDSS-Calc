from __future__ import annotations

import json
import subprocess
import sys
import textwrap


def test_full_chain_dc_fault_integration() -> None:
    script = textwrap.dedent(
        """
        import json
        import re
        from unittest.mock import patch

        import pandas as pd

        from engine.db_manager import EquipmentManager
        from engine.models import BatteryModel, LineModel, LoadModel
        from engine.solver import DSSSolver

        def pick_id(available, preferred):
            available = [str(x).strip() for x in available if str(x).strip()]
            for item in preferred:
                if item in available:
                    return item
            low = {x.lower(): x for x in available}
            for item in preferred:
                if item.lower() in low:
                    return low[item.lower()]
            return None

        def pick_by_pattern(available, pattern):
            rx = re.compile(pattern)
            for item in available:
                if rx.search(item):
                    return item
            return available[0]

        def init_manager():
            try:
                return EquipmentManager()
            except ValueError as exc:
                if "Missing required sheet(s):" not in str(exc):
                    raise
                sheets = pd.read_excel(EquipmentManager._default_db_path(), sheet_name=None, engine="odf", skiprows=1)
                sheets["CircuitBreakers"] = pd.DataFrame({"ID": pd.Series(dtype=str)})
                sheets["Fuses"] = pd.DataFrame({"ID": pd.Series(dtype=str)})
                with patch("pandas.read_excel", return_value=sheets):
                    return EquipmentManager()

        def fix_schema(manager):
            if "ID" not in manager.cables_df.columns:
                if len(manager.cables_df.columns) >= 12:
                    cols = ["ID","Name","R20","X1","R0","X0","C1","I_adm","I_sc_1s","cond_material","insul_material","Diameter"]
                else:
                    cols = ["ID","R20","X1","R0","X0","C1","I_adm","I_sc_1s","cond_material","insul_material","Diameter"]
                n = min(len(cols), len(manager.cables_df.columns))
                manager.cables_df.columns = cols[:n] + [f"extra_{i}" for i in range(len(manager.cables_df.columns) - n)]

            if "ID" not in manager.batteries_df.columns:
                cols = ["ID", "Name", "Capacity", "U_nom", "Ri_cell", "I_sc_cell"]
                n = min(len(cols), len(manager.batteries_df.columns))
                manager.batteries_df.columns = cols[:n] + [f"extra_{i}" for i in range(len(manager.batteries_df.columns) - n)]

            manager.cables_df["ID"] = manager.cables_df["ID"].astype(str).str.strip()
            manager.batteries_df["ID"] = manager.batteries_df["ID"].astype(str).str.strip()

        try:
            print("[STEP 1] Database Loaded.")
            manager = init_manager()
            fix_schema(manager)

            battery_ids = manager.get_all_battery_ids()
            cable_ids = manager.get_all_cable_ids()

            battery_id = pick_id(battery_ids, ["BAE 6 OGi 300", "BAE_6_OGi_300"]) or pick_by_pattern(battery_ids, r"300")
            cable_id = pick_id(cable_ids, ["ВВГнг 2х35", "ВВГнг 2x35", "VVGng_2x35"]) or pick_by_pattern(cable_ids, r"35")

            raw_cell = manager.get_raw_cell(battery_id)
            raw_cable = manager.get_raw_cable(cable_id)

            battery_model = BatteryModel(raw_cell)
            line_model = LineModel(raw_cable)
            battery_params = battery_model.get_params(n_cells=104, jumpers_mohm=0.5)
            line_model.get_params(mode="DC", length_km=0.05, temperature=70.0)
            print("[STEP 2] Physical Models Created.")

            solver = DSSSolver()
            solver.setup_simulation(mode="DC")
            source_kv = float(battery_params["U_total_V"]) / 1000.0
            solver.add_element(battery_model, name="BAT1", bus1="SourceBus", n_cells=104, jumpers_mohm=0.5, phases=1, base_kv=source_kv)
            solver.add_element(line_model, name="L1", bus1="SourceBus", bus2="LoadBus", length_km=0.05, temperature=70.0, phases=1)
            solver.add_element(LoadModel(power_kw=5.0, voltage_kv=source_kv, pf=1.0), name="LD1", bus1="LoadBus", phases=1)
            print("[STEP 3] OpenDSS Model Built.")

            results = solver.run_fault_study(source_bus="SourceBus", load_bus="LoadBus")
            print(f"[STEP 4] Results: SourceBus = {results['source_fault_current_a']:.2f} Amps, LoadBus = {results['load_fault_current_a']:.2f} Amps.")

            payload = {
                "U_total_V": float(battery_params["U_total_V"]),
                "source_fault_current_a": float(results["source_fault_current_a"]),
                "load_fault_current_a": float(results["load_fault_current_a"]),
                "results": {k: float(v) for k, v in results.items()},
            }
            print("RESULT_JSON=" + json.dumps(payload, ensure_ascii=False))
        except KeyError as exc:
            print(f"ERROR: Missing ID or schema field: {exc}")
            raise
        except Exception as exc:
            print(f"ERROR: OpenDSS script error: {exc}")
            raise
        """
    )

    proc = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, f"Integration script failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"

    result_line = next((line for line in proc.stdout.splitlines() if line.startswith("RESULT_JSON=")), None)
    assert result_line is not None, f"No RESULT_JSON in output:\n{proc.stdout}"

    payload = json.loads(result_line.split("=", 1)[1])
    u_total = float(payload["U_total_V"])
    source_fault = float(payload["source_fault_current_a"])
    load_fault = float(payload["load_fault_current_a"])
    results = payload["results"]

    assert (216.0 <= u_total <= 228.0) or (200.0 <= u_total <= 212.0), f"Unexpected U_total_V={u_total}"
    assert source_fault > load_fault, f"Expected SourceBus > LoadBus. source={source_fault}, load={load_fault}"
    assert isinstance(results, dict) and results
    assert all(isinstance(value, float) for value in results.values())
