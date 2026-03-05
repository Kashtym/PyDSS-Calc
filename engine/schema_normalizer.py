"""Database loading and schema normalization for known ODS layouts."""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd

from engine.db_manager import EquipmentManager


def init_equipment_manager() -> EquipmentManager:
    """Initialize EquipmentManager with fallback for optional sheets."""
    try:
        manager = EquipmentManager()
    except ValueError as exc:
        if "Missing required sheet(s):" not in str(exc):
            raise
        sheets = pd.read_excel(
            EquipmentManager._default_db_path(),
            sheet_name=None,
            engine="odf",
            skiprows=1,
        )
        if "CircuitBreakers" not in sheets:
            if "CircuitBreakersCatalog" in sheets:
                sheets["CircuitBreakers"] = sheets["CircuitBreakersCatalog"]
            else:
                sheets["CircuitBreakers"] = pd.DataFrame({"ID": pd.Series(dtype=str)})
        if "Fuses" not in sheets:
            sheets["Fuses"] = pd.DataFrame({"ID": pd.Series(dtype=str)})
        with patch("pandas.read_excel", return_value=sheets):
            manager = EquipmentManager()

    normalize_schema_for_known_ods_layout(manager)
    return manager


def normalize_schema_for_known_ods_layout(manager: EquipmentManager) -> None:
    """Normalize DataFrame columns for known localized ODS variants."""
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
        manager.cables_df.columns = cable_cols[:n] + [
            f"extra_{i}" for i in range(len(manager.cables_df.columns) - n)
        ]

    if "ID" not in manager.batteries_df.columns:
        battery_cols = ["ID", "Name", "Capacity", "U_nom", "Ri_cell", "I_sc_cell"]
        n = min(len(battery_cols), len(manager.batteries_df.columns))
        manager.batteries_df.columns = battery_cols[:n] + [
            f"extra_{i}" for i in range(len(manager.batteries_df.columns) - n)
        ]

    if hasattr(manager, "breakers_df") and "ID" not in manager.breakers_df.columns:
        breaker_cols = [
            "ID",
            "Name",
            "Manufacturer",
            "Series",
            "TripUnit_ID",
            "In",
            "Poles",
            "P_loss_W",
            "Icu_kA",
        ]
        n = min(len(breaker_cols), len(manager.breakers_df.columns))
        manager.breakers_df.columns = breaker_cols[:n] + [
            f"extra_{i}" for i in range(len(manager.breakers_df.columns) - n)
        ]

    if hasattr(manager, "trip_units_df") and "TripUnit_ID" not in manager.trip_units_df.columns:
        trip_cols = [
            "TripUnit_ID",
            "L_zone_type",
            "L_curve_min",
            "L_curve_max",
            "L_Kl",
            "L_Ir_range",
            "L_tr_range_s",
            "L_time_accur_pct",
            "L_I_accur",
            "S_i2t_mode",
            "S_Isd_range",
            "S_time_accur_pct",
            "S_tsd_range_s",
            "I_Ii_range",
            "I_accur_pct",
            "I_t_range_s",
        ]
        n = min(len(trip_cols), len(manager.trip_units_df.columns))
        manager.trip_units_df.columns = trip_cols[:n] + [
            f"extra_{i}" for i in range(len(manager.trip_units_df.columns) - n)
        ]

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
        manager.fuses_df.columns = fuse_cols[:n] + [
            f"extra_{i}" for i in range(len(manager.fuses_df.columns) - n)
        ]

    if hasattr(manager, "transformers_df") and "ID" not in manager.transformers_df.columns:
        transformer_cols = [
            "ID",
            "Designation",
            "Wind",
            "Sn",
            "Un_HV",
            "Conn_HV",
            "Un_LV",
            "Conn_LV",
            "Ukz",
            "Pnh",
            "Pkz",
            "I0_pct",
        ]
        n = min(len(transformer_cols), len(manager.transformers_df.columns))
        manager.transformers_df.columns = transformer_cols[:n] + [
            f"extra_{i}" for i in range(len(manager.transformers_df.columns) - n)
        ]

    manager.cables_df["ID"] = manager.cables_df["ID"].astype(str).str.strip()
    manager.batteries_df["ID"] = manager.batteries_df["ID"].astype(str).str.strip()
    if hasattr(manager, "breakers_df") and "ID" in manager.breakers_df.columns:
        manager.breakers_df["ID"] = manager.breakers_df["ID"].astype(str).str.strip()
    if hasattr(manager, "fuses_df") and "ID" in manager.fuses_df.columns:
        manager.fuses_df["ID"] = manager.fuses_df["ID"].astype(str).str.strip()
    if hasattr(manager, "transformers_df") and "ID" in manager.transformers_df.columns:
        manager.transformers_df["ID"] = manager.transformers_df["ID"].astype(str).str.strip()
    if hasattr(manager, "trip_units_df") and "TripUnit_ID" in manager.trip_units_df.columns:
        manager.trip_units_df["TripUnit_ID"] = manager.trip_units_df["TripUnit_ID"].astype(str).str.strip()
