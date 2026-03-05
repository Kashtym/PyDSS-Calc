"""Database access layer for equipment catalogs stored in ODS format."""

from __future__ import annotations

import os
from typing import Any

import pandas as pd


class EquipmentManager:
    """Load and provide raw equipment data from ``data/library.ods``.

    Notes:
        - The second row in each sheet contains units/notes and is skipped
          using ``skiprows=1``.
        - This class performs no physical calculations and returns raw values
          from the database.
    """

    REQUIRED_SHEETS = {
        "cables": "Cables",
        "batteries": "Batteries",
        "fuses": "Fuses",
    }

    OPTIONAL_SHEETS = {
        "transformers": "Transformers",
        "tripunitscatalog": "TripUnitsCatalog",
    }

    def __init__(self, db_path: str | None = None) -> None:
        """Initialize manager and load all required sheets into DataFrames.

        Args:
            db_path: Optional path to ``library.ods``. If not provided, uses
                ``<project_root>/data/library.ods``.

        Raises:
            FileNotFoundError: If the ODS file does not exist.
            ValueError: If any required sheet is missing.
        """

        self.db_path = db_path or self._default_db_path()
        if not os.path.isfile(self.db_path):
            raise FileNotFoundError(f"Database file not found: {self.db_path}")

        sheets = pd.read_excel(
            self.db_path,
            sheet_name=None,
            engine="odf",
            skiprows=1,
        )
        normalized = {self._normalize_name(name): df for name, df in sheets.items()}

        missing = [
            original
            for key, original in self.REQUIRED_SHEETS.items()
            if key not in normalized
        ]
        has_breakers = "circuitbreakers" in normalized or "circuitbreakerscatalog" in normalized
        if not has_breakers:
            missing.append("CircuitBreakers or CircuitBreakersCatalog")
        if missing:
            raise ValueError("Missing required sheet(s): " + ", ".join(missing))

        self.cables_df = self._sanitize_dataframe(normalized["cables"])
        self.batteries_df = self._sanitize_dataframe(normalized["batteries"])
        breakers_key = "circuitbreakers" if "circuitbreakers" in normalized else "circuitbreakerscatalog"
        self.breakers_df = self._sanitize_dataframe(normalized[breakers_key])
        self.fuses_df = self._sanitize_dataframe(normalized["fuses"])

        # Optional sheets — не вызывают ошибку если отсутствуют
        if "transformers" in normalized:
            self.transformers_df = self._sanitize_dataframe(normalized["transformers"])
        else:
            self.transformers_df = pd.DataFrame(columns=[
                "ID", "Designation", "Wind", "Sn", "Un_HV", "Conn_HV",
                "Un_LV", "Conn_LV", "Ukz", "Pnh", "Pkz", "I0_pct",
            ])

        trip_key = "triputitscatalog" if "triputitscatalog" in normalized else "tripunitscatalog"
        if trip_key in normalized:
            self.trip_units_df = self._sanitize_dataframe(normalized[trip_key])
        else:
            self.trip_units_df = pd.DataFrame(columns=["TripUnit_ID"])

    # ------------------------------------------------------------------
    # Cables
    # ------------------------------------------------------------------

    def get_raw_cable(self, cable_id: str) -> dict[str, Any]:
        """Return raw cable row from the ``Cables`` sheet by ID."""
        row = self._get_row_by_id(self.cables_df, cable_id, "Cables")
        return self._series_to_dict(row)

    def get_all_cable_ids(self) -> list[str]:
        """Return all cable IDs for UI selectors."""
        return self._get_all_ids(self.cables_df, "Cables")

    # ------------------------------------------------------------------
    # Batteries
    # ------------------------------------------------------------------

    def get_raw_cell(self, cell_id: str) -> dict[str, Any]:
        """Return raw battery cell row from the ``Batteries`` sheet by ID."""
        row = self._get_row_by_id(self.batteries_df, cell_id, "Batteries")
        return self._series_to_dict(row)

    def get_all_battery_ids(self) -> list[str]:
        """Return all battery IDs for UI selectors."""
        return self._get_all_ids(self.batteries_df, "Batteries")

    # ------------------------------------------------------------------
    # Circuit breakers
    # ------------------------------------------------------------------

    def get_raw_breaker(self, breaker_id: str) -> dict[str, Any]:
        """Return raw breaker row from the ``CircuitBreakers`` sheet by ID."""
        row = self._get_row_by_id(self.breakers_df, breaker_id, "CircuitBreakers")
        return self._series_to_dict(row)

    def get_raw_breaker_with_trip_unit(self, breaker_id: str) -> dict[str, Any]:
        """Return breaker row merged with linked trip unit row when available."""
        breaker = self.get_raw_breaker(breaker_id)
        trip_unit_id = str(breaker.get("TripUnit_ID", breaker.get("trip_unit_id", ""))).strip()
        if not trip_unit_id:
            return breaker
        try:
            trip = self.get_raw_trip_unit(trip_unit_id)
        except (KeyError, ValueError):
            return breaker

        merged = dict(breaker)
        for key, value in trip.items():
            if key == "TripUnit_ID":
                continue
            merged.setdefault(key, value)
            merged[f"trip_{key}"] = value
        return merged

    def get_all_breaker_ids(self) -> list[str]:
        """Return all breaker IDs for UI selectors."""
        return self._get_all_ids(self.breakers_df, "CircuitBreakers")

    def get_raw_trip_unit(self, trip_unit_id: str) -> dict[str, Any]:
        """Return raw trip unit row from ``TripUtitsCatalog`` by ID."""
        row = self._get_row_by_id(self.trip_units_df, trip_unit_id, "TripUtitsCatalog", id_column="TripUnit_ID")
        return self._series_to_dict(row)

    def get_all_trip_unit_ids(self) -> list[str]:
        """Return all trip unit IDs for UI selectors."""
        return self._get_all_ids(self.trip_units_df, "TripUtitsCatalog", id_column="TripUnit_ID")

    def load_db(self) -> pd.DataFrame:
        """Return merged breaker + trip unit table by ``TripUnit_ID``."""
        if self.breakers_df.empty:
            return self.breakers_df.copy()
        if self.trip_units_df.empty or "TripUnit_ID" not in self.trip_units_df.columns:
            return self.breakers_df.copy()

        left = self.breakers_df.copy()
        right = self.trip_units_df.copy()
        merged = left.merge(right, on="TripUnit_ID", how="left", suffixes=("", "_trip"))
        return merged

    # ------------------------------------------------------------------
    # Fuses
    # ------------------------------------------------------------------

    def get_raw_fuse(self, fuse_id: str) -> dict[str, Any]:
        """Return raw fuse row from the ``Fuses`` sheet by ID."""
        row = self._get_row_by_id(self.fuses_df, fuse_id, "Fuses")
        return self._series_to_dict(row)

    def get_all_fuse_ids(self) -> list[str]:
        """Return all fuse IDs for UI selectors."""
        return self._get_all_ids(self.fuses_df, "Fuses")

    # ------------------------------------------------------------------
    # Transformers
    # ------------------------------------------------------------------

    def get_raw_transformer(self, transformer_id: str) -> dict[str, Any]:
        """Return raw transformer row from the ``Transformers`` sheet by ID.

        Expected columns:
            ID, Designation, Wind, Sn (kVA), Un_HV (kV), Conn_HV (D/Y),
            Un_LV (kV), Conn_LV (D/Y), Ukz (%), Pnh (kW), Pkz (kW), I0_pct (%)
        """
        row = self._get_row_by_id(self.transformers_df, transformer_id, "Transformers")
        return self._series_to_dict(row)

    def get_all_transformer_ids(self) -> list[str]:
        """Return all transformer IDs for UI selectors."""
        return self._get_all_ids(self.transformers_df, "Transformers")

    def has_transformers(self) -> bool:
        """Return True if Transformers sheet is loaded and non-empty."""
        return not self.transformers_df.empty

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _default_db_path() -> str:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(base_dir, "data", "library.ods")

    @staticmethod
    def _normalize_name(name: str) -> str:
        return "".join(ch for ch in str(name).strip().lower() if ch.isalnum())

    @staticmethod
    def _sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        cleaned = df.copy()
        cleaned.columns = [str(col).strip() for col in cleaned.columns]
        cleaned = cleaned.dropna(how="all")
        if "ID" in cleaned.columns:
            cleaned["ID"] = cleaned["ID"].astype(str).str.strip()
        return cleaned

    @staticmethod
    def _get_row_by_id(
        df: pd.DataFrame,
        item_id: str,
        sheet_name: str,
        id_column: str = "ID",
    ) -> pd.Series:
        if id_column not in df.columns:
            raise ValueError(f"Sheet '{sheet_name}' does not contain required '{id_column}' column")

        target_id = str(item_id).strip()
        mask = df[id_column].astype(str).str.strip() == target_id
        if not mask.any():
            raise KeyError(f"ID '{target_id}' not found in sheet '{sheet_name}'")

        return df.loc[mask].iloc[0]

    @staticmethod
    def _get_all_ids(df: pd.DataFrame, sheet_name: str, id_column: str = "ID") -> list[str]:
        if id_column not in df.columns:
            raise ValueError(f"Sheet '{sheet_name}' does not contain required '{id_column}' column")
        ids = df[id_column].dropna().astype(str).str.strip()
        return [item for item in ids.tolist() if item]

    @staticmethod
    def _series_to_dict(row: pd.Series) -> dict[str, Any]:
        return {str(key): value for key, value in row.items()}
