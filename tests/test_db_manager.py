from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from engine.db_manager import EquipmentManager


def _mock_sheets() -> dict[str, pd.DataFrame]:
    return {
        "Cables": pd.DataFrame(
            [
                {
                    "ID": "VVGng_2x1.5",
                    "R20": 12.1,
                    "X1": 0.08,
                    "R0": 12.1,
                    "X0": 0.08,
                    "C1": 45.0,
                    "I_adm": 27,
                    "I_sc_1s": 0.5,
                    "cond_material": "Cu",
                    "insul_material": "PVC",
                    "Diameter": 9.2,
                }
            ]
        ),
        "Batteries": pd.DataFrame(
            [
                {
                    "ID": "BAE_6_OGi_300",
                    "Capacity": 300,
                    "U_nom": 2.0,
                    "Ri_cell": 0.42,
                    "I_sc_cell": 4700,
                }
            ]
        ),
        "CircuitBreakers": pd.DataFrame(
            [
                {
                    "ID": "QF1",
                    "I_n": 250,
                    "Type": "MCCB",
                    "Curve": "curve_a.csv",
                }
            ]
        ),
        "Fuses": pd.DataFrame(
            [
                {
                    "ID": "FU1",
                    "In": 10,
                    "P_loss_W": 2.0,
                    "curve": "fuse_curve.csv",
                }
            ]
        ),
    }


def _build_manager() -> EquipmentManager:
    with patch("os.path.isfile", return_value=True), patch(
        "pandas.read_excel", return_value=_mock_sheets()
    ):
        return EquipmentManager("/tmp/library.ods")


def test_init_uses_odf_and_skiprows() -> None:
    with patch("os.path.isfile", return_value=True), patch(
        "pandas.read_excel", return_value=_mock_sheets()
    ) as read_excel_mock:
        EquipmentManager("/tmp/library.ods")

    read_excel_mock.assert_called_once_with(
        "/tmp/library.ods",
        sheet_name=None,
        engine="odf",
        skiprows=1,
    )


def test_init_raises_when_file_missing() -> None:
    with patch("os.path.isfile", return_value=False):
        with pytest.raises(FileNotFoundError):
            EquipmentManager("/missing/library.ods")


def test_init_raises_when_sheet_missing() -> None:
    incomplete = _mock_sheets()
    incomplete.pop("CircuitBreakers")

    with patch("os.path.isfile", return_value=True), patch(
        "pandas.read_excel", return_value=incomplete
    ):
        with pytest.raises(ValueError, match="Missing required sheet"):
            EquipmentManager("/tmp/library.ods")


def test_get_raw_cable_returns_raw_row() -> None:
    manager = _build_manager()

    result = manager.get_raw_cable("VVGng_2x1.5")
    assert result["ID"] == "VVGng_2x1.5"
    assert result["R20"] == pytest.approx(12.1)
    assert result["X1"] == pytest.approx(0.08)
    assert result["cond_material"] == "Cu"


def test_get_raw_cell_returns_raw_row() -> None:
    manager = _build_manager()

    result = manager.get_raw_cell("BAE_6_OGi_300")
    assert result["ID"] == "BAE_6_OGi_300"
    assert result["U_nom"] == pytest.approx(2.0)
    assert result["Ri_cell"] == pytest.approx(0.42)


def test_get_raw_breaker_returns_raw_row() -> None:
    manager = _build_manager()

    result = manager.get_raw_breaker("QF1")
    assert result == {"ID": "QF1", "I_n": 250, "Type": "MCCB", "Curve": "curve_a.csv"}


def test_get_raw_fuse_returns_raw_row() -> None:
    manager = _build_manager()

    result = manager.get_raw_fuse("FU1")
    assert result["ID"] == "FU1"
    assert result["In"] == pytest.approx(10)


def test_list_id_methods() -> None:
    manager = _build_manager()

    assert manager.get_all_cable_ids() == ["VVGng_2x1.5"]
    assert manager.get_all_battery_ids() == ["BAE_6_OGi_300"]
    assert manager.get_all_breaker_ids() == ["QF1"]
    assert manager.get_all_fuse_ids() == ["FU1"]


def test_raw_methods_raise_keyerror_for_unknown_ids() -> None:
    manager = _build_manager()

    with pytest.raises(KeyError, match="not found"):
        manager.get_raw_cable("NO_CABLE")

    with pytest.raises(KeyError, match="not found"):
        manager.get_raw_cell("NO_CELL")

    with pytest.raises(KeyError, match="not found"):
        manager.get_raw_breaker("NO_BREAKER")

    with pytest.raises(KeyError, match="not found"):
        manager.get_raw_fuse("NO_FUSE")
