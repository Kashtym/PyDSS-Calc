"""Shared constants used across AC/DC calculation modules."""

from __future__ import annotations

# OpenDSS workaround: use near-zero frequency for DC snapshots.
DC_EQUIVALENT_FREQUENCY_HZ = 0.001

# Default study configuration.
DEFAULT_FAULT_RESISTANCE_OHM = 0.001
DEFAULT_MINIMUM_TRIP_TIME_S = 0.001

# Plot defaults for TCC curves.
TCC_PLOT_I_MIN_A = 1.0
TCC_PLOT_I_MAX_A = 10000.0
TCC_PLOT_TIME_MIN_S = 0.001
TCC_PLOT_TIME_MAX_S = 1000.0
TCC_PLOT_POINTS = 700

# Voltage reference values for reports.
NOMINAL_DC_VOLTAGE_V = 220.0
AC_VNOM_400V_V = 400.0
AC_VNOM_6KV_V = 6000.0
