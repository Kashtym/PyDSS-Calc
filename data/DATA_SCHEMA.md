## Database Schema: library.ods

The equipment database is stored in a multi-sheet `.ods` file. All electrical parameters follow OpenDSS standards (Ohm/km, nF/km).

### Sheet 1: `Cables` (Universal library for DC and AC)

Contains physical and electrical properties of conductors. Data source: Manufacturer catalogs (e.g., Odeskabel, Yuzhcable).

-   **ID**: Unique string identifier (e.g., `VVGng_2x1.5`).
-   **Designation**: Designation (e.g., `ВВГнг 2х1.5`).
-   **R20**: Positive sequence active resistance at 20°C [Ohm/km].
-   **X1**: Positive sequence inductive reactance at 50Hz [Ohm/km]. (Used for AC; forced to 0 for DC).
-   **R0**: Zero sequence active resistance [Ohm/km]. (Critical for single-phase-to-ground AC faults).
-   **X0**: Zero sequence inductive reactance [Ohm/km].
-   **C1**: Positive sequence capacitance [nF/km].
-   **I_adm**: Maximum continuous current rating in air [A].
-   **I_sc_1s**: 1-second short-circuit thermal limit [kA].
-   **cond_material**: Conductor material (`Cu` or `Al`). Used for temperature coefficient calculation.
-   **insul_material**: Insulation material type (`PVC` or `XLPE`). Defines maximum operating temperature (70°C for PVC, 90°C for XLPE).
-   **Diameter**: Outer cable diameter [mm]. (Used for cable tray filling checks).

### Sheet 2: `Batteries` (DC Source components)

Contains parameters for 2V battery cells (units to be scaled by N_cells in code).

-   **ID**: Cell model name (e.g., `BAE_2_OGi_50`).
-   **Designation**: Designation (e.g., `BAE 2 OGi 50`).
-   **Capacity**: Nominal capacity at C10 rate [Ah].
-   **U_nom**: Nominal cell voltage [V] (typically 2V).
-   **Ri_cell**: Internal resistance per cell [mOhm] (Critical for SC calc).
-   **I_sc_cell**: Manufacturer-stated short-circuit current [A].
