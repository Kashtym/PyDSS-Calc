# Open DSS Calculator

Simple CLI application for AC/DC network calculations with OpenDSS:
- power flow,
- fault currents,
- protection trip times,
- text report and TCC plot generation.

## Requirements

- Python 3.11+
- Poetry

## Install

1. Clone/download repository.
2. Install dependencies:

```bash
poetry install
```

## Run Calculation

Use a YAML project file:

```bash
poetry run python main_cli.py projects/test_system_AC_04kV/config.yaml
```

Or for another project:

```bash
poetry run python main_cli.py projects/test_system_AC_10kV/config.yaml

# Standalone TCC debug (curves/intersections)
poetry run python tcc_debug_cli.py projects/tcc_debug/config.yaml
```

## Output Files

After successful run, files are created in the same folder as the YAML config:

- `{project_name}_report.txt` - calculation report
- `{project_name}_tcc.png` - protection TCC plot

Example:

- `projects/test_system_AC_04kV/Substation_LV_Panel_report.txt`
- `projects/test_system_AC_04kV/Substation_LV_Panel_tcc.png`

## YAML Notes (Minimum)

- `project_name`
- `calculation_mode`: `AC` or `DC`
- `source` (for AC) or `battery` (for DC)
- `topology`: list of lines
- `load`: list of loads

### Protection settings (new format)

You can define protection per topology segment using `protection_settings`.

Breaker example (catalog + runtime setpoints):

```yaml
protection_settings:
  name: "QF1"
  type: Breaker
  breaker_id: "NSX630N_3P_ML_5.3_630A"
  L_stage:
    active: true
    Ir: 0.8
    tr: 16.0
  S_stage:
    active: true
    mode: i2t_on
    Isd: 5.0
    tsd: 0.2
  I_stage:
    active: true
    Ii: 10.0
```

Relay example (absolute current units):

```yaml
protection_settings:
  name: "Relay feeder"
  type: Relay
  L_stage:
    active: true
    mode: iec_inverse
    curve_type: standard_inverse
    I1_A: 1000
    t1_s: 0.5
  S_stage:
    active: true
    mode: flat
    I2_A: 4000
    tsd: 0.1
```

Fuse example:

```yaml
protection_settings:
  name: "FU1"
  type: Fuse
  fuse_id: "PNA000_10A_2P"
```

Phase options (AC):
- By default source/line/load are 3-phase.
- For single-phase set:
  - `phases: 1`
  - `phase: 1|2|3`

## Run Tests

```bash
poetry run pytest -q
```

## Typical Issues

- **File not found**: check YAML path and file extension (`.yaml` vs `.yalm`).
- **ID not found**: verify `cable_id` / `breaker_id` / `fuse_id` against `data/library.ods`.
- **No output image/report**: ensure the project folder is writable.
