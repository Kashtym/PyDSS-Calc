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
