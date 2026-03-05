from __future__ import annotations

import argparse
import traceback

from engine.app.run_project import run


def main() -> int:
    parser = argparse.ArgumentParser(description="Run AC/DC substation calculations from YAML project file")
    parser.add_argument("project", help="Path to project YAML file")
    args = parser.parse_args()

    try:
        report_path = run(args.project)
        print(f"Report generated: {report_path}")
        return 0
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
