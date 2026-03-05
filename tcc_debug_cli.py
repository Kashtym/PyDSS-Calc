from __future__ import annotations

import argparse
import traceback

from engine.tcc_debug import run_tcc_debug


def main() -> int:
    parser = argparse.ArgumentParser(description="Run standalone TCC debug from YAML")
    parser.add_argument("config", help="Path to TCC debug YAML file")
    args = parser.parse_args()
    try:
        report = run_tcc_debug(args.config)
        print(f"TCC debug report: {report}")
        return 0
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
