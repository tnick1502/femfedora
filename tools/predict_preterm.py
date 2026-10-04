"""Run the preterm-birth model for one JSON input from a file or stdin."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference.preterm_woman import get_feature_importance, predict_woman


def main() -> None:
    parser = argparse.ArgumentParser(description="Forecast preterm birth for one ongoing singleton pregnancy")
    parser.add_argument("input", nargs="?", default="-", help="Input JSON file, or - for stdin")
    parser.add_argument("--importance", action="store_true", help="Include global model feature importance")
    args = parser.parse_args()
    source = sys.stdin if args.input == "-" else Path(args.input).open(encoding="utf-8")
    try:
        snapshot = json.load(source)
    finally:
        if source is not sys.stdin:
            source.close()
    result = predict_woman(snapshot)
    if args.importance and result["status"] == "OK":
        result["feature_importance"] = get_feature_importance()
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
