"""Run all demo users and examples through the engine and compare with the stored outputs.
Usage (from the module root): python scripts/run_demo.py [--write]"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.ring_health_v3.engine import RingHealthV3  # noqa: E402


def short(res):
    out = []
    f = res.get("fertility")
    if f:
        out.append(f"fertility={f['status']}" + (f" day {f['cycle_day']} {f['phase']} P(fertile)={f['fertile_window']['today_probability']} ovulation={f['ovulation']['status']} {f['ovulation']['date']} next={f['next_period']['date']}" if f["status"] == "ok" else ""))
    m = res.get("menopause")
    if m:
        out.append(f"menopause={m['status']}" + (f" stage={m['stage']['label']} FMP<=12mo={m['final_menstrual_period']['probability_by_months']['12']}" if m["status"] == "ok" else ""))
    return " | ".join(out)


def main():
    eng = RingHealthV3()
    write = "--write" in sys.argv
    ok = bad = 0
    cases = [(p, p / "input.json", p / "output.json") for p in sorted((ROOT / "demo_users").iterdir()) if p.is_dir()]
    cases += [(p, p, Path(str(p).replace("_input.json", "_output.json"))) for p in sorted((ROOT / "examples").glob("*_input.json"))]
    for name, fin, fout in cases:
        res = eng.analyze(json.loads(fin.read_text()))
        if write:
            fout.write_text(json.dumps(res, ensure_ascii=False, indent=1))
        same = json.loads(fout.read_text()) == res
        ok += same; bad += (not same)
        print(("OK  " if same else "DIFF"), name.name, "->", short(res))
    print(f"\n{ok} identical, {bad} different")
    sys.exit(1 if bad and not write else 0)


if __name__ == "__main__":
    main()
