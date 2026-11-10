"""Single entry point: RingHealthV3().analyze(payload) -> dict (JSON-serialisable).

Artifacts are verified against models/registry/ring_health_v3.json (SHA256) and loaded once.
The engine is stateless and thread-safe (no per-request state on the instance).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import VERSION
from .cycle_engine import CycleModel
from .fertility import analyze_fertility
from .menopause import analyze_menopause
from .schema import AnalyzeRequest, Night

import os
# Module root (contains models/exported/ring_health/v3 and models/registry/ring_health_v3.json).
# Override with RING_HEALTH_V3_ROOT when the package is vendored elsewhere.
ROOT = Path(os.getenv("RING_HEALTH_V3_ROOT", Path(__file__).resolve().parents[2]))
NIGHT_COLS = ["temperature_deviation", "resting_hr", "hrv_ms", "respiratory_rate", "sleep_minutes", "sleep_efficiency", "awake_minutes"]


def normalize_oura(oura: dict) -> list[dict]:
    """Oura API v2 documents -> nights. Uses the longest long_sleep per Oura 'day'; temperature from daily_readiness."""
    rec = {}
    for r in oura.get("daily_readiness", []):
        day = r.get("day")
        if day is None:
            continue
        rec.setdefault(day, {"date": day})["temperature_deviation"] = r.get("temperature_deviation")
    best = {}
    for r in oura.get("sleep", []):
        if r.get("type") != "long_sleep" or r.get("day") is None:
            continue
        dur = r.get("total_sleep_duration") or 0
        if r["day"] not in best or dur > (best[r["day"]].get("total_sleep_duration") or 0):
            best[r["day"]] = r
    for day, r in best.items():
        x = rec.setdefault(day, {"date": day})
        x.update(resting_hr=r.get("lowest_heart_rate"), hrv_ms=r.get("average_hrv"), respiratory_rate=r.get("average_breath"),
                 sleep_minutes=(r["total_sleep_duration"] / 60) if r.get("total_sleep_duration") is not None else None,
                 sleep_efficiency=r.get("efficiency"), awake_minutes=(r["awake_time"] / 60) if r.get("awake_time") is not None else None)
    return [Night.model_validate(rec[k]).model_dump(mode="json") for k in sorted(rec)]


def nights_frame(rows, as_of, first_date=None):
    """Continuous daily calendar from min(first night, first_date) to as_of; unobserved nights stay NaN."""
    as_of = pd.Timestamp(as_of)
    if rows:
        d = pd.DataFrame(rows)
        d["date"] = pd.to_datetime(d["date"])
        for c in NIGHT_COLS:
            d[c] = pd.to_numeric(d[c], errors="coerce") if c in d else np.nan
        d = d.set_index("date").sort_index()
        start = d.index.min()
    else:
        d = pd.DataFrame(columns=NIGHT_COLS, index=pd.DatetimeIndex([], name="date"), dtype=float)
        start = as_of
    if first_date is not None:
        start = min(start, pd.Timestamp(first_date))
    start = max(start, as_of - pd.Timedelta(days=800))
    d = d.reindex(pd.date_range(start, as_of, freq="D")).rename_axis("date").reset_index()
    for c in NIGHT_COLS:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d[["date"] + NIGHT_COLS]


def coverage(d):
    out = {}
    for c in NIGHT_COLS:
        x = d[c].notna().to_numpy()
        tail = x[-30:]
        gaps = np.diff(np.r_[0, np.flatnonzero(np.r_[x, True])]) - 1 if len(x) else [0]
        out[c] = {"nights_last30": int(tail.sum()), "coverage_last30": round(float(tail.mean()), 2) if len(tail) else 0.0}
    t = d["temperature_deviation"].notna().to_numpy()
    runs, cur = [], 0
    for v in t[-90:]:
        cur = 0 if v else cur + 1; runs.append(cur)
    out["longest_temperature_gap_last90"] = int(max(runs) if runs else 0)
    return out


class RingHealthV3:
    def __init__(self, root: Path = ROOT, verify=True):
        import joblib
        import sklearn
        self.root = Path(root)
        self.dir = self.root / "models/exported/ring_health/v3"
        reg = json.loads((self.root / "models/registry/ring_health_v3.json").read_text())
        if verify:
            if reg["sklearn"] != sklearn.__version__:
                raise RuntimeError(f"scikit-learn {sklearn.__version__} != artifact version {reg['sklearn']}")
            for name, sha in reg["artifacts"].items():
                if hashlib.sha256((self.dir / name).read_bytes()).hexdigest() != sha:
                    raise RuntimeError("Artifact checksum mismatch: " + name)
        self.registry = reg
        self.cycle = CycleModel(json.loads((self.dir / "cycle_engine_params.json").read_text()))
        vms = joblib.load(self.dir / "menopause_vms_night.joblib")
        mm = joblib.load(self.dir / "menopause_models.joblib")
        self.meno = {"models": mm, "vms": vms}
        self.digest = hashlib.sha256((self.root / "models/registry/ring_health_v3.json").read_bytes()).hexdigest()

    def analyze(self, payload):
        req = payload if isinstance(payload, AnalyzeRequest) else AnalyzeRequest.model_validate(payload)
        prof = req.profile.model_dump()
        rows = [n.model_dump(mode="json") for n in req.nights] if req.nights is not None else (normalize_oura(req.oura.model_dump()) if req.oura else [])
        for r in rows:
            if pd.Timestamp(r["date"]).date() > req.as_of:
                raise ValueError("Night after as_of")
        starts_all = sorted(pd.Timestamp(x) for x in req.cycle.period_starts)
        d = nights_frame(rows, req.as_of, first_date=starts_all[0] if starts_all else None)
        starts = [s for s in starts_all if s >= d["date"].iloc[0]]  # cycle model window (<=800 days); menopause gets all starts
        lh = [(pd.Timestamp(t.date), t.result == "positive") for t in req.lh_tests]
        out = {"model_version": VERSION, "artifact_registry_sha256": self.digest, "as_of": req.as_of.isoformat(), "mode": "demonstration",
               "clinical_validation": "not_established", "training_data": "synthetic (literature- and real-data-calibrated generator)",
               "recommendations": [], "data_quality": {"nights_supplied": len(rows), "coverage": coverage(d)}}
        if "fertility" in req.modules:
            excl = [k for k in ["pregnant", "breastfeeding", "hormonal_contraception", "menopausal_hormone_therapy", "hysterectomy", "bilateral_oophorectomy"] if prof.get(k)]
            if excl:
                out["fertility"] = {"status": "out_of_scope", "reason": "Cycle model not applicable: " + ", ".join(excl)}
            elif prof["age"] > 50 or prof["age"] < 18:
                out["fertility"] = {"status": "not_applicable_age", "reason": "Cycle model evaluated for age 18-50"}
            elif not starts:
                out["fertility"] = {"status": "anchor_required", "needed": ["at least one period start date"]}
            else:
                rel = 0.97 if req.cycle.period_logs_complete else 0.9
                f = analyze_fertility(self.cycle, d, starts, req.as_of, age=prof["age"], lh=lh, log_reliability=rel, menses_length=req.cycle.menses_length)
                f.pop("_internal", None)
                out["fertility"] = f
        if "menopause" in req.modules:
            sym = req.symptoms.model_dump()
            labs = {"fsh": req.labs.fsh_iu_l.value if req.labs.fsh_iu_l else None, "amh": req.labs.amh_pg_ml.value if req.labs.amh_pg_ml else None}
            out["menopause"] = analyze_menopause(self.meno, self.cycle, d if rows else None, starts_all, req.as_of, prof,
                                                 last_period_date=req.cycle.last_period_date, logs_complete=req.cycle.period_logs_complete,
                                                 logging_since=req.cycle.logging_since, symptoms=sym, labs=labs)
        return json.loads(json.dumps(out, default=_json))


def _json(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp,)):
        return str(o.date())
    return str(o)
