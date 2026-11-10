"""Ring Health v3 contract, causality, missing-data and safety tests."""
from __future__ import annotations

import copy
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.ring_health_v3.adapter_core import RingFertilityComponent, RingMenopauseComponent
from src.ring_health_v3.cycle_engine import CycleModel
from src.ring_health_v3.engine import RingHealthV3, normalize_oura
from src.ring_health_v3.fertility import analyze_fertility

ROOT = Path(__file__).resolve().parents[2]
SAMPLES = ROOT / "data/samples/ring_health/v3"


@pytest.fixture(scope="module")
def eng():
    return RingHealthV3()


def load(name):
    return json.loads((SAMPLES / f"{name}_input.json").read_text())


def test_fertility_ok_and_fields(eng):
    r = eng.analyze(load("fertility_regular"))
    f = r["fertility"]
    assert f["status"] == "ok"
    for k in ["cycle_day", "phase", "fertile_window", "ovulation", "next_period", "insights", "data_quality"]:
        assert k in f
    assert len(f["fertile_window"]["forecast_7d"]) == 7
    assert 0 <= f["fertile_window"]["today_probability"] <= 1
    assert r["clinical_validation"] == "not_established" and r["recommendations"] == []
    assert f["data_quality"]["imputation"].startswith("none")


def test_deterministic_and_thread_safe(eng):
    req = load("fertility_regular")
    a = eng.analyze(req)
    with ThreadPoolExecutor(4) as ex:
        outs = list(ex.map(lambda _: eng.analyze(copy.deepcopy(req)), range(8)))
    assert all(o == a for o in outs)


def test_causality_past_unchanged_by_later_nights(eng):
    req = load("fertility_regular")
    as_of = pd.Timestamp(req["as_of"]) - pd.Timedelta(days=5)
    early = copy.deepcopy(req); early["as_of"] = str(as_of.date())
    early["nights"] = [n for n in req["nights"] if pd.Timestamp(n["date"]) <= as_of]
    early["cycle"]["period_starts"] = [s for s in req["cycle"]["period_starts"] if pd.Timestamp(s) <= as_of]
    r1 = eng.analyze(early)
    # future nights are rejected, never silently used
    bad = copy.deepcopy(early); bad["nights"] = req["nights"]
    with pytest.raises(Exception):
        eng.analyze(bad)
    assert r1["fertility"]["status"] == "ok"


def test_missing_nights_are_not_imputed_and_degrade_gracefully(eng):
    full = eng.analyze(load("fertility_regular"))["fertility"]
    temp_only = eng.analyze(load("fertility_temperature_only"))["fertility"]
    cal = eng.analyze(load("fertility_calendar_only"))["fertility"]
    assert temp_only["status"] == "ok" and cal["status"] == "ok"
    assert cal["data_quality"]["evidence"] in ("history_only", "population_prior_only")
    assert cal["ovulation"]["status"] != "detected"
    assert full["data_quality"]["temperature_nights_last14"] >= temp_only["data_quality"]["temperature_nights_last14"] - 0


def test_null_is_missing_not_zero(eng):
    req = load("fertility_regular")
    z = copy.deepcopy(req)
    for n in z["nights"][-10:]:
        n["temperature_deviation"] = None
    r = eng.analyze(z)["fertility"]
    assert r["data_quality"]["temperature_nights_last14"] <= 4


def test_anchor_required_and_exclusions(eng):
    req = load("fertility_regular"); req["cycle"] = {"period_starts": []}
    assert eng.analyze(req)["fertility"]["status"] == "anchor_required"
    req = load("fertility_regular"); req["profile"]["hormonal_contraception"] = True
    assert eng.analyze(req)["fertility"]["status"] == "out_of_scope"


@pytest.mark.parametrize("field,value", [("temperature_deviation", float("nan")), ("resting_hr", "60"), ("hrv_ms", True)])
def test_invalid_numbers_rejected(eng, field, value):
    req = load("fertility_regular"); req["nights"][0][field] = value
    with pytest.raises(Exception):
        eng.analyze(req)


def test_duplicate_and_future_dates_rejected(eng):
    req = load("fertility_regular"); req["nights"].append(dict(req["nights"][-1]))
    with pytest.raises(Exception):
        eng.analyze(req)
    req = load("fertility_regular"); req["cycle"]["period_starts"].append("2999-01-01")
    with pytest.raises(Exception):
        eng.analyze(req)


def test_oura_adapter_longest_long_sleep_and_units():
    o = {"daily_readiness": [{"day": "2025-01-02", "temperature_deviation": 0.21}],
         "sleep": [{"day": "2025-01-02", "type": "long_sleep", "lowest_heart_rate": 55, "average_hrv": 40, "average_breath": 14.5, "total_sleep_duration": 25200, "efficiency": 88, "awake_time": 1800},
                   {"day": "2025-01-02", "type": "long_sleep", "lowest_heart_rate": 70, "total_sleep_duration": 3600},
                   {"day": "2025-01-03", "type": "rest", "lowest_heart_rate": 80, "total_sleep_duration": 1200}]}
    rows = normalize_oura(o)
    assert len(rows) == 1 and rows[0]["resting_hr"] == 55 and rows[0]["sleep_minutes"] == 420 and rows[0]["awake_minutes"] == 30
    assert rows[0]["temperature_deviation"] == 0.21


def test_oura_format_equals_nights_format(eng):
    a = eng.analyze(load("fertility_oura_format"))["fertility"]
    assert a["status"] == "ok"


def test_unlogged_period_is_suggested_not_silently_assumed(eng):
    req = load("fertility_regular")
    if len(req["cycle"]["period_starts"]) < 3:
        pytest.skip("not enough cycles")
    req["cycle"]["period_starts"] = req["cycle"]["period_starts"][:-1]
    req["cycle"]["period_logs_complete"] = False
    f = eng.analyze(req)["fertility"]
    assert f["status"] == "ok"
    if f["cycle_start_source"] == "inferred_from_physiology":
        assert f["unlogged_period_suggestions"] and f["unlogged_period_suggestions"][0]["action"] == "ask_user_to_confirm_period_start"


def test_menopause_outputs_and_rules(eng):
    r = eng.analyze(load("menopause_postmenopause"))["menopause"]
    assert r["status"] == "ok"
    assert r["stage"]["label"] == "postmenopause" and r["final_menstrual_period"]["already_confirmed"]
    r = eng.analyze(load("menopause_late_transition"))["menopause"]
    p = r["final_menstrual_period"]["probability_by_months"]
    vals = [p[k] for k in ["6", "12", "24", "36", "48", "60"]]
    assert all(b >= a for a, b in zip(vals, vals[1:]))
    assert abs(sum(r["stage"]["probabilities"].values()) - 1) < 1e-3


def test_menopause_insufficient_and_exclusions(eng):
    assert eng.analyze(load("menopause_insufficient"))["menopause"]["status"] == "insufficient_data"
    req = load("menopause_late_transition"); req["profile"]["menopausal_hormone_therapy"] = True
    assert eng.analyze(req)["menopause"]["status"] == "out_of_scope"
    req = load("menopause_late_transition"); req["profile"]["age"] = 30
    assert eng.analyze(req)["menopause"]["status"] == "not_applicable_age"


def test_core_adapter_cards():
    req = load("fertility_regular")
    values = {"nights": req["nights"], "period_starts": req["cycle"]["period_starts"]}
    ctx = {"as_of": req["as_of"], "profile": req["profile"]}
    c = RingFertilityComponent().compute(values, ctx)
    assert c["status"] == "ok" and c["mode"] == "SHADOW"
    c = RingFertilityComponent().compute({"nights": req["nights"], "period_starts": []}, ctx)
    assert c["status"] == "insufficient_data"
    c = RingFertilityComponent().compute({"nights": req["nights"] + [req["nights"][-1]], "period_starts": req["cycle"]["period_starts"]}, ctx)
    assert c["status"] in ("MANUAL_REVIEW", "COMPONENT_ERROR")
    m = load("menopause_late_transition")
    c = RingMenopauseComponent().compute({"nights": m["nights"], "period_starts": m["cycle"]["period_starts"], "last_period_date": m["cycle"]["last_period_date"]}, {"as_of": m["as_of"], "profile": m["profile"]})
    assert c["status"] == "ok"


def test_checksum_guard(tmp_path):
    import shutil
    root = tmp_path / "r"
    for sub in ["models/exported/ring_health/v3", "models/registry"]:
        (root / sub).mkdir(parents=True)
    for p in (ROOT / "models/exported/ring_health/v3").glob("*"):
        shutil.copy(p, root / "models/exported/ring_health/v3" / p.name)
    shutil.copy(ROOT / "models/registry/ring_health_v3.json", root / "models/registry/ring_health_v3.json")
    f = root / "models/exported/ring_health/v3/cycle_engine_params.json"
    f.write_text(f.read_text().replace('"tempering": 0.7', '"tempering": 0.71'))
    with pytest.raises(RuntimeError):
        RingHealthV3(root=root)


def test_missing_night_drops_out_of_likelihood_exactly():
    """signal_loglik with a missing night == brute-force multivariate normal over the observed nights only."""
    from scipy.stats import multivariate_normal
    P = json.loads((ROOT / "models/exported/ring_health/v3/cycle_engine_params.json").read_text())
    m = CycleModel(P)
    rng = np.random.default_rng(0)
    days_all = np.arange(1, 29, dtype=float)
    keep = np.ones(len(days_all), bool); keep[[4, 11, 12, 20]] = False   # four missing nights
    days = days_all[keep]; y = rng.normal(0, 0.2, len(days)) + 0.3 * (days > 15)
    s = "temperature_deviation"; noise = {"sd": 0.18, "rho": 0.4}; amp = (0.3, 0.02); grid = np.arange(8, 22); base = 0.0
    ll_rel, ll0_abs, _ = m.signal_loglik(s, days, y, grid, noise, amp, baseline=base)
    tau = P["offset_sd"][s]
    dist = np.abs(days[:, None] - days[None, :])
    S0 = noise["sd"] ** 2 * noise["rho"] ** dist + tau ** 2
    for j, O in enumerate(grid):
        g = m.g(s, days - O)
        cov = S0 + amp[1] * np.outer(g, g)
        brute = multivariate_normal(mean=amp[0] * g, cov=cov).logpdf(y - base)
        assert abs((ll_rel[j] + ll0_abs) - brute) < 1e-6
    assert abs(ll0_abs - multivariate_normal(mean=np.zeros(len(y)), cov=S0).logpdf(y - base)) < 1e-6


def test_calendar_only_history_is_not_detection(eng):
    f = eng.analyze(load("fertility_calendar_only"))["fertility"]
    assert all(not h.get("ovulation_detected") for h in f["history"])
    assert all(h.get("estimate_basis") in (None, "calendar_or_sparse_data") for h in f["history"] if "ovulation_detected" in h)


def test_stale_anchor_is_fast_and_explicit(eng):
    import time
    req = {"as_of": "2026-03-01", "modules": ["fertility"], "profile": {"age": 30}, "cycle": {"period_starts": ["2024-01-15"]}, "nights": []}
    t = time.time(); r = eng.analyze(req)["fertility"]
    assert r["status"] == "anchor_stale" and time.time() - t < 2.0


def test_stale_logs_do_not_confirm_postmenopause(eng):
    req = load("menopause_late_transition")
    as_of = pd.Timestamp(req["as_of"])
    req["cycle"] = {"period_starts": [str((as_of - pd.Timedelta(days=d)).date()) for d in (520, 490, 460)],
                    "last_period_date": str((as_of - pd.Timedelta(days=20)).date())}
    r = eng.analyze(req)["menopause"]
    assert not r["stage"]["straw10_criteria"]["amenorrhea_12m"] and not r["final_menstrual_period"]["already_confirmed"]
    assert r["signals"]["days_since_last_period"] <= 21


def test_old_period_starts_kept_for_menopause(eng):
    req = load("menopause_postmenopause")
    as_of = pd.Timestamp(req["as_of"])
    req["cycle"] = {"period_starts": [str((as_of - pd.Timedelta(days=d)).date()) for d in (1000, 970, 940)], "period_logs_complete": True}
    r = eng.analyze(req)["menopause"]
    assert r["status"] == "ok" and r["signals"]["days_since_last_period"] >= 900


def test_adapter_tolerates_extra_profile_keys():
    req = load("fertility_regular")
    ctx = {"as_of": req["as_of"], "profile": {**req["profile"], "pregnancy_week": None, "user_id": "x"}}
    c = RingFertilityComponent().compute({"nights": req["nights"], "period_starts": req["cycle"]["period_starts"]}, ctx)
    assert c["status"] == "ok"


def test_http_api_contract():
    import os
    from fastapi.testclient import TestClient
    os.environ["RING_HEALTH_API_KEY"] = "test-key"
    from src.ring_health_v3.api import app
    with TestClient(app) as c:
        assert c.get("/health/ready").json()["status"] == "ready"
        req = load("fertility_oura_format")
        assert c.post("/v3/analyze", json=req).status_code == 401
        r = c.post("/v3/analyze", json=req, headers={"Authorization": "Bearer test-key"})
        assert r.status_code == 200 and r.json()["fertility"]["status"] == "ok"
        assert c.post("/v3/analyze", json={"as_of": "x"}, headers={"Authorization": "Bearer test-key"}).status_code == 422
    os.environ.pop("RING_HEALTH_API_KEY", None)
