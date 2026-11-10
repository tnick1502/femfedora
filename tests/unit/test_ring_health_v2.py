import copy
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from src.ring_health.api_v2 import app
from src.ring_health.engine_v2 import Engine
from src.ring_health.missing_v2 import causal_features,normalize_oura
from src.ring_health.schema_v2 import AnalyzeRequest

ROOT=Path(__file__).resolve().parents[2]


def payload():
    p=json.loads((ROOT/"data/samples/ring_health/fertility_demo_input.json").read_text())
    p["context"]["period_start_dates"]=p["context"]["period_start_dates"][-1:]
    return p


@pytest.fixture(scope="module")
def engine():return Engine()


@pytest.mark.parametrize("invalid",[None,"NaT","2024-04-01T12:00:00","2024-04-01T00:00:00Z",1700000000,"2024-02-30"])
def test_reject_malformed_dates(invalid):
    p=payload();p["as_of"]=invalid
    with pytest.raises(ValidationError):AnalyzeRequest.model_validate(p)


@pytest.mark.parametrize("value",["36.5",float("nan"),float("inf"),True,36.5])
def test_reject_invalid_numbers(value):
    p=payload();p["daily"][-1]["temperature_deviation"]=value
    with pytest.raises(ValidationError):AnalyzeRequest.model_validate(p)


def test_reject_null_day():
    p=payload();p["daily"][0]["date"]=None
    with pytest.raises(ValidationError):AnalyzeRequest.model_validate(p)


def test_explicit_as_of_required():
    p=payload();del p["as_of"]
    with pytest.raises(ValidationError):AnalyzeRequest.model_validate(p)


def test_exactly_one_input():
    p=payload();p["oura"]={}
    with pytest.raises(ValidationError):AnalyzeRequest.model_validate(p)


def test_null_oura_input():
    p=payload();p.pop("daily");p["oura"]=None
    with pytest.raises(ValidationError):AnalyzeRequest.model_validate(p)


def test_bounded_calendar_span():
    p=payload();p["as_of"]="2100-01-01"
    with pytest.raises(ValidationError):AnalyzeRequest.model_validate(p)


def test_temperature_only_is_supported(engine):
    p=payload();p["daily"]=[{"date":r["date"],"temperature_deviation":r.get("temperature_deviation")} for r in p["daily"]]
    r=engine.analyze(p)
    assert r["fertility"]["status"] in ["estimated_with_gaps","thermal_shift_detected"]
    assert len(r["fertility"]["forecast"])==7


def test_missing_current_temperature_is_not_blanket_refusal(engine):
    p=payload();p["daily"][-1]["temperature_deviation"]=None
    r=engine.analyze(p)
    assert "forecast" in r["fertility"]
    assert r["fertility"]["days_since_last_temperature"]>=1


def test_no_temperature_has_prior_fallback(engine):
    p=payload()
    for row in p["daily"]:row["temperature_deviation"]=None
    r=engine.analyze(p)
    assert r["fertility"]["status"].startswith("calendar_only")
    assert r["fertility"]["reference_event_date"] is None
    assert r["fertility"]["imputed_measurement_count"]==0


def test_empty_observations_can_return_only_prior(engine):
    p=payload();p["daily"]=[]
    r=engine.analyze(p)
    assert r["fertility"]["evidence_source"]=="cycle_prior_only"


def test_no_period_logs_require_anchor(engine):
    p=payload();p["context"]["period_start_dates"]=[]
    assert engine.analyze(p)["fertility"]["status"]=="anchor_required"


def test_cycle_day_can_replace_date_log(engine):
    p=payload();p["context"]["period_start_dates"]=[];p["context"]["cycle_day"]=11
    assert len(engine.analyze(p)["fertility"]["forecast"])==7


def test_oura_union_preserves_readiness_only_temperature():
    r=normalize_oura({"daily_readiness":[{"day":"2024-01-01","temperature_deviation":.2}],"daily_activity":[{"day":"2024-01-02","steps":5}]})
    assert len(r)==2 and r[0]["temperature_deviation"]==.2


def test_oura_duplicates_are_rejected():
    with pytest.raises(ValueError):normalize_oura({"daily_readiness":[{"day":"2024-01-01"}]*2})


def test_step_count_cannot_leak_into_morning_fertility_features():
    d=pd.DataFrame({"temperature_deviation":np.arange(40)/100,"steps":np.arange(40)})
    f=causal_features(d);d["steps"]=99999
    pd.testing.assert_frame_equal(f,causal_features(d))


def test_future_mutation_does_not_change_feature_prefix():
    d=pd.DataFrame({"temperature_deviation":np.sin(np.arange(50))})
    f=causal_features(d);d.loc[30:,"temperature_deviation"]=100
    pd.testing.assert_frame_equal(f.iloc[:30],causal_features(d).iloc[:30])


def test_long_cycle_does_not_get_false_high_confidence(engine):
    p=payload();p["context"]["period_start_dates"]=[];p["context"]["cycle_day"]=60
    assert engine.analyze(p)["fertility"]["status"]=="uncertain_cycle_anchor"


def test_probabilities_are_finite_and_deterministic(engine):
    p=payload();a=engine.analyze(p);b=engine.analyze(p)
    assert a==b
    json.dumps(a,allow_nan=False)
    assert all(0<=x["fertile_day_probability"]<=1 for x in a["fertility"]["forecast"])


def test_api_and_readiness():
    with TestClient(app) as client:
        assert client.get("/health/ready").status_code==200
        r=client.post("/v2/analyze",json=payload())
        assert r.status_code==200 and r.json()["model_version"]=="ring-health-v2"
        assert client.post("/v2/analyze",json={"as_of":"NaT"}).status_code==422
        assert "/v2/analyze" in client.get("/openapi.json").json()["paths"]


def test_api_key_enforced(monkeypatch):
    monkeypatch.setenv("RING_HEALTH_API_KEY","test-secret")
    with TestClient(app) as c:
        assert c.post("/v2/analyze",json=payload()).status_code==401
        assert c.post("/v2/analyze",json=payload(),headers={"Authorization":"Bearer test-secret"}).status_code==200


def test_artifact_hash_mismatch_fails_startup(tmp_path):
    root=tmp_path
    (root/"models/registry").mkdir(parents=True)
    (root/"models/exported/ring_health/v2").mkdir(parents=True)
    m=json.loads((ROOT/"models/registry/ring_health_v2.json").read_text())
    m["artifacts"]={"bad.joblib":{"sha256":"0"*64}}
    (root/"models/registry/ring_health_v2.json").write_text(json.dumps(m))
    (root/"models/exported/ring_health/v2/bad.joblib").write_bytes(b"bad")
    with pytest.raises(RuntimeError,match="checksum"):Engine(root)


def test_old_event_does_not_establish_current_cycle(engine):
    p=payload();p["context"]["period_start_dates"]=[];p["context"]["cycle_day"]=40
    start=pd.Timestamp(p["as_of"])-pd.Timedelta(days=39)
    p["daily"]=[{"date":str((start+pd.Timedelta(days=i)).date()),"temperature_deviation":0. if i<15 else .4} for i in range(40)]
    r=engine.analyze(p)["fertility"]
    assert r["status"]=="uncertain_cycle_anchor"
    assert r["reference_event_date"] is None and "forecast" not in r


def test_chunked_body_limit():
    with TestClient(app) as client:
        r=client.post("/v2/analyze",content=iter([b"x"*1_000_001,b"x"*1_000_001]))
        assert r.status_code==413
