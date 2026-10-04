"""Final full-cohort search must stay separate from outer evaluation."""
import tempfile
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.training import preeclampsia_cdc_final as final


def test_full_cohort_selection_refit_resume_and_corruption():
    rows = [{"source_year": year, "source_row": fold * 10 + index,
             "fold": fold, "target_hdp_proxy": index % 2}
            for year in (2023, 2024) for fold in range(5) for index in range(10)]
    source = pd.DataFrame(rows)
    fits = []

    def fit(name, train, description, params):
        if description.startswith("Final grid"):
            held_out = int(description.split("fold ")[1].split(":")[0])
            assert set(train.fold) == set(range(5)) - {held_out}
        else:
            assert set(train.fold) == set(range(5))
        assert set(train.source_year) == {2023, 2024}
        fits.append(description)
        return {"C": params["C"]}

    def predict(name, model, frame):
        high = .8 if model["C"] == 1.0 else .6
        return np.where(frame.source_row.to_numpy() % 2, high, 1 - high).astype(np.float32)

    grid = {"prevalence": [{}], "logistic": [{"C": .1}, {"C": 1.0}]}
    with tempfile.TemporaryDirectory() as temporary, \
         patch.object(final, "FINAL_DIR", Path(temporary) / "search"), \
         patch.object(final, "MODEL_DIR", Path(temporary) / "models"), \
         patch.object(final, "GRID", grid), \
         patch.object(final, "MODEL_NAMES", list(grid)), \
         patch.object(final, "_input_data", return_value=source), \
         patch.object(final, "final_fingerprint", return_value="test-fingerprint"), \
         patch.object(final, "_fit_model", side_effect=fit), \
         patch.object(final, "_predict_model", side_effect=predict):
        results, summary, best = final.run_full_cohort_search()
        assert len(results) == 15
        assert len(summary) == 3
        assert (best["development_n"], best["development_events"]) == (100, 50)
        assert best["overall_winner"] == "logistic"
        assert len(fits) == 10
        final.run_full_cohort_search(resume=True)
        assert len(fits) == 10
        manifest = final.refit_full_cohort_best_models()
        assert manifest["overall_winner"] == "logistic"
        assert manifest["n"] == 100
        assert len(fits) == 11
        final.refit_full_cohort_best_models(resume=True)
        assert len(fits) == 11

        selection_path = final.FINAL_DIR / "best_configurations.json"
        selection = json.loads(selection_path.read_text())
        altered_selection = json.loads(selection_path.read_text())
        altered_selection["best_by_model"]["logistic"]["params"] = {"C": .1}
        selection_path.write_text(json.dumps(altered_selection))
        with pytest.raises(ValueError, match="does not minimize"):
            final.load_full_cohort_best()
        selection_path.write_text(json.dumps(selection))

        score = final.FINAL_DIR / "fold_0" / f"logistic_{final.candidate_id('logistic', {'C': 1.0})}.parquet"
        damaged = pd.read_parquet(score)
        damaged.loc[0, "prediction"] = .5
        damaged.to_parquet(score, index=False)
        with pytest.raises(ValueError, match="score hash differs"):
            final.run_full_cohort_search(resume=True)


def test_final_selection_rejects_modified_fold_summary():
    records = []
    for fold in range(5):
        for model, params in (("prevalence", {}), ("logistic", {"C": 1.0})):
            records.append({"fold": fold, "model": model,
                "candidate_id": final.candidate_id(model, params), "n": 10,
                "events": 2, "log_loss": .5 if model == "logistic" else .6,
                "brier": .2, "auroc": .6, "auprc": .4, "fit_seconds": 1.0})
    with patch.object(final, "GRID", {"prevalence": [{}], "logistic": [{"C": 1.0}]}), \
         patch.object(final, "MODEL_NAMES", ["prevalence", "logistic"]):
        summary, best = final._choose(pd.DataFrame(records), "test")
        assert best["overall_winner"] == "logistic"
        assert summary.n_oof.eq(50).all()
        broken = pd.DataFrame(records).drop(index=0)
        with pytest.raises(ValueError, match="missing or duplicated"):
            final._choose(broken, "test")


def test_full_cohort_refit_resumes_after_interrupted_model():
    source = pd.DataFrame({"source_year": [2023, 2024] * 5,
                           "target_hdp_proxy": [0, 1] * 5})
    names = ["prevalence", "logistic", "hist_gb"]
    best = {"fingerprint": "resume-test", "development_n": len(source),
            "development_events": int(source.target_hdp_proxy.sum()),
            "overall_winner": "logistic",
            "best_by_model": {name: {"candidate_id": final.candidate_id(name, {}),
                                     "params": {}} for name in names}}
    calls = []

    def fit(name, train, description, params):
        calls.append(name)
        if name == "hist_gb" and calls.count(name) == 1:
            raise RuntimeError("interrupted second model")
        return {"model": name}

    with tempfile.TemporaryDirectory() as temporary, \
         patch.object(final, "MODEL_DIR", Path(temporary)), \
         patch.object(final, "MODEL_NAMES", names), \
         patch.object(final, "load_full_cohort_best", return_value=best), \
         patch.object(final, "_input_data", return_value=source), \
         patch.object(final, "_fit_model", side_effect=fit):
        with pytest.raises(RuntimeError, match="interrupted second model"):
            final.refit_full_cohort_best_models(resume=True)
        result = final.refit_full_cohort_best_models(resume=True)
        assert calls == ["logistic", "hist_gb", "hist_gb"]
        assert set(result["artifacts"]) == set(names)
        final.refit_full_cohort_best_models(resume=True)
        assert calls == ["logistic", "hist_gb", "hist_gb"]
