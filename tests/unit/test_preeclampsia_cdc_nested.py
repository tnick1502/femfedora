"""Small end-to-end checks of the nested selection and checkpoint contract."""
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.data.preeclampsia_cdc import MISSING_MASKS, QUALITY_MASKS
from src.training import preeclampsia_cdc_nested as nested


def _cohort():
    rows = []
    for year in (2023, 2024):
        for fold in range(5):
            for index in range(12):
                row = {"source_year": year, "source_row": 12 * fold + index,
                    "fold": fold, "target_hdp_proxy": index % 2,
                    "target_hdp_preterm37": index % 2,
                    "target_hdp_preterm34": index % 2,
                    "mager": 25 + index, "bmi": 20 + index / 2,
                    "priorlive": 0, "priordead": 0, "rf_pdiab": 0,
                    "rf_inftr": 0, "rf_artec": 0, "f_rf_inf_art": 1,
                    "mrace6": 1, "mhisp_r": 1, "meduc": 1}
                row.update({name: 0 for name in MISSING_MASKS + QUALITY_MASKS})
                rows.append(row)
    return pd.DataFrame(rows)


def test_nested_cv_excludes_outer_labels_and_resumes_without_refitting():
    source = _cohort()
    trained_on = []

    def fit(name, train, description, params):
        outer = int(description.split("Outer ")[1].split()[0])
        assert outer not in set(train.fold)
        if "/ inner " in description:
            inner = int(description.split("/ inner ")[1].split(":")[0])
            assert inner not in set(train.fold)
            assert set(train.fold) == set(range(5)) - {outer, inner}
        else:
            assert set(train.fold) == set(range(5)) - {outer}
        trained_on.append(description)
        return {"C": params.get("C", 1.0), "train_n": len(train)}

    def predict(name, model, frame):
        return np.where(frame.source_row.to_numpy() % 2, .7, .3).astype(np.float32)

    with tempfile.TemporaryDirectory() as temporary, \
         patch.object(nested, "NESTED_DIR", Path(temporary)), \
         patch.object(nested, "GRID", {"prevalence": [{}], "logistic": [{"C": .1}, {"C": 1.0}]}), \
         patch.object(nested, "MODEL_NAMES", ["prevalence", "logistic"]), \
         patch.object(nested, "run_fingerprint", return_value="synthetic-source"), \
         patch.object(nested, "_nested_input_data", return_value=source), \
         patch.object(nested, "_fit_model", side_effect=fit), \
         patch.object(nested, "_predict_model", side_effect=predict):
        inner, outer, choices, oof = nested.run_nested_cv()
        assert len(inner) == 5 * 4 * 3
        assert len(outer) == 5 * 3  # prevalence, logistic, selected procedure
        assert len(choices) == 5
        assert len(oof) == len(source)
        assert not oof.duplicated(["source_year", "source_row"]).any()
        assert oof.groupby(["source_year", "fold"]).size().eq(12).all()
        fitted = len(trained_on)
        assert fitted == 5 * 4 * 2 + 5
        nested.run_nested_cv(resume=True)
        assert len(trained_on) == fitted
        loaded_inner, loaded_outer, loaded_choices, loaded_oof = nested.load_nested_results()
        assert len(loaded_inner) == len(inner)
        assert len(loaded_outer) == len(outer)
        assert len(loaded_choices) == len(choices)
        assert len(loaded_oof) == len(oof)
        assert len(trained_on) == fitted  # The evaluation reader did not fit.
        path = nested.NESTED_DIR / "nested_oof.parquet"
        damaged = pd.read_parquet(path)
        damaged.loc[0, "label"] ^= 1
        damaged.to_parquet(path, index=False)
        with pytest.raises(ValueError, match="differs from source"):
            nested.load_nested_results()


def test_inner_choice_does_not_see_outer_labels():
    source = _cohort()
    changed = source.copy()
    changed.loc[changed.fold.eq(0), "target_hdp_proxy"] ^= 1

    def fit(name, train, description, params):
        return {"C": params["C"]}

    def predict(name, model, frame):
        return np.where(frame.source_row.to_numpy() % 2, .65, .35).astype(np.float32)

    class Progress:
        def set_postfix_str(self, value, refresh=True):
            pass
        def update(self):
            pass

    with tempfile.TemporaryDirectory() as temporary, \
         patch.object(nested, "NESTED_DIR", Path(temporary)), \
         patch.object(nested, "GRID", {"prevalence": [{}], "logistic": [{"C": .1}, {"C": 1.0}]}), \
         patch.object(nested, "MODEL_NAMES", ["prevalence", "logistic"]), \
         patch.object(nested, "_fit_model", side_effect=fit), \
         patch.object(nested, "_predict_model", side_effect=predict):
        first, summary_a, choice_a = nested._inner_grid(source, 0, "test", Progress(), resume=False)
        second, summary_b, choice_b = nested._inner_grid(changed, 0, "test", Progress(), resume=False)
        pd.testing.assert_frame_equal(first.drop(columns=["fit_seconds", "rss_gb"]),
                                      second.drop(columns=["fit_seconds", "rss_gb"]))
        pd.testing.assert_frame_equal(summary_a.drop(columns="fit_seconds_total"),
                                      summary_b.drop(columns="fit_seconds_total"))
        assert choice_a["best_by_model"] == choice_b["best_by_model"]
        assert choice_a["overall_winner"] == choice_b["overall_winner"]


def test_nested_checkpoint_rejects_changed_predictions():
    valid = _cohort().loc[lambda frame: frame.fold.eq(0)].reset_index(drop=True)
    probabilities = np.where(valid.target_hdp_proxy.to_numpy(), .7, .3).astype(np.float32)
    scored = nested._prediction_table(valid, probabilities)
    identity = {"fingerprint": "test", "outer_fold": 1, "inner_fold": 0,
                "model": "logistic", "candidate_id": "example", "params": {"C": 1.0},
                "train_n": 30, "train_events": 15,
                "n": len(valid), "events": int(valid.target_hdp_proxy.sum())}
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "scores.parquet"
        metadata_path = path.with_suffix(".json")
        nested._save_fit(scored, path, metadata_path,
                         {**identity, **nested.metrics(scored.label, scored.prediction)})
        assert nested._cached_fit(path, metadata_path, valid, identity) is not None
        altered = scored.copy()
        altered.loc[0, "prediction"] = .9
        altered.to_parquet(path, index=False)
        with pytest.raises(ValueError, match="score hash differs"):
            nested._cached_fit(path, metadata_path, valid, identity)
