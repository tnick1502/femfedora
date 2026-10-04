"""Guard the user-facing order of tuning, training, and evaluation notebooks."""

import ast
import gc
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_curve


NOTEBOOKS = Path(__file__).resolve().parents[2] / "research/notebooks/preeclampsia_cdc"


def _calls(book):
    calls = []
    for index, cell in enumerate(book.cells):
        if cell.cell_type != "code":
            continue
        tree = ast.parse(cell.source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                calls.append((node.func.id, index))
    return calls


def test_tuned_workflow_is_in_training_and_temporal_notebooks():
    assert not (NOTEBOOKS / "07_hyperparameter_ablations.ipynb").exists()
    books = {}
    for name in ("02_train_baselines", "03_evaluate_folds", "06_temporal_validation"):
        book = nbformat.read(NOTEBOOKS / f"{name}.ipynb", as_version=4)
        nbformat.validate(book)
        books[name] = (book, dict(_calls(book)))

    train = books["02_train_baselines"][1]
    evaluation = books["03_evaluate_folds"][1]
    temporal_book, temporal = books["06_temporal_validation"]
    assert train["fit_folds"] < train["run_nested_cv"] < train["run_full_cohort_search"]
    assert train["run_full_cohort_search"] < train["refit_full_cohort_best_models"]
    assert temporal["run_grid_search"] < temporal["fit_temporal"]
    assert temporal["run_grid_search"] < temporal["score_selected_on_2024"]
    # The 2023 OOF frame is released before the large temporal workflow, then
    # reloaded before the final two-year ROC/PR chart uses it again.
    reloads = [i for i, cell in enumerate(temporal_book.cells)
               if cell.cell_type == "code" and "selected_oof = selected_oof_predictions()" in cell.source]
    chart = next(i for i, cell in enumerate(temporal_book.cells)
                 if cell.cell_type == "code" and "(2023,selected_oof)" in cell.source)
    assert max(reloads) < chart and max(reloads) > temporal["fit_temporal"]
    assert "load_nested_results" in evaluation
    assert "load_full_cohort_best" in evaluation
    evaluate_code = "\n".join(cell.source for cell in books["03_evaluate_folds"][0].cells
                              if cell.cell_type == "code")
    temporal_code = "\n".join(cell.source for cell in temporal_book.cells
                              if cell.cell_type == "code")
    assert "RocCurveDisplay.from_predictions" in evaluate_code
    assert "secondary_outcome_metrics(oof_nested" in evaluate_code
    assert "sns.heatmap" in temporal_code
    assert "ablation_results(fold_results)" in temporal_code
    for forbidden in ("run_nested_cv", "run_full_cohort_search", "refit_full_cohort_best_models",
                      "run_grid_search", "score_selected_on_2024", "fit_folds"):
        assert forbidden not in evaluation


def test_temporal_roc_pr_cell_reuses_reloaded_2023_scores():
    book = nbformat.read(NOTEBOOKS / "06_temporal_validation.ipynb", as_version=4)
    rng = np.random.default_rng(20261001)
    names = ("logistic", "logistic_spline", "hist_gb", "catboost", "lightgbm")
    labels = np.tile([0, 1], 200)
    frame = pd.DataFrame({"label": labels})
    for name in names:
        frame[name] = np.clip(.15 + .35 * labels + rng.normal(0, .1, len(labels)), .01, .99)
    scope = {"gc": gc, "selected_oof_predictions": lambda: frame,
             "dev": frame, "test": frame, "sub": frame,
             "temporal_scores": frame, "plt": plt, "np": np,
             "roc_curve": roc_curve, "precision_recall_curve": precision_recall_curve,
             "auc": auc, "GREY": "#666D78",
             "MODEL_PALETTE": [("logistic_spline", "#255B8B"),
                               ("hist_gb", "#D17A31"), ("catboost", "#287D69"),
                               ("lightgbm", "#8062A5")]}
    reload_cell = next(cell for cell in book.cells if cell.cell_type == "code" and
                       "selected_oof = selected_oof_predictions()" in cell.source)
    chart_cell = next(cell for cell in book.cells if cell.cell_type == "code" and
                      "(2023,selected_oof)" in cell.source)
    exec(reload_cell.source, scope)
    assert scope["selected_oof"] is frame
    assert "dev" not in scope and "test" not in scope
    with patch.object(plt, "show"):
        exec(chart_cell.source, scope)
    plt.close("all")
