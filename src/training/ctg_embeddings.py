"""Additional inductive CTG embeddings; outcome-free search, preserved old results.

No outcome argument exists in fit/search. NSP is read only after the configuration
has been locked. All fitting, including early stopping, respects source files.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import trustworthiness
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             pairwise_distances, silhouette_samples)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GroupShuffleSplit, ParameterGrid
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from umap import UMAP

from src.training.ctg_clustering import FuzzyCMeans, _membership
from src.training.ctg_experiments import CTG_FEATURES, ROOT, OUT, MODELS, SEED, digest, load_data, write_json

RESULTS = OUT / "embeddings"
ARTIFACT = MODELS / "clustering_embeddings.joblib"
VERSION = "ctg_embeddings_v1"
MIN_FRACTION = .03
MIN_TRUST = .80
UMAP_GRID = list(ParameterGrid({"n_components": [2, 5, 10], "n_neighbors": [10, 30, 60],
                                "min_dist": [0., .1, .5]}))
AE_GRID = list(ParameterGrid({"n_components": [2, 5, 10], "width": [16, 32],
                              "alpha": [.001, .1], "activation": ["tanh", "relu"]}))
EMBEDDINGS = ([dict(method="RAW", params={})] +
              [dict(method="PCA", params={"n_components": d}) for d in [2, 5, 10]] +
              [dict(method="UMAP", params=p) for p in UMAP_GRID] +
              [dict(method="AE", params=p) for p in AE_GRID])


def versions():
    return {p: importlib.metadata.version(p) for p in
            ["numpy", "scipy", "pandas", "scikit-learn", "umap-learn", "numba", "pynndescent", "joblib"]}


def legacy_snapshot():
    """Only pre-existing clustering evidence, never new extension outputs."""
    files = sorted(OUT.glob("cluster_*")) + [OUT / "fcm_convergence.json",
        ROOT / "research/notebooks/ctg_uci/04_clustering_without_outcomes.ipynb",
        MODELS / "clustering.joblib"]
    return {str(p.relative_to(ROOT)): digest(p) for p in files if p.is_file()}


class EmbeddingInput:
    """The same distance representation as 04, before its PCA (21 columns)."""
    def fit(self, X):
        self.preprocessor_ = clone(joblib.load(ROOT / "data/processed/ctg_uci/preprocessor.joblib"))
        z = self.preprocessor_.fit_transform(X)
        names = self.preprocessor_.get_feature_names_out()
        self.numeric_idx_ = [i for i, n in enumerate(names) if not n.startswith(("rare_present__", "tendency__"))]
        self.category_idx_ = [i for i, n in enumerate(names) if n.startswith("tendency__")]
        self.names_ = [str(names[i]) for i in self.numeric_idx_ + self.category_idx_]
        self.scaler_ = StandardScaler().fit(z[:, self.numeric_idx_])
        return self

    def transform(self, X):
        z = self.preprocessor_.transform(X)
        result = np.column_stack([self.scaler_.transform(z[:, self.numeric_idx_]),
                                  z[:, self.category_idx_] / np.sqrt(2)])
        if not np.isfinite(result).all():
            raise ValueError("Non-finite embedding input")
        return result


class DenoisingAutoencoder:
    """Small undercomplete MLP: p -> width -> latent -> width -> p.

    sklearn supplies Adam/backpropagation and linear reconstruction outputs.
    Explicit group-held-out early stopping avoids its random-row validation split.
    Noise is applied to inputs only; reconstruction target always stays clean.
    """
    def __init__(self, n_components=5, width=32, alpha=.001, activation="tanh",
                 noise=.05, max_epochs=1200, patience=60, seed=SEED):
        self.n_components, self.width, self.alpha = n_components, width, alpha
        self.activation, self.noise = activation, noise
        self.max_epochs, self.patience, self.seed = max_epochs, patience, seed

    def _new_model(self):
        return MLPRegressor(hidden_layer_sizes=(self.width, self.n_components, self.width),
            activation=self.activation, solver="adam", alpha=self.alpha,
            batch_size=128, learning_rate_init=.001, shuffle=True,
            early_stopping=False, random_state=self.seed)

    def fit(self, X, X_stop=None, epochs=None):
        X = np.asarray(X, dtype=float)
        if not 0 < self.n_components < X.shape[1]:
            raise ValueError("An undercomplete bottleneck is required")
        self.model_ = self._new_model()
        rng = np.random.default_rng(self.seed)
        self.history_ = []
        best, best_model, stale = np.inf, None, 0
        for epoch in range(1, (epochs or self.max_epochs) + 1):
            noisy = X + rng.normal(0., self.noise, X.shape)
            self.model_.partial_fit(noisy, X)
            training_mse = float(np.mean((self.model_.predict(X) - X) ** 2))
            stop_mse = float(np.mean((self.model_.predict(X_stop) - X_stop) ** 2)) if X_stop is not None else np.nan
            self.history_.append(dict(epoch=epoch, train_mse=training_mse, stop_mse=stop_mse))
            if X_stop is not None:
                if stop_mse < best - 1e-5:
                    best, best_model, stale = stop_mse, copy.deepcopy(self.model_), 0
                    self.best_epoch_ = epoch
                else:
                    stale += 1
                if epoch >= 30 and stale >= self.patience:
                    break
        if X_stop is not None:
            self.model_ = best_model
            self.best_stop_mse_ = best
        else:
            self.best_epoch_ = len(self.history_)
        self.hit_epoch_cap_ = len(self.history_) == self.max_epochs and X_stop is not None
        return self

    def transform(self, X):
        z = np.asarray(X, dtype=float)
        for w, b in zip(self.model_.coefs_[:2], self.model_.intercepts_[:2]):
            z = z @ w + b
            z = np.tanh(z) if self.activation == "tanh" else np.maximum(z, 0.)
        return z

    def inverse_transform(self, Z):
        z = np.asarray(Z, dtype=float)
        w, b = self.model_.coefs_[2], self.model_.intercepts_[2]
        z = z @ w + b
        z = np.tanh(z) if self.activation == "tanh" else np.maximum(z, 0.)
        return z @ self.model_.coefs_[3] + self.model_.intercepts_[3]


def make_reducer(method, params, seed=SEED):
    if method == "RAW":
        return None
    if method == "PCA":
        return PCA(**params, svd_solver="full")
    if method == "UMAP":
        return UMAP(**params, metric="euclidean", n_epochs=300, n_jobs=1,
                    random_state=seed, transform_seed=seed, init="spectral")
    if method == "AE":
        return DenoisingAutoencoder(**params, seed=seed)
    raise ValueError(method)


def fit_embedding(spec, prepared, stop_data=None, seed=SEED, epochs=None):
    method, params = spec["method"], spec["params"]
    reducer = make_reducer(method, params, seed)
    metadata = {}
    if method == "AE":
        if epochs is None:
            a, b = stop_data
            probe = make_reducer(method, params, seed).fit(a, X_stop=b)
            epochs = probe.best_epoch_
            metadata = dict(best_epoch=epochs, stop_mse=probe.best_stop_mse_,
                            hit_epoch_cap=probe.hit_epoch_cap_, history=probe.history_)
        reducer.fit(prepared, epochs=epochs)
    elif reducer is not None:
        reducer.fit(prepared)
    return reducer, metadata


def project(reducer, X, training=False):
    if reducer is None:
        return X.copy()
    # UMAP.transform(X_train) is deliberately replaced by its fitted embedding.
    if training and isinstance(reducer, UMAP):
        return reducer.embedding_.copy()
    return reducer.transform(X)


def cluster_grid():
    for k in range(2, 7):
        for m in [1.5, 2., 2.5]:
            yield dict(algorithm="FCM", k=k, variant=m)
        yield dict(algorithm="KMeans", k=k, variant=None)
        for cov in ["diag", "full"]:
            yield dict(algorithm="GMM", k=k, variant=cov)


def fit_cluster(params, X, seed=SEED):
    algorithm, k, variant = params["algorithm"], int(params["k"]), params["variant"]
    if algorithm == "FCM":
        return FuzzyCMeans(k, m=float(variant), seed=seed).fit(X)
    if algorithm == "KMeans":
        return KMeans(n_clusters=k, n_init=20, random_state=seed).fit(X)
    return GaussianMixture(n_components=k, covariance_type=variant, n_init=5,
        max_iter=300, reg_covar=1e-4, random_state=seed).fit(X)


def silhouette(distances, labels, groups=None):
    if not 1 < len(np.unique(labels)) < len(labels):
        return -1.
    values = silhouette_samples(distances, labels, metric="precomputed")
    if groups is None:
        return float(values.mean())
    return float(pd.Series(values).groupby(np.asarray(groups)).mean().mean())


def geometry(original, embedded):
    k = 10
    a = NearestNeighbors(n_neighbors=k).fit(original).kneighbors(return_distance=False)
    b = NearestNeighbors(n_neighbors=k).fit(embedded).kneighbors(return_distance=False)
    overlap = np.mean([len(set(x) & set(y)) / k for x, y in zip(a, b)])
    return dict(trustworthiness=float(trustworthiness(original, embedded, n_neighbors=k)),
                neighbor_overlap=float(overlap))


def choose_configs(grid):
    eligible = grid[grid.eligible].copy()
    required = set(grid.method.unique())
    if set(eligible.method.unique()) != required:
        raise ValueError("An embedding family has no eligible candidate; inspect the grid")
    order = ["validation_group_silhouette_common", "dimensions", "k", "embedding_id", "key"]
    ranked = eligible.sort_values(order, ascending=[False, True, True, True, True])
    chosen = ranked.groupby(["method", "algorithm"], sort=True).head(1).copy()
    if len(chosen) != 3 * len(required):
        raise ValueError("A method/algorithm pair has no eligible candidate")
    return chosen.sort_values(["method", "algorithm"]).reset_index(drop=True)


def risk_bootstrap(frame, labels, k, repetitions=1000):
    """Block bootstrap source files; pointwise, descriptive intervals only."""
    groups, group_index = np.unique(frame.source_file, return_inverse=True)
    counts = np.zeros((len(groups), k, 3))
    np.add.at(counts, (group_index, labels, frame.NSP.to_numpy(dtype=int) - 1), 1)
    rng = np.random.default_rng(SEED + 900)
    samples = rng.integers(len(groups), size=(repetitions, len(groups)))
    totals = counts[samples].sum(axis=1)
    den = totals.sum(axis=2)
    result = {}
    for label, name in [(2, "path"), (1, "suspect"), (None, "abnormal")]:
        num = totals[:, :, label] if label is not None else totals[:, :, 1:].sum(axis=2)
        rate = np.divide(num, den, out=np.full_like(den, np.nan), where=den > 0)
        baseline = num.sum(axis=1) / den.sum(axis=1)
        enrichment = np.divide(rate, baseline[:, None], out=np.full_like(rate, np.nan), where=baseline[:, None] > 0)
        for c in range(k):
            result.setdefault(c, {}).update({
                f"{name}_fraction_low": float(np.nanquantile(rate[:, c], .025)),
                f"{name}_fraction_high": float(np.nanquantile(rate[:, c], .975)),
                f"{name}_enrichment_low": float(np.nanquantile(enrichment[:, c], .025)),
                f"{name}_enrichment_high": float(np.nanquantile(enrichment[:, c], .975))})
    return result


def run_embeddings(force=False):
    RESULTS.mkdir(parents=True, exist_ok=True)
    data, data_fingerprint = load_data()
    old = legacy_snapshot()
    preservation = RESULTS / "preserved_legacy.json"
    if preservation.exists():
        if json.loads(preservation.read_text())["sha256"] != old:
            raise ValueError("Pre-existing clustering results changed since extension started")
    else:
        write_json(preservation, dict(sha256=old))
    fingerprint = hashlib.sha256((data_fingerprint + digest(__file__) +
        digest(ROOT / "src/training/ctg_clustering.py") + json.dumps(versions(), sort_keys=True)).encode()).hexdigest()
    receipt_path = RESULTS / "validation_receipt.json"
    if not force and receipt_path.exists() and ARTIFACT.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt["fingerprint"] == fingerprint and receipt["legacy_unchanged"]:
            print("Loaded complete, code/data/version-matched extension results.")
            return receipt
    cache = ROOT / "tmp/ctg_embeddings" / fingerprint[:16]
    cache.mkdir(parents=True, exist_ok=True)
    train = data[data.split == "train"].reset_index(drop=True)
    valid = data[data.split == "validation"].reset_index(drop=True)
    X, V = train[list(CTG_FEATURES)], valid[list(CTG_FEATURES)]
    input_model = EmbeddingInput().fit(X)
    common_train, common_valid = input_model.transform(X), input_model.transform(V)
    # Unsupervised group split: no NSP or CLASS in this operation.
    a, b = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=SEED + 800).split(X, groups=train.source_file))
    assert set(train.iloc[a].source_file).isdisjoint(train.iloc[b].source_file)
    stop_input = EmbeddingInput().fit(X.iloc[a])
    stop_data = stop_input.transform(X.iloc[a]), stop_input.transform(X.iloc[b])
    stop_roles = train[["source_excel_row", "source_file"]].copy()
    stop_roles["role"] = "fit"
    stop_roles.loc[b, "role"] = "early_stop"
    stop_roles.to_csv(RESULTS / "ae_early_stopping_groups.csv", index=False)
    common_distance = pairwise_distances(common_valid)
    np.fill_diagonal(common_distance, 0.)
    all_rows, embedding_rows, reducers = [], [], {}
    start = time.time()
    for number, spec in enumerate(EMBEDDINGS):
        embedding_id = f"{spec['method']}_{number:02d}"
        checkpoint = cache / f"{embedding_id}.joblib"
        if checkpoint.exists():
            saved = joblib.load(checkpoint)
        else:
            reducer, metadata = fit_embedding(spec, common_train, stop_data)
            z_train = project(reducer, common_train, training=True)
            z_valid = project(reducer, common_valid)
            geom = geometry(common_valid, z_valid)
            embedding_record = dict(embedding_id=embedding_id, method=spec["method"],
                params=json.dumps(spec["params"], sort_keys=True), dimensions=z_train.shape[1], **geom,
                best_epoch=metadata.get("best_epoch"), stop_mse=metadata.get("stop_mse"),
                hit_epoch_cap=metadata.get("hit_epoch_cap", False))
            if spec["method"] in ["AE", "PCA"]:
                embedding_record["validation_reconstruction_mse"] = float(np.mean(
                    (reducer.inverse_transform(z_valid) - common_valid) ** 2))
                embedding_record["train_reconstruction_mse"] = float(np.mean(
                    (reducer.inverse_transform(z_train) - common_train) ** 2))
            embedded_distance = pairwise_distances(z_valid)
            np.fill_diagonal(embedded_distance, 0.)
            rows = []
            for ci, cp in enumerate(cluster_grid()):
                model = fit_cluster(cp, z_train)
                lt, lv = model.predict(z_train), model.predict(z_valid)
                k = cp["k"]
                ft, fv = np.bincount(lt, minlength=k).min() / len(lt), np.bincount(lv, minlength=k).min() / len(lv)
                converged = bool(getattr(model, "converged_", True))
                center_ok = cp["algorithm"] != "FCM" or model.minimum_center_distance_ >= 1e-3
                rows.append(dict(key=f"{embedding_id}_{ci:02d}", embedding_id=embedding_id,
                    method=spec["method"], dimensions=z_train.shape[1], **cp,
                    validation_group_silhouette_common=silhouette(common_distance, lv, valid.source_file),
                    validation_silhouette_common=silhouette(common_distance, lv),
                    validation_silhouette_embedding=silhouette(embedded_distance, lv),
                    minimum_train_fraction=ft, minimum_validation_fraction=fv, converged=converged,
                    eligible=bool(ft >= MIN_FRACTION and fv >= MIN_FRACTION and converged and center_ok and geom["trustworthiness"] >= MIN_TRUST),
                    trustworthiness=geom["trustworthiness"], neighbor_overlap=geom["neighbor_overlap"],
                    bic=float(model.bic(z_train)) if cp["algorithm"] == "GMM" else np.nan,
                    fpc=getattr(model, "fpc_", np.nan),
                    initialization_ari=getattr(model, "initialization_ari_", np.nan)))
            saved = dict(reducer=reducer, metadata=metadata, embedding_record=embedding_record, rows=rows)
            joblib.dump(saved, checkpoint)
        reducers[embedding_id] = saved["reducer"]
        all_rows.extend(saved["rows"])
        embedding_rows.append(saved["embedding_record"])
        if saved["metadata"].get("history"):
            pd.DataFrame(saved["metadata"]["history"]).to_csv(RESULTS / f"{embedding_id}_learning_curve.csv", index=False)
        pd.DataFrame(all_rows).to_csv(RESULTS / "joint_grid.csv", index=False)
        pd.DataFrame(embedding_rows).to_csv(RESULTS / "embedding_grid.csv", index=False)
        write_json(RESULTS / "progress.json", dict(completed=number + 1, total=len(EMBEDDINGS),
            elapsed_seconds=time.time() - start, last=embedding_id, fingerprint=fingerprint))
        print(f"{embedding_id}: {number+1}/{len(EMBEDDINGS)} representations; 30 clustering configurations each", flush=True)
    grid = pd.DataFrame(all_rows)
    chosen = choose_configs(grid)
    chosen.to_csv(RESULTS / "chosen_configurations.csv", index=False)
    write_json(RESULTS / "locked_config.json", dict(fingerprint=fingerprint, version=VERSION, versions=versions(),
        selection="Highest source-file-mean validation silhouette in common 21-column space; ties: dimensions, k, IDs",
        minimum_cluster_fraction=MIN_FRACTION, minimum_validation_trustworthiness=MIN_TRUST,
        early_stop_split="GroupShuffleSplit .2 inside train, seed+800; preprocessing fit on inner fit only",
        chosen=chosen.to_dict("records"), embeddings=EMBEDDINGS,
        noise=.05, ae_max_epochs=1200, ae_patience=60,
        budget_note="AE budget raised from 300 to 1200 after reconstruction-only pilot: 300-epoch cap was binding; no NSP criterion",
        outcomes_in_selection=False, test_in_selection=False,
        limitation="Exploratory extension on a previously inspected benchmark; not a fresh blind cohort"))
    # Configuration is saved above before touching test features or any outcomes.
    bundles, assignments, associations, summaries, projections = {}, [], [], [], []
    for row in chosen.to_dict("records"):
        eid, algorithm = row["embedding_id"], row["algorithm"]
        key = row["key"]
        reducer = reducers[eid]
        z_train = project(reducer, common_train, training=True)
        model = fit_cluster(row, z_train)
        spec = EMBEDDINGS[int(eid.rsplit("_", 1)[1])]
        best_epoch = next(e.get("best_epoch") for e in embedding_rows if e["embedding_id"] == eid)
        bundles[key] = dict(reducer=reducer, clusterer=model, config=row, spec=spec, epochs=best_epoch)
        for split, frame in data.groupby("split"):
            common = input_model.transform(frame[list(CTG_FEATURES)])
            z = project(reducer, common, training=split == "train")
            membership = np.asarray(_membership(model, z, algorithm), dtype=np.float64)
            membership /= membership.sum(axis=1, keepdims=True)
            labels = membership.argmax(axis=1)
            k = membership.shape[1]
            assignment = frame[["source_excel_row", "source_file", "NSP"]].copy()
            assignment["key"], assignment["method"], assignment["algorithm"] = key, row["method"], algorithm
            assignment["split"], assignment["cluster"] = split, labels
            assignment["membership_entropy"] = -np.sum(membership * np.log(np.clip(membership, 1e-12, 1)), axis=1) / np.log(k)
            for c in range(k):
                assignment[f"membership_{c}"] = membership[:, c]
            assignments.append(assignment)
            distances = pairwise_distances(common)
            np.fill_diagonal(distances, 0.)
            associations.append(dict(key=key, method=row["method"], algorithm=algorithm, split=split, k=k,
                ari=adjusted_rand_score(frame.NSP, labels), nmi=normalized_mutual_info_score(frame.NSP, labels),
                silhouette_common=silhouette(distances, labels),
                group_silhouette_common=silhouette(distances, labels, frame.source_file)))
            if split == "test":
                intervals = risk_bootstrap(frame, labels, k)
                visual = z[:, :2] if row["dimensions"] == 2 else PCA(n_components=2).fit(z_train).transform(z)
                projections.append(pd.DataFrame(dict(key=key, axis1=visual[:, 0], axis2=visual[:, 1],
                    cluster=labels, NSP=frame.NSP.to_numpy(), source_excel_row=frame.source_excel_row.to_numpy())))
                for c in range(k):
                    inside = frame.iloc[np.flatnonzero(labels == c)]
                    soft_rate = float(membership[:, c] @ (frame.NSP.to_numpy() == 3) / membership[:, c].sum())
                    summary = dict(key=key, method=row["method"], algorithm=algorithm, cluster=c,
                        segments=len(inside), source_files=inside.source_file.nunique(),
                        pathological=int((inside.NSP == 3).sum()),
                        small_cluster=bool(inside.source_file.nunique() < 10 or (inside.NSP == 3).sum() < 5),
                        membership_weighted_path_fraction=soft_rate)
                    for name, label in [("path", 3), ("suspect", 2), ("abnormal", None)]:
                        rate = float((inside.NSP == label).mean()) if label is not None else float((inside.NSP != 1).mean())
                        baseline = float((frame.NSP == label).mean()) if label is not None else float((frame.NSP != 1).mean())
                        summary.update({f"{name}_fraction": rate, f"{name}_enrichment": rate / baseline})
                    summaries.append({**summary, **intervals[c]})
    pd.concat(assignments, ignore_index=True).to_parquet(RESULTS / "assignments.parquet", index=False)
    pd.DataFrame(associations).to_csv(RESULTS / "association.csv", index=False)
    pd.DataFrame(summaries).to_csv(RESULTS / "test_risk_summary.csv", index=False)
    pd.concat(projections, ignore_index=True).to_parquet(RESULTS / "test_projections.parquet", index=False)
    joblib.dump(dict(input_model=input_model, bundles=bundles, fingerprint=fingerprint,
                     feature_columns=list(CTG_FEATURES), versions=versions()), ARTIFACT)
    stability = evaluate_stability(train, valid, input_model, bundles)
    stability.to_csv(RESULTS / "stability.csv", index=False)
    assert legacy_snapshot() == old, "Existing clustering evidence was modified"
    receipt = dict(fingerprint=fingerprint, data_fingerprint=data_fingerprint, versions=versions(),
        representations=len(EMBEDDINGS), joint_configurations=len(grid), selected_models=len(chosen),
        legacy_unchanged=True, legacy_files=len(old), feature_columns=list(CTG_FEATURES),
        distance_columns=input_model.names_, early_stop_fit_files=train.iloc[a].source_file.nunique(),
        early_stop_validation_files=train.iloc[b].source_file.nunique(),
        elapsed_seconds=time.time() - start, notebook_execution="Result computation complete; notebook independently checks saved evidence")
    write_json(receipt_path, receipt)
    return receipt


def evaluate_stability(train, valid, input_model, bundles):
    """Fixed-config sensitivity, never outcome-based reselection.

    Two additional seeds and five 80% file subsamples. Each resample refits all
    preprocessing and reduction; comparison is on the same unchanged validation.
    This diagnoses sensitivity, not a confidence interval or new CV estimate.
    """
    rows, fitted = [], {}
    X, V = train[list(CTG_FEATURES)], valid[list(CTG_FEATURES)]
    common_train, common_valid = input_model.transform(X), input_model.transform(V)
    for key, bundle in bundles.items():
        reference = bundle["clusterer"].predict(project(bundle["reducer"], common_valid))
        for mode, count in [("seed", 2), ("file_subsample", 5)]:
            for repeat in range(count):
                seed = SEED + 1000 + repeat
                if mode == "file_subsample":
                    groups = np.array(sorted(train.source_file.unique()))
                    selected = np.random.default_rng(seed).choice(groups, int(.8 * len(groups)), replace=False)
                    ix = np.flatnonzero(train.source_file.isin(selected))
                else:
                    ix = np.arange(len(X))
                cache_key = (bundle["config"]["embedding_id"], mode, repeat)
                if cache_key not in fitted:
                    prep = input_model if mode == "seed" else EmbeddingInput().fit(X.iloc[ix])
                    ct = common_train if mode == "seed" else prep.transform(X.iloc[ix])
                    cv = common_valid if mode == "seed" else prep.transform(V)
                    reducer, _ = fit_embedding(bundle["spec"], ct,
                        seed=seed if mode == "seed" else SEED, epochs=bundle["epochs"])
                    fitted[cache_key] = project(reducer, ct, training=True), project(reducer, cv)
                zt, zv = fitted[cache_key]
                model = fit_cluster(bundle["config"], zt, seed=seed if mode == "seed" else SEED)
                labels = model.predict(zv)
                rows.append(dict(key=key, method=bundle["config"]["method"], algorithm=bundle["config"]["algorithm"],
                    mode=mode, repeat=repeat, train_segments=len(ix), train_files=train.iloc[ix].source_file.nunique(),
                    validation_ari_vs_reference=adjusted_rand_score(reference, labels),
                    converged=bool(getattr(model, "converged_", True))))
        print(f"Stability complete: {key}", flush=True)
    return pd.DataFrame(rows)
