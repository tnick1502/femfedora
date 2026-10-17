"""Inductive, outcome-free CTG clustering and held-out descriptive associations."""
from __future__ import annotations

import json
import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from src.training.ctg_experiments import CTG_FEATURES, ROOT, OUT, MODELS, SEED, load_data, write_json


class FuzzyCMeans:
    def __init__(self, n_clusters=3, m=2., n_init=5, max_iter=300, tolerance=1e-6, seed=SEED):
        self.n_clusters, self.m, self.n_init = n_clusters, m, n_init
        self.max_iter, self.tolerance, self.seed = max_iter, tolerance, seed

    def membership(self, X, centers):
        d2 = np.maximum(cdist(X, centers, metric='sqeuclidean'), 1e-12)
        log_weights = -np.log(d2) / (self.m - 1)
        log_weights -= log_weights.max(axis=1, keepdims=True)
        weights = np.exp(log_weights)
        return weights / weights.sum(axis=1, keepdims=True)

    def fit(self, X):
        if self.m <= 1:
            raise ValueError('FCM requires m > 1')
        X = np.asarray(X, dtype=float)
        solutions = []
        for run in range(self.n_init):
            centers = KMeans(n_clusters=self.n_clusters, n_init=1, random_state=self.seed + run).fit(X).cluster_centers_
            u, trace = self.membership(X, centers), []
            converged = False
            for iteration in range(self.max_iter):
                w = u ** self.m
                centers = w.T @ X / np.maximum(w.sum(axis=0)[:, None], 1e-12)
                u = self.membership(X, centers)
                objective = float(np.sum(u ** self.m * cdist(X, centers, metric='sqeuclidean')))
                trace.append(objective)
                if len(trace) > 1 and abs(trace[-2] - trace[-1]) <= self.tolerance * max(1., abs(trace[-2])):
                    converged = True
                    break
            assert np.all(np.diff(trace) <= 1e-7 * max(1., trace[0]))
            solutions.append(dict(centers=centers, u=u, trace=trace, converged=converged))
        best = min(solutions, key=lambda s: s['trace'][-1])
        self.centers_, self.membership_, self.objective_history_ = best['centers'], best['u'], best['trace']
        self.converged_ = best['converged']
        labels = self.membership_.argmax(axis=1)
        others = [s for s in solutions if s is not best]
        self.initialization_ari_ = float(np.mean([adjusted_rand_score(labels, s['u'].argmax(axis=1)) for s in others])) if others else 1.
        self.fpc_ = float(np.mean(np.sum(self.membership_ ** 2, axis=1)))
        self.minimum_center_distance_ = float(pdist(self.centers_).min())
        return self

    def predict_proba(self, X):
        return self.membership(np.asarray(X), self.centers_)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


class ClusterProjector:
    def fit(self, X):
        self.preprocessor_ = clone(joblib.load(ROOT / 'data/processed/ctg_uci/preprocessor.joblib'))
        transformed = self.preprocessor_.fit_transform(X)
        names = self.preprocessor_.get_feature_names_out()
        self.numeric_idx_ = [i for i, name in enumerate(names) if not name.startswith(('rare_present__', 'tendency__'))]
        self.category_idx_ = [i for i, name in enumerate(names) if name.startswith('tendency__')]
        self.scaler_ = StandardScaler().fit(transformed[:, self.numeric_idx_])
        encoded = np.column_stack([self.scaler_.transform(transformed[:, self.numeric_idx_]),
                                   transformed[:, self.category_idx_] / np.sqrt(2)])
        self.pca_ = PCA(n_components=.90, svd_solver='full').fit(encoded)
        return self

    def transform(self, X):
        z = self.preprocessor_.transform(X)
        encoded = np.column_stack([self.scaler_.transform(z[:, self.numeric_idx_]), z[:, self.category_idx_] / np.sqrt(2)])
        return self.pca_.transform(encoded)


def _score(X, labels, k):
    counts = np.bincount(labels, minlength=k)
    if len(np.unique(labels)) < 2 or len(np.unique(labels)) >= len(labels):
        return -1., float(counts.min() / len(labels))
    return float(silhouette_score(X, labels)), float(counts.min() / len(labels))


def _membership(model, X, algorithm):
    if algorithm == 'KMeans':
        return np.eye(model.n_clusters)[model.predict(X)]
    return model.predict_proba(X)


def _cluster_bootstrap(part, repetitions=500):
    rng = np.random.default_rng(SEED + 400)
    groups = part.source_file.unique()
    indices = {g: np.flatnonzero(part.source_file.to_numpy() == g) for g in groups}
    clusters = sorted(part.cluster.unique())
    rates = {(c, outcome): [] for c in clusters for outcome in ['path', 'suspect', 'abnormal']}
    ratios = {(c, outcome): [] for c in clusters for outcome in ['path', 'suspect', 'abnormal']}
    for _ in range(repetitions):
        ix = np.concatenate([indices[g] for g in rng.choice(groups, len(groups), replace=True)])
        sample = part.iloc[ix]
        for c in clusters:
            inside = sample[sample.cluster == c]
            for outcome, label in [('path', 3), ('suspect', 2), ('abnormal', None)]:
                baseline = float((sample.NSP != 1).mean()) if label is None else float((sample.NSP == label).mean())
                rate = (float((inside.NSP != 1).mean()) if label is None else float((inside.NSP == label).mean())) if len(inside) else np.nan
                rates[(c, outcome)].append(rate)
                ratios[(c, outcome)].append(rate / baseline if baseline > 0 else np.nan)
    result = {}
    for c in clusters:
        result[c] = {}
        for outcome in ['path', 'suspect', 'abnormal']:
            prefix = '' if outcome == 'path' else outcome + '_'
            result[c][outcome + '_fraction_low'] = float(np.nanquantile(rates[(c,outcome)], .025))
            result[c][outcome + '_fraction_high'] = float(np.nanquantile(rates[(c,outcome)], .975))
            result[c][prefix + 'enrichment_low'] = float(np.nanquantile(ratios[(c,outcome)], .025))
            result[c][prefix + 'enrichment_high'] = float(np.nanquantile(ratios[(c,outcome)], .975))
    return result


def run_clustering():
    data, fingerprint = load_data()
    train = data[data.split == 'train'].reset_index(drop=True)
    valid = data[data.split == 'validation'].reset_index(drop=True)
    projector = ClusterProjector().fit(train[list(CTG_FEATURES)])
    z_train, z_valid = projector.transform(train[list(CTG_FEATURES)]), projector.transform(valid[list(CTG_FEATURES)])
    rows, candidates = [], {}
    for algorithm in ['FCM', 'KMeans', 'GMM']:
        for k in range(2, 7):
            variants = [1.5, 2., 2.5] if algorithm == 'FCM' else (['diag', 'full'] if algorithm == 'GMM' else [None])
            for variant in variants:
                if algorithm == 'FCM':
                    model = FuzzyCMeans(k, m=variant).fit(z_train)
                elif algorithm == 'KMeans':
                    model = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(z_train)
                else:
                    model = GaussianMixture(n_components=k, covariance_type=variant, n_init=5,
                                            max_iter=300, reg_covar=1e-4, random_state=SEED).fit(z_train)
                train_labels, valid_labels = model.predict(z_train), model.predict(z_valid)
                train_score, train_frac = _score(z_train, train_labels, k)
                valid_score, valid_frac = _score(z_valid, valid_labels, k)
                key = f'{algorithm}_{k}_{variant}'
                candidates[key] = model
                rows.append(dict(key=key, algorithm=algorithm, k=k, variant=variant,
                    train_silhouette=train_score, validation_silhouette=valid_score,
                    minimum_train_fraction=train_frac, minimum_validation_fraction=valid_frac,
                    converged=bool(getattr(model, 'converged_', True)),
                    fpc=getattr(model, 'fpc_', np.nan), initialization_ari=getattr(model, 'initialization_ari_', np.nan),
                    minimum_center_distance=getattr(model, 'minimum_center_distance_', np.nan),
                    bic=float(model.bic(z_train)) if algorithm == 'GMM' else np.nan))
        print(f'{algorithm}: full unsupervised grid complete', flush=True)
    table = pd.DataFrame(rows)
    chosen = {}
    for algorithm, part in table.groupby('algorithm'):
        eligible = part[(part.minimum_train_fraction >= .03) & (part.minimum_validation_fraction >= .03) & part.converged]
        if algorithm == 'FCM':
            eligible = eligible[eligible.minimum_center_distance >= 1e-3]
        if eligible.empty:
            raise ValueError(f'No non-degenerate converged clustering for {algorithm}')
        chosen[algorithm] = eligible.sort_values(['validation_silhouette','k'], ascending=[False,True]).iloc[0]['key']
    # No NSP or CLASS has entered any estimator or criterion above.
    write_json(OUT / 'cluster_locked_config.json', dict(fingerprint=fingerprint, selection='validation silhouette without outcomes',
        chosen=chosen, pca_components=int(projector.pca_.n_components_),
        explained_variance=float(projector.pca_.explained_variance_ratio_.sum()),
        removed_distance_columns=['rare_present__DP','rare_present__Nzeros'],
        minimum_cluster_fraction=.03))
    table['chosen'] = table.key.isin(chosen.values())
    table.to_csv(OUT / 'cluster_grid.csv', index=False)
    association, summaries, assignments = [], [], []
    models = {}
    for algorithm, key in chosen.items():
        models[algorithm] = candidates[key]
        for split, frame in data.groupby('split'):
            z = projector.transform(frame[list(CTG_FEATURES)])
            memberships = _membership(candidates[key], z, algorithm)
            labels = memberships.argmax(axis=1)
            n_clusters = memberships.shape[1]
            assignment = frame[['source_excel_row','source_file','NSP']].copy()
            assignment['algorithm'], assignment['split'], assignment['cluster'] = algorithm, split, labels
            entropy = -np.sum(memberships * np.log(np.clip(memberships, 1e-12, 1)), axis=1) / np.log(n_clusters)
            assignment['membership_entropy'] = entropy
            for j in range(n_clusters):
                assignment[f'membership_{j}'] = memberships[:, j]
            assignments.append(assignment)
            association.append(dict(algorithm=algorithm, split=split, k=n_clusters,
                ari=adjusted_rand_score(frame.NSP, labels), nmi=normalized_mutual_info_score(frame.NSP, labels),
                silhouette=_score(z, labels, n_clusters)[0], mean_membership_entropy=float(entropy.mean())))
            if split == 'test':
                intervals = _cluster_bootstrap(assignment)
                baseline = float((frame.NSP == 3).mean())
                for cluster, inside in assignment.groupby('cluster'):
                    rate = float((inside.NSP == 3).mean())
                    weights = memberships[:, int(cluster)]
                    soft_rate = float(np.sum(weights * (frame.NSP.to_numpy() == 3)) / weights.sum())
                    summaries.append(dict(algorithm=algorithm, cluster=int(cluster), segments=len(inside),
                        source_files=inside.source_file.nunique(), pathological=int((inside.NSP == 3).sum()),
                        small_cluster=(inside.source_file.nunique() < 10 or (inside.NSP == 3).sum() < 5),
                        membership_mass=float(weights.sum()), membership_weighted_path_fraction=soft_rate,
                        suspect_fraction=float((inside.NSP == 2).mean()),
                        abnormal_fraction=float((inside.NSP != 1).mean()),
                        suspect_enrichment=float((inside.NSP == 2).mean() / (frame.NSP == 2).mean()),
                        abnormal_enrichment=float((inside.NSP != 1).mean() / (frame.NSP != 1).mean()),
                        path_fraction=rate, enrichment=rate / baseline, **intervals[cluster]))
                if algorithm == 'FCM':
                    projection = pd.DataFrame({'PC1':z[:,0], 'PC2':z[:,1], 'cluster':labels,
                                               'NSP':frame.NSP.to_numpy(), 'source_excel_row':frame.source_excel_row.to_numpy()})
                    projection.to_parquet(OUT / 'cluster_projection.parquet', index=False)
    pd.concat(assignments, ignore_index=True).to_parquet(OUT / 'cluster_assignments.parquet', index=False)
    pd.DataFrame(association).to_csv(OUT / 'cluster_association.csv', index=False)
    pd.DataFrame(summaries).to_csv(OUT / 'cluster_test_summary.csv', index=False)
    joblib.dump(dict(projector=projector, models=models), MODELS / 'clustering.joblib')
    write_json(OUT / 'fcm_convergence.json', {key:candidates[key].objective_history_ for key in chosen.values() if key.startswith('FCM')})
    return table, pd.DataFrame(association), pd.DataFrame(summaries)
