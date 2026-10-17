"""Deterministic folds balancing label rows and positive recording support.

Only NSP and source_file (and a duplicate-group safety check) are consulted.
No predictor, prediction, or model score participates in split construction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def balanced_group_folds(frame: pd.DataFrame, n_splits: int, seed: int):
    """Return positional (fit, hold) arrays, with recording-level stratification.

    A recording can contain multiple NSP classes. Assignment therefore balances
    a six-dimensional vector: its three row counts and three class presences.
    We also balance total rows and total recording counts. Twelve deterministic
    randomized greedy restarts plus local swaps reduce the multiobjective loss.
    Rare-class recording support is weighted more strongly than row balance;
    perfect row balance is neither promised nor possible for oversized groups.
    """
    if n_splits < 2 or frame.empty:
        raise ValueError('At least two nonempty folds are required')
    required = {'NSP', 'source_file'}
    if not required.issubset(frame.columns) or frame[list(required)].isna().any().any():
        raise ValueError('NSP and source_file must be present and nonmissing')
    if set(frame.NSP.unique()) != {1, 2, 3}:
        raise ValueError('NSP must contain all three classes with codes 1, 2, 3')
    if 'feature_duplicate_group' in frame:
        dup = frame.dropna(subset=['feature_duplicate_group'])
        if dup.groupby('feature_duplicate_group').source_file.nunique().max() > 1:
            raise ValueError('Feature duplicates span recordings; merge recording groups before splitting')
    group_codes, group_names = pd.factorize(frame.source_file, sort=False)
    n_groups = len(group_names)
    if n_groups < n_splits:
        raise ValueError('Fewer recordings than folds')
    counts = np.zeros((n_groups, 3), dtype=float)
    np.add.at(counts, (group_codes, frame.NSP.to_numpy(dtype=int) - 1), 1.)
    presence = (counts > 0).astype(float)
    if (presence.sum(axis=0) < n_splits).any():
        raise ValueError('Every class needs at least n_splits distinct recordings')
    vectors = np.column_stack([counts, presence, counts.sum(axis=1), np.ones(n_groups)])
    target = vectors.sum(axis=0) / n_splits
    # Comparable relative deviations; emphasize independent class support.
    weight = np.array([1., 1., 1., 4., 4., 6., .5, .5]) / np.square(target)
    minimum_presence = np.maximum(1., np.floor(presence.sum(axis=0) / n_splits) - 1.)
    rng = np.random.default_rng(seed)

    def loss(load):
        deficit = np.maximum(0., minimum_presence - load[:, 3:6])
        return float(np.sum(np.square(load - target) * weight)
                     + 20. * np.square(deficit).sum())

    best = None
    rarity = np.sum(counts / counts.sum(axis=0), axis=1) + np.sum(presence / presence.sum(axis=0), axis=1)
    for restart in range(12):
        order = np.argsort(-(rarity * rng.uniform(.85, 1.15, size=n_groups)), kind='stable')
        load = np.zeros((n_splits, 8))
        assignment = np.full(n_groups, -1, dtype=int)
        # Greedy partial-load objective avoids applying support deficits before
        # there have been enough assignments to fill every fold.
        for group in order:
            candidate = np.sum((2 * (load - target) * vectors[group] + np.square(vectors[group])) * weight, axis=1)
            tied = np.flatnonzero(np.isclose(candidate, candidate.min(), rtol=0, atol=1e-12))
            fold = int(rng.choice(tied))
            assignment[group] = fold
            load[fold] += vectors[group]
        current = loss(load)
        # Moves and swaps preserve indivisible recordings. Swaps permit escaping
        # a local optimum in which no individual recording can move profitably.
        for step in range(1400):
            a = int(rng.integers(n_groups))
            old = assignment[a]
            if step % 4 == 0:
                new = int(rng.integers(n_splits))
                if old == new or load[old, -1] <= 1:
                    continue
                delta = vectors[a]
                proposal = load.copy()
                proposal[old] -= delta
                proposal[new] += delta
                score = loss(proposal)
                if score < current - 1e-12:
                    load, current = proposal, score
                    assignment[a] = new
            else:
                b = int(rng.integers(n_groups))
                new = assignment[b]
                if old == new:
                    continue
                delta = vectors[a] - vectors[b]
                proposal = load.copy()
                proposal[old] -= delta
                proposal[new] += delta
                score = loss(proposal)
                if score < current - 1e-12:
                    load, current = proposal, score
                    assignment[a], assignment[b] = new, old
        if best is None or current < best[0] - 1e-12:
            best = (current, assignment.copy())
    row_folds = best[1][group_codes]
    folds = []
    for k in range(n_splits):
        fit, hold = np.flatnonzero(row_folds != k), np.flatnonzero(row_folds == k)
        if set(frame.iloc[fit].NSP) != {1, 2, 3} or set(frame.iloc[hold].NSP) != {1, 2, 3}:
            raise ValueError('Unable to allocate all classes to every fold')
        assert set(frame.iloc[fit].source_file).isdisjoint(frame.iloc[hold].source_file)
        if 'feature_duplicate_group' in frame:
            a = set(frame.iloc[fit].feature_duplicate_group.dropna())
            b = set(frame.iloc[hold].feature_duplicate_group.dropna())
            assert a.isdisjoint(b)
        folds.append((fit, hold))
    return folds


def fold_inventory(frame: pd.DataFrame, folds) -> pd.DataFrame:
    """Human-readable independent-group support alongside segment counts."""
    rows = []
    for k, (fit, hold) in enumerate(folds):
        part = frame.iloc[hold]
        row = {'fold': k, 'n_fit': len(fit), 'n_hold': len(hold),
               'source_files': int(part.source_file.nunique())}
        for code, label in [(1, 'normal'), (2, 'suspect'), (3, 'path')]:
            inside = part[part.NSP == code]
            row[f'{label}_segments'] = len(inside)
            row[f'{label}_files'] = int(inside.source_file.nunique())
        rows.append(row)
    return pd.DataFrame(rows)
