import numpy as np
from sklearn.metrics import adjusted_rand_score
from src.training.ctg_clustering import FuzzyCMeans


def test_fcm_objective_memberships_and_inductive_assignment():
    rng = np.random.default_rng(42)
    x = np.vstack([rng.normal(-3,.2,(30,2)),rng.normal(3,.2,(30,2))])
    model = FuzzyCMeans(n_clusters=2,n_init=2,max_iter=100).fit(x)
    assert model.converged_
    assert np.all(np.diff(model.objective_history_) <= 1e-7)
    np.testing.assert_allclose(model.predict_proba(x).sum(axis=1),1)
    assert adjusted_rand_score(np.repeat([0,1],30),model.predict(x)) == 1
    assert model.predict_proba(np.array([[-3,-3],[3,3]])).shape == (2,2)
