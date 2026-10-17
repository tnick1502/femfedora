import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from src.training.ctg_embeddings import DenoisingAutoencoder, choose_configs, risk_bootstrap


def test_autoencoder_bottleneck_decoder_and_serialization(tmp_path):
    rng = np.random.default_rng(8)
    latent = rng.normal(size=(180, 2))
    x = latent @ rng.normal(size=(2, 6)) / 2
    model = DenoisingAutoencoder(n_components=2, width=12, activation="relu",
        max_epochs=150, noise=.02, seed=8).fit(x[:140], X_stop=x[140:])
    assert model.transform(x).shape == (180, 2)
    np.testing.assert_allclose(model.inverse_transform(model.transform(x)), model.model_.predict(x), atol=1e-12)
    assert mean_squared_error(x[140:], model.model_.predict(x[140:])) < np.mean(x[140:] ** 2) * .6
    path = tmp_path / "ae.joblib"
    joblib.dump(model, path)
    np.testing.assert_allclose(joblib.load(path).transform(x), model.transform(x))
    assert 1 <= model.best_epoch_ <= 150


def test_selection_uses_common_geometry_and_rejects_ineligible():
    rows = []
    for algorithm in ["FCM", "GMM", "KMeans"]:
        for key, common, embedded, eligible in [("a", .2, .8, True), ("b", .3, .4, True), ("c", .9, .9, False)]:
            rows.append(dict(method="UMAP", algorithm=algorithm, key=key, embedding_id=key,
                dimensions=5, k=2, validation_group_silhouette_common=common,
                validation_silhouette_embedding=embedded, eligible=eligible, outcome_ari=1-common))
    selected = choose_configs(pd.DataFrame(rows))
    assert set(selected.key) == {"b"}


def test_block_bootstrap_constant_composition_has_exact_rates():
    frame = pd.DataFrame({"source_file": np.repeat(["a", "b", "c"], 6),
                          "NSP": [1, 2, 3, 1, 2, 3] * 3})
    labels = np.tile([0, 0, 0, 1, 1, 1], 3)
    result = risk_bootstrap(frame, labels, 2, repetitions=20)
    for c in range(2):
        assert result[c]["path_fraction_low"] == 1/3
        assert result[c]["path_fraction_high"] == 1/3
        assert result[c]["path_enrichment_low"] == 1
