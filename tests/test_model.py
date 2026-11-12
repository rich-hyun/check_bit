import json

import numpy as np
import pytest

from checkbit_ai.features import FEATURES, aggregate_minutes, build_features, load_trades
from checkbit_ai.lstm_ae import LSTMAutoEncoder, TrainConfig
from checkbit_ai.train import split_windows, train


def test_gradient_check():
    rng = np.random.default_rng(0)
    m = LSTMAutoEncoder(3, 5, 4, seed=1)
    X = rng.normal(size=(2, 5, 3))
    _, g = m.loss_and_grads(X)
    for k, p in m.params.items():
        for idx in list(np.ndindex(p.shape))[:12]:
            o = p[idx]
            p[idx] = o + 1e-6
            lp, _ = m.loss_and_grads(X)
            p[idx] = o - 1e-6
            lm, _ = m.loss_and_grads(X)
            p[idx] = o
            num = (lp - lm) / 2e-6
            assert abs(num - g[k][idx]) <= 1e-6 + 1e-3 * abs(num), (k, idx)


def test_training_reduces_loss_and_roundtrip(tmp_path):
    rng = np.random.default_rng(3)
    t = np.linspace(0, 6, 12)
    X = np.stack([np.stack([np.sin(t + p), np.cos(t + p)], -1) for p in rng.uniform(0, 6, 128)])
    m = LSTMAutoEncoder(2, 12, 8)
    before = m.scores(X).mean()
    m.fit(X[:100], X[100:], TrainConfig(hidden=8, epochs=25, lr=1e-2), log=lambda *a: None)
    after = m.scores(X).mean()
    assert after < before * 0.5
    m.save(tmp_path / "m.npz")
    m2 = LSTMAutoEncoder.load(tmp_path / "m.npz")
    np.testing.assert_allclose(m.scores(X), m2.scores(X))


def test_split_has_no_overlap():
    ends = np.arange(19, 500)
    tr, va, te = split_windows(ends, 20)
    assert va.min() - 19 > tr.max() and te.min() - 19 > va.max()


def test_train_artifacts_and_no_leakage(trained_model, synthetic_csv):
    meta = json.loads((trained_model / "meta.json").read_text(encoding="utf-8"))
    report = json.loads((trained_model / "report.json").read_text(encoding="utf-8"))
    assert (trained_model / "model.npz").exists() and (trained_model / "report.md").exists()
    assert meta["features"] == FEATURES and meta["data"]["label"] == "synthetic"
    assert meta["threshold"] > 0 and meta["score_steps"] == 5
    # Scaler mean must come from training rows only.
    trades, _ = load_trades([synthetic_csv])
    from checkbit_ai.features import window_end_indices
    f = build_features(aggregate_minutes(trades))
    ends = window_end_indices(f, meta["window"])
    tr, _, _ = split_windows(ends, meta["window"])
    rows = np.unique((tr[:, None] + np.arange(-meta["window"] + 1, 1)).ravel())
    np.testing.assert_allclose(meta["scaler"]["mean"], np.nanmean(f.X[rows], axis=0))
    assert not np.allclose(meta["scaler"]["mean"], np.nanmean(f.X[f.valid], axis=0))
    ev = report["evaluation"]
    assert "Accuracy" in ev["note"] and "synthetic_injection" in ev
    big = [r for r in ev["synthetic_injection"]["results"] if r["delta_std"] == 5.0][0]
    assert big["lstm_auc"] > 0.8  # strong synthetic perturbations must be separable


def test_injected_bursts_score_higher(trained_model, synthetic_csv):
    from checkbit_ai.features import Scaler, window_end_indices
    from checkbit_ai.lstm_ae import score_from_errors
    from conftest import SYN_ANOMALIES, SYN_START
    meta = json.loads((trained_model / "meta.json").read_text(encoding="utf-8"))
    m = LSTMAutoEncoder.load(trained_model / "model.npz")
    trades, _ = load_trades([synthetic_csv])
    f = build_features(aggregate_minutes(trades))
    ends = window_end_indices(f, meta["window"])
    Xs = Scaler.from_dict(meta["scaler"]).transform(np.nan_to_num(f.X))
    idx = ends[:, None] + np.arange(-meta["window"] + 1, 1)
    s = score_from_errors(m.errors(Xs[idx]), meta["score_steps"])
    burst_minutes = {int(SYN_START // 60) + a for a in SYN_ANOMALIES}
    at_burst = np.array([int(f.minutes[e]) in burst_minutes for e in ends])
    assert at_burst.sum() == 2
    assert s[at_burst].min() > meta["threshold"]       # injected bursts are flagged
    assert np.mean(s[~at_burst] > meta["threshold"]) < 0.1


def test_insufficient_data_is_reported(tmp_path, synthetic_csv):
    from checkbit_ai.synthetic import generate
    small = generate(tmp_path / "synthetic_small.csv", minutes=60)
    with pytest.raises(SystemExit, match="데이터 부족"):
        train([str(small)], tmp_path / "m", log=lambda *a: None)
