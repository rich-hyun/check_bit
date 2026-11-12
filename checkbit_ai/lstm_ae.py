"""Lightweight LSTM AutoEncoder implemented with NumPy only.

Why NumPy instead of PyTorch/TensorFlow: the production server is a small
AWS Lightsail instance. Inference needs only `numpy` (a few MB of RAM, a few ms
per window) and training is a separate offline CLI (`python -m checkbit_ai.train`).

Architecture (sequence-to-sequence reconstruction):
    x[0..T-1] (F features) --encoder LSTM(H)--> z = h_enc[T-1]
    z repeated T times --decoder LSTM(H)--> s[t] --linear--> x_hat[t]
Loss: mean squared reconstruction error. Gradients via full BPTT (verified by a
numerical gradient check in tests/test_lstm_ae.py).
"""
from __future__ import annotations

import io
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * x))  # numerically stable


def _lstm_init(rng: np.random.Generator, n_in: int, hidden: int) -> dict[str, np.ndarray]:
    lim_x = math.sqrt(6.0 / (n_in + 4 * hidden))
    lim_h = math.sqrt(6.0 / (hidden + 4 * hidden))
    b = np.zeros(4 * hidden)
    b[hidden:2 * hidden] = 1.0  # forget-gate bias = 1 (standard trick)
    return {"Wx": rng.uniform(-lim_x, lim_x, (n_in, 4 * hidden)),
            "Wh": rng.uniform(-lim_h, lim_h, (hidden, 4 * hidden)), "b": b}


def lstm_forward(X: np.ndarray, p: dict[str, np.ndarray]):
    """X: (B, T, In) -> hs: (B, T, H) plus cache for backward."""
    B, T, _ = X.shape
    H = p["Wh"].shape[0]
    h = np.zeros((B, H))
    c = np.zeros((B, H))
    hs = np.zeros((B, T, H))
    cache = []
    xw = X @ p["Wx"] + p["b"]  # (B, T, 4H) input projection for all steps at once
    for t in range(T):
        a = xw[:, t] + h @ p["Wh"]
        i = _sigmoid(a[:, :H])
        f = _sigmoid(a[:, H:2 * H])
        o = _sigmoid(a[:, 2 * H:3 * H])
        g = np.tanh(a[:, 3 * H:])
        c_prev, h_prev = c, h
        c = f * c_prev + i * g
        tc = np.tanh(c)
        h = o * tc
        hs[:, t] = h
        cache.append((i, f, o, g, c_prev, h_prev, tc))
    return hs, cache


def lstm_backward(dhs: np.ndarray, X: np.ndarray, p: dict[str, np.ndarray], cache):
    """dhs: (B, T, H) gradient w.r.t. every hidden output. Returns dX, grads."""
    B, T, _ = X.shape
    H = p["Wh"].shape[0]
    dWx = np.zeros_like(p["Wx"])
    dWh = np.zeros_like(p["Wh"])
    db = np.zeros_like(p["b"])
    dX = np.zeros_like(X, dtype=float)
    dh_next = np.zeros((B, H))
    dc_next = np.zeros((B, H))
    for t in reversed(range(T)):
        i, f, o, g, c_prev, h_prev, tc = cache[t]
        dh = dhs[:, t] + dh_next
        do = dh * tc
        dc = dh * o * (1.0 - tc * tc) + dc_next
        di = dc * g
        dg = dc * i
        df = dc * c_prev
        dc_next = dc * f
        da = np.concatenate([di * i * (1 - i), df * f * (1 - f), do * o * (1 - o), dg * (1 - g * g)], axis=1)
        dWx += X[:, t].T @ da
        dWh += h_prev.T @ da
        db += da.sum(axis=0)
        dX[:, t] = da @ p["Wx"].T
        dh_next = da @ p["Wh"].T
    return dX, {"Wx": dWx, "Wh": dWh, "b": db}


@dataclass
class TrainConfig:
    hidden: int = 24
    epochs: int = 60
    batch_size: int = 64
    lr: float = 3e-3
    patience: int = 8
    clip_norm: float = 5.0
    seed: int = 42


class LSTMAutoEncoder:
    PARAM_NAMES = ("enc_Wx", "enc_Wh", "enc_b", "dec_Wx", "dec_Wh", "dec_b", "out_W", "out_b")

    def __init__(self, n_features: int, window: int, hidden: int = 24, seed: int = 42) -> None:
        self.n_features, self.window, self.hidden = int(n_features), int(window), int(hidden)
        rng = np.random.default_rng(seed)
        enc = _lstm_init(rng, n_features, hidden)
        dec = _lstm_init(rng, hidden, hidden)
        lim = math.sqrt(6.0 / (hidden + n_features))
        self.params: dict[str, np.ndarray] = {
            "enc_Wx": enc["Wx"], "enc_Wh": enc["Wh"], "enc_b": enc["b"],
            "dec_Wx": dec["Wx"], "dec_Wh": dec["Wh"], "dec_b": dec["b"],
            "out_W": rng.uniform(-lim, lim, (hidden, n_features)), "out_b": np.zeros(n_features)}

    # ------------------------------------------------------------------ core
    def _sub(self, prefix: str) -> dict[str, np.ndarray]:
        return {k: self.params[f"{prefix}_{k}"] for k in ("Wx", "Wh", "b")}

    def forward(self, X: np.ndarray):
        X = np.asarray(X, dtype=float)
        hs_e, cache_e = lstm_forward(X, self._sub("enc"))
        z = hs_e[:, -1]
        U = np.repeat(z[:, None, :], X.shape[1], axis=1)
        hs_d, cache_d = lstm_forward(U, self._sub("dec"))
        Y = hs_d @ self.params["out_W"] + self.params["out_b"]
        return Y, (X, hs_e, cache_e, U, hs_d, cache_d)

    def reconstruct(self, X: np.ndarray, batch: int = 512) -> np.ndarray:
        out = [self.forward(X[i:i + batch])[0] for i in range(0, len(X), batch)]
        return np.concatenate(out) if out else np.zeros_like(X, dtype=float)

    def loss_and_grads(self, X: np.ndarray):
        Y, (X, hs_e, cache_e, U, hs_d, cache_d) = self.forward(X)
        diff = Y - X
        loss = float(np.mean(diff * diff))
        dY = 2.0 * diff / diff.size
        grads = {"out_W": np.einsum("bth,btf->hf", hs_d, dY), "out_b": dY.sum(axis=(0, 1))}
        dhs_d = dY @ self.params["out_W"].T
        dU, g_dec = lstm_backward(dhs_d, U, self._sub("dec"), cache_d)
        dz = dU.sum(axis=1)
        dhs_e = np.zeros_like(hs_e)
        dhs_e[:, -1] = dz
        _, g_enc = lstm_backward(dhs_e, X, self._sub("enc"), cache_e)
        for k, v in g_dec.items():
            grads[f"dec_{k}"] = v
        for k, v in g_enc.items():
            grads[f"enc_{k}"] = v
        return loss, grads

    # --------------------------------------------------------------- scoring
    def errors(self, X: np.ndarray) -> np.ndarray:
        """Squared reconstruction error per window, time step and feature: (N, T, F)."""
        Y = self.reconstruct(X)
        return (Y - X) ** 2

    def scores(self, X: np.ndarray, last_steps: int | None = None) -> np.ndarray:
        """Anomaly score = mean squared reconstruction error, over the whole
        window or only its `last_steps` most recent minutes."""
        if len(X) == 0:
            return np.zeros(0)
        return score_from_errors(self.errors(X), last_steps)

    # -------------------------------------------------------------- training
    def fit(self, X_train: np.ndarray, X_val: np.ndarray, cfg: TrainConfig, log=print) -> dict:
        rng = np.random.default_rng(cfg.seed)
        m = {k: np.zeros_like(v) for k, v in self.params.items()}
        v2 = {k: np.zeros_like(v) for k, v in self.params.items()}
        b1, b2, eps, step = 0.9, 0.999, 1e-8, 0
        best = (math.inf, None, 0)
        history = []
        t0 = time.time()
        for epoch in range(1, cfg.epochs + 1):
            order = rng.permutation(len(X_train))
            losses = []
            for s in range(0, len(order), cfg.batch_size):
                batch = X_train[order[s:s + cfg.batch_size]]
                loss, grads = self.loss_and_grads(batch)
                losses.append(loss * len(batch))
                norm = math.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
                scale = min(1.0, cfg.clip_norm / (norm + 1e-12))
                step += 1
                for k, g in grads.items():
                    g = g * scale
                    m[k] = b1 * m[k] + (1 - b1) * g
                    v2[k] = b2 * v2[k] + (1 - b2) * g * g
                    mh = m[k] / (1 - b1 ** step)
                    vh = v2[k] / (1 - b2 ** step)
                    self.params[k] -= cfg.lr * mh / (np.sqrt(vh) + eps)
            train_loss = sum(losses) / max(1, len(X_train))
            val_loss = float(self.scores(X_val).mean()) if len(X_val) else train_loss
            history.append({"epoch": epoch, "train_loss": round(train_loss, 6), "val_loss": round(val_loss, 6)})
            log(f"epoch {epoch:3d}  train_mse={train_loss:.5f}  val_mse={val_loss:.5f}")
            if val_loss < best[0] - 1e-6:
                best = (val_loss, {k: v.copy() for k, v in self.params.items()}, epoch)
            elif epoch - best[2] >= cfg.patience:
                log(f"early stopping (best epoch {best[2]})")
                break
        if best[1] is not None:
            self.params = best[1]
        return {"history": history, "best_epoch": best[2], "best_val_loss": round(best[0], 6),
                "train_seconds": round(time.time() - t0, 1)}

    # ----------------------------------------------------------- persistence
    def n_params(self) -> int:
        return int(sum(v.size for v in self.params.values()))

    def save(self, path: Path) -> None:
        buf = io.BytesIO()
        np.savez_compressed(buf, **self.params,
                            _shape=np.array([self.n_features, self.window, self.hidden]))
        Path(path).write_bytes(buf.getvalue())

    @classmethod
    def load(cls, path: Path) -> "LSTMAutoEncoder":
        with np.load(Path(path), allow_pickle=False) as data:
            nf, w, h = (int(x) for x in data["_shape"])
            model = cls(nf, w, h)
            for k in cls.PARAM_NAMES:
                arr = data[k]
                if arr.shape != model.params[k].shape:
                    raise ValueError(f"parameter {k} has shape {arr.shape}, expected {model.params[k].shape}")
                if not np.all(np.isfinite(arr)):
                    raise ValueError(f"parameter {k} contains non-finite values")
                model.params[k] = arr.astype(float)
        return model


def score_from_errors(err: np.ndarray, last_steps: int | None = None) -> np.ndarray:
    """err: (N, T, F) squared errors -> (N,) scores."""
    if last_steps:
        err = err[:, -last_steps:]
    return err.mean(axis=(1, 2))


def write_json(path: Path, obj) -> None:
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
