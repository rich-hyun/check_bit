"""Offline training CLI. Never run automatically on the production server.

Example:
    python -m checkbit_ai.train --csv data/btc_anomaly_dataset.csv --out models/lstm_ae

Steps: load & clean CSV -> 1-minute bars -> causal features -> chronological
train/val/test split (no overlapping windows across splits) -> scaler fitted on
train only -> LSTM AutoEncoder training with early stopping on validation ->
threshold = quantile of validation scores -> evaluation -> artifacts:
    model.npz, meta.json, report.json, report.md
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .baseline import baseline_scores, roc_auc, spearman
from .features import (FEATURES, FEATURE_LABELS, Scaler, aggregate_minutes, build_features, load_trades,
                       make_windows, minute_iso, window_end_indices)
from .lstm_ae import LSTMAutoEncoder, TrainConfig, write_json

MODEL_TYPE = "lstm_autoencoder_numpy"


def _describe(x: np.ndarray) -> dict:
    if len(x) == 0:
        return {"n": 0}
    q = np.quantile(x, [0.5, 0.9, 0.95, 0.99])
    return {"n": int(len(x)), "mean": float(x.mean()), "std": float(x.std()), "p50": float(q[0]),
            "p90": float(q[1]), "p95": float(q[2]), "p99": float(q[3]), "max": float(x.max())}


def robust_threshold(scores: np.ndarray, k: float) -> float:
    """median + k * 1.4826 * MAD: insensitive to the anomalies that may be
    contained in the (unlabelled) validation split."""
    med = float(np.median(scores))
    mad = float(np.median(np.abs(scores - med))) * 1.4826
    return med + k * mad


def pick_threshold(scores: np.ndarray, method: str, quantile: float, k: float) -> tuple[float, str]:
    if method == "quantile":
        return float(np.quantile(scores, quantile)), f"validation score quantile q={quantile}"
    return robust_threshold(scores, k), f"validation median + {k}*MAD (robust)"


def split_windows(ends: np.ndarray, window: int, ratios=(0.70, 0.15, 0.15)):
    """Chronological split of window end indices; windows of a later split never
    share input rows with an earlier split (purging)."""
    n = len(ends)
    n_tr = int(n * ratios[0])
    n_va = int(n * ratios[1])
    tr = ends[:n_tr]
    rest_va = ends[n_tr:n_tr + n_va]
    rest_te = ends[n_tr + n_va:]
    last_tr = tr[-1] if len(tr) else -1
    va = rest_va[rest_va - window + 1 > last_tr]
    last_va = va[-1] if len(va) else last_tr
    te = rest_te[rest_te - window + 1 > last_va]
    return tr, va, te


def inject(Xs: np.ndarray, rng: np.random.Generator, delta: float, last_steps: int = 3, n_feat: int = 3):
    """Synthetic anomaly: shift `n_feat` random features by +-delta std over the
    last `last_steps` minutes of each window (in standardized space)."""
    out = Xs.copy()
    for k in range(len(out)):
        feats = rng.choice(out.shape[2], size=n_feat, replace=False)
        signs = rng.choice([-1.0, 1.0], size=n_feat)
        out[k, -last_steps:, feats] += (signs * delta)[:, None]
    return out


def train(csv_paths: list[str], out_dir: Path, window: int = 20, hidden: int = 24, epochs: int = 60,
          quantile: float = 0.99, threshold_method: str = "mad", mad_k: float = 4.0, score_steps: int = 5, min_train_windows: int = 200, data_label: str = "collected",
          market: str = "KRW-BTC", seed: int = 42, log=print) -> dict:
    trades, clean = load_trades(csv_paths, market)
    log(f"trades: {len(trades):,} valid / clean stats: {clean.as_dict()}")
    bars = aggregate_minutes(trades)
    frame = build_features(bars)
    ends = window_end_indices(frame, window)
    log(f"minute bars: {len(frame):,} (filled {int(frame.filled.sum())}), segments: "
        f"{int(frame.segments.max()) + 1 if len(frame) else 0}, valid windows: {len(ends):,}")
    tr, va, te = split_windows(ends, window)
    if len(tr) < min_train_windows or len(va) < 10:
        raise SystemExit(f"데이터 부족: 학습 윈도우 {len(tr)}개 / 검증 {len(va)}개 "
                         f"(최소 학습 {min_train_windows}개, 검증 10개 필요). 더 오래 수집한 뒤 다시 실행하세요.")

    train_rows = np.unique((tr[:, None] + np.arange(-window + 1, 1)[None, :]).ravel())
    scaler = Scaler.fit(frame.X[train_rows])
    Xs = scaler.transform(np.nan_to_num(frame.X, nan=0.0))
    W_tr, W_va, W_te = (make_windows(Xs, e, window) for e in (tr, va, te))

    model = LSTMAutoEncoder(len(FEATURES), window, hidden, seed=seed)
    cfg = TrainConfig(hidden=hidden, epochs=epochs, seed=seed)
    log(f"training LSTM AE: params={model.n_params():,}, train={len(W_tr)}, val={len(W_va)}, test={len(W_te)}")
    fit = model.fit(W_tr, W_va, cfg, log=log)

    s_tr, s_va, s_te = (model.scores(x, score_steps) for x in (W_tr, W_va, W_te))
    threshold, threshold_desc = pick_threshold(s_va, threshold_method, quantile, mad_k)
    alt_thresholds = {"quantile": float(np.quantile(s_va, quantile)), "mad": robust_threshold(s_va, mad_k)}
    b_tr, b_va, b_te = (baseline_scores(x, score_steps) for x in (W_tr, W_va, W_te))
    b_threshold, _ = pick_threshold(b_va, threshold_method, quantile, mad_k)
    val_feat_err = model.errors(W_va).mean(axis=(0, 1))

    raw_train = frame.X[train_rows]
    feature_stats = {}
    for j, name in enumerate(FEATURES):
        q = np.quantile(raw_train[:, j], [0.05, 0.5, 0.95])
        feature_stats[name] = {"p05": float(q[0]), "p50": float(q[1]), "p95": float(q[2]),
                               "mean": float(raw_train[:, j].mean()), "std": float(raw_train[:, j].std())}

    # ---- evaluation (unsupervised: no ground-truth anomaly labels exist) ----
    evaluation: dict = {
        "note": ("실제 이상 라벨이 없으므로 Accuracy/Precision/Recall은 산출하지 않습니다. "
                 "아래는 (1) 재구성 오차 분포, (2) 테스트 구간 경보율, (3) 통계 기준 모델과의 비교, "
                 "(4) 합성 이상 주입 시험 결과입니다. (4)는 인위적 교란에 대한 민감도이며 실제 시장 이상 탐지 성능이 아닙니다."),
        "score_distribution": {"train": _describe(s_tr), "val": _describe(s_va), "test": _describe(s_te)},
        "alarm_rate": {"val": float(np.mean(s_va > threshold)),
                       "test": float(np.mean(s_te > threshold)) if len(s_te) else None,
                       "threshold_candidates": alt_thresholds},
        "baseline": {"method": "표준화 특징량 제곱 평균(시간 구조 미반영)", "threshold": b_threshold,
                     "test_alarm_rate": float(np.mean(b_te > b_threshold)) if len(b_te) else None,
                     "spearman_vs_lstm_test": spearman(s_te, b_te) if len(s_te) else None},
    }
    if len(s_te):
        a, b = s_te > threshold, b_te > b_threshold
        union = int(np.sum(a | b))
        evaluation["baseline"]["flag_jaccard_test"] = (float(np.sum(a & b)) / union) if union else None
        rng = np.random.default_rng(seed + 1)
        synth = []
        for delta in (2.0, 3.0, 5.0):
            Xi = inject(W_te, rng, delta)
            si, bi = model.scores(Xi, score_steps), baseline_scores(Xi, score_steps)
            synth.append({"delta_std": delta, "n": int(len(Xi)),
                          "lstm_detect_rate": float(np.mean(si > threshold)),
                          "lstm_auc": roc_auc(s_te, si),
                          "baseline_detect_rate": float(np.mean(bi > b_threshold)),
                          "baseline_auc": roc_auc(b_te, bi)})
        evaluation["synthetic_injection"] = {
            "method": "테스트 윈도우 마지막 3분의 무작위 특징량 3개를 ±δ 표준편차만큼 교란 (합성 데이터)",
            "results": synth}

    version = "lstm-ae-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    data_info = {
        "label": data_label, "files": [Path(p).name for p in csv_paths], "clean": clean.as_dict(),
        "trades": len(trades), "minute_bars": int(len(frame)), "filled_bars": int(frame.filled.sum()),
        "first_minute": minute_iso(int(frame.minutes[0])), "last_minute": minute_iso(int(frame.minutes[-1])),
        "windows": {"train": int(len(tr)), "val": int(len(va)), "test": int(len(te))},
        "split_end": {"train": minute_iso(int(frame.minutes[tr[-1]])),
                      "val": minute_iso(int(frame.minutes[va[-1]])),
                      "test": minute_iso(int(frame.minutes[te[-1]])) if len(te) else None},
    }
    meta = {"version": version, "model_type": MODEL_TYPE, "created_at": datetime.now(timezone.utc).isoformat(),
            "market": market, "features": FEATURES, "feature_labels": FEATURE_LABELS, "window": window, "score_steps": score_steps,
            "hidden": hidden, "n_params": model.n_params(), "scaler": scaler.to_dict(),
            "threshold": threshold, "threshold_method": threshold_desc,
            "threshold_candidates": alt_thresholds, "quantile": quantile, "mad_k": mad_k, "baseline_threshold": b_threshold, "feature_stats": feature_stats,
            "val_feature_error_mean": val_feat_err.tolist(), "data": data_info,
            "training": {k: v for k, v in fit.items() if k != "history"}}
    report = {"version": version, "data": data_info, "training": fit, "threshold": threshold,
              "evaluation": evaluation}

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_dir / "model.npz")
    write_json(out_dir / "meta.json", meta)
    write_json(out_dir / "report.json", report)
    (out_dir / "report.md").write_text(render_report_md(meta, report), encoding="utf-8")
    log(f"saved model {version} -> {out_dir}")
    return report


def render_report_md(meta: dict, report: dict) -> str:
    ev, d = report["evaluation"], report["data"]
    sd = ev["score_distribution"]
    lines = [f"# CheckBit LSTM AutoEncoder 학습 보고서 — {meta['version']}", "",
             f"- 데이터 구분: **{d['label']}** ({', '.join(d['files'])})",
             f"- 기간: {d['first_minute']} ~ {d['last_minute']} (UTC), 체결 {d['trades']:,}건, 1분 봉 {d['minute_bars']:,}개",
             f"- 정제: {json.dumps(d['clean'], ensure_ascii=False)}",
             f"- 윈도우(길이 {meta['window']}분): 학습 {d['windows']['train']} / 검증 {d['windows']['val']} / 테스트 {d['windows']['test']}",
             f"- 모델: 인코더 LSTM({meta['hidden']}) → 디코더 LSTM({meta['hidden']}) → Linear({len(meta['features'])}), 파라미터 {meta['n_params']:,}개",
             f"- 학습: best epoch {report['training']['best_epoch']}, val MSE {report['training']['best_val_loss']}, {report['training']['train_seconds']}초",
             f"- 이상 점수: 윈도우 마지막 {meta['score_steps']}분의 평균 제곱 재구성 오차",
             f"- 임계값: {meta['threshold']:.6f} ({meta['threshold_method']})", "",
             "## 평가 (라벨 없음 — 비지도 기준)", "", ev["note"], "",
             "| 구분 | n | mean | p50 | p95 | p99 | max |", "|---|---|---|---|---|---|---|"]
    for k in ("train", "val", "test"):
        s = sd[k]
        if s.get("n"):
            lines.append(f"| {k} | {s['n']} | {s['mean']:.4f} | {s['p50']:.4f} | {s['p95']:.4f} | {s['p99']:.4f} | {s['max']:.4f} |")
    ar = ev["alarm_rate"]
    lines += ["", f"- 경보율: 검증 {ar['val']:.3%}, 테스트 {ar['test']:.3%}" if ar["test"] is not None else
              f"- 경보율: 검증 {ar['val']:.3%}", f"- 통계 기준 모델: {json.dumps(ev['baseline'], ensure_ascii=False)}"]
    if "synthetic_injection" in ev:
        lines += ["", "### 합성 이상 주입 시험 (인위적 교란, 실제 이상 아님)", "", ev["synthetic_injection"]["method"], "",
                  "| δ(σ) | LSTM 탐지율 | LSTM AUC | 기준모델 탐지율 | 기준모델 AUC |", "|---|---|---|---|---|"]
        for r in ev["synthetic_injection"]["results"]:
            lines.append(f"| {r['delta_std']} | {r['lstm_detect_rate']:.1%} | {r['lstm_auc']:.3f} | "
                         f"{r['baseline_detect_rate']:.1%} | {r['baseline_auc']:.3f} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Train the CheckBit LSTM AutoEncoder (offline).")
    ap.add_argument("--csv", action="append", required=True, help="collected CSV (repeatable)")
    ap.add_argument("--out", default="models/lstm_ae")
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--hidden", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--threshold-method", choices=["mad", "quantile"], default="mad")
    ap.add_argument("--mad-k", type=float, default=4.0)
    ap.add_argument("--quantile", type=float, default=0.99)
    ap.add_argument("--score-steps", type=int, default=5, help="score = mean error of the last N minutes")
    ap.add_argument("--min-train-windows", type=int, default=200)
    ap.add_argument("--data-label", default="collected",
                    help="'collected' for real Upbit data, 'synthetic' for test fixtures")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)
    train(a.csv, Path(a.out), a.window, a.hidden, a.epochs, a.quantile, a.threshold_method, a.mad_k,
          a.score_steps, a.min_train_windows, a.data_label, seed=a.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
