"""Rule-based evidence generation for a scored window.

Every sentence is derived from numbers that were actually computed (feature
values, reconstruction error shares, training-period percentiles). No causes,
news or price forecasts are guessed.
"""
from __future__ import annotations

import math

import numpy as np

from .features import FEATURE_LABELS, FEATURES, minute_iso

DISCLAIMER = ("이 설명은 모델 재구성 오차와 실제 계산된 특징량을 근거로 한 통계적 해석이며, "
              "원인 단정이나 시세 예측·투자 조언이 아닙니다.")


def _kst(minute: int) -> str:
    # Upbit is a Korean exchange; display times in KST (UTC+9).
    m = (minute + 9 * 60) % (24 * 60)
    return f"{m // 60:02d}:{m % 60:02d}"


def _fmt(name: str, v: float) -> str:
    if v is None or not math.isfinite(v):
        return "—"
    if name == "log_return":
        return f"{math.expm1(v) * 100:+.3f}%"
    if name in ("hl_range", "rv_10"):
        return f"{v * 100:.3f}%"
    if name in ("d_log_volume", "d_log_value"):
        return f"직전 1분 대비 {math.exp(v):.2f}배"
    if name == "log_value":
        return f"{math.expm1(v) / 100:.2f}억 원"  # value was in million KRW
    if name == "log_trade_count":
        return f"{math.expm1(v):.0f}건"
    if name == "spread_bps":
        return f"{v:.2f}bp"
    return f"{v:+.2f}"


def _direction(name: str, v: float, p50: float | None) -> str:
    if name == "buy_sell_imbalance":
        return "매수 체결 우위" if v > 0 else "매도 체결 우위" if v < 0 else "균형"
    if name in ("l1_imbalance", "depth_imbalance"):
        return "매수 호가 잔량 우위" if v > 0 else "매도 호가 잔량 우위" if v < 0 else "균형"
    if p50 is None:
        return ""
    return "평소보다 높음" if v > p50 else "평소보다 낮음"


def explain_window(*, minutes: np.ndarray, raw: np.ndarray, err: np.ndarray, score: float, threshold: float,
                   meta: dict, close: np.ndarray, value: np.ndarray, top_k: int = 3) -> dict:
    """minutes/raw/err/close/value describe one window (T rows)."""
    stats = meta.get("feature_stats", {})
    val_err = np.asarray(meta.get("val_feature_error_mean") or [np.nan] * len(FEATURES), float)
    per_feat = err.sum(axis=0)
    total = float(per_feat.sum()) or 1e-12
    order = np.argsort(per_feat)[::-1]
    step_err = err.sum(axis=1)
    peak = int(np.argmax(step_err))
    top = []
    for j in order[:top_k]:
        name = FEATURES[j]
        t = int(np.argmax(err[:, j]))
        v = float(raw[t, j])
        s = stats.get(name, {})
        p05, p50, p95 = s.get("p05"), s.get("p50"), s.get("p95")
        mean_err = float(err[:, j].mean())
        ratio = mean_err / val_err[j] if math.isfinite(val_err[j]) and val_err[j] > 0 else None
        outside = (p05 is not None and p95 is not None and not (p05 <= v <= p95))
        top.append({"feature": name, "label": FEATURE_LABELS[name], "_minute": int(minutes[t]),
                    "contribution_pct": round(per_feat[j] / total * 100, 1),
                    "at": minute_iso(int(minutes[t])), "value": v, "value_text": _fmt(name, v),
                    "normal_p05": p05, "normal_p50": p50, "normal_p95": p95,
                    "normal_text": (f"{_fmt(name, p05)} ~ {_fmt(name, p95)}" if p05 is not None else None),
                    "outside_normal_range": bool(outside),
                    "error_vs_validation": round(ratio, 2) if ratio else None,
                    "direction": _direction(name, v, p50)})

    price_change = (close[-1] / close[0] - 1) * 100 if close[0] > 0 else 0.0
    window_value = float(np.sum(value))
    normal_value = stats.get("log_value", {}).get("p50")
    normal_value_krw = math.expm1(normal_value) * 1e6 * len(minutes) if normal_value is not None else None
    ratio_score = score / threshold if threshold > 0 else None
    is_anomaly = bool(score > threshold)

    evidence = [f"이상 점수 {score:.4f}는 판정 임계값 {threshold:.4f}의 {ratio_score:.2f}배입니다."
                if ratio_score is not None else f"이상 점수 {score:.4f}"]
    evidence.append(f"분석 구간 {_kst(int(minutes[0]))}~{_kst(int(minutes[-1]))} KST ({len(minutes)}분) 동안 가격은 "
                    f"{price_change:+.3f}% 변했고, 최고 ₩{np.max(close):,.0f} / 최저 ₩{np.min(close):,.0f}입니다.")
    if normal_value_krw:
        evidence.append(f"구간 거래대금 합계는 {window_value / 1e8:.2f}억 원으로, 학습 기간 같은 길이의 "
                        f"중앙값 수준({normal_value_krw / 1e8:.2f}억 원)의 {window_value / normal_value_krw:.2f}배입니다.")
    evidence.append(f"재구성 오차가 가장 큰 시점은 {_kst(int(minutes[peak]))} KST입니다.")
    for item in top:
        line = f"{item['label']}: {_kst(item.pop('_minute'))} 값 {item['value_text']}"
        if item["normal_text"]:
            line += f" (학습 기간 평소 5~95% 범위 {item['normal_text']}"
            line += ", 범위 밖)" if item["outside_normal_range"] else ", 범위 안)"
        if item["direction"]:
            line += f" · {item['direction']}"
        line += f" — 전체 오차의 {item['contribution_pct']}% 기여"
        if item["error_vs_validation"]:
            line += f", 검증 데이터 평균 오차의 {item['error_vs_validation']}배"
        evidence.append(line + ".")

    if is_anomaly:
        lead = top[0]["label"] if top else "복합 특징"
        summary = (f"{_kst(int(minutes[-1]))} KST 기준 최근 {len(minutes)}분 패턴이 학습된 정상 패턴과 크게 달라 "
                   f"이상으로 판정했습니다. 주된 근거는 {lead}입니다.")
    else:
        summary = (f"{_kst(int(minutes[-1]))} KST 기준 최근 {len(minutes)}분 패턴의 재구성 오차가 임계값 이하로, "
                   "정상 범위로 판정했습니다.")
    return {"summary": summary, "evidence": evidence, "top_features": top,
            "peak_at": minute_iso(int(minutes[peak])),
            "window": {"start": minute_iso(int(minutes[0])), "end": minute_iso(int(minutes[-1])),
                       "minutes": int(len(minutes)), "price_change_pct": round(price_change, 4),
                       "price_high": float(np.max(close)), "price_low": float(np.min(close)),
                       "value_sum_krw": window_value},
            "disclaimer": DISCLAIMER}
