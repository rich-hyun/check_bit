"""Feature pool definitions (6 axes) and their computation.

compute() takes consecutive MinuteData (lookback + targets) and returns the raw
feature values for the target minutes only. Every value at minute t depends only
on minutes [t - LOOKBACK_MINUTES, t], so computing one minute alone gives the
same result as computing it inside a replay.

Conventions
- close_t: last trade price of minute t; a no-trade minute takes the last price
  from up to FFILL_MAX minutes back, never across a gap minute.
- r_t = ln(close_t / close_{t-1}).
- A gap minute (collector not connected for > GAP_TOLERANCE_MS) has every feature NaN.
- Rolling features are NaN when any minute in their window is a gap.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import params as P
from .minute import MinuteData


@dataclass(frozen=True)
class FeatureDef:
    name: str
    axis: str
    unit: str
    definition: str
    nan_when: str
    mad_floor: float


FEATURES: list[FeatureDef] = [
    # 1. price
    FeatureDef("price_return_1m", "가격", "log", "ln(close_t / close_{t-1})", "양 끝 close 미정의", 1e-5),
    FeatureDef("price_return_5m", "가격", "log", "ln(close_t / close_{t-5})", "양 끝 close 미정의", 1e-5),
    FeatureDef("price_return_15m", "가격", "log", "ln(close_t / close_{t-15})", "양 끝 close 미정의", 1e-5),
    FeatureDef("moving_average_gap", "가격", "ratio", "close_t / mean(close_{t-29..t}) − 1 (SMA 30분)",
               "30분 중 close 미정의(공백 포함)", 1e-5),
    # 2. volatility
    FeatureDef("price_range_1m", "변동성", "log", "ln(high_t / low_t)", "체결 0건", 1e-5),
    FeatureDef("volatility", "변동성", "log", "std(r_{t-14..t}), ddof=1 (현재 포함 15개)",
               "15개 수익률 중 미정의", 1e-6),
    FeatureDef("bipower_var", "변동성", "log²",
               f"Lee-Mykland σ̂²: Σ_(j=t-K+2)^(t-1) |r_j||r_(j-1)| / (K−2), K={P.BIPOWER_K} (현재 r_t 제외)",
               "r_(t-K+1..t-1) 중 미정의", 1e-10),
    # 3. activity
    FeatureDef("volume", "거래 활동", "BTC", "Σ trade_volume", "공백", 1e-3),
    FeatureDef("volume_krw", "거래 활동", "KRW", "Σ trade_price × trade_volume", "공백", 1e5),
    FeatureDef("trade_count", "거래 활동", "건", "체결 건수", "공백", 1.0),
    FeatureDef("volume_change_rate", "거래 활동", "log",
               f"ln(volume_t + ε) − ln(volume_(t-1) + ε), ε={P.VOLUME_EPS} BTC", "t 또는 t−1 공백", 0.01),
    # 4. liquidity
    FeatureDef("spread_mean_bp", "유동성", "bp", "스냅샷별 (ask₁−bid₁)/mid×10⁴ 의 평균", "스냅샷 0개", 0.1),
    FeatureDef("spread_max_bp", "유동성", "bp", "스냅샷별 스프레드의 최대 (1초 샘플링 범위 내)", "스냅샷 0개", 0.1),
    FeatureDef("spread_wide_ratio", "유동성", "ratio",
               "스프레드 > 1틱 인 스냅샷 비율 (틱 = 그 스냅샷의 인접 단계 간격 최솟값)", "스냅샷 0개 또는 틱 미정", 0.01),
    FeatureDef("depth_total_15", "유동성", "BTC",
               "스냅샷별 Σ(15단계 bid_size + ask_size)의 평균 (업비트 total_*_size 아님)", "스냅샷 0개", 0.01),
    FeatureDef("amihud", "유동성", "log / 1억원", "|price_return_1m| / (volume_krw / 10⁸)",
               "거래대금 0 또는 수익률 미정의", 1e-6),
    # 5. order flow
    FeatureDef("buy_ratio", "주문 흐름", "ratio", "BID 거래량 / 전체 거래량 (taker 기준)", "체결 0건", 0.01),
    FeatureDef("net_taker_volume", "주문 흐름", "BTC", "BID 거래량 − ASK 거래량", "공백", 1e-3),
    FeatureDef("order_imbalance_5", "주문 흐름", "ratio",
               "스냅샷별 (Σbid₁₋₅ − Σask₁₋₅)/(Σbid₁₋₅ + Σask₁₋₅)의 평균", "스냅샷 0개", 0.01),
    FeatureDef("roll_measure", "주문 흐름", "bp",
               f"2√(−Cov(Δp_i, Δp_(i−1))) / mean(p) × 10⁴, 직전 {P.ROLL_MINUTES}분(t 포함) 체결 단위 가격 변화",
               f"윈도우 중 공백, 가격 변화 < {P.ROLL_MIN_CHANGES}건, 또는 Cov ≥ 0", 0.1),
    # 6. trade composition
    FeatureDef("average_trade_size", "체결 구성", "BTC", "volume / trade_count", "체결 0건", 1e-4),
    FeatureDef("large_trade_ratio", "체결 구성", "ratio",
               f"1건 금액 ≥ {P.LARGE_TRADE_KRW:,.0f}원 체결의 거래량 / 전체 거래량", "체결 0건", 0.01),
    FeatureDef("round_number_ratio_btc", "체결 구성", "ratio",
               f"수량이 {P.ROUND_BTC_UNIT} BTC 배수인 체결 건수 비율", "체결 0건", 0.01),
    FeatureDef("round_number_ratio_krw", "체결 구성", "ratio",
               f"금액이 {P.ROUND_KRW_UNIT:,.0f}원 배수(±{P.ROUND_KRW_TOLERANCE:g}원)인 체결 건수 비율", "체결 0건", 0.01),
]
FEATURE_NAMES = [f.name for f in FEATURES]
MAD_FLOORS = {f.name: f.mad_floor for f in FEATURES}


@dataclass
class MinuteMeta:
    minute: int
    covered_ms: int
    is_gap: bool
    n_trades: int
    n_snapshots: int
    low_snapshots: bool
    roll_cov_nonneg: bool


def low_snapshot_threshold(orderbook_interval_ms: int) -> float:
    expected = P.MINUTE_MS / orderbook_interval_ms if orderbook_interval_ms > 0 else 60.0
    return 0.5 * min(expected, 60.0)


def _is_multiple(x: np.ndarray, unit: float, tol: float) -> np.ndarray:
    q = x / unit
    return np.abs(q - np.round(q)) * unit <= tol


def compute(minutes: list[MinuteData], first_target: int, orderbook_interval_ms: int = 1000
            ) -> tuple[list[MinuteMeta], dict[str, np.ndarray]]:
    """Raw features for minutes[first_target:]. minutes must be consecutive."""
    n = len(minutes)
    nan = math.nan
    gap = np.array([P.MINUTE_MS - m.covered_ms > P.GAP_TOLERANCE_MS for m in minutes], bool)
    ntr = np.array([m.n_trades for m in minutes], int)
    last = np.array([m.prices[-1] if m.n_trades else nan for m in minutes], float)

    close = np.full(n, nan)
    for i in range(n):
        if gap[i]:
            continue
        if ntr[i]:
            close[i] = last[i]
            continue
        for j in range(i - 1, max(-1, i - P.FFILL_MAX - 1), -1):
            if gap[j]:
                break
            if ntr[j]:
                close[i] = last[j]
                break
    with np.errstate(divide="ignore", invalid="ignore"):
        r1 = np.full(n, nan)
        r1[1:] = np.log(close[1:] / close[:-1])

    targets = range(first_target, n)
    out = {name: np.full(len(targets), nan) for name in FEATURE_NAMES}
    metas: list[MinuteMeta] = []
    low_thr = low_snapshot_threshold(orderbook_interval_ms)

    def window_ok(arr: np.ndarray, lo: int, hi: int) -> bool:  # arr[lo..hi] all finite
        return lo >= 0 and bool(np.all(np.isfinite(arr[lo:hi + 1])))

    for k, i in enumerate(targets):
        m = minutes[i]
        roll_nonneg = False
        if not gap[i]:
            f = {}
            f["price_return_1m"] = r1[i]
            for lag, name in ((5, "price_return_5m"), (15, "price_return_15m")):
                f[name] = math.log(close[i] / close[i - lag]) if i - lag >= 0 and np.isfinite(close[i]) \
                    and np.isfinite(close[i - lag]) else nan
            lo = i - P.SMA_MINUTES + 1
            f["moving_average_gap"] = close[i] / np.mean(close[lo:i + 1]) - 1 if window_ok(close, lo, i) else nan

            f["price_range_1m"] = math.log(m.prices.max() / m.prices.min()) if m.n_trades else nan
            lo = i - P.VOL_MINUTES + 1
            f["volatility"] = float(np.std(r1[lo:i + 1], ddof=1)) if window_ok(r1, lo, i) else nan
            K = P.BIPOWER_K
            if window_ok(r1, i - K + 1, i - 1):
                seg = np.abs(r1[i - K + 1:i])          # r_{t-K+1} .. r_{t-1}
                f["bipower_var"] = float(np.sum(seg[1:] * seg[:-1]) / (K - 2))
            else:
                f["bipower_var"] = nan

            vol = float(m.volumes.sum())
            krw = float(np.sum(m.prices * m.volumes))
            buy = float(m.volumes[m.is_buy].sum())
            f["volume"], f["volume_krw"], f["trade_count"] = vol, krw, float(m.n_trades)
            if i >= 1 and not gap[i - 1]:
                pv = float(minutes[i - 1].volumes.sum())
                f["volume_change_rate"] = math.log(vol + P.VOLUME_EPS) - math.log(pv + P.VOLUME_EPS)
            else:
                f["volume_change_rate"] = nan

            if m.n_snapshots:
                f["spread_mean_bp"] = float(np.nanmean(m.spread_bp)) if np.isfinite(m.spread_bp).any() else nan
                f["spread_max_bp"] = float(np.nanmax(m.spread_bp)) if np.isfinite(m.spread_bp).any() else nan
                f["spread_wide_ratio"] = float(np.nanmean(m.spread_wide)) if np.isfinite(m.spread_wide).any() else nan
                f["depth_total_15"] = float(np.mean(m.depth15))
                f["order_imbalance_5"] = float(np.nanmean(m.oi5)) if np.isfinite(m.oi5).any() else nan
            else:
                for name in ("spread_mean_bp", "spread_max_bp", "spread_wide_ratio", "depth_total_15",
                             "order_imbalance_5"):
                    f[name] = nan
            f["amihud"] = abs(r1[i]) / (krw / 1e8) if krw > 0 and np.isfinite(r1[i]) else nan

            f["net_taker_volume"] = buy - (vol - buy)
            if m.n_trades:
                f["buy_ratio"] = buy / vol if vol > 0 else nan
                f["average_trade_size"] = vol / m.n_trades
                notional = m.prices * m.volumes
                f["large_trade_ratio"] = float(m.volumes[notional >= P.LARGE_TRADE_KRW].sum() / vol) if vol > 0 else nan
                f["round_number_ratio_btc"] = float(np.mean(_is_multiple(m.volumes, P.ROUND_BTC_UNIT, 1e-9)))
                f["round_number_ratio_krw"] = float(np.mean(
                    _is_multiple(notional, P.ROUND_KRW_UNIT, P.ROUND_KRW_TOLERANCE)))
            else:
                for name in ("buy_ratio", "average_trade_size", "large_trade_ratio",
                             "round_number_ratio_btc", "round_number_ratio_krw"):
                    f[name] = nan

            lo = i - P.ROLL_MINUTES + 1
            f["roll_measure"] = nan
            if lo >= 0 and not gap[lo:i + 1].any():
                px = np.concatenate([minutes[j].prices for j in range(lo, i + 1)])
                dp = np.diff(px)
                if len(dp) >= P.ROLL_MIN_CHANGES:
                    cov = float(np.cov(dp[1:], dp[:-1])[0, 1])
                    if cov < 0:
                        f["roll_measure"] = 2 * math.sqrt(-cov) / float(np.mean(px)) * 1e4
                    else:
                        roll_nonneg = True
            for name, v in f.items():
                out[name][k] = v if v is not None and np.isfinite(v) else nan
        metas.append(MinuteMeta(minute=m.minute, covered_ms=int(m.covered_ms), is_gap=bool(gap[i]),
                                n_trades=int(m.n_trades), n_snapshots=int(m.n_snapshots),
                                low_snapshots=bool(not gap[i] and m.n_snapshots < low_thr),
                                roll_cov_nonneg=roll_nonneg))
    return metas, out


def render_markdown() -> str:
    """docs/FEATURES.md is generated from FEATURES (a test keeps them in sync)."""
    lines = [
        "# Feature pool (`checkbit_ai/featurepool`)",
        "",
        f"정의 버전 `{P.DEFS_VERSION}`. 이 파일은 `python -m checkbit_ai.featurepool.docs`로 생성한다 (직접 수정 금지).",
        "",
        "## 공통 규칙",
        "",
        "- 버킷: 업비트 `trade_timestamp`/호가 `timestamp` 기준 UTC 1분 `[t, t+60s)`. "
        f"t+60s+{P.GRACE_MS // 1000}s에 확정.",
        f"- 수집 공백: `collector_sessions`로 덮이지 않은 시간이 {P.GAP_TOLERANCE_MS // 1000}초를 넘는 분. "
        "그 분의 모든 feature는 NaN.",
        "- 보조 점검: 스냅샷 수가 예상치의 절반 미만이면 `feature_minutes.low_snapshots = 1` (값은 계산).",
        f"- close_t: 그 분 마지막 체결가. 체결이 없으면 최대 {P.FFILL_MAX}분 전까지의 마지막 체결가 (공백을 넘지 않음). "
        "r_t = ln(close_t / close_(t−1)).",
        "- 롤링 feature는 윈도우 안에 공백이 하나라도 있으면 NaN.",
        "",
        "## 정규화",
        "",
        f"- robust z = (x − median) / max({P.MAD_SCALE} × MAD, MAD 하한). 현재 분은 윈도우에서 제외.",
        f"- 단기: 직전 {P.SHORT_WINDOW}분 (유효값 {P.SHORT_MIN_PERIODS}개 이상), "
        f"장기: 직전 {P.LONG_WINDOW}분 (유효값 {P.LONG_MIN_PERIODS}개 이상).",
        "- `z` = 장기 z, 장기가 NaN이면 단기 z.",
        "",
        "## Feature",
        "",
        "| 축 | feature | 단위 | 정의 | NaN 조건 | MAD 하한 |",
        "|---|---|---|---|---|---|",
    ]
    cell = lambda text: text.replace("|", "\\|")  # noqa: E731  (|r| inside a table cell)
    for f in FEATURES:
        lines.append(f"| {f.axis} | `{f.name}` | {cell(f.unit)} | {cell(f.definition)} | {cell(f.nan_when)} "
                     f"| {f.mad_floor:g} |")
    return "\n".join(lines) + "\n"
