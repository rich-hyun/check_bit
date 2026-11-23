"""Trigger parameters (confirmed 2026-11-23). Bump TRIGGER_VERSION when a rule changes."""
from __future__ import annotations

TRIGGER_VERSION = "tr-v2"

# --- Lee-Mykland jump test (realtime decision) ------------------------------
JUMP_ALPHA = 0.01          # familywise error per day
JUMP_N = 1440              # tests per day (1-minute returns)
# replay report: Benjamini-Hochberg over each UTC day
BH_Q = 0.05

# --- rolling robust z ---------------------------------------------------------
Z_THRESHOLD = 4.0
# Per-feature thresholds replace Z_THRESHOLD once calibrated from collected data
# (target exceedance rate per minute). Empty = Z_THRESHOLD for every feature.
Z_THRESHOLDS: dict[str, float] = {}
MIN_TRADES_FOR_RATIOS = 20

# direction in which a large z is anomalous
UP, DOWN, BOTH = "up", "down", "both"
Z_DIRECTIONS: dict[str, str] = {
    "volume": UP, "volume_krw": UP, "trade_count": UP, "volume_change_rate": UP,
    "price_range_1m": UP, "volatility": UP,
    "spread_mean_bp": UP, "spread_max_bp": UP, "spread_wide_ratio": UP, "amihud": UP, "roll_measure": UP,
    "average_trade_size": UP, "large_trade_ratio": UP,
    "depth_total_15": DOWN,
    "price_return_5m": BOTH, "price_return_15m": BOTH, "moving_average_gap": BOTH,
    "buy_ratio": BOTH, "net_taker_volume": BOTH, "order_imbalance_5": BOTH,
    "round_number_ratio_btc": BOTH, "round_number_ratio_krw": BOTH,
}
# not judged by the z detector: price_return_1m (jump test), bipower_var (its input)

# ratio-type features are noisy with few trades
RATIO_FEATURES = {"buy_ratio", "large_trade_ratio", "average_trade_size",
                  "round_number_ratio_btc", "round_number_ratio_krw"}
# orderbook features are skipped when the minute had too few snapshots
BOOK_FEATURES = {"spread_mean_bp", "spread_max_bp", "spread_wide_ratio", "depth_total_15", "order_imbalance_5"}

# L2 type hints attached to candidates (the agent makes the final call)
TYPE_HINTS: dict[str, str] = {
    "volume": "거래량 급증", "volume_krw": "거래량 급증", "trade_count": "거래량 급증",
    "volume_change_rate": "거래량 급증",
    "spread_mean_bp": "유동성 경색", "spread_max_bp": "유동성 경색", "spread_wide_ratio": "유동성 경색",
    "depth_total_15": "유동성 경색", "amihud": "유동성 경색", "roll_measure": "유동성 경색",
    "order_imbalance_5": "호가 불균형",
    "buy_ratio": "한쪽 쏠림 체결", "net_taker_volume": "한쪽 쏠림 체결",
    "average_trade_size": "체결 규모 분포 이탈", "large_trade_ratio": "체결 규모 분포 이탈",
    "round_number_ratio_btc": "체결 규모 분포 이탈", "round_number_ratio_krw": "체결 규모 분포 이탈",
    "price_range_1m": "가격 급변", "volatility": "가격 급변", "price_return_5m": "가격 급변",
    "price_return_15m": "가격 급변", "moving_average_gap": "가격 급변",
}
JUMP_TYPE = "점프"
LSTM_TYPE = "복합 패턴 이탈(LSTM)"

# --- candidates -----------------------------------------------------------------
MERGE_GAP = 2              # up to this many hit-free minutes inside one candidate
MAX_CANDIDATE_MINUTES = 60 # force-close and start a new candidate after this span
STRONG_RATIO = 1.5         # any detector at >= 1.5x its threshold -> strong
STRONG_MIN_FAMILIES = 2    # or >= 2 detector families -> strong

# --- LSTM AE wrapper ------------------------------------------------------------
# Off since tr-v2 (2026-11-23): the deployed model was trained on ~2 hours of data
# and flags ~27% of minutes, so it would open candidates on its own all day. Turn on
# again after retraining brings its alarm rate down.
LSTM_ENABLED = False
# The LSTM service scores minute t at about the same time features finalize t,
# so with LSTM on, triggers run one minute behind to see its result (keeps realtime == replay).
LSTM_LAG_MINUTES = 1


def lag_minutes() -> int:
    return LSTM_LAG_MINUTES if LSTM_ENABLED else 0
