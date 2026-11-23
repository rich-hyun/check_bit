"""Feature-pool parameters. Thresholds that depend on the market can be overridden
with environment variables; window lengths are part of the definitions (DEFS_VERSION)."""
from __future__ import annotations

import os

DEFS_VERSION = "fp-v1"

# --- coverage / bucketing -------------------------------------------------
MINUTE_MS = 60_000
GAP_TOLERANCE_MS = 3_000      # a minute with more uncovered time than this is a gap
GRACE_MS = 3_000              # a minute [t, t+60s) is finalized at t+60s+3s
FFILL_MAX = 60                # a no-trade minute takes the last price from at most this many minutes back

# --- windows (minutes) -----------------------------------------------------
SMA_MINUTES = 30
VOL_MINUTES = 15              # returns r_{t-14}..r_t
BIPOWER_K = 120               # Lee-Mykland window: products over r_{t-K+1}..r_{t-1}
ROLL_MINUTES = 30             # trade-level price changes in minutes t-29..t
ROLL_MIN_CHANGES = 200

# Raw-data lookback needed so that a single minute computed alone equals the same
# minute inside a replay: close() looks back FFILL_MAX, returns look back 1 more,
# and the longest close-based window is the bipower one.
LOOKBACK_MINUTES = BIPOWER_K + 1 + FFILL_MAX + 1

# --- thresholds --------------------------------------------------------------
LARGE_TRADE_KRW = float(os.getenv("CHECKBIT_LARGE_TRADE_KRW", "10000000"))
ROUND_BTC_UNIT = float(os.getenv("CHECKBIT_ROUND_BTC_UNIT", "0.001"))
ROUND_KRW_UNIT = float(os.getenv("CHECKBIT_ROUND_KRW_UNIT", "10000"))
ROUND_KRW_TOLERANCE = 1.0     # KRW
VOLUME_EPS = 0.001            # BTC, for volume_change_rate

# --- normalization ---------------------------------------------------------
SHORT_WINDOW, SHORT_MIN_PERIODS = 60, 30
LONG_WINDOW, LONG_MIN_PERIODS = 1440, 720
MAD_SCALE = 1.4826
