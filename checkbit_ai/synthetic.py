"""SYNTHETIC test data generator — NOT real market data.

Produces a CSV in the exact collector schema so the full pipeline can be tested
offline. Every row is marked: sequential_id starts with "SYN-" and the default
file name contains "synthetic". Never mix these files with collected data.

    python -m checkbit_ai.synthetic --minutes 900 --out tests/fixtures/synthetic_trades.csv
"""
from __future__ import annotations

import argparse
import csv
import math
from datetime import datetime
from pathlib import Path

import numpy as np

from .features import CSV_HEADER


def generate(out: Path, minutes: int = 900, start_ts: float = 1_790_000_000.0, seed: int = 7,
             anomalies: tuple[int, ...] = (), trades_per_min: float = 25.0, gap_at: int | None = None) -> Path:
    """`anomalies`: minute offsets where a burst (price jump, volume x8,
    one-sided flow, wide spread) is injected. `gap_at`: minute where 30 minutes
    of data are missing (simulates a collector stop)."""
    rng = np.random.default_rng(seed)
    price = 100_000_000.0
    out.parent.mkdir(parents=True, exist_ok=True)
    seq = 0
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADER)
        for m in range(minutes):
            if gap_at is not None and gap_at <= m < gap_at + 30:
                continue
            burst = m in anomalies
            n = rng.poisson(trades_per_min * (8 if burst else 1)) + 1
            drift = (0.012 if burst else 0.0) * rng.choice([-1, 1])
            buy_p = 0.92 if burst else 0.5 + 0.1 * math.sin(m / 37)
            spread_ticks = 12 if burst else rng.choice([1, 1, 1, 2])
            for k in range(n):
                ts = start_ts + m * 60 + (k + rng.random()) * 60 / (n + 1)
                price *= math.exp(rng.normal(drift / n, 0.00012))
                price = round(price / 1000) * 1000
                side = "BID" if rng.random() < buy_p else "ASK"
                bid = price - 1000 * (0 if side == "ASK" else spread_ticks)
                ask = bid + 1000 * spread_ticks
                vol = float(rng.lognormal(-4.5, 1.1))
                seq += 1
                w.writerow([datetime.fromtimestamp(ts), "KRW-BTC", price, round(vol, 8), side, bid, ask,
                            round(float(rng.lognormal(-3, 1)), 8), round(float(rng.lognormal(-3, 1)), 8),
                            round(float(rng.uniform(1, 6)) * (0.3 if burst else 1), 8),
                            round(float(rng.uniform(1, 6)), 8), int(ts * 1000), f"SYN-{seq}"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate SYNTHETIC collector-format CSV (test only).")
    ap.add_argument("--out", default="tests/fixtures/synthetic_trades.csv")
    ap.add_argument("--minutes", type=int, default=900)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    print(generate(Path(a.out), a.minutes, seed=a.seed, anomalies=(700, 780)))


if __name__ == "__main__":
    main()
