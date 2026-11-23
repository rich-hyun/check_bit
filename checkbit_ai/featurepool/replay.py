"""Recompute stored features for a period from raw.sqlite3.

    python -m checkbit_ai.featurepool.replay                       # everything collected so far
    python -m checkbit_ai.featurepool.replay --from 2026-11-23T00:00Z --to 2026-11-24T00:00Z

Minutes are processed in time order, so normalization always sees recomputed
history. Start the range at least LONG_WINDOW minutes before the period you care
about, or the long z of its first day uses whatever history is already stored.
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone

from .. import config
from . import params as P
from . import store
from .pipeline import first_session_minute, last_final_minute, now_ms, open_raw, process_minutes


def _minute(text: str) -> int:
    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() // 60)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="start", help="ISO time (UTC if no offset); default: first collector session")
    ap.add_argument("--to", dest="end", help="ISO time, exclusive; default: last finalized minute")
    ap.add_argument("--market", default=config.MARKET)
    ap.add_argument("--raw-db", default=str(config.raw_db_path()))
    ap.add_argument("--features-db", default=str(config.features_db_path()))
    args = ap.parse_args(argv)

    raw = open_raw(args.raw_db)
    feat = store.connect(args.features_db)
    try:
        now = now_ms()
        m0 = _minute(args.start) if args.start else first_session_minute(raw, args.market)
        if m0 is None:
            print("no collector sessions in raw DB; nothing to replay")
            return 1
        m1 = (_minute(args.end) - 1) if args.end else last_final_minute(now)
        t = time.time()
        n = process_minutes(raw, feat, args.market, m0, m1, source="replay", now=now,
                            orderbook_interval_ms=config.ORDERBOOK_INTERVAL_MS)
        gaps = feat.execute("SELECT SUM(is_gap), SUM(low_snapshots) FROM feature_minutes"
                            " WHERE market = ? AND minute BETWEEN ? AND ?", (args.market, m0, m1)).fetchone()
        print(f"{args.market}: {n} minutes ({P.DEFS_VERSION}) in {time.time() - t:.1f}s; "
              f"gap minutes {gaps[0] or 0}, low-snapshot minutes {gaps[1] or 0}")
        return 0
    finally:
        raw.close()
        feat.close()


if __name__ == "__main__":
    raise SystemExit(main())
