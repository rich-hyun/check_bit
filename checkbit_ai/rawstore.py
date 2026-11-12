"""Raw trade CSV storage used by the collector.

- Same column layout as before plus two appended columns (trade_timestamp,
  sequential_id) so old analysis code reading the first 11 columns keeps working.
- Legacy files (11 columns, or the old git-tracked file in the repo root) are
  migrated into DATA_DIR once; the original is kept as a backup.
- A partially written last line (crash / power loss) is cut off on startup and
  saved to backup/corrupt_fragments.log instead of being silently lost.
- Appends and download snapshots share one lock, so a download never contains a
  half-written row.
"""
from __future__ import annotations

import csv
import io
import logging
import os
import shutil
import threading
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Iterator

from .features import CSV_HEADER, LEGACY_HEADER

LOG = logging.getLogger("checkbit.rawstore")


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _read_header(path: Path) -> list[str]:
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        first = f.readline()
    return next(csv.reader([first]), []) if first else []


def count_rows(path: Path) -> int:
    if not path.exists():
        return 0
    n = 0
    last = b"\n"
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            n += chunk.count(b"\n")
            last = chunk[-1:]
    if last != b"\n":
        n += 1
    return max(0, n - 1)


class RawCsvStore:
    def __init__(self, path: Path, backup_dir: Path, legacy_candidates: list[Path] | None = None) -> None:
        self.path = Path(path)
        self.backup_dir = Path(backup_dir)
        self.legacy_candidates = legacy_candidates or []
        self.lock = threading.RLock()
        self._fh: io.TextIOWrapper | None = None
        self._writer = None
        self._pending: deque[list] = deque(maxlen=20000)  # rows kept while the disk is failing
        self.rows = 0
        self.write_errors = 0
        self.last_error: str | None = None
        self.notes: list[str] = []
        self.prepare()

    # --------------------------------------------------------------- setup
    def _backup(self, src: Path, tag: str) -> Path:
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        dst = self.backup_dir / f"{src.stem}_{tag}_{_stamp()}{src.suffix}"
        shutil.copy2(src, dst)
        return dst

    def _convert_legacy(self, src: Path) -> int:
        """Write src (11 columns) into self.path with the new header."""
        tmp = self.path.with_suffix(".migrating")
        n = 0
        with open(src, newline="", encoding="utf-8", errors="replace") as fi, \
                open(tmp, "w", newline="", encoding="utf-8") as fo:
            reader = csv.reader(fi)
            next(reader, None)
            w = csv.writer(fo)
            w.writerow(CSV_HEADER)
            for row in reader:
                if len(row) == len(LEGACY_HEADER):
                    w.writerow(row + ["", ""])
                    n += 1
                elif len(row) == len(CSV_HEADER):
                    w.writerow(row)
                    n += 1
        os.replace(tmp, self.path)
        return n

    def _repair_tail(self) -> None:
        size = self.path.stat().st_size
        if size == 0:
            return
        with open(self.path, "rb+") as f:
            f.seek(max(0, size - (1 << 16)))
            tail = f.read()
            if tail.endswith(b"\n"):
                return
            cut = tail.rfind(b"\n")
            if cut < 0:
                return  # a single (header) line without newline; leave it
            fragment = tail[cut + 1:]
            new_size = size - len(tail) + cut + 1
            f.truncate(new_size)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        with open(self.backup_dir / "corrupt_fragments.log", "ab") as log:
            log.write(f"{datetime.now().isoformat()} {self.path.name}: ".encode() + fragment + b"\n")
        self.notes.append(f"incomplete last line removed ({len(fragment)} bytes, saved to backup)")
        LOG.warning("Removed incomplete last CSV line (%d bytes)", len(fragment))

    def prepare(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            for cand in self.legacy_candidates:
                if cand.exists() and _read_header(cand)[:11] == LEGACY_HEADER:
                    n = self._convert_legacy(cand)
                    self.notes.append(f"migrated {n} rows from {cand.name}")
                    LOG.info("Migrated %d legacy rows from %s", n, cand)
                    break
        if self.path.exists():
            header = _read_header(self.path)
            if header == LEGACY_HEADER:
                bk = self._backup(self.path, "legacy")
                n = self._convert_legacy(bk)
                self.notes.append(f"converted legacy header ({n} rows), original kept at {bk.name}")
            elif header != CSV_HEADER:
                bk = self.backup_dir / f"{self.path.stem}_unknown_header_{_stamp()}{self.path.suffix}"
                self.backup_dir.mkdir(parents=True, exist_ok=True)
                os.replace(self.path, bk)
                self.notes.append(f"unexpected header; file moved to {bk.name}")
                LOG.error("Unexpected CSV header; moved file to %s", bk)
        if not self.path.exists():
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(CSV_HEADER)
        else:
            self._repair_tail()
        self.rows = count_rows(self.path)

    # --------------------------------------------------------------- write
    def _open(self) -> None:
        if self._fh is None:
            self._fh = open(self.path, "a", newline="", encoding="utf-8")
            self._writer = csv.writer(self._fh)

    def append(self, row: list) -> bool:
        with self.lock:
            self._pending.append(row)
            try:
                self._open()
                while self._pending:
                    self._writer.writerow(self._pending[0])
                    self._pending.popleft()
                    self.rows += 1
                self._fh.flush()
                return True
            except OSError as exc:
                self.write_errors += 1
                self.last_error = f"CSV 저장 오류: {exc}"
                LOG.error("CSV write failed (%d rows pending): %s", len(self._pending), exc)
                self.close()
                return False

    def close(self) -> None:
        with self.lock:
            if self._fh is not None:
                try:
                    self._fh.close()
                except OSError:
                    pass
            self._fh = None
            self._writer = None

    @property
    def pending(self) -> int:
        return len(self._pending)

    # ---------------------------------------------------------------- read
    def snapshot_size(self) -> int:
        with self.lock:
            if self._fh is not None:
                self._fh.flush()
            return self.path.stat().st_size if self.path.exists() else 0

    def iter_snapshot(self, chunk: int = 1 << 16) -> Iterator[bytes]:
        """Bytes of the file up to the size at call time (always ends at a row boundary)."""
        size = self.snapshot_size()
        with open(self.path, "rb") as f:
            remaining = size
            while remaining > 0:
                data = f.read(min(chunk, remaining))
                if not data:
                    break
                remaining -= len(data)
                yield data

    def tail_rows(self, max_bytes: int = 16 << 20) -> Iterator[dict]:
        """Yield the rows in the last `max_bytes` of the file (for bootstrap), streaming."""
        size = self.snapshot_size()
        with open(self.path, "rb") as f:
            start = max(0, size - max_bytes)
            f.seek(start)
            data = f.read(size - start)
        if start > 0:
            nl = data.find(b"\n")
            data = data[nl + 1:] if nl >= 0 else b""
        else:
            nl = data.find(b"\n")  # drop header
            data = data[nl + 1:] if nl >= 0 else b""
        text = io.TextIOWrapper(io.BytesIO(data), encoding="utf-8", errors="replace", newline="")
        del data
        for r in csv.reader(text):
            if len(r) == len(CSV_HEADER):
                yield dict(zip(CSV_HEADER, r))
