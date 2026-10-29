"""Append-only CSV ledgers, one file per leg.

Rows are only ever appended. Floats are written with Python's shortest
round-trip repr, so a ledger read back is bit-identical to what was written
and reconciliation can demand exact agreement.
"""
import csv
import json
from pathlib import Path

import pandas as pd

COLUMNS = ["date", "prev_close", "open", "close", "carried", "held", "cost",
           "equity", "target_next", "source", "recorded_at", "recorded_late"]
LOCK = "spec.lock.json"


def leg_path(ledger_dir: Path, symbol: str) -> Path:
    return ledger_dir / f"{symbol}.csv"


def read_leg(ledger_dir: Path, symbol: str) -> pd.DataFrame:
    path = leg_path(ledger_dir, symbol)
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(path, parse_dates=["date"])
    df["recorded_late"] = df["recorded_late"].astype(str).str.lower() == "true"
    return df


def append_leg(ledger_dir: Path, symbol: str, rows: list[dict]) -> None:
    if not rows:
        return
    path = leg_path(ledger_dir, symbol)
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(COLUMNS)
        for r in rows:
            w.writerow([_cell(r[c]) for c in COLUMNS])


def _cell(v):
    if isinstance(v, pd.Timestamp):
        return v.strftime("%Y-%m-%d")
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):  # includes numpy floats, whose repr is not a plain number
        return repr(float(v))
    return "" if v is None else v


def read_lock(ledger_dir: Path) -> dict | None:
    path = ledger_dir / LOCK
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def write_lock(ledger_dir: Path, lock: dict) -> None:
    ledger_dir.mkdir(parents=True, exist_ok=True)
    (ledger_dir / LOCK).write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
