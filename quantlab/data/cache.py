"""Simple parquet/csv file cache so repeated backtests don't re-hit APIs."""
import hashlib
import time
from pathlib import Path

import pandas as pd

CACHE_DIR = Path(__file__).resolve().parents[2] / "data_cache"
CACHE_DIR.mkdir(exist_ok=True)

DEFAULT_TTL = 60 * 60 * 12  # 12 hours


def _key_to_path(key: str) -> Path:
    digest = hashlib.md5(key.encode()).hexdigest()[:10]
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in key)[:60]
    return CACHE_DIR / f"{safe}_{digest}.csv"


def load(key: str, ttl: int = DEFAULT_TTL) -> pd.DataFrame | None:
    path = _key_to_path(key)
    if not path.exists():
        return None
    if time.time() - path.stat().st_mtime > ttl:
        return None
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df if not df.empty else None
    except Exception:
        return None


def save(key: str, df: pd.DataFrame) -> None:
    if df is None or df.empty:
        return
    df.to_csv(_key_to_path(key))
