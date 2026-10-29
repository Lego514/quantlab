"""Forward ledger: trade a frozen spec on each new completed bar, append-only.

Each run, for every leg:
  1. download fresh daily bars and keep only the completed ones
  2. for each completed bar after the last recorded one, in order: fill the
     target set at the previous close at this bar's open, mark to the close
     (with `engine.step_return`, the backtester's own formula), then compute
     the next target from the history through this bar
  3. append the rows. Nothing already written is ever rewritten.

The first run writes an anchor row: the last completed bar, flat, equity 1.0,
holding the first target. Trading starts on the bar after it.

The gap into each bar is measured from the freshly downloaded previous close,
not the recorded one, because Yahoo re-adjusts past prices for dividends; the
price actually used is stored in `prev_close`.

Bars processed in a catch-up (a run was missed) are marked `recorded_late`.
The rule is mechanical, so the numbers are what an on-time run would have
written, but the git history shows when they were really recorded.
"""
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from ..engine import Backtester, step_return
from ..research.controls import FixedPositions
from ..research.portfolio import leg_returns
from ..strategies import make_strategy
from . import ledger
from .spec import Spec, load_spec

ROOT = Path(__file__).resolve().parents[2]
SPECS = Path("forward") / "specs"
STOCK_DATA_SETTLED = 17  # New York hour; the close is 16:00, give the data an hour

EQUITY_TOL = 1e-12   # ledger arithmetic must match its own inputs exactly
ENGINE_TOL = 1e-6    # ledger vs backtester on freshly downloaded data
TARGET_TOL = 1e-6


def ledger_dir(spec: Spec, root: Path | None = None) -> Path:
    return (root or ROOT) / "forward" / "ledgers" / spec.name


def completed_bars(bars: pd.DataFrame, market: str, now: datetime) -> pd.DataFrame:
    """Drop bars that may still be forming at `now`."""
    now = pd.Timestamp(now).tz_convert("UTC") if pd.Timestamp(now).tzinfo else \
        pd.Timestamp(now).tz_localize("UTC")
    if market == "stock":
        local = now.tz_convert("America/New_York")
        last_done = local.date() if local.hour >= STOCK_DATA_SETTLED else \
            local.date() - timedelta(days=1)
        keep = bars.index.date <= last_done
    else:  # crypto daily bars close at 00:00 UTC
        keep = bars.index.date < now.date()
    return bars[keep]


def default_fetch(leg, tries: int = 3) -> pd.DataFrame:
    from ..data import fetch
    kwargs = {"use_cache": False, **leg.history}
    if leg.market == "crypto" and leg.source:
        kwargs["source"] = leg.source
    for i in range(tries):
        try:
            return fetch(leg.market, leg.symbol, **kwargs)
        except Exception as e:  # yfinance rate limits are common on CI runners
            if i == tries - 1:
                raise
            print(f"  ! {leg.symbol}: fetch failed ({str(e)[:80]}), retrying", file=sys.stderr)
            time.sleep(10 * (i + 1))


def _target(strategy, history: pd.DataFrame) -> float:
    """Long-only, at most 1x: the same clip the backtester applies."""
    pos = float(strategy.generate_positions(history).iloc[-1])
    return min(max(pos, 0.0), 1.0)


def advance_leg(leg, spec: Spec, rows: pd.DataFrame, bars: pd.DataFrame,
                recorded_at: str) -> list[dict]:
    """New ledger rows for one leg, given its existing rows and completed bars."""
    if len(bars) < 2:
        raise RuntimeError(f"{leg.symbol}: not enough completed bars")
    strategy = make_strategy(leg.strategy, **leg.params)
    source = bars.attrs.get("source", "")

    if rows.empty:
        t = bars.index[-1]
        return [dict(date=t, prev_close=float(bars["close"].iloc[-2]),
                     open=float(bars["open"].loc[t]), close=float(bars["close"].loc[t]),
                     carried=0.0, held=0.0, cost=0.0, equity=1.0,
                     target_next=_target(strategy, bars), source=source,
                     recorded_at=recorded_at, recorded_late=False)]

    last = rows.iloc[-1]
    prev_date = pd.Timestamp(last["date"])
    if prev_date not in bars.index:
        raise RuntimeError(f"{leg.symbol}: last recorded bar {prev_date.date()} is missing "
                           "from the fresh download; refusing to guess the gap")
    equity, held, target = float(last["equity"]), float(last["held"]), float(last["target_next"])
    newest = bars.index[-1]
    out = []
    for t in bars.index[bars.index > prev_date]:
        prev_close = float(bars["close"].loc[prev_date])
        o, c = float(bars["open"].loc[t]), float(bars["close"].loc[t])
        carried, held = held, target
        equity *= 1 + step_return(carried, held, o / prev_close - 1, c / o - 1, spec.cost_rate)
        target = _target(strategy, bars.loc[:t])
        out.append(dict(date=t, prev_close=prev_close, open=o, close=c,
                        carried=carried, held=held,
                        cost=abs(held - carried) * spec.cost_rate, equity=equity,
                        target_next=target, source=source,
                        recorded_at=recorded_at, recorded_late=bool(t != newest)))
        prev_date = t
    return out


def run(spec_path, now: datetime | None = None, fetch=default_fetch,
        root: Path | None = None) -> dict:
    """Record every new completed bar. Returns {symbol: rows appended}.

    All legs are computed before anything is written, so a failed download
    leaves the ledger untouched."""
    spec = load_spec(spec_path)
    now = now or datetime.now(timezone.utc)
    recorded_at = pd.Timestamp(now).strftime("%Y-%m-%dT%H:%M:%SZ")
    d = ledger_dir(spec, root)

    lock = ledger.read_lock(d)
    if lock and lock["sha256"] != spec.sha256:
        raise RuntimeError(f"spec {spec.name} changed since its ledger started "
                           f"(locked {lock['sha256'][:12]}, now {spec.sha256[:12]}). "
                           "Give the new rules a new spec name.")

    pending = {}
    for leg in spec.legs:
        bars = completed_bars(fetch(leg), leg.market, now)
        pending[leg.symbol] = advance_leg(leg, spec, ledger.read_leg(d, leg.symbol), bars,
                                          recorded_at)
    if not lock:
        ledger.write_lock(d, {"spec": spec.name, "sha256": spec.sha256,
                              "started_at": recorded_at})
    for symbol, rows in pending.items():
        ledger.append_leg(d, symbol, rows)
    write_summary(spec, root)
    return {s: len(r) for s, r in pending.items()}


def reconcile(spec_path, fetch=default_fetch, root: Path | None = None,
              now: datetime | None = None) -> list[str]:
    """Problems found, or [] when the ledger checks out.

    1. its own arithmetic: each row follows from the previous one and its
       recorded prices, exactly
    2. the engine: the backtester, run on fresh data over the same bars with
       the strategy's own positions, gives the same equity and targets
    """
    spec = load_spec(spec_path)
    d = ledger_dir(spec, root)
    now = now or datetime.now(timezone.utc)
    problems = []
    for leg in spec.legs:
        rows = ledger.read_leg(d, leg.symbol)
        if rows.empty:
            problems.append(f"{leg.symbol}: no ledger yet")
            continue
        problems += _check_arithmetic(leg.symbol, rows, spec.cost_rate)
        try:
            fresh = fetch(leg)
        except Exception as e:
            # The arithmetic check above needs no data; the engine check does.
            # A download outage skips it with a warning rather than failing.
            print(f"  ! {spec.name}/{leg.symbol}: engine check skipped, fetch failed "
                  f"({str(e)[:80]})", file=sys.stderr)
            continue
        problems += _check_engine(leg, spec, rows, completed_bars(fresh, leg.market, now))
    return problems


def _check_arithmetic(symbol, rows, cost_rate) -> list[str]:
    out = []
    first = rows.iloc[0]
    if (first["equity"], first["held"], first["carried"]) != (1.0, 0.0, 0.0):
        out.append(f"{symbol}: anchor row is not flat at equity 1.0")
    if not rows["date"].is_monotonic_increasing or rows["date"].duplicated().any():
        out.append(f"{symbol}: dates are not strictly increasing")
    for i in range(1, len(rows)):
        p, r = rows.iloc[i - 1], rows.iloc[i]
        day = r["date"].date()
        if r["carried"] != p["held"] or r["held"] != p["target_next"]:
            out.append(f"{symbol} {day}: position does not follow from the previous row")
        step = step_return(r["carried"], r["held"], r["open"] / r["prev_close"] - 1,
                           r["close"] / r["open"] - 1, cost_rate)
        if abs(r["equity"] / (p["equity"] * (1 + step)) - 1) > EQUITY_TOL:
            out.append(f"{symbol} {day}: equity does not follow from the recorded prices")
    return out


def _check_engine(leg, spec, rows, bars) -> list[str]:
    anchor = rows["date"].iloc[0]
    missing = [t.date() for t in rows["date"] if t not in bars.index]
    if missing:
        return [f"{leg.symbol}: fresh data lacks recorded bars {missing[:3]}"]
    # Replay exactly the bars the ledger saw. Yahoo sometimes skips a day and
    # fills it in later; a bar that appeared after the fact is not one the
    # ledger could have traded, and must not make every later check fail.
    seen = bars[(bars.index <= anchor) | bars.index.isin(rows["date"])]
    window = seen.loc[anchor:]
    positions = make_strategy(leg.strategy, **leg.params).generate_positions(seen)
    positions = positions.clip(0.0, 1.0).loc[window.index]
    result = Backtester(fee=spec.fee, slippage=spec.slippage, fill="next_open").run(
        window, FixedPositions(positions))
    equity = result.equity.loc[rows["date"]].to_numpy()
    out = []
    worst = abs(equity / rows["equity"].to_numpy() - 1).max()
    if worst > ENGINE_TOL:
        out.append(f"{leg.symbol}: backtester equity differs from the ledger by up to "
                   f"{worst:.2e} (data revised, or the ledger drifted from the engine)")
    gap = abs(positions.loc[rows["date"]].to_numpy() - rows["target_next"].to_numpy()).max()
    if gap > TARGET_TOL:
        out.append(f"{leg.symbol}: targets differ from the strategy's by up to {gap:.4f}")
    return out


def portfolio_equity(spec: Spec, root: Path | None = None) -> pd.Series:
    """Weighted legs, rebalanced daily on the days every leg trades, the way
    the case study combines them. Derived: rebuilt from the legs each run."""
    d = ledger_dir(spec, root)
    legs = {}
    for leg in spec.legs:
        rows = ledger.read_leg(d, leg.symbol)
        if rows.empty:
            return pd.Series(dtype=float)
        legs[leg.symbol] = rows.set_index("date")["equity"].astype(float)
    first_common = pd.DataFrame(legs).dropna().index
    if len(first_common) == 0:
        return pd.Series(dtype=float)
    start = pd.Series([1.0], index=[first_common[0]])
    if len(first_common) == 1:
        return start
    rets = leg_returns(legs)
    port = sum(leg.weight * rets[leg.symbol] for leg in spec.legs)
    return pd.concat([start, (1 + port).cumprod()])


def summary_text(spec: Spec, root: Path | None = None) -> str:
    d = ledger_dir(spec, root)
    lock = ledger.read_lock(d) or {}
    lines = [f"# Forward record: {spec.name}", "",
             "Generated by `python main.py forward run` after every recorded day; do not edit.",
             "",
             f"Rules frozen in [`forward/specs/{spec.name}.json`](../../specs/{spec.name}.json) "
             f"(rules sha256 `{spec.sha256[:12]}`), recorded daily since "
             f"{lock.get('started_at', 'n/a')}. Each day is a commit: "
             f"`git log -p forward/ledgers/{spec.name}` shows when it was written.", "",
             "| Leg | Anchor bar | Last bar | Bars | Return | Position now | Recorded late |",
             "|---|---|---|---|---|---|---|"]
    for leg in spec.legs:
        rows = ledger.read_leg(d, leg.symbol)
        if rows.empty:
            continue
        lines.append(f"| {leg.symbol} ({leg.weight:.0%}) | {rows['date'].iloc[0].date()} | "
                     f"{rows['date'].iloc[-1].date()} | {len(rows) - 1} | "
                     f"{rows['equity'].iloc[-1] - 1:+.2%} | {rows['target_next'].iloc[-1]:.2f} | "
                     f"{int(rows['recorded_late'].sum())} |")
    port = portfolio_equity(spec, root)
    if len(port):
        lines.append(f"| **Portfolio** | {port.index[0].date()} | {port.index[-1].date()} | "
                     f"{len(port) - 1} | **{port.iloc[-1] - 1:+.2%}** | | |")
    if spec.description:
        lines += ["", spec.description]
    exp = spec.expected
    if exp:
        claim = []
        if "sharpe" in exp:
            claim.append(f"Sharpe {exp['sharpe']}")
        if "sharpe_ci" in exp:
            claim.append(f"95% interval {exp['sharpe_ci'][0]}–{exp['sharpe_ci'][1]}")
        if "note" in exp:
            claim.append(exp["note"])
        lines += ["", f"Backtest expectation ({exp.get('source', 'n/a')}): "
                  f"{', '.join(claim)}."]
        days = max(len(port) - 1, 0)
        if days < 120:
            lines.append(f"{days} portfolio days so far: far too few to compare a Sharpe "
                         "with that interval. Read this as a running log, not a verdict.")
        else:
            r = port.pct_change().dropna()
            sharpe = r.mean() / r.std() * (252 ** 0.5) if r.std() > 0 else 0.0
            lines.append(f"Sharpe over {days} portfolio days: {sharpe:.2f}.")
    return "\n".join(lines) + "\n"


def spec_paths(root: Path | None = None) -> list[Path]:
    return sorted(((root or ROOT) / SPECS).glob("*.json"))


def run_all(now: datetime | None = None, fetch=default_fetch,
            root: Path | None = None) -> dict:
    """Run every spec. One ledger's failed download must not stop the others:
    returns {name: rows appended per leg, or the error}."""
    out = {}
    for path in spec_paths(root):
        try:
            out[path.stem] = run(path, now=now, fetch=fetch, root=root)
        except Exception as e:
            out[path.stem] = e
    write_index(root)
    return out


def reconcile_all(fetch=default_fetch, root: Path | None = None,
                  now: datetime | None = None) -> dict:
    """{name: problems} for every spec whose ledger has started."""
    out = {}
    for path in spec_paths(root):
        if ledger.read_lock(ledger_dir(load_spec(path), root)) is None:
            continue  # never recorded (its first run failed): nothing to check
        out[path.stem] = reconcile(path, fetch=fetch, root=root, now=now)
    return out


def write_index(root: Path | None = None) -> None:
    """forward/README.md: one line per ledger."""
    base = root or ROOT
    lines = ["# Forward records", "",
             "Ledgers trading frozen rules, one completed bar at a time, recorded daily by "
             "GitHub Actions (`.github/workflows/forward.yml`). Generated; do not edit.", "",
             "| Ledger | Legs | Started | Last bar | Portfolio return |",
             "|---|---|---|---|---|"]
    for path in spec_paths(root):
        spec = load_spec(path)
        lock = ledger.read_lock(ledger_dir(spec, root))
        legs = ", ".join(f"{l.symbol} {l.weight:.0%}" for l in spec.legs)
        if lock is None:
            lines.append(f"| {spec.name} | {legs} | not yet | | |")
            continue
        port = portfolio_equity(spec, root)
        last = port.index[-1].date() if len(port) else ""
        ret = f"{port.iloc[-1] - 1:+.2%}" if len(port) else ""
        lines.append(f"| [{spec.name}](ledgers/{spec.name}/) | {legs} | "
                     f"{lock['started_at'][:10]} | {last} | {ret} |")
    (base / "forward").mkdir(parents=True, exist_ok=True)
    (base / "forward" / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(spec: Spec, root: Path | None = None) -> None:
    d = ledger_dir(spec, root)
    port = portfolio_equity(spec, root)
    if len(port):
        port.rename("equity").rename_axis("date").to_frame().to_csv(
            d / "portfolio.csv", float_format="%.12g", date_format="%Y-%m-%d")
    (d / "README.md").write_text(summary_text(spec, root), encoding="utf-8")
