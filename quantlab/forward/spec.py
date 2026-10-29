"""Frozen forward-test specifications.

A spec is the whole rule set a forward ledger trades: legs, strategies,
parameters, costs and the fill. Its rule fields are hashed when the ledger
starts, and the ledger refuses to run under a different hash. Changing a rule
after seeing results would turn a forward record back into a backtest, so a
new rule set gets a new spec name and a new ledger.

`description` and `expected` are documentation and stay outside the hash.
"""
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from ..strategies import REGISTRY

RULE_FIELDS = ("name", "fee", "slippage", "fill", "legs")
MARKETS = ("stock", "crypto")


@dataclass(frozen=True)
class Leg:
    symbol: str
    market: str
    weight: float
    strategy: str
    params: dict
    history: dict        # download window for signals: {"period": "2y"} or {"days": 400}
    source: str | None   # crypto only: pin one data provider


@dataclass(frozen=True)
class Spec:
    name: str
    description: str
    fee: float
    slippage: float
    fill: str
    legs: tuple
    expected: dict
    sha256: str

    @property
    def cost_rate(self) -> float:
        return self.fee + self.slippage


def rules_hash(data: dict) -> str:
    rules = {k: data[k] for k in RULE_FIELDS}
    canonical = json.dumps(rules, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def load_spec(path) -> Spec:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    missing = [k for k in RULE_FIELDS if k not in data]
    if missing:
        raise ValueError(f"spec is missing {missing}")
    if data["fill"] != "next_open":
        raise ValueError("forward ledgers trade at the next open, like the backtests")
    legs = tuple(Leg(symbol=l["symbol"], market=l["market"], weight=float(l["weight"]),
                     strategy=l["strategy"], params=dict(l.get("params", {})),
                     history=dict(l["history"]), source=l.get("source"))
                 for l in data["legs"])
    for leg in legs:
        if leg.market not in MARKETS:
            raise ValueError(f"{leg.symbol}: market must be one of {MARKETS}")
        if leg.strategy not in REGISTRY:
            raise ValueError(f"{leg.symbol}: unknown strategy {leg.strategy!r}")
    if abs(sum(l.weight for l in legs) - 1.0) > 1e-9:
        raise ValueError("leg weights must sum to 1")
    return Spec(name=data["name"], description=data.get("description", ""),
                fee=float(data["fee"]), slippage=float(data["slippage"]), fill=data["fill"],
                legs=legs, expected=dict(data.get("expected", {})), sha256=rules_hash(data))
