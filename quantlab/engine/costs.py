"""Venue cost presets.

A backtest's cost assumption is a claim about where it would trade. These
presets name that claim instead of hiding it in two loose floats. Taker fees
only: a strategy that trades on a signal at the open takes liquidity.

Fees change. Every preset records where and when its number was taken; check
the venue's current schedule before leaning on one.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Venue:
    fee: float        # taker fee per fill, fraction of notional
    slippage: float   # per fill, fraction of notional
    note: str


VENUES = {
    "default": Venue(0.001, 0.0005,
                     "the engine's original generic assumption"),
    "okx_spot": Venue(0.0015, 0.0003,
                      "OKX VIP0 spot taker, recorded 2026-09-12"),
    "okx_perp": Venue(0.0005, 0.0003,
                      "OKX VIP0 perpetual taker, recorded 2026-09-12"),
    "kraken_pro": Venue(0.0026, 0.0005,
                        "Kraken Pro taker, recorded 2026-07-29"),
    "coinbase_advanced": Venue(0.006, 0.0005,
                               "Coinbase Advanced lowest-tier taker, recorded 2026-07-29"),
    "max_tw": Venue(0.0015, 0.0005,
                    "MAX (Taiwan) taker, recorded 2026-07-29"),
}


def venue(name: str) -> Venue:
    if name not in VENUES:
        raise ValueError(f"unknown venue {name!r}; available: {list(VENUES)}")
    return VENUES[name]
