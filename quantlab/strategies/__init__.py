from .base import Strategy
from .buy_hold import BuyHold
from .sma_cross import SmaCross
from .rsi_reversion import RsiReversion
from .momentum import Momentum
from .bollinger import BollingerReversion
from .vol_momentum import VolMomentum
from .donchian import DonchianBreakout

REGISTRY = {
    "donchian": DonchianBreakout,
    "buy_hold": BuyHold,
    "sma_cross": SmaCross,
    "rsi_reversion": RsiReversion,
    "momentum": Momentum,
    "bollinger": BollingerReversion,
    "vol_momentum": VolMomentum,
}


def make_strategy(name: str, **params) -> Strategy:
    if name not in REGISTRY:
        raise ValueError(f"unknown strategy {name!r}; available: {list(REGISTRY)}")
    return REGISTRY[name](**params)
