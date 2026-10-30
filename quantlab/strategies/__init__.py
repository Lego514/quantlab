from .base import Strategy
from .buy_hold import BuyHold
from .sma_cross import SmaCross
from .rsi_reversion import RsiReversion
from .momentum import Momentum
from .bollinger import BollingerReversion
from .vol_momentum import VolMomentum
from .donchian import DonchianBreakout
from .trend_vol import TrendVolTarget
from .rsi2 import Rsi2

REGISTRY = {
    "donchian": DonchianBreakout,
    "buy_hold": BuyHold,
    "sma_cross": SmaCross,
    "rsi_reversion": RsiReversion,
    "momentum": Momentum,
    "bollinger": BollingerReversion,
    "vol_momentum": VolMomentum,
    "trend_vol": TrendVolTarget,
    "rsi2": Rsi2,
}


def make_strategy(name: str, **params) -> Strategy:
    if name not in REGISTRY:
        raise ValueError(f"unknown strategy {name!r}; available: {list(REGISTRY)}")
    return REGISTRY[name](**params)
