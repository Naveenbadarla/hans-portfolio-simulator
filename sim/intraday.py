import numpy as np
import pandas as pd

class IntradayState:
    def __init__(self, idx: pd.DatetimeIndex):
        self.idx = idx
        self.trades_kw = pd.Series(0.0, index=idx)     # net buy(+)/sell(-) adjustments
        self.flex_shift_kw = pd.Series(0.0, index=idx) # effect on net load (negative reduces net load)

def apply_flex_shift(now_idx: int, idx: pd.DatetimeIndex, now: pd.Timestamp,
                     flex_strength: float, hours_window: int = 6) -> pd.Series:
    """
    Simple portfolio flex model:
    - We can shave peaks in next few hours by shifting EV charging within a window.
    - Implement as: reduce net load in high-price evening hours and move it to later night.
    flex_strength in [0,1] controls amplitude.
    Returns a Series of kW adjustments (negative = reduces net load).
    """
    adj = pd.Series(0.0, index=idx)
    if flex_strength <= 0:
        return adj

    # target window from now to now+hours_window
    end = now + pd.Timedelta(hours=hours_window)
    w = (idx >= now) & (idx <= end)

    # reduce during 17-21, add back during 22-02
    hours = idx.hour.values + idx.minute.values / 60.0
    peak = (hours >= 17.0) & (hours <= 21.0) & w
    night = ((hours >= 22.0) | (hours <= 2.0)) & w

    peak_n = peak.sum()
    night_n = night.sum()
    if peak_n == 0 or night_n == 0:
        return adj

    # portfolio-scale shift capacity (kW): tie it loosely to flex_strength
    # This is an MVP proxy: assume 1000 customers => ~100-600 kW flexible at any time depending on strength
    shift_kw = 600.0 * flex_strength
    # distribute
    adj.loc[peak] = -shift_kw
    adj.loc[night] = +shift_kw * (peak_n / night_n)  # energy balance within window
    return adj

def step_intraday(state: IntradayState, now: pd.Timestamp, nowcast_kw: pd.Series,
                  contracted_kw: pd.Series,
                  do_trade_kwh: float, trade_delivery_hours: int,
                  flex_strength: float,
                  freq_min: int = 15):
    """
    One decision step:
    - Trade: buy/sell constant power over next N hours to reduce delta
    - Flex: apply a shift pattern to reduce peaks (energy-balanced locally)
    """
    idx = state.idx
    now = now if now.tzinfo else now.tz_localize(idx.tz)

    # apply flex adjustments
    flex_adj = apply_flex_shift(now_idx=idx.get_indexer([now], method="nearest")[0],
                                idx=idx, now=now, flex_strength=flex_strength, hours_window=6)
    state.flex_shift_kw = state.flex_shift_kw.add(flex_adj, fill_value=0.0)

    # Trade implementation: spread kWh over delivery window
    if do_trade_kwh != 0.0:
        hours = trade_delivery_hours
        # convert kWh to constant kW over window
        kw = do_trade_kwh / hours
        end = now + pd.Timedelta(hours=hours)
        m = (idx > now) & (idx <= end)
        state.trades_kw.loc[m] += kw
