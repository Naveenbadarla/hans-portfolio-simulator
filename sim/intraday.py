import numpy as np
import pandas as pd


class IntradayState:
    """
    Stores cumulative decisions for a single delivery day (96 x 15-min).
    trades_kw: signed kW trades (buy + / sell -)
    ev_plan_kw: the planned EV charging schedule (kW) after shifts/enforcement
    flex_shift_kw: net-load adjustment implied by EV plan vs baseline EV (kW)
    """
    def __init__(self, idx: pd.DatetimeIndex):
        self.idx = idx
        self.trades_kw = pd.Series(0.0, index=idx)
        self.ev_plan_kw = None
        self.flex_shift_kw = pd.Series(0.0, index=idx)


def _hour_of_day(idx: pd.DatetimeIndex) -> np.ndarray:
    return idx.hour.values + idx.minute.values / 60.0


def _mask_window(idx: pd.DatetimeIndex, start_hour: float, end_hour: float) -> np.ndarray:
    h = _hour_of_day(idx)
    return (h >= start_hour) & (h < end_hour)


def _mask_multi_windows(idx: pd.DatetimeIndex, windows) -> np.ndarray:
    h = _hour_of_day(idx)
    m = np.zeros(len(idx), dtype=bool)
    for a, b in windows:
        m |= (h >= a) & (h < b)
    return m


def _kwh_from_kw(kw: pd.Series, freq_min: int) -> float:
    return float(kw.sum() * (freq_min / 60.0))


def enforce_ev_deadline(
    idx: pd.DatetimeIndex,
    ev_plan_kw: pd.Series,
    ev_cap_series_kw: pd.Series,
    required_kwh_by_deadline: float,
    deadline_ts: pd.Timestamp,
    prices_eur_mwh: pd.Series,
    freq_min: int = 15,
) -> pd.Series:
    """
    Ensures the EV plan delivers at least required_kwh_by_deadline by deadline_ts,
    without exceeding ev_cap_series_kw.
    If short, it adds energy into the CHEAPEST available slots before deadline using headroom.

    Returns a feasible ev_plan_kw (kW).
    """
    ev_plan_kw = ev_plan_kw.reindex(idx).fillna(0.0)
    cap = ev_cap_series_kw.reindex(idx).fillna(0.0)

    # Cap the plan
    ev_plan_kw = pd.Series(np.minimum(ev_plan_kw.values, cap.values), index=idx).clip(lower=0.0)

    # Identify pre-deadline slots (inclusive up to deadline)
    deadline_ts = deadline_ts if deadline_ts.tzinfo else deadline_ts.tz_localize(idx.tz)
    pre = idx <= deadline_ts
    if pre.sum() == 0:
        return ev_plan_kw

    delivered_kwh = _kwh_from_kw(ev_plan_kw.loc[pre], freq_min=freq_min)
    short_kwh = max(0.0, float(required_kwh_by_deadline) - delivered_kwh)
    if short_kwh <= 1e-9:
        return ev_plan_kw

    # Headroom before deadline
    headroom_kw = (cap - ev_plan_kw).clip(lower=0.0)
    headroom_kwh = _kwh_from_kw(headroom_kw.loc[pre], freq_min=freq_min)
    if headroom_kwh <= 1e-9:
        # Cannot fix infeasibility; return capped plan (still short)
        return ev_plan_kw

    add_kwh = min(short_kwh, headroom_kwh)

    # Allocate added kWh to cheapest intervals before deadline
    # Sort pre-deadline indices by price ascending
    prices_pre = prices_eur_mwh.reindex(idx).fillna(prices_eur_mwh.mean()).loc[pre]
    order = prices_pre.sort_values(ascending=True).index

    hours_per_step = freq_min / 60.0
    remaining = add_kwh

    ev_plan_arr = ev_plan_kw.copy()
    for t in order:
        if remaining <= 1e-9:
            break
        hr_kw = float(headroom_kw.loc[t])
        if hr_kw <= 1e-9:
            continue
        # how much kWh can we add in this slot
        can_add_kwh = hr_kw * hours_per_step
        delta_kwh = min(remaining, can_add_kwh)
        delta_kw = delta_kwh / hours_per_step
        ev_plan_arr.loc[t] += delta_kw
        remaining -= delta_kwh

    # Final cap safety
    ev_plan_arr = pd.Series(np.minimum(ev_plan_arr.values, cap.values), index=idx).clip(lower=0.0)
    return ev_plan_arr


def build_ev_plan_from_shift(
    idx: pd.DatetimeIndex,
    base_ev_kw: pd.Series,
    ev_cap_series_kw: pd.Series,
    prices_eur_mwh: pd.Series,
    flex_strength: float,
    from_window=(17.0, 21.0),
    to_windows=((22.0, 24.0), (0.0, 2.0)),
    freq_min: int = 15,
    max_shift_fraction_of_from_ev: float = 0.80,
) -> pd.Series:
    """
    Creates an EV plan by shifting EV energy from 'from_window' into 'to_windows',
    enforcing capacity feasibility and energy conservation.
    Does NOT enforce deadline. Use enforce_ev_deadline after this if needed.
    """
    if flex_strength <= 0:
        # just cap baseline
        cap = ev_cap_series_kw.reindex(idx).fillna(0.0)
        base = base_ev_kw.reindex(idx).fillna(0.0).clip(lower=0.0)
        return pd.Series(np.minimum(base.values, cap.values), index=idx)

    hours_per_step = freq_min / 60.0
    base = base_ev_kw.reindex(idx).fillna(0.0).clip(lower=0.0)
    cap = ev_cap_series_kw.reindex(idx).fillna(0.0).clip(lower=0.0)

    from_mask = _mask_window(idx, from_window[0], from_window[1])
    to_mask = _mask_multi_windows(idx, to_windows)

    if from_mask.sum() == 0 or to_mask.sum() == 0:
        return pd.Series(np.minimum(base.values, cap.values), index=idx)

    # Energy available to move (kWh) from from_window
    from_energy_kwh = float(base.loc[from_mask].sum() * hours_per_step)
    if from_energy_kwh <= 1e-9:
        return pd.Series(np.minimum(base.values, cap.values), index=idx)

    target_shift_kwh = from_energy_kwh * float(np.clip(flex_strength, 0.0, 1.0)) * max_shift_fraction_of_from_ev

    # Headroom in to_windows (kWh)
    headroom_kw = (cap - base).clip(lower=0.0)
    headroom_kwh = float(headroom_kw.loc[to_mask].sum() * hours_per_step)
    feasible_shift_kwh = min(target_shift_kwh, headroom_kwh)
    if feasible_shift_kwh <= 1e-9:
        return pd.Series(np.minimum(base.values, cap.values), index=idx)

    # Reduce proportional to base EV in from_window (never below 0)
    from_vals = base.loc[from_mask]
    from_sum = float(from_vals.sum())
    if from_sum <= 1e-9:
        return pd.Series(np.minimum(base.values, cap.values), index=idx)

    reduce_weights = from_vals / from_sum
    reduce_kwh_each = feasible_shift_kwh * reduce_weights
    reduce_kw_each = reduce_kwh_each / hours_per_step

    # Add to cheapest slots within to_window using headroom weights guided by prices (cheapest first)
    plan = base.copy()
    plan.loc[from_mask] = (plan.loc[from_mask] - reduce_kw_each).clip(lower=0.0)

    # Add back energy: greedy by low DA prices within to_mask
    prices_to = prices_eur_mwh.reindex(idx).fillna(prices_eur_mwh.mean()).loc[to_mask]
    order = prices_to.sort_values(ascending=True).index

    remaining = feasible_shift_kwh
    for t in order:
        if remaining <= 1e-9:
            break
        hr_kw = float((cap.loc[t] - plan.loc[t]))
        if hr_kw <= 1e-9:
            continue
        can_add_kwh = hr_kw * hours_per_step
        delta_kwh = min(remaining, can_add_kwh)
        plan.loc[t] += (delta_kwh / hours_per_step)
        remaining -= delta_kwh

    # Final cap safety
    plan = pd.Series(np.minimum(plan.values, cap.values), index=idx).clip(lower=0.0)
    return plan


def step_intraday(
    state: IntradayState,
    now: pd.Timestamp,
    idx: pd.DatetimeIndex,
    # Baselines / feasibility inputs:
    base_ev_kw: pd.Series,
    ev_cap_series_kw: pd.Series,
    required_kwh_by_deadline: float,
    deadline_ts: pd.Timestamp,
    prices_id_eur_mwh: pd.Series,
    # User actions:
    flex_strength: float,
    do_trade_kwh: float,
    trade_delivery_hours: int,
    # Shift preferences:
    from_window=(17.0, 21.0),
    to_windows=((22.0, 24.0), (0.0, 2.0)),
    freq_min: int = 15,
):
    """
    Intraday step:
      1) Update EV plan using shifting strength (price-guided) and enforce deadline.
      2) Add trades over next N hours (constant kW).

    The EV plan is global for the day in this MVP. A V2 would make it receding-horizon
    from 'now' with connected-EV forecasts and multiple departure cohorts.
    """
    now = now if now.tzinfo else now.tz_localize(idx.tz)

    # Start from previous plan if exists, otherwise baseline
    if state.ev_plan_kw is None:
        state.ev_plan_kw = base_ev_kw.reindex(idx).fillna(0.0)

    # Build a new plan using baseline (not cumulative stacking) to keep it stable to play with
    ev_plan = build_ev_plan_from_shift(
        idx=idx,
        base_ev_kw=base_ev_kw,
        ev_cap_series_kw=ev_cap_series_kw,
        prices_eur_mwh=prices_id_eur_mwh,
        flex_strength=float(flex_strength),
        from_window=from_window,
        to_windows=to_windows,
        freq_min=freq_min,
    )

    # Enforce energy-by-deadline constraint
    ev_plan = enforce_ev_deadline(
        idx=idx,
        ev_plan_kw=ev_plan,
        ev_cap_series_kw=ev_cap_series_kw,
        required_kwh_by_deadline=float(required_kwh_by_deadline),
        deadline_ts=deadline_ts,
        prices_eur_mwh=prices_id_eur_mwh,
        freq_min=freq_min,
    )

    state.ev_plan_kw = ev_plan

    # Net-load adjustment implied by EV plan vs baseline EV
    state.flex_shift_kw = (state.ev_plan_kw - base_ev_kw.reindex(idx).fillna(0.0)).fillna(0.0)

    # Trades: spread kWh over delivery window
    if do_trade_kwh != 0.0:
        hours = int(trade_delivery_hours)
        if hours <= 0:
            return
        kw = float(do_trade_kwh) / float(hours)
        end = now + pd.Timedelta(hours=hours)
        m = (idx > now) & (idx <= end)
        state.trades_kw.loc[m] += kw
