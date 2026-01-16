import numpy as np
import pandas as pd


def make_price_curves(
    idx: pd.DatetimeIndex,
    base_da: float,
    base_id: float,
    base_imb: float,
    regime_mult: float,
    seed: int = 123,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(idx)
    hour = idx.hour.values + idx.minute.values / 60.0

    # DA: smoother, mild evening uplift
    da = base_da * regime_mult * (1.0 + 0.10 * np.exp(-0.5 * ((hour - 19.0) / 2.2) ** 2))
    da *= (1.0 + rng.normal(0, 0.04, size=n))

    # ID: more volatile, stronger uplift
    idp = base_id * regime_mult * (1.0 + 0.14 * np.exp(-0.5 * ((hour - 19.0) / 2.0) ** 2))
    idp *= (1.0 + rng.normal(0, 0.08, size=n))

    # Imbalance: spiky
    imb = base_imb * regime_mult * (1.0 + 0.25 * np.exp(-0.5 * ((hour - 19.0) / 1.8) ** 2))
    imb *= (1.0 + rng.normal(0, 0.15, size=n))
    imb = np.clip(imb, 20.0, None)

    return pd.DataFrame({"da_eur_mwh": da, "id_eur_mwh": idp, "imb_eur_mwh": imb}, index=idx)


def kwh_from_kw_series(kw: pd.Series, freq_min: int = 15) -> float:
    return float(kw.sum() * (freq_min / 60.0))


def cost_eur(kw: pd.Series, price_eur_mwh: pd.Series, freq_min: int = 15) -> float:
    hours = freq_min / 60.0
    return float(((kw * hours) * (price_eur_mwh / 1000.0)).sum())


# -----------------------------
# E.ON-style layered hedging
# -----------------------------

def _season_peak_multiplier(month: int) -> float:
    """
    Winter (Nov–Feb) => more peak hedging
    Shoulder (Mar/Apr, Sep/Oct) => medium
    Summer (May–Aug) => less
    """
    if month in [11, 12, 1, 2]:
        return 1.25
    if month in [3, 4, 9, 10]:
        return 1.05
    return 0.85


def _daytype_peak_multiplier(dayofweek: int) -> float:
    """
    0=Mon ... 6=Sun
    Weekends: reduce peak overlay (less pronounced commuter peak)
    """
    if dayofweek >= 5:
        return 0.80
    return 1.00


def build_layered_hedge_curve_eon_style_seasonal(
    idx: pd.DatetimeIndex,
    da_forecast_kw: pd.Series,
    hedge_ratio_total: float,
    # Base shares before seasonal adjustments
    base_share_of_hedge: float = 0.70,
    peak_share_of_hedge: float = 0.30,
    # Default weekday peak window
    weekday_peak_start_hour: float = 17.0,
    weekday_peak_end_hour: float = 21.0,
    # Weekend peak window (often later / flatter)
    weekend_peak_start_hour: float = 18.0,
    weekend_peak_end_hour: float = 22.0,
    # Shape sizing knobs
    peak_shape_strength: float = 1.00,
    cap_peak_kw_fraction_of_forecast: float = 0.60,
    # Risk / optionality knobs
    forecast_confidence: float = 0.70,  # 0..1
    ev_flex_available: float = 0.60,    # 0..1
    pv_uncertainty: float = 0.50,       # 0..1
    # Enable/disable seasonal logic
    enable_seasonality: bool = True,
    enable_weekend_logic: bool = True,
    freq_min: int = 15,
) -> pd.Series:
    """
    E.ON-style layered hedge curve for one day (96 points):
      - Baseload layer: flat kW covering structural energy
      - Peak layer: overlay during peak window sized by forecast "peaky-ness"
      - Peak share is adjusted by season (winter more) and weekend (less)
      - Peak shaping reduces when PV uncertainty is high or EV flexibility is high
      - Total hedged daily energy = hedge_ratio_total * forecast daily energy

    Returns:
      forward_hedge_kw (pd.Series indexed by idx)
    """
    assert 0.0 <= hedge_ratio_total <= 1.0
    assert 0.0 <= base_share_of_hedge <= 1.0
    assert 0.0 <= peak_share_of_hedge <= 1.0
    if abs((base_share_of_hedge + peak_share_of_hedge) - 1.0) > 1e-9:
        raise ValueError("Base+Peak shares must sum to 1")

    hours_per_step = freq_min / 60.0
    forecast_kwh = float(da_forecast_kw.sum() * hours_per_step)
    target_hedged_kwh = forecast_kwh * hedge_ratio_total

    # Determine the day from first timestamp (idx should be one day)
    day = idx[0]
    month = int(day.month)
    dow = int(day.dayofweek)  # 0=Mon ... 6=Sun

    # Select peak window
    if enable_weekend_logic and dow >= 5:
        peak_start_hour = weekend_peak_start_hour
        peak_end_hour = weekend_peak_end_hour
    else:
        peak_start_hour = weekday_peak_start_hour
        peak_end_hour = weekday_peak_end_hour

    # Seasonal/daytype multipliers for peak share
    season_mult = _season_peak_multiplier(month) if enable_seasonality else 1.0
    day_mult = _daytype_peak_multiplier(dow) if enable_weekend_logic else 1.0

    # Adjust peak share but keep a sane range
    peak_share_adj = float(np.clip(peak_share_of_hedge * season_mult * day_mult, 0.05, 0.60))
    base_share_adj = 1.0 - peak_share_adj

    # Shaping permission: shape more when confidence high & uncertainty low,
    # shape less when EV flexibility high (keep optionality).
    shaping_permission = (
        0.55 * float(np.clip(forecast_confidence, 0.0, 1.0))
        + 0.25 * (1.0 - float(np.clip(pv_uncertainty, 0.0, 1.0)))
        + 0.20 * (1.0 - float(np.clip(ev_flex_available, 0.0, 1.0)))
    )
    shaping_permission = float(np.clip(shaping_permission, 0.0, 1.0))
    shaping_permission *= float(np.clip(peak_shape_strength, 0.0, 1.0))

    # Peak mask
    hour = idx.hour.values + idx.minute.values / 60.0
    peak_mask = (hour >= peak_start_hour) & (hour < peak_end_hour)
    peak_steps = int(peak_mask.sum())
    total_steps = len(idx)

    # Fallback to flat if peak invalid
    if peak_steps <= 0 or peak_steps >= total_steps:
        avg_kw = target_hedged_kwh / (total_steps * hours_per_step)
        return pd.Series(avg_kw, index=idx)

    # Allocate energy to layers using adjusted shares
    base_kwh = target_hedged_kwh * base_share_adj
    peak_kwh_budget = target_hedged_kwh * peak_share_adj

    base_kw = base_kwh / (total_steps * hours_per_step)

    # Size peak add-on by forecast peaky-ness
    forecast_arr = da_forecast_kw.values.astype(float)
    peak_mean = float(np.mean(forecast_arr[peak_mask]))
    off_mean = float(np.mean(forecast_arr[~peak_mask])) if (~peak_mask).sum() > 0 else peak_mean

    uplift = 0.0 if off_mean <= 1e-9 else (peak_mean - off_mean) / off_mean
    uplift = float(np.clip(uplift, 0.0, 1.0))

    peak_add_kw_budget = peak_kwh_budget / (peak_steps * hours_per_step)
    peak_add_kw = peak_add_kw_budget * uplift * shaping_permission

    # Cap peak add-on (desk risk control)
    peak_cap_kw = float(cap_peak_kw_fraction_of_forecast) * peak_mean
    peak_add_kw = float(np.clip(peak_add_kw, 0.0, peak_cap_kw))

    curve = np.full(total_steps, base_kw, dtype=float)
    curve[peak_mask] += peak_add_kw

    # Normalize to hit exact kWh target
    current_kwh = float(curve.sum() * hours_per_step)
    if current_kwh <= 1e-9:
        avg_kw = target_hedged_kwh / (total_steps * hours_per_step)
        return pd.Series(avg_kw, index=idx)

    curve *= (target_hedged_kwh / current_kwh)
    return pd.Series(curve, index=idx)
