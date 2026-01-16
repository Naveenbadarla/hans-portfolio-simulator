import numpy as np
import pandas as pd

def typical_day_profile(series: pd.Series, tz: str) -> pd.DataFrame:
    s = series.copy()
    # Ensure tz
    if s.index.tz is None:
        s.index = s.index.tz_localize(tz)
    else:
        s.index = s.index.tz_convert(tz)

    df = pd.DataFrame({"v": s})
    df["tod"] = df.index.hour * 4 + (df.index.minute // 15)
    df["date"] = df.index.date

    # Compute percentiles by time-of-day
    g = df.groupby("tod")["v"]
    out = pd.DataFrame({
        "p10": g.quantile(0.10),
        "p50": g.quantile(0.50),
        "p90": g.quantile(0.90),
        "mean": g.mean(),
    })
    out.index.name = "tod"
    return out

def day_ahead_forecast(port_df: pd.DataFrame, tz: str, target_date: pd.Timestamp,
                       pv_forecast_bias: float = 0.0, ev_timing_shift_qh: int = 0,
                       noise_scale: float = 0.03) -> pd.DataFrame:
    """
    Simple DA forecast:
    - Start from typical day (p50)
    - Adjust PV by bias and EV timing by shifting curve
    - Add small noise
    """
    target_date = target_date.tz_localize(tz) if target_date.tzinfo is None else target_date.tz_convert(tz)

    net = pd.Series(port_df["net_kw"].values, index=port_df.index)
    base = typical_day_profile(net, tz=tz)["p50"].values  # 96 points

    # Construct a 96-index for target day
    day_start = target_date.normalize()
    idx = pd.date_range(day_start, day_start + pd.Timedelta(days=1), freq="15min", inclusive="left", tz=tz)

    # crude component adjustments using portfolio components typical patterns
    pv = pd.Series(port_df["pv_kw"].values, index=port_df.index)
    ev = pd.Series(port_df["ev_kw"].values, index=port_df.index)

    pv_base = typical_day_profile(pv, tz=tz)["p50"].values
    ev_base = typical_day_profile(ev, tz=tz)["p50"].values

    # apply pv bias (positive bias means "expect more PV" => net forecast lower)
    pv_adj = pv_base * (1.0 + pv_forecast_bias)

    # shift EV timing in quarter-hours
    ev_adj = np.roll(ev_base, ev_timing_shift_qh)

    # to keep net consistent, adjust net = base - pv_base + pv_adj - ev_base + ev_adj
    net_adj = base - pv_base + pv_adj - ev_base + ev_adj

    rng = np.random.default_rng(int(target_date.value % (2**32 - 1)))
    noise = rng.normal(0, noise_scale, size=96) * np.maximum(1.0, net_adj)
    forecast = np.clip(net_adj + noise, 0.0, None)

    # uncertainty band (simple)
    sigma = np.maximum(0.05 * forecast, 20.0)  # at least 20 kW portfolio
    return pd.DataFrame(
        {
            "forecast_kw": forecast,
            "low_kw": np.clip(forecast - 1.28 * sigma, 0.0, None),
            "high_kw": forecast + 1.28 * sigma,
        },
        index=idx
    )

def intraday_nowcast(da_forecast: pd.DataFrame, actual_so_far: pd.Series, now: pd.Timestamp, tz: str) -> pd.DataFrame:
    """
    Very simple nowcast:
    - Up to 'now': match actual
    - Beyond 'now': DA forecast + bias correction based on recent error
    """
    now = now.tz_localize(tz) if now.tzinfo is None else now.tz_convert(tz)

    f = da_forecast.copy()
    # compute recent bias from last 8 intervals if available
    recent = actual_so_far[actual_so_far.index <= now].tail(8)
    if len(recent) >= 2:
        f_recent = f.loc[recent.index, "forecast_kw"]
        bias = (recent.values - f_recent.values).mean()
    else:
        bias = 0.0

    f["nowcast_kw"] = f["forecast_kw"]
    # overwrite past with actual
    common = f.index.intersection(actual_so_far.index)
    f.loc[common, "nowcast_kw"] = actual_so_far.loc[common].values
    # apply bias to future
    future_mask = f.index > now
    f.loc[future_mask, "nowcast_kw"] = np.clip(f.loc[future_mask, "forecast_kw"].values + bias, 0.0, None)
    return f
