import numpy as np
import pandas as pd

def make_time_index(end_date: pd.Timestamp, days: int, freq_min: int, tz: str) -> pd.DatetimeIndex:
    # end_date inclusive-ish: generate [end_date-days, end_date)
    end = end_date.tz_localize(tz) if end_date.tzinfo is None else end_date.tz_convert(tz)
    start = end - pd.Timedelta(days=days)
    return pd.date_range(start=start, end=end, freq=f"{freq_min}min", inclusive="left", tz=tz)

def seasonality_factor(ts: pd.DatetimeIndex) -> np.ndarray:
    # Simple yearly seasonality: higher in winter
    day_of_year = ts.dayofyear.values
    # winter peak around day 15, trough around day 200
    return 1.0 + 0.20 * np.cos(2 * np.pi * (day_of_year - 15) / 365.25)

def day_type_factor(ts: pd.DatetimeIndex) -> np.ndarray:
    # weekend slightly different consumption
    dow = ts.dayofweek.values
    weekend = (dow >= 5).astype(float)
    return 1.0 + 0.05 * weekend

def typical_household_shape(ts: pd.DatetimeIndex) -> np.ndarray:
    # 15-min shape: morning & evening peaks (in kW multiplier)
    hour = ts.hour.values + ts.minute.values / 60.0
    morning = np.exp(-0.5 * ((hour - 7.5) / 1.5) ** 2)
    evening = np.exp(-0.5 * ((hour - 19.0) / 2.0) ** 2)
    night_base = 0.35 + 0.15 * np.exp(-0.5 * ((hour - 2.0) / 2.5) ** 2)
    return night_base + 0.55 * morning + 0.90 * evening

def pv_shape(ts: pd.DatetimeIndex, weather_factor: np.ndarray) -> np.ndarray:
    # midday bell curve scaled by seasonality + weather
    hour = ts.hour.values + ts.minute.values / 60.0
    sun = np.exp(-0.5 * ((hour - 13.0) / 2.5) ** 2)
    # seasonal daylight intensity proxy
    seas = seasonality_factor(ts)
    # invert so summer higher: cos winter high -> use (2 - seas)
    summer_boost = np.clip(2.0 - seas, 0.7, 1.4)
    return sun * summer_boost * weather_factor

def make_weather(ts: pd.DatetimeIndex, regime: str, rng: np.random.Generator) -> np.ndarray:
    # weather_factor ~ [0,1.2]
    base = 0.9
    if regime == "Sunny":
        base = 1.05
        noise = rng.normal(0, 0.08, size=len(ts))
    elif regime == "Cloudy":
        base = 0.65
        noise = rng.normal(0, 0.12, size=len(ts))
    else:  # Variable
        base = 0.85
        noise = rng.normal(0, 0.18, size=len(ts))
    # add slow drift (fronts)
    drift = pd.Series(rng.normal(0, 0.03, size=len(ts))).rolling(32, min_periods=1).mean().values
    wf = np.clip(base + noise + drift, 0.0, 1.2)
    return wf

def sample_ev_params(n: int, rng: np.random.Generator, mix=(0.25, 0.35, 0.40)):
    # charger power in kW
    probs = np.array(mix) / np.sum(mix)
    choices = rng.choice([3.7, 7.4, 11.0], size=n, p=probs)
    return choices

def simulate_ev_load(ts: pd.DatetimeIndex, has_ev: np.ndarray, charger_kw: np.ndarray,
                     commuter_share: float, rng: np.random.Generator) -> np.ndarray:
    """
    Returns EV charging load in kW for each customer aggregated? No: we'll generate per customer energy need,
    then aggregate later in portfolio.py for performance (vectorized per customer-day).
    Here we generate a portfolio-level EV curve by sampling connected EVs + charging windows daily.
    """
    # We'll create an approximate portfolio EV curve:
    # For each day, a fraction of EVs charge with energy need ~ N(8,3) kWh clipped.
    # Arrival time: commuter around 18:00; irregular uniform evening.
    # Charging scheduled as "uncontrolled": start at arrival at full power.
    nT = len(ts)
    ev_curve = np.zeros(nT, dtype=float)

    # Index helpers
    df = pd.DataFrame(index=ts)
    df["date"] = ts.date
    unique_dates = pd.unique(df["date"])
    # typical departure 07:00 next day, but for uncontrolled we just charge on arrival
    for d in unique_dates:
        mask = (df["date"] == d).values
        tday = ts[mask]
        if len(tday) == 0:
            continue

        ev_n = int(has_ev.sum())
        if ev_n == 0:
            continue

        # fraction that needs charging this day
        need_frac = np.clip(rng.normal(0.55, 0.12), 0.25, 0.85)
        chargers_today = charger_kw[has_ev]
        n_need = int(np.round(ev_n * need_frac))
        if n_need <= 0:
            continue
        idx = rng.choice(np.arange(ev_n), size=n_need, replace=False)
        pwr = chargers_today[idx]

        # energy need per EV (kWh)
        kwh_need = np.clip(rng.normal(9.0, 3.0, size=n_need), 2.0, 25.0)

        # arrival time distribution
        is_commuter = rng.random(n_need) < commuter_share
        arr = np.empty(n_need)
        arr[is_commuter] = np.clip(rng.normal(18.2, 1.1, size=is_commuter.sum()), 15.0, 23.0)
        arr[~is_commuter] = np.clip(rng.normal(20.0, 2.0, size=(~is_commuter).sum()), 12.0, 23.75)

        # build charging blocks on this date (uncontrolled)
        for i in range(n_need):
            start_hour = arr[i]
            # duration hours = kWh / kW
            dur_h = kwh_need[i] / pwr[i]
            # map into indices of tday
            start_idx = int(np.floor(start_hour * 4))  # 15-min buckets
            end_idx = int(np.ceil((start_hour + dur_h) * 4))
            start_idx = np.clip(start_idx, 0, len(tday)-1)
            end_idx = np.clip(end_idx, start_idx+1, len(tday))
            ev_curve[np.where(mask)[0][start_idx:end_idx]] += pwr[i]

    return ev_curve

def simulate_household_load(ts: pd.DatetimeIndex, n_customers: int, rng: np.random.Generator) -> np.ndarray:
    # Base per-customer scale ~ lognormal
    scale = rng.lognormal(mean=np.log(0.9), sigma=0.35, size=n_customers)  # kW scale
    shape = typical_household_shape(ts) * seasonality_factor(ts) * day_type_factor(ts)
    # portfolio aggregate: sum_i scale_i * shape + noise
    port = shape[:, None] * scale[None, :]
    # add correlated noise (portfolio) + idiosyncratic
    corr = pd.Series(rng.normal(0, 0.03, size=len(ts))).rolling(16, min_periods=1).mean().values
    port = port * (1.0 + corr[:, None])
    # sum across customers (kW)
    agg = port.sum(axis=1)
    return agg

def simulate_pv_generation(ts: pd.DatetimeIndex, has_pv: np.ndarray, rng: np.random.Generator, weather: np.ndarray) -> np.ndarray:
    n_pv = int(has_pv.sum())
    if n_pv == 0:
        return np.zeros(len(ts), dtype=float)
    # PV size distribution (kWp)
    pv_kwp = rng.lognormal(mean=np.log(5.0), sigma=0.35, size=n_pv)  # typical 3-8 kWp
    pv_unit = pv_shape(ts, weather)  # 0..~1.2
    # generation kW approx = kWp * unit * efficiency
    eff = 0.85
    gen = (pv_unit[:, None] * pv_kwp[None, :] * eff)
    # aggregate
    return gen.sum(axis=1)

def simulate_battery_effect(ts: pd.DatetimeIndex, has_bat: np.ndarray, pv_kw: np.ndarray, load_kw: np.ndarray,
                            rng: np.random.Generator, cap_kwh_mean=8.0, pwr_kw_mean=3.0) -> np.ndarray:
    """
    Simple heuristic at portfolio level:
    - If PV surplus (pv > load), charge up to power and remaining capacity.
    - If evening peak (17-21) and load high, discharge up to power and SOC.
    Returns battery power in kW (positive = adds to net load (charging), negative = reduces net load (discharging)).
    """
    n_bat = int(has_bat.sum())
    if n_bat == 0:
        return np.zeros(len(ts), dtype=float)

    # portfolio equivalent battery capacity/power
    cap = n_bat * cap_kwh_mean
    pwr = n_bat * pwr_kw_mean
    soc = 0.5 * cap

    bat_p = np.zeros(len(ts), dtype=float)

    hour = ts.hour.values + ts.minute.values / 60.0
    for t in range(len(ts)):
        dt_h = 0.25
        surplus = pv_kw[t] - load_kw[t]
        evening = (hour[t] >= 17.0) and (hour[t] <= 21.0)

        if surplus > 0.0:
            # charge (positive power increases net load? actually charging increases load;
            # but if charging from surplus PV behind meter, net export reduces, meaning net load increases.
            # We model battery power as seen at meter: charging consumes PV surplus first, so net meter moves toward 0.
            # We'll approximate: charging absorbs surplus, so it increases net load (less negative / more positive).
            charge_kw = min(pwr, surplus, (cap - soc) / dt_h)
            soc += charge_kw * dt_h
            bat_p[t] = charge_kw
        elif evening:
            # discharge to reduce net load
            discharge_kw = min(pwr, -surplus * 0.0 + pwr, soc / dt_h)  # allow discharge up to pwr
            soc -= discharge_kw * dt_h
            bat_p[t] = -discharge_kw
        else:
            bat_p[t] = 0.0

        soc = np.clip(soc, 0.0, cap)

    return bat_p
