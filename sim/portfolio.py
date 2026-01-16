import numpy as np
import pandas as pd
from dataclasses import dataclass
from .data_gen import (
    make_time_index, make_weather, sample_ev_params,
    simulate_household_load, simulate_ev_load, simulate_pv_generation, simulate_battery_effect
)

@dataclass
class Portfolio:
    ts: pd.DatetimeIndex
    n: int
    has_ev: np.ndarray
    has_pv: np.ndarray
    has_bat: np.ndarray
    charger_kw: np.ndarray
    weather: np.ndarray

    # portfolio-level series (kW)
    household_kw: np.ndarray
    ev_kw: np.ndarray
    pv_kw: np.ndarray
    bat_kw: np.ndarray
    net_kw: np.ndarray

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "household_kw": self.household_kw,
                "ev_kw": self.ev_kw,
                "pv_kw": self.pv_kw,
                "battery_kw": self.bat_kw,
                "net_kw": self.net_kw,
            },
            index=self.ts
        )

def build_portfolio(end_date: pd.Timestamp, days: int, freq_min: int, tz: str,
                    n_customers: int,
                    ev_pen: float, pv_pen: float, bat_pen: float,
                    charger_mix=(0.25, 0.35, 0.40),
                    commuter_share=0.75,
                    weather_regime="Variable",
                    seed=7) -> Portfolio:
    rng = np.random.default_rng(seed)
    ts = make_time_index(end_date=end_date, days=days, freq_min=freq_min, tz=tz)

    has_ev = rng.random(n_customers) < ev_pen
    has_pv = rng.random(n_customers) < pv_pen
    has_bat = rng.random(n_customers) < bat_pen

    charger_kw = sample_ev_params(n_customers, rng=rng, mix=charger_mix)

    weather = make_weather(ts, regime=weather_regime, rng=rng)

    household_kw = simulate_household_load(ts, n_customers=n_customers, rng=rng)
    ev_kw = simulate_ev_load(ts, has_ev=has_ev, charger_kw=charger_kw,
                            commuter_share=commuter_share, rng=rng)
    pv_kw = simulate_pv_generation(ts, has_pv=has_pv, rng=rng, weather=weather)

    # battery depends on load & pv (very simplified)
    bat_kw = simulate_battery_effect(ts, has_bat=has_bat, pv_kw=pv_kw, load_kw=household_kw + ev_kw, rng=rng)

    # Net meter load: household + EV - PV + battery_power
    net_kw = household_kw + ev_kw - pv_kw + bat_kw

    return Portfolio(
        ts=ts, n=n_customers,
        has_ev=has_ev, has_pv=has_pv, has_bat=has_bat, charger_kw=charger_kw,
        weather=weather,
        household_kw=household_kw, ev_kw=ev_kw, pv_kw=pv_kw, bat_kw=bat_kw, net_kw=net_kw
    )
