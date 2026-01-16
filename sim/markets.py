import numpy as np
import pandas as pd

def make_price_curves(idx: pd.DatetimeIndex, base_da: float, base_id: float, base_imb: float,
                      regime_mult: float, seed: int = 123) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(idx)
    hour = idx.hour.values + idx.minute.values / 60.0

    # DA: smoother
    da = base_da * regime_mult * (1.0 + 0.10 * np.exp(-0.5 * ((hour - 19.0) / 2.2) ** 2))
    da *= (1.0 + rng.normal(0, 0.04, size=n))

    # ID: more volatile
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
    # cost = sum(kW * hours * €/kWh); €/MWh -> €/kWh = /1000
    hours = freq_min / 60.0
    return float(((kw * hours) * (price_eur_mwh / 1000.0)).sum())
