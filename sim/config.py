from dataclasses import dataclass

@dataclass
class SimConfig:
    # Time & geography
    tz: str = "Europe/Berlin"
    freq_min: int = 15

    # Portfolio
    portfolio_n: int = 1000
    history_days: int = 365

    # Technology penetrations (fractions of customers)
    ev_penetration: float = 0.35
    pv_penetration: float = 0.20
    bat_penetration: float = 0.10

    # EV charger mix probabilities (must sum to ~1)
    charger_mix_37: float = 0.25   # 3.7 kW
    charger_mix_74: float = 0.35   # 7.4 kW
    charger_mix_11: float = 0.40   # 11 kW

    # EV behaviour
    commuter_share: float = 0.75   # share of EV owners with commuter pattern

    # Base price levels (€/MWh) — synthetic but realistic
    base_da_price: float = 90.0
    base_id_price: float = 95.0
    base_imb_price: float = 120.0

    # Price regime multipliers
    regime_normal: float = 1.0
    regime_volatile: float = 1.3
    regime_crisis: float = 2.0
