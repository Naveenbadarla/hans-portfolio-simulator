import pandas as pd
from .markets import cost_eur

def compute_settlement(actual_net_kw: pd.Series,
                       forward_kw: pd.Series,
                       da_kw: pd.Series,
                       id_trades_kw: pd.Series,
                       flex_shift_kw: pd.Series,
                       prices: pd.DataFrame,
                       freq_min: int = 15) -> dict:
    """
    Positions:
    - forward_kw: base hedged schedule (kW curve)
    - da_kw: day-ahead incremental schedule (kW curve)
    - id_trades_kw: intraday adjustments (kW curve)
    Flex:
    - flex_shift_kw affects actual net load (negative reduces load)
    """
    contracted = forward_kw + da_kw + id_trades_kw
    actual_after_flex = (actual_net_kw + flex_shift_kw).clip(lower=0.0)

    # residual imbalance = actual - contracted
    imb = actual_after_flex - contracted

    # costs (positive means pay)
    da_cost = cost_eur(da_kw, prices["da_eur_mwh"], freq_min=freq_min)
    # forward cost: approximate at DA price level for MVP
    fw_cost = cost_eur(forward_kw, prices["da_eur_mwh"], freq_min=freq_min)
    id_cost = cost_eur(id_trades_kw, prices["id_eur_mwh"], freq_min=freq_min)
    imb_cost = cost_eur(imb, prices["imb_eur_mwh"], freq_min=freq_min)

    total = fw_cost + da_cost + id_cost + imb_cost

    return {
        "fw_cost_eur": fw_cost,
        "da_cost_eur": da_cost,
        "id_cost_eur": id_cost,
        "imb_cost_eur": imb_cost,
        "total_cost_eur": total,
        "contracted_kw": contracted,
        "actual_after_flex_kw": actual_after_flex,
        "imbalance_kw": imb,
    }
