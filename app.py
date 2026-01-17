import streamlit as st
import numpy as np
import pandas as pd

from sim.config import SimConfig
from sim.portfolio import build_portfolio
from sim.forecasting import day_ahead_forecast, typical_day_profile
from sim.markets import make_price_curves, build_layered_hedge_curve_eon_style_seasonal
from sim.intraday import IntradayState, build_ev_plan_from_shift, enforce_ev_deadline, step_intraday
from sim.settlement import compute_settlement
from sim.visuals import portfolio_components, line_with_band, compare_positions, price_chart

st.set_page_config(page_title="Hans Portfolio Walk-Forward Simulator", layout="wide")


def regime_mult(name: str, cfg: SimConfig) -> float:
    return {"Normal": cfg.regime_normal, "Volatile": cfg.regime_volatile, "Crisis": cfg.regime_crisis}.get(name, 1.0)


@st.cache_data(show_spinner=False)
def cached_portfolio(cfg_dict: dict, end_date_str: str, weather_regime: str, seed: int):
    cfg = SimConfig(**cfg_dict)
    end_date = pd.Timestamp(end_date_str)
    port = build_portfolio(
        end_date=end_date,
        days=cfg.history_days,
        freq_min=cfg.freq_min,
        tz=cfg.tz,
        n_customers=cfg.portfolio_n,
        ev_pen=cfg.ev_penetration,
        pv_pen=cfg.pv_penetration,
        bat_pen=cfg.bat_penetration,
        charger_mix=(cfg.charger_mix_37, cfg.charger_mix_74, cfg.charger_mix_11),
        commuter_share=cfg.commuter_share,
        weather_regime=weather_regime,
        seed=seed,
    )
    return port.as_frame()


def init_state():
    if "cfg" not in st.session_state:
        st.session_state.cfg = SimConfig()
    for k in [
        "portfolio_df",
        "selected_day",
        "forward_kw",
        "da_kw",
        "da_forecast",
        "prices",
        "id_state",
        "actual_day_kw",
        "weather_regime",
        "price_regime",
        "ev_cap_kw",
        # EV feasibility objects for delivery day
        "base_ev_kw_day",
        "ev_cap_series_kw_day",
        "ev_required_kwh_deadline",
        "ev_deadline_ts",
    ]:
        if k not in st.session_state:
            st.session_state[k] = None
    if st.session_state.weather_regime is None:
        st.session_state.weather_regime = "Variable"
    if st.session_state.price_regime is None:
        st.session_state.price_regime = "Normal"


init_state()
cfg: SimConfig = st.session_state.cfg
tz = cfg.tz

st.title("Hans Portfolio Walk-Forward Simulator (Germany, 15-min, 1,000 customers)")

with st.sidebar:
    st.header("Portfolio Builder")
    cfg.portfolio_n = st.slider("Portfolio size (customers)", 200, 5000, cfg.portfolio_n, step=100)
    cfg.history_days = st.slider("History length (days)", 30, 730, cfg.history_days, step=15)

    st.subheader("Penetrations")
    cfg.ev_penetration = st.slider("EV penetration", 0.0, 1.0, float(cfg.ev_penetration), step=0.05)
    cfg.pv_penetration = st.slider("PV penetration", 0.0, 1.0, float(cfg.pv_penetration), step=0.05)
    cfg.bat_penetration = st.slider("Battery penetration", 0.0, 1.0, float(cfg.bat_penetration), step=0.05)

    st.subheader("EV behavior")
    cfg.commuter_share = st.slider("Commuter share (EV owners)", 0.0, 1.0, float(cfg.commuter_share), step=0.05)

    st.subheader("World")
    st.session_state.weather_regime = st.selectbox("Weather regime", ["Sunny", "Cloudy", "Variable"], index=2)
    st.session_state.price_regime = st.selectbox("Price regime", ["Normal", "Volatile", "Crisis"], index=0)

    seed = st.number_input("Random seed", min_value=1, max_value=999999, value=7, step=1)
    end_date = st.date_input("History end date", value=pd.Timestamp("2026-01-16").date())

    if st.button("Generate / Regenerate Portfolio", type="primary"):
        cfg_dict = cfg.__dict__.copy()
        st.session_state.portfolio_df = cached_portfolio(cfg_dict, str(end_date), st.session_state.weather_regime, int(seed))

        # reset run state
        st.session_state.selected_day = None
        st.session_state.forward_kw = None
        st.session_state.da_kw = None
        st.session_state.da_forecast = None
        st.session_state.prices = None
        st.session_state.id_state = None
        st.session_state.actual_day_kw = None

        # reset EV feasibility objects
        st.session_state.base_ev_kw_day = None
        st.session_state.ev_cap_series_kw_day = None
        st.session_state.ev_required_kwh_deadline = None
        st.session_state.ev_deadline_ts = None

        # Derive a global EV cap from history (kW)
        df_tmp = st.session_state.portfolio_df
        ev_series = pd.Series(df_tmp["ev_kw"].values, index=df_tmp.index)
        st.session_state.ev_cap_kw = float(max(50.0, ev_series.quantile(0.98) * 1.15))


df = st.session_state.portfolio_df
if df is None:
    st.info("Use the sidebar to generate a synthetic portfolio first.")
    st.stop()

df = df.copy()
if df.index.tz is None:
    df.index = df.index.tz_localize(tz)

available_days = pd.Index(pd.to_datetime(pd.Series(df.index.date).unique())).sort_values()


def get_day_slice(day_ts: pd.Timestamp) -> pd.DataFrame:
    day_ts = day_ts.tz_localize(tz) if day_ts.tzinfo is None else day_ts.tz_convert(tz)
    start = day_ts.normalize()
    end = start + pd.Timedelta(days=1)
    return df.loc[(df.index >= start) & (df.index < end)]


def build_ev_baseline_and_capacity(idx_day: pd.DatetimeIndex) -> tuple[pd.Series, pd.Series]:
    """
    Baseline EV profile from historical typical-day p50.
    Connected capacity series: time-varying cap, bounded by global cap.
    """
    ev_hist = pd.Series(df["ev_kw"].values, index=df.index)
    ev_typ = typical_day_profile(ev_hist, tz=tz)["p50"].values
    base_ev = pd.Series(ev_typ, index=idx_day).clip(lower=0.0)

    global_cap = float(st.session_state.ev_cap_kw or 600.0)

    # Time-varying connected capacity proxy:
    cap = (base_ev * 1.8 + global_cap * 0.15).clip(lower=0.0)
    cap = cap.clip(upper=global_cap)
    return base_ev, cap


def required_kwh_by_deadline_from_baseline(base_ev_kw: pd.Series, deadline_ts: pd.Timestamp, freq_min: int = 15) -> float:
    deadline_ts = deadline_ts if deadline_ts.tzinfo else deadline_ts.tz_localize(base_ev_kw.index.tz)
    pre = base_ev_kw.index <= deadline_ts
    return float(base_ev_kw.loc[pre].sum() * (freq_min / 60.0))


tabs = st.tabs(
    [
        "1) Meter history",
        "2) Profiling & disaggregation",
        "3) Forecasting",
        "4) Forwards hedging",
        "5) Day-ahead (DA)",
        "6) Intraday (ID) game loop",
        "7) Settlement & learning",
    ]
)

with tabs[0]:
    st.subheader("1) Meter history")
    c1, c2 = st.columns([2, 1])
    with c2:
        chosen_day = st.selectbox("Choose a day to simulate forward from", available_days, index=len(available_days) - 2)
        st.session_state.selected_day = pd.Timestamp(chosen_day).tz_localize(tz)

    last7 = df.tail(7 * 24 * 4)
    st.plotly_chart(portfolio_components(last7, title="Last 7 days: components and net meter (kW)"), use_container_width=True)

    if st.session_state.ev_cap_kw is not None:
        st.info(f"Estimated portfolio EV max connected charging cap: ~{st.session_state.ev_cap_kw:.0f} kW")


with tabs[1]:
    st.subheader("2) Profiling & disaggregation")
    day_df = get_day_slice(st.session_state.selected_day)
    st.plotly_chart(portfolio_components(day_df, title=f"Selected day components (kW): {st.session_state.selected_day.date()}"), use_container_width=True)


with tabs[2]:
    st.subheader("3) Forecasting")
    pv_bias = st.slider("PV forecast bias (expect more PV → lower net)", -0.5, 0.5, 0.0, step=0.05)
    ev_shift = st.slider("EV timing shift (quarter-hours)", -16, 16, 0, step=1)
    noise = st.slider("Forecast noise", 0.0, 0.10, 0.03, step=0.01)

    target_day = st.session_state.selected_day + pd.Timedelta(days=1)
    da_f = day_ahead_forecast(
        df,
        tz=tz,
        target_date=target_day,
        pv_forecast_bias=pv_bias,
        ev_timing_shift_qh=ev_shift,
        noise_scale=noise
    )
    st.session_state.da_forecast = da_f
    st.plotly_chart(line_with_band(da_f, title=f"DA forecast for {target_day.date()} (96×15-min)"), use_container_width=True)

    actual_day = get_day_slice(target_day)
    if len(actual_day) == 96:
        st.session_state.actual_day_kw = pd.Series(actual_day["net_kw"].values, index=actual_day.index)


with tabs[3]:
    st.subheader("4) Forwards hedging (E.ON-style layered + seasonal)")
    if st.session_state.da_forecast is None:
        st.warning("Go to Forecasting tab first.")
        st.stop()

    da_curve = st.session_state.da_forecast["forecast_kw"]
    hedge_ratio = st.slider("Total hedge ratio (energy volume)", 0.0, 1.0, 0.85, step=0.05)

    base_share = st.slider("Baseload share (before season/day adjustment)", 0.40, 0.95, 0.70, step=0.05)
    peak_share = 1.0 - float(base_share)

    forecast_conf = st.slider("Forecast confidence", 0.0, 1.0, 0.70, step=0.05)
    ev_flex = st.slider("EV flexibility available", 0.0, 1.0, 0.60, step=0.05)
    pv_unc = st.slider("PV uncertainty", 0.0, 1.0, 0.50, step=0.05)
    peak_strength = st.slider("Peak shaping strength", 0.0, 1.0, 1.00, step=0.05)

    enable_seasonality = st.toggle("Enable seasonality (winter more peak)", value=True)
    enable_weekend = st.toggle("Enable weekend logic (later/weaker peak)", value=True)

    forward_kw = build_layered_hedge_curve_eon_style_seasonal(
        idx=da_curve.index,
        da_forecast_kw=da_curve,
        hedge_ratio_total=float(hedge_ratio),
        base_share_of_hedge=float(base_share),
        peak_share_of_hedge=float(peak_share),
        peak_shape_strength=float(peak_strength),
        forecast_confidence=float(forecast_conf),
        ev_flex_available=float(ev_flex),
        pv_uncertainty=float(pv_unc),
        enable_seasonality=bool(enable_seasonality),
        enable_weekend_logic=bool(enable_weekend),
        freq_min=cfg.freq_min,
    )

    st.session_state.forward_kw = forward_kw
    st.plotly_chart(compare_positions(da_curve.index, forward_kw.values, da_curve.values, title="Forward hedge vs DA forecast (kW)"), use_container_width=True)


with tabs[4]:
    st.subheader("5) Day-ahead (DA): buy/sell delta + EV scheduling with energy-by-departure deadline")
    if st.session_state.forward_kw is None or st.session_state.da_forecast is None:
        st.warning("Run Forecasting + Hedging first.")
        st.stop()

    da_forecast_kw = st.session_state.da_forecast["forecast_kw"]
    idx_day = da_forecast_kw.index
    forward_kw = st.session_state.forward_kw

    prices = make_price_curves(
        idx_day,
        cfg.base_da_price,
        cfg.base_id_price,
        cfg.base_imb_price,
        regime_mult=regime_mult(st.session_state.price_regime, cfg),
        seed=42,
    )
    st.session_state.prices = prices
    st.plotly_chart(price_chart(prices, title="DA / ID / Imbalance price curves (synthetic)"), use_container_width=True)

    base_ev_kw, ev_cap_series_kw = build_ev_baseline_and_capacity(idx_day)

    deadline_hour = st.selectbox("EV departure deadline (hour)", [6.0, 7.0, 8.0], index=1)

    # ---- FIXED TIMEZONE HANDLING ----
    deadline_ts = idx_day[0].normalize() + pd.Timedelta(hours=float(deadline_hour))
    if deadline_ts.tzinfo is None:
        deadline_ts = deadline_ts.tz_localize(tz)
    else:
        deadline_ts = deadline_ts.tz_convert(tz)
    # --------------------------------

    required_kwh = required_kwh_by_deadline_from_baseline(base_ev_kw, deadline_ts, freq_min=cfg.freq_min)
    st.info(f"EV deadline: {deadline_ts.strftime('%H:%M')} | Required energy by deadline (kWh): ~{required_kwh:.1f}")

    use_da_shift = st.toggle("Shift EV charging away from expensive evening hours in DA", value=True)
    flex_strength_da = st.slider("DA EV shift strength (0..1)", 0.0, 1.0, 0.5, step=0.05)

    if use_da_shift:
        ev_plan_da = build_ev_plan_from_shift(
            idx=idx_day,
            base_ev_kw=base_ev_kw,
            ev_cap_series_kw=ev_cap_series_kw,
            prices_eur_mwh=prices["da_eur_mwh"],
            flex_strength=float(flex_strength_da),
            from_window=(17.0, 21.0),
            to_windows=((22.0, 24.0), (0.0, 2.0)),
            freq_min=cfg.freq_min,
        )
        ev_plan_da = enforce_ev_deadline(
            idx=idx_day,
            ev_plan_kw=ev_plan_da,
            ev_cap_series_kw=ev_cap_series_kw,
            required_kwh_by_deadline=float(required_kwh),
            deadline_ts=deadline_ts,
            prices_eur_mwh=prices["da_eur_mwh"],
            freq_min=cfg.freq_min,
        )
    else:
        ev_plan_da = pd.Series(np.minimum(base_ev_kw.values, ev_cap_series_kw.values), index=idx_day)

    ev_shift_adj = (ev_plan_da - base_ev_kw).fillna(0.0)
    da_forecast_for_bidding = (da_forecast_kw + ev_shift_adj).clip(lower=0.0)

    st.plotly_chart(
        compare_positions(idx_day, da_forecast_for_bidding.values, da_forecast_kw.values, title="DA forecast after feasible EV scheduling (kW)"),
        use_container_width=True
    )

    da_strategy = st.selectbox("DA strategy", ["Follow forecast", "Conservative (buy extra)", "Leave for intraday"], index=0)
    buy_mult = {"Follow forecast": 1.0, "Conservative (buy extra)": 1.05, "Leave for intraday": 0.92}[da_strategy]

    raw_delta = da_forecast_for_bidding - forward_kw
    da_order = raw_delta.copy()
    da_order[da_order > 0] = da_order[da_order > 0] * buy_mult
    da_kw = da_order  # allow negative sell
    st.session_state.da_kw = da_kw

    contracted = forward_kw + da_kw
    st.plotly_chart(
        compare_positions(idx_day, contracted.values, da_forecast_for_bidding.values, title="Contracted vs DA (after EV scheduling) — buy/sell enabled"),
        use_container_width=True
    )

    # store EV feasibility objects for ID tab
    st.session_state.base_ev_kw_day = base_ev_kw
    st.session_state.ev_cap_series_kw_day = ev_cap_series_kw
    st.session_state.ev_required_kwh_deadline = required_kwh
    st.session_state.ev_deadline_ts = deadline_ts


with tabs[5]:
    st.subheader("6) Intraday (ID): trade vs EV scheduling with deadline constraint")
    if st.session_state.da_kw is None or st.session_state.prices is None:
        st.warning("Run Day-ahead first.")
        st.stop()

    da_forecast = st.session_state.da_forecast.copy()
    prices = st.session_state.prices.copy()
    idx = da_forecast.index

    forward_kw = st.session_state.forward_kw.reindex(idx).fillna(0.0)
    da_kw = st.session_state.da_kw.reindex(idx).fillna(0.0)

    if st.session_state.actual_day_kw is not None and len(st.session_state.actual_day_kw) == 96:
        actual = st.session_state.actual_day_kw.reindex(idx).fillna(da_forecast["forecast_kw"])
    else:
        rng = np.random.default_rng(1234)
        actual = pd.Series(np.clip(da_forecast["forecast_kw"].values * (1.0 + rng.normal(0, 0.05, size=96)), 0, None), index=idx)

    base_ev_kw = st.session_state.base_ev_kw_day if st.session_state.base_ev_kw_day is not None else pd.Series(0.0, index=idx)
    ev_cap_series_kw = st.session_state.ev_cap_series_kw_day if st.session_state.ev_cap_series_kw_day is not None else pd.Series(0.0, index=idx)
    required_kwh = float(st.session_state.ev_required_kwh_deadline or 0.0)
    deadline_ts = st.session_state.ev_deadline_ts if st.session_state.ev_deadline_ts is not None else (idx[0].normalize() + pd.Timedelta(hours=7))

    if st.session_state.id_state is None or st.session_state.id_state.idx[0] != idx[0]:
        st.session_state.id_state = IntradayState(idx=idx)
    id_state: IntradayState = st.session_state.id_state

    colL, colR = st.columns([1, 2])
    with colL:
        now_i = st.slider("Simulation time (15-min step)", 0, 95, 32, step=1)
        now = idx[now_i]
        st.caption(f"Now: {now.strftime('%Y-%m-%d %H:%M')}")

        trade_kwh = st.number_input("Trade (kWh): +buy / -sell", value=0.0, step=50.0)
        delivery_h = st.selectbox("Trade delivery window (hours)", [1, 2, 4, 6], index=1)
        flex_strength = st.slider("EV scheduling strength (0..1)", 0.0, 1.0, 0.4, step=0.05)

        if st.button("Apply intraday step"):
            step_intraday(
                state=id_state,
                now=now,
                idx=idx,
                base_ev_kw=base_ev_kw,
                ev_cap_series_kw=ev_cap_series_kw,
                required_kwh_by_deadline=required_kwh,
                deadline_ts=deadline_ts,
                prices_id_eur_mwh=prices["id_eur_mwh"],
                flex_strength=float(flex_strength),
                do_trade_kwh=float(trade_kwh),
                trade_delivery_hours=int(delivery_h),
                from_window=(17.0, 21.0),
                to_windows=((22.0, 24.0), (0.0, 2.0)),
                freq_min=cfg.freq_min,
            )
            st.success("Applied intraday trade + EV scheduling (deadline enforced).")

        pre = idx <= deadline_ts
        delivered_kwh = float(id_state.ev_plan_kw.loc[pre].sum() * (cfg.freq_min / 60.0)) if id_state.ev_plan_kw is not None else 0.0
        st.write(f"EV delivered by deadline (kWh): **{delivered_kwh:.1f} / {required_kwh:.1f}**")

    contracted = (forward_kw + da_kw + id_state.trades_kw).reindex(idx)
    actual_after_ev = (actual + id_state.flex_shift_kw).clip(lower=0.0)

    with colR:
        st.plotly_chart(compare_positions(idx, contracted.values, actual_after_ev.values, title="Contracted vs Actual (after EV scheduling) — intraday view"), use_container_width=True)


with tabs[6]:
    st.subheader("7) Settlement & learning")
    if st.session_state.id_state is None or st.session_state.da_kw is None or st.session_state.prices is None:
        st.warning("Run intraday at least once.")
        st.stop()

    da_forecast = st.session_state.da_forecast
    prices = st.session_state.prices
    idx = da_forecast.index

    forward_kw = st.session_state.forward_kw.reindex(idx).fillna(0.0)
    da_kw = st.session_state.da_kw.reindex(idx).fillna(0.0)
    id_state: IntradayState = st.session_state.id_state

    day_df = get_day_slice(idx[0])
    if len(day_df) == 96:
        actual = pd.Series(day_df["net_kw"].values, index=idx)
    else:
        rng = np.random.default_rng(1234)
        actual = pd.Series(np.clip(da_forecast["forecast_kw"].values * (1.0 + rng.normal(0, 0.05, size=96)), 0, None), index=idx)

    result = compute_settlement(
        actual_net_kw=actual,
        forward_kw=forward_kw,
        da_kw=da_kw,
        id_trades_kw=id_state.trades_kw.reindex(idx).fillna(0.0),
        flex_shift_kw=id_state.flex_shift_kw.reindex(idx).fillna(0.0),
        prices=prices,
        freq_min=cfg.freq_min,
    )

    baseline = compute_settlement(
        actual_net_kw=actual,
        forward_kw=forward_kw,
        da_kw=da_kw,
        id_trades_kw=pd.Series(0.0, index=idx),
        flex_shift_kw=pd.Series(0.0, index=idx),
        prices=prices,
        freq_min=cfg.freq_min,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Forward cost (€)", f"{result['fw_cost_eur']:.0f}")
    c2.metric("DA cost (€)", f"{result['da_cost_eur']:.0f}")
    c3.metric("ID cost (€)", f"{result['id_cost_eur']:.0f}")
    c4.metric("Imbalance cost (€)", f"{result['imb_cost_eur']:.0f}")

    st.metric("Total cost (€)", f"{result['total_cost_eur']:.0f}",
              delta=f"{(baseline['total_cost_eur'] - result['total_cost_eur']):.0f} vs no-ID/no-EV-scheduling")

    st.plotly_chart(
        compare_positions(idx=idx, contracted=result["contracted_kw"].values, actual=result["actual_after_flex_kw"].values,
                          title="Final contracted vs actual after EV scheduling (kW)"),
        use_container_width=True
    )

    st.markdown("### What changed vs previous version")
    st.write("- EV scheduling is constrained by a **hard energy-by-departure deadline** (e.g., 07:00).")
    st.write("- DA can **buy or sell** (negative DA = sell).")
    st.write("- Tight deadlines reduce how much you can shift out of peak.")
