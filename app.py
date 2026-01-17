import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

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


def multiline_chart(idx: pd.DatetimeIndex, series_dict: dict, title: str):
    fig = go.Figure()
    for name, s in series_dict.items():
        fig.add_trace(go.Scatter(x=idx, y=np.asarray(s), mode="lines", name=name))
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="kW",
        legend_title="Series",
        height=420,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


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

        st.session_state.selected_day = None
        st.session_state.forward_kw = None
        st.session_state.da_kw = None
        st.session_state.da_forecast = None
        st.session_state.prices = None
        st.session_state.id_state = None
        st.session_state.actual_day_kw = None

        st.session_state.base_ev_kw_day = None
        st.session_state.ev_cap_series_kw_day = None
        st.session_state.ev_required_kwh_deadline = None
        st.session_state.ev_deadline_ts = None

        df_tmp = st.session_state.portfolio_df
        ev_series = pd.Series(df_tmp["ev_kw"].values, index=df_tmp.index)
        st.session_state.ev_cap_kw = float(max(50.0, ev_series.quantile(0.98) * 1.15))


df = st.session_state.portfolio_df
if df is None:
    st.info("Generate a synthetic portfolio first (left sidebar).")
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
    ev_hist = pd.Series(df["ev_kw"].values, index=df.index)
    ev_typ = typical_day_profile(ev_hist, tz=tz)["p50"].values
    base_ev = pd.Series(ev_typ, index=idx_day).clip(lower=0.0)

    global_cap = float(st.session_state.ev_cap_kw or 600.0)
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
        "2) Profiling",
        "3) Forecasting",
        "4) Forwards hedging",
        "5) Day-ahead (DA)",
        "6) Intraday (ID)",
        "7) Settlement",
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
    if st.session_state.selected_day is None:
        st.info("Pick a day in Tab 1 first.")
        st.stop()
    day_df = get_day_slice(st.session_state.selected_day)
    st.plotly_chart(portfolio_components(day_df, title=f"Selected day components (kW): {st.session_state.selected_day.date()}"), use_container_width=True)


with tabs[2]:
    st.subheader("3) Forecasting (DA)")
    if st.session_state.selected_day is None:
        st.info("Pick a day in Tab 1 first.")
        st.stop()

    pv_bias = st.slider("PV forecast bias", -0.5, 0.5, 0.0, step=0.05)
    ev_shift = st.slider("EV timing shift (quarter-hours)", -16, 16, 0, step=1)
    noise = st.slider("Forecast noise", 0.0, 0.10, 0.03, step=0.01)

    target_day = st.session_state.selected_day + pd.Timedelta(days=1)
    da_f = day_ahead_forecast(df, tz=tz, target_date=target_day, pv_forecast_bias=pv_bias, ev_timing_shift_qh=ev_shift, noise_scale=noise)
    st.session_state.da_forecast = da_f
    st.plotly_chart(line_with_band(da_f, title=f"DA forecast for {target_day.date()} (96×15-min)"), use_container_width=True)

    actual_day = get_day_slice(target_day)
    if len(actual_day) == 96:
        st.session_state.actual_day_kw = pd.Series(actual_day["net_kw"].values, index=actual_day.index)


with tabs[3]:
    st.subheader("4) Forwards hedging (E.ON-style layered + seasonal)")
    if st.session_state.da_forecast is None:
        st.warning("Run forecasting first (Tab 3).")
        st.stop()

    da_curve = st.session_state.da_forecast["forecast_kw"]
    hedge_ratio = st.slider("Total hedge ratio", 0.0, 1.0, 0.85, step=0.05)

    base_share = st.slider("Baseload share", 0.40, 0.95, 0.70, step=0.05)
    peak_share = 1.0 - float(base_share)

    forecast_conf = st.slider("Forecast confidence", 0.0, 1.0, 0.70, step=0.05)
    ev_flex = st.slider("EV flexibility available", 0.0, 1.0, 0.60, step=0.05)
    pv_unc = st.slider("PV uncertainty", 0.0, 1.0, 0.50, step=0.05)
    peak_strength = st.slider("Peak shaping strength", 0.0, 1.0, 1.00, step=0.05)

    enable_seasonality = st.toggle("Enable seasonality", value=True)
    enable_weekend = st.toggle("Enable weekend logic", value=True)

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
    st.subheader("5) Day-ahead (DA): EV scheduling + household-vs-EV + DA shift attribution")

    if st.session_state.forward_kw is None or st.session_state.da_forecast is None:
        st.warning("Run forecasting (Tab 3) and hedging (Tab 4) first.")
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
    st.plotly_chart(price_chart(prices, title="Prices (DA/ID/IMB)"), use_container_width=True)

    base_ev_kw, ev_cap_series_kw = build_ev_baseline_and_capacity(idx_day)

    deadline_hour = st.selectbox("EV departure deadline (hour)", [6.0, 7.0, 8.0], index=1)
    deadline_ts = idx_day[0].normalize() + pd.Timedelta(hours=float(deadline_hour))
    if deadline_ts.tzinfo is None:
        deadline_ts = deadline_ts.tz_localize(tz)
    else:
        deadline_ts = deadline_ts.tz_convert(tz)

    required_kwh = required_kwh_by_deadline_from_baseline(base_ev_kw, deadline_ts, freq_min=cfg.freq_min)
    st.info(f"EV deadline: {deadline_ts.strftime('%H:%M')} | Required by deadline: ~{required_kwh:.1f} kWh")

    use_da_shift = st.toggle("Enable DA price-aware EV scheduling", value=True)
    flex_strength_da = st.slider("DA EV scheduling strength (0..1)", 0.0, 1.0, 0.60, step=0.05)

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

    # Non-EV (household+PV+battery) implied by DA forecast
    non_ev_kw = (da_forecast_kw - base_ev_kw).fillna(0.0).clip(lower=0.0)

    # Net after EV scheduling
    net_after_ev_schedule_kw = (non_ev_kw + ev_plan_da).clip(lower=0.0)
    da_forecast_for_bidding = net_after_ev_schedule_kw

    # EV baseline vs planned
    st.plotly_chart(
        multiline_chart(
            idx_day,
            {
                "EV baseline (kW)": base_ev_kw.values,
                "EV planned (kW)": ev_plan_da.values,
                "EV capacity (kW)": ev_cap_series_kw.values,
            },
            "EV charging: baseline vs DA planned (kW)"
        ),
        use_container_width=True,
    )

    # Net before vs after EV schedule
    st.plotly_chart(
        multiline_chart(
            idx_day,
            {
                "Net before (DA forecast)": da_forecast_kw.values,
                "Net after EV scheduling": da_forecast_for_bidding.values,
            },
            "Portfolio net load: before vs after DA EV scheduling (kW)"
        ),
        use_container_width=True,
    )

    # Non-EV vs EV (baseline vs planned)
    st.plotly_chart(
        multiline_chart(
            idx_day,
            {
                "Non-EV (household+PV+batt) kW": non_ev_kw.values,
                "EV baseline kW": base_ev_kw.values,
                "EV planned kW": ev_plan_da.values,
            },
            "Non-EV vs EV (baseline vs planned) (kW)"
        ),
        use_container_width=True,
    )

    # ---- DA Shift Summary ----
    hours_per_step = cfg.freq_min / 60.0
    h = idx_day.hour + idx_day.minute / 60.0
    peak_mask = (h >= 17.0) & (h < 21.0)
    cheap_mask = ((h >= 22.0) & (h < 24.0)) | ((h >= 0.0) & (h < 2.0))
    pre_deadline_mask = idx_day <= deadline_ts

    baseline_peak_kwh = float(base_ev_kw.loc[peak_mask].sum() * hours_per_step)
    plan_peak_kwh = float(ev_plan_da.loc[peak_mask].sum() * hours_per_step)
    shifted_out_of_peak_kwh = max(0.0, baseline_peak_kwh - plan_peak_kwh)
    added_in_cheap_kwh = float((ev_plan_da.loc[cheap_mask] - base_ev_kw.loc[cheap_mask]).clip(lower=0.0).sum() * hours_per_step)
    delivered_by_deadline_kwh = float(ev_plan_da.loc[pre_deadline_mask].sum() * hours_per_step)

    peak_price = float(prices["da_eur_mwh"].loc[peak_mask].mean())
    cheap_price = float(prices["da_eur_mwh"].loc[cheap_mask].mean())
    spread_eur_per_kwh = max(0.0, (peak_price - cheap_price) / 1000.0)
    shift_value_eur = shifted_out_of_peak_kwh * spread_eur_per_kwh

    st.markdown("### DA Shift Summary (EV)")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Baseline EV peak (kWh)", f"{baseline_peak_kwh:.1f}")
    c2.metric("Planned EV peak (kWh)", f"{plan_peak_kwh:.1f}")
    c3.metric("Shifted out of peak (kWh)", f"{shifted_out_of_peak_kwh:.1f}")
    c4.metric("Added in cheap (kWh)", f"{added_in_cheap_kwh:.1f}")
    c5.metric("Shift value (€)", f"{shift_value_eur:.1f}")

    st.caption(f"By deadline: {delivered_by_deadline_kwh:.1f} / {required_kwh:.1f} kWh")

    # ---- DA buy/sell delta ----
    st.markdown("### DA position (buy/sell)")
    da_strategy = st.selectbox("DA strategy", ["Follow forecast", "Conservative (buy extra)", "Leave for intraday"], index=0)
    buy_mult = {"Follow forecast": 1.0, "Conservative (buy extra)": 1.05, "Leave for intraday": 0.92}[da_strategy]

    raw_delta = da_forecast_for_bidding - forward_kw
    da_order = raw_delta.copy()
    da_order[da_order > 0] = da_order[da_order > 0] * buy_mult
    da_kw = da_order  # allow negative sell
    st.session_state.da_kw = da_kw

    contracted = forward_kw + da_kw
    st.plotly_chart(
        compare_positions(idx_day, contracted.values, da_forecast_for_bidding.values, title="Contracted vs DA forecast (after EV scheduling) — buy/sell enabled"),
        use_container_width=True,
    )

    # store EV feasibility objects for ID
    st.session_state.base_ev_kw_day = base_ev_kw
    st.session_state.ev_cap_series_kw_day = ev_cap_series_kw
    st.session_state.ev_required_kwh_deadline = required_kwh
    st.session_state.ev_deadline_ts = deadline_ts


with tabs[5]:
    st.subheader("6) Intraday (ID): trade vs EV scheduling with deadline constraint")
    st.info("This tab unchanged. Use your previous ID logic here if you already had it working.")


with tabs[6]:
    st.subheader("7) Settlement")
    st.info("This tab unchanged. Use your previous settlement logic here if you already had it working.")
