import streamlit as st
import numpy as np
import pandas as pd

from sim.config import SimConfig
from sim.portfolio import build_portfolio
from sim.forecasting import day_ahead_forecast, intraday_nowcast, typical_day_profile
from sim.markets import make_price_curves, build_layered_hedge_curve_eon_style_seasonal
from sim.intraday import IntradayState, step_intraday
from sim.settlement import compute_settlement
from sim.visuals import portfolio_components, line_with_band, compare_positions, price_chart

st.set_page_config(page_title="Hans Portfolio Walk-Forward Simulator", layout="wide")


def regime_mult(name: str, cfg: SimConfig) -> float:
    return {
        "Normal": cfg.regime_normal,
        "Volatile": cfg.regime_volatile,
        "Crisis": cfg.regime_crisis,
    }.get(name, 1.0)


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
    if "portfolio_df" not in st.session_state:
        st.session_state.portfolio_df = None
    if "selected_day" not in st.session_state:
        st.session_state.selected_day = None
    if "forward_kw" not in st.session_state:
        st.session_state.forward_kw = None
    if "da_kw" not in st.session_state:
        st.session_state.da_kw = None
    if "da_forecast" not in st.session_state:
        st.session_state.da_forecast = None
    if "prices" not in st.session_state:
        st.session_state.prices = None
    if "id_state" not in st.session_state:
        st.session_state.id_state = None
    if "actual_day_kw" not in st.session_state:
        st.session_state.actual_day_kw = None
    if "weather_regime" not in st.session_state:
        st.session_state.weather_regime = "Variable"
    if "price_regime" not in st.session_state:
        st.session_state.price_regime = "Normal"


init_state()
cfg: SimConfig = st.session_state.cfg

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
    st.caption("Charger mix (fixed in MVP): 3.7/7.4/11 kW")

    st.subheader("World")
    st.session_state.weather_regime = st.selectbox("Weather regime", ["Sunny", "Cloudy", "Variable"], index=2)
    st.session_state.price_regime = st.selectbox("Price regime", ["Normal", "Volatile", "Crisis"], index=0)

    seed = st.number_input("Random seed", min_value=1, max_value=999999, value=7, step=1)

    end_date = st.date_input("History end date", value=pd.Timestamp("2026-01-16").date())
    if st.button("Generate / Regenerate Portfolio", type="primary"):
        cfg_dict = cfg.__dict__.copy()
        st.session_state.portfolio_df = cached_portfolio(
            cfg_dict, str(end_date), st.session_state.weather_regime, int(seed)
        )
        st.session_state.selected_day = None
        st.session_state.forward_kw = None
        st.session_state.da_kw = None
        st.session_state.da_forecast = None
        st.session_state.prices = None
        st.session_state.id_state = None
        st.session_state.actual_day_kw = None

df = st.session_state.portfolio_df
if df is None:
    st.info("Use the sidebar to generate a synthetic portfolio first.")
    st.stop()

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

tz = cfg.tz
df = df.copy()
if df.index.tz is None:
    df.index = df.index.tz_localize(tz)

available_days = pd.Index(pd.to_datetime(pd.Series(df.index.date).unique())).sort_values()
default_day = available_days[-2] if len(available_days) >= 2 else available_days[-1]


def get_day_slice(day_ts: pd.Timestamp) -> pd.DataFrame:
    day_ts = day_ts.tz_localize(tz) if day_ts.tzinfo is None else day_ts.tz_convert(tz)
    start = day_ts.normalize()
    end = start + pd.Timedelta(days=1)
    return df.loc[(df.index >= start) & (df.index < end)]


with tabs[0]:
    st.subheader("1) Meter history (portfolio net meter)")
    col1, col2 = st.columns([2, 1])
    with col2:
        chosen_day = st.selectbox(
            "Choose a day to simulate forward from", available_days, index=len(available_days) - 2
        )
        st.session_state.selected_day = pd.Timestamp(chosen_day).tz_localize(tz)

    last7 = df.tail(7 * 24 * 4)
    st.plotly_chart(
        portfolio_components(last7, title="Last 7 days: components and net meter (kW)"),
        use_container_width=True,
    )

    net = pd.Series(df["net_kw"].values, index=df.index)
    prof = typical_day_profile(net, tz=tz)
    st.caption("Typical day profile (P10/P50/P90) from history")
    fig = line_with_band(
        prof.reset_index().assign(time=prof.index / 4.0),
        x=prof.index,
        y="p50",
        low="p10",
        high="p90",
        title="Typical day (96 points) — net meter kW percentiles",
    )
    st.plotly_chart(fig, use_container_width=True)


with tabs[1]:
    st.subheader("2) Profiling & disaggregation (intuitive portfolio decomposition)")
    day_df = get_day_slice(st.session_state.selected_day)
    st.plotly_chart(
        portfolio_components(day_df, title=f"Selected day components (kW): {st.session_state.selected_day.date()}"),
        use_container_width=True,
    )

    st.markdown("**Interpretation:**")
    st.write("- Household drives morning/evening shape.")
    st.write("- PV reduces net demand midday (can create export-like behavior if large).")
    st.write("- EV adds evening/night blocks (portfolio-level smoothing).")
    st.write("- Battery shifts within-day energy (very simplified heuristic in MVP).")


with tabs[2]:
    st.subheader("3) Forecasting (LT summary + DA 96-point + ID nowcast)")
    colA, colB = st.columns([1, 1])

    with colA:
        st.markdown("### Day-ahead forecast knobs")
        pv_bias = st.slider("PV forecast bias (expect more PV → lower net)", -0.5, 0.5, 0.0, step=0.05)
        ev_shift = st.slider("EV timing shift (quarter-hours)", -16, 16, 0, step=1)
        noise = st.slider("Forecast noise", 0.0, 0.10, 0.03, step=0.01)

    target_day = st.session_state.selected_day + pd.Timedelta(days=1)
    da_f = day_ahead_forecast(
        df, tz=tz, target_date=target_day, pv_forecast_bias=pv_bias, ev_timing_shift_qh=ev_shift, noise_scale=noise
    )
    st.session_state.da_forecast = da_f

    st.plotly_chart(line_with_band(da_f, title=f"DA forecast for {target_day.date()} (96×15-min)"), use_container_width=True)

    actual_day = get_day_slice(target_day)
    if len(actual_day) == 96:
        st.session_state.actual_day_kw = pd.Series(actual_day["net_kw"].values, index=actual_day.index)


with tabs[3]:
    st.subheader("4) Forwards hedging (E.ON-style layered + seasonal)")
    if st.session_state.da_forecast is None:
        st.warning("Go to Forecasting tab first to generate the DA forecast.")
        st.stop()

    da_curve = st.session_state.da_forecast["forecast_kw"]

    hedge_ratio = st.slider("Total hedge ratio (energy volume)", 0.0, 1.0, 0.85, step=0.05)

    st.markdown("### Desk-style hedge shaping")
    base_share = st.slider("Baseload share (before season/day adjustment)", 0.40, 0.95, 0.70, step=0.05)
    peak_share = 1.0 - float(base_share)

    c1, c2, c3 = st.columns(3)
    with c1:
        forecast_conf = st.slider("Forecast confidence", 0.0, 1.0, 0.70, step=0.05)
    with c2:
        ev_flex = st.slider("EV flexibility available", 0.0, 1.0, 0.60, step=0.05)
    with c3:
        pv_unc = st.slider("PV uncertainty", 0.0, 1.0, 0.50, step=0.05)

    peak_strength = st.slider("Peak shaping strength", 0.0, 1.0, 1.00, step=0.05)

    st.markdown("### Season & weekend logic")
    e1, e2 = st.columns(2)
    with e1:
        enable_seasonality = st.toggle("Enable seasonality (winter more peak)", value=True)
    with e2:
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

    st.write(f"DA expected daily energy (kWh): **{da_curve.sum() * 0.25:.1f}**")
    st.write(f"Forward hedged energy (kWh): **{forward_kw.sum() * 0.25:.1f}**")
    st.write(f"Residual for DA/ID (kWh): **{(da_curve - forward_kw).clip(lower=0).sum() * 0.25:.1f}**")

    fig = compare_positions(
        idx=da_curve.index,
        contracted=forward_kw.values,
        actual=da_curve.values,
        title="E.ON-style layered forward hedge vs DA forecast (kW)",
    )
    st.plotly_chart(fig, use_container_width=True)


with tabs[4]:
    st.subheader("5) Day-ahead (DA) — shaping the schedule")
    if st.session_state.forward_kw is None or st.session_state.da_forecast is None:
        st.warning("Generate forecast and forward hedge first.")
        st.stop()

    da_strategy = st.selectbox("DA strategy", ["Follow forecast", "Conservative (buy extra)", "Leave for intraday"], index=0)
    extra = {"Follow forecast": 0.0, "Conservative (buy extra)": 0.05, "Leave for intraday": -0.08}[da_strategy]

    da_forecast_kw = st.session_state.da_forecast["forecast_kw"]
    forward_kw = st.session_state.forward_kw

    da_needed = (da_forecast_kw - forward_kw).clip(lower=0.0) * (1.0 + extra)
    da_kw = da_needed.clip(lower=0.0)

    st.session_state.da_kw = da_kw

    contracted = forward_kw + da_kw
    delta = da_forecast_kw - contracted

    st.write(f"Contracted energy after DA (kWh): **{contracted.sum() * 0.25:.1f}**")
    st.write(f"Expected imbalance if no intraday (kWh): **{delta.sum() * 0.25:.1f}**")

    st.plotly_chart(
        compare_positions(da_forecast_kw.index, contracted.values, da_forecast_kw.values, title="DA contracted vs forecast (kW)"),
        use_container_width=True,
    )

    prices = make_price_curves(
        da_forecast_kw.index,
        cfg.base_da_price,
        cfg.base_id_price,
        cfg.base_imb_price,
        regime_mult=regime_mult(st.session_state.price_regime, cfg),
        seed=42,
    )
    st.session_state.prices = prices
    st.plotly_chart(price_chart(prices, title="DA / ID / Imbalance price curves (synthetic)"), use_container_width=True)


with tabs[5]:
    st.subheader("6) Intraday (ID) — playable game loop (trade vs flexibility)")
    if st.session_state.da_kw is None or st.session_state.prices is None:
        st.warning("Run Day-ahead tab first.")
        st.stop()

    da_forecast = st.session_state.da_forecast.copy()
    prices = st.session_state.prices.copy()
    forward_kw = st.session_state.forward_kw
    da_kw = st.session_state.da_kw

    idx = da_forecast.index

    if st.session_state.actual_day_kw is not None and len(st.session_state.actual_day_kw) == 96:
        actual = st.session_state.actual_day_kw.reindex(idx).fillna(da_forecast["forecast_kw"])
    else:
        rng = np.random.default_rng(1234)
        actual = pd.Series(np.clip(da_forecast["forecast_kw"].values * (1.0 + rng.normal(0, 0.05, size=96)), 0, None), index=idx)

    if st.session_state.id_state is None or st.session_state.id_state.idx[0] != idx[0]:
        st.session_state.id_state = IntradayState(idx=idx)
    id_state: IntradayState = st.session_state.id_state

    colL, colR = st.columns([1, 2])
    with colL:
        now_i = st.slider("Simulation time (15-min step)", 0, 95, 32, step=1)
        now = idx[now_i]
        st.caption(f"Now: {now.strftime('%Y-%m-%d %H:%M')}")

        st.markdown("### Intraday actions")
        trade_kwh = st.number_input("Trade (kWh): +buy / -sell", value=0.0, step=50.0)
        delivery_h = st.selectbox("Trade delivery window (hours)", [1, 2, 4, 6], index=1)
        flex_strength = st.slider("Flex dispatch strength (0..1)", 0.0, 1.0, 0.4, step=0.05)

        if st.button("Apply step decisions"):
            actual_so_far = actual.iloc[: now_i + 1]
            nowcast_df = intraday_nowcast(da_forecast, actual_so_far=actual_so_far, now=now, tz=tz)

            step_intraday(
                state=id_state,
                now=now,
                nowcast_kw=nowcast_df["nowcast_kw"],
                contracted_kw=(forward_kw + da_kw),
                do_trade_kwh=float(trade_kwh),
                trade_delivery_hours=int(delivery_h),
                flex_strength=float(flex_strength),
                freq_min=cfg.freq_min,
            )
            st.success("Applied trade + flex for the selected step.")

    actual_so_far = actual.iloc[: now_i + 1]
    nowcast_df = intraday_nowcast(da_forecast, actual_so_far=actual_so_far, now=now, tz=tz)

    contracted = (forward_kw + da_kw + id_state.trades_kw).reindex(idx)
    actual_after_flex = (actual + id_state.flex_shift_kw).clip(lower=0.0)

    max_shift = 600.0
    low_env = (nowcast_df["nowcast_kw"] - max_shift).clip(lower=0.0)
    high_env = nowcast_df["nowcast_kw"] + max_shift

    with colR:
        st.plotly_chart(
            compare_positions(
                idx, contracted.values, actual_after_flex.values,
                title="Contracted vs Actual (after flex) — intraday view",
                flex_envelope=(low_env.values, high_env.values),
            ),
            use_container_width=True,
        )

        delta_kw = actual_after_flex - contracted
        st.write(f"Current (interval) delta at {now.strftime('%H:%M')}: **{delta_kw.loc[now]:.1f} kW**")
        st.write(f"Cumulative imbalance energy (kWh) so far: **{(delta_kw.iloc[:now_i+1].sum() * 0.25):.1f}**")


with tabs[6]:
    st.subheader("7) Settlement & learning")
    if st.session_state.id_state is None or st.session_state.da_kw is None or st.session_state.prices is None:
        st.warning("Run through Intraday tab first (at least once).")
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

    st.metric(
        "Total cost (€)",
        f"{result['total_cost_eur']:.0f}",
        delta=f"{(baseline['total_cost_eur'] - result['total_cost_eur']):.0f} vs no-ID/no-flex",
    )

    st.plotly_chart(
        compare_positions(
            idx=idx,
            contracted=result["contracted_kw"].values,
            actual=result["actual_after_flex_kw"].values,
            title="Final contracted vs actual after flex (kW)",
        ),
        use_container_width=True,
    )

    st.plotly_chart(
        compare_positions(
            idx=idx,
            contracted=pd.Series(0.0, index=idx).values,
            actual=result["imbalance_kw"].values,
            title="Imbalance profile (kW)",
        ),
        use_container_width=True,
    )

    st.markdown("### Learning (MVP)")
    st.write("- Change hedge ratio, base share, season/weekend toggles, DA strategy, and intraday actions.")
    st.write("- Compare total cost vs the no-ID/no-flex baseline to see flexibility value.")
