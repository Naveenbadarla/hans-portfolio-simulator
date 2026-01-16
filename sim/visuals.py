import plotly.graph_objects as go
import pandas as pd

def line_with_band(df: pd.DataFrame, x=None, y="forecast_kw", low="low_kw", high="high_kw", title=""):
    if x is None:
        x = df.index
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=df[high], line=dict(width=0), name="High", showlegend=False))
    fig.add_trace(go.Scatter(x=x, y=df[low], fill="tonexty", line=dict(width=0), name="Uncertainty", opacity=0.2))
    fig.add_trace(go.Scatter(x=x, y=df[y], name="Forecast"))
    fig.update_layout(title=title, height=380, margin=dict(l=10, r=10, t=35, b=10))
    return fig

def portfolio_components(frame: pd.DataFrame, title="Portfolio components (kW)"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frame.index, y=frame["household_kw"], name="Household"))
    fig.add_trace(go.Scatter(x=frame.index, y=frame["ev_kw"], name="EV"))
    fig.add_trace(go.Scatter(x=frame.index, y=-frame["pv_kw"], name="-PV (reduces net)"))
    fig.add_trace(go.Scatter(x=frame.index, y=frame["battery_kw"], name="Battery"))
    fig.add_trace(go.Scatter(x=frame.index, y=frame["net_kw"], name="Net meter", line=dict(width=3)))
    fig.update_layout(title=title, height=420, margin=dict(l=10, r=10, t=35, b=10))
    return fig

def compare_positions(idx, contracted, actual, title="Contracted vs Actual (kW)", flex_envelope=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=idx, y=contracted, name="Contracted", line=dict(width=3)))
    fig.add_trace(go.Scatter(x=idx, y=actual, name="Actual (after flex)", mode="lines", line=dict(width=2)))
    if flex_envelope is not None:
        low, high = flex_envelope
        fig.add_trace(go.Scatter(x=idx, y=high, line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=idx, y=low, fill="tonexty", line=dict(width=0), name="Flex envelope", opacity=0.15))
    fig.update_layout(title=title, height=420, margin=dict(l=10, r=10, t=35, b=10))
    return fig

def price_chart(prices: pd.DataFrame, title="Prices (€/MWh)"):
    fig = go.Figure()
    for c in prices.columns:
        fig.add_trace(go.Scatter(x=prices.index, y=prices[c], name=c))
    fig.update_layout(title=title, height=300, margin=dict(l=10, r=10, t=35, b=10))
    return fig
