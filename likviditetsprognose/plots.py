"""Plotting helpers using Plotly."""
from __future__ import annotations

import pandas as pd
import plotly.express as px


def cash_balance_timeseries(df: pd.DataFrame) -> px.line:
    title = "Likvid beholdning over tid"
    fig = px.line(
        df,
        x="year",
        y="cash_balance",
        color="kommune",
        title=title,
        markers=True,
    )
    fig.update_layout(xaxis_title="År", yaxis_title="Likvid beholdning (1.000 kr.)")
    return fig


def liquidity_per_capita_timeseries(df: pd.DataFrame) -> px.line:
    title = "Likviditet pr. indbygger"
    fig = px.line(
        df,
        x="year",
        y="liquidity_per_capita",
        title=title,
        markers=True,
    )
    fig.update_layout(xaxis_title="År", yaxis_title="Kr. pr. indbygger")
    return fig


def forecast_plot(history: pd.DataFrame, forecast: pd.DataFrame) -> px.line:
    history = history.copy()
    history["type"] = "Historik"
    forecast = forecast.copy()
    forecast["type"] = "Prognose"
    combined = pd.concat([history, forecast], ignore_index=True)
    fig = px.line(
        combined,
        x="year",
        y="cash_balance",
        color="type",
        markers=True,
        title=f"Likviditetsprognose - {history['kommune'].iloc[0] if not history.empty else ''}",
    )
    fig.update_layout(xaxis_title="År", yaxis_title="Likvid beholdning (1.000 kr.)")
    return fig
