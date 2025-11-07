"""Streamlit application for Danish municipal liquidity forecasts."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from likviditetsprognose.data import (
    available_municipalities,
    calculate_aggregates,
    filter_by_municipalities,
    load_dataset,
)
from likviditetsprognose.forecast import BaselineForecaster
from likviditetsprognose.plots import (
    cash_balance_timeseries,
    forecast_plot,
    liquidity_per_capita_timeseries,
)

st.set_page_config(page_title="Likviditetsprognose", layout="wide")


def main() -> None:
    st.title("Likviditetsprognoser for danske kommuner")
    st.markdown(
        """
        Denne applikation kombinerer offentligt tilgængelige regnskabs-, budget- og
        demografidata og beregner en treårig prognose for kommunernes likvide
        beholdninger. Brug sidepanelet til at vælge kommuner og parametrer for
        prognosen.
        """
    )

    dataset = load_dataset()
    if dataset.empty:
        st.error("Der blev ikke fundet nogen data. Kør ETL-scriptet for at hente data.")
        st.stop()

    dataset["year"] = dataset["year"].astype(int)
    min_year, max_year = int(dataset["year"].min()), int(dataset["year"].max())

    with st.sidebar:
        st.header("Filtre")
        selected_municipalities = st.multiselect(
            "Kommuner",
            options=available_municipalities(dataset),
            default=[],
        )
        year_range = st.slider(
            "År",
            min_value=min_year,
            max_value=max_year,
            value=(max(min_year, max_year - 5), max_year),
        )
        horizon = st.slider("Prognosehorisont (år)", 1, 5, 3)
        feature_options = {
            "Befolkning": "population",
            "Nye virksomheder": "new_businesses",
            "Fuldtidsansatte": "employees",
            "Tilskud og udligning": "grants",
            "Budgetbalance": "budget_balance",
        }
        selected_features = st.multiselect("Forklarende variable", options=list(feature_options.keys()))
        features = [feature_options[label] for label in selected_features]

    filtered = filter_by_municipalities(dataset, selected_municipalities)
    filtered = filtered[(filtered["year"] >= year_range[0]) & (filtered["year"] <= year_range[1])]

    st.subheader("Udvikling i likvide beholdninger")
    st.plotly_chart(cash_balance_timeseries(filtered), use_container_width=True)

    aggregates = calculate_aggregates(filtered)
    st.subheader("Likviditet pr. indbygger")
    st.plotly_chart(liquidity_per_capita_timeseries(aggregates), use_container_width=True)

    st.subheader("Prognose pr. kommune")
    if filtered.empty:
        st.info("Vælg mindst én kommune for at se prognoser.")
    else:
        forecaster = BaselineForecaster(features=features, horizon=horizon)
        for result in forecaster.forecast(filtered):
            st.plotly_chart(forecast_plot(result.history, result.forecast), use_container_width=True)

    st.subheader("Dataudtræk")
    st.dataframe(filtered)
    st.download_button(
        "Download filtreret data (CSV)",
        data=_dataframe_to_csv(filtered),
        file_name="likviditetsprognose.csv",
        mime="text/csv",
    )


@st.cache_data(show_spinner=False)
def _dataframe_to_csv(df: pd.DataFrame) -> str:
    return df.to_csv(index=False)


if __name__ == "__main__":
    main()
