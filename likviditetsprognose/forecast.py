"""Forecasting utilities for liquidity projections."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


@dataclass
class ForecastResult:
    municipality: str
    history: pd.DataFrame
    forecast: pd.DataFrame


class BaselineForecaster:
    """Baseline forecaster using a simple linear regression model.

    The model fits a regression on the logarithm of the cash balance to avoid
    negative predictions and to capture percentage growth. For small or missing
    data the forecaster falls back to a naive last-value projection.
    """

    min_years: int = 3

    def __init__(self, features: Optional[Iterable[str]] = None, horizon: int = 3):
        self.features = list(features) if features else []
        self.horizon = horizon

    def forecast(self, df: pd.DataFrame) -> list[ForecastResult]:
        results: list[ForecastResult] = []
        for municipality, group in df.groupby("kommune"):
            group = group.sort_values("year")
            forecast_df = self._forecast_single(group)
            results.append(ForecastResult(municipality, group, forecast_df))
        return results

    def _forecast_single(self, group: pd.DataFrame) -> pd.DataFrame:
        if group.shape[0] < self.min_years:
            return self._naive_forecast(group)

        X, y = self._prepare_xy(group)
        if X is None or y is None:
            return self._naive_forecast(group)

        model = LinearRegression()
        model.fit(X, y)

        last_year = int(group["year"].max())
        future_years = np.arange(last_year + 1, last_year + self.horizon + 1)
        future_X = self._prepare_future_X(group, future_years)
        y_pred = model.predict(future_X)

        # Reverse the log transform and ensure realistic values.
        cash_pred = np.exp(y_pred)
        forecast_df = pd.DataFrame({
            "kommune": group["kommune"].iloc[0],
            "year": future_years,
            "cash_balance": cash_pred,
        })
        return forecast_df

    def _prepare_xy(self, group: pd.DataFrame) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        group = group.dropna(subset=["cash_balance"]).copy()
        if group.empty:
            return None, None

        y = np.log(group["cash_balance"].to_numpy())
        base_features = [group["year"].to_numpy()]

        for feature in self.features:
            if feature in group.columns:
                base_features.append(group[feature].to_numpy())

        X = np.vstack(base_features).T.astype(float)
        mask = ~np.isnan(X).any(axis=1)
        X = X[mask]
        y = y[mask]
        if X.size == 0:
            return None, None
        return X, y

    def _prepare_future_X(self, group: pd.DataFrame, future_years: np.ndarray) -> np.ndarray:
        feature_matrix = [future_years]
        for feature in self.features:
            if feature not in group.columns:
                continue
            trend = self._extend_trend(group[feature].to_numpy(), len(future_years))
            feature_matrix.append(trend)
        return np.vstack(feature_matrix).T

    @staticmethod
    def _extend_trend(values: np.ndarray, horizon: int) -> np.ndarray:
        values = values.astype(float)
        values = values[~np.isnan(values)]
        if values.size == 0:
            return np.zeros(horizon)
        if values.size == 1:
            return np.repeat(values[-1], horizon)

        diffs = np.diff(values)
        avg_diff = diffs[-3:].mean() if diffs.size else 0.0
        steps = np.arange(1, horizon + 1)
        return values[-1] + avg_diff * steps

    def _naive_forecast(self, group: pd.DataFrame) -> pd.DataFrame:
        last_value = group["cash_balance"].iloc[-1]
        last_year = int(group["year"].max())
        future_years = np.arange(last_year + 1, last_year + self.horizon + 1)
        forecast_df = pd.DataFrame({
            "kommune": group["kommune"].iloc[0],
            "year": future_years,
            "cash_balance": last_value,
        })
        return forecast_df
