"""Data loading and preparation utilities for the liquidity forecast app."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PROCESSED_FILE = DATA_DIR / "liquidity_dataset.parquet"


@dataclass(frozen=True)
class DataSourceConfig:
    """Configuration for locating processed ETL output."""

    processed_file: Path = PROCESSED_FILE


def load_dataset(config: Optional[DataSourceConfig] = None) -> pd.DataFrame:
    """Load the processed dataset produced by the ETL script.

    If the processed dataset is not available, a small synthetic example dataset is
    generated so the application remains usable in development environments.
    """

    cfg = config or DataSourceConfig()
    if cfg.processed_file.exists():
        df = _load_from_parquet(cfg.processed_file)
        return _ensure_municipality_column(df)

    return _load_example_dataset()


def _load_from_parquet(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if not {"kommunekode", "kommune", "year", "cash_balance"}.issubset(df.columns):
        raise ValueError(
            "Processed dataset is missing one or more required columns: "
            "kommunekode, kommune, year, cash_balance"
        )
    return _ensure_municipality_column(df)


def _load_example_dataset() -> pd.DataFrame:
    """Return a deterministic example dataset covering a handful of municipalities."""

    # The numbers below are illustrative and loosely based on published public data.
    records = [
        {"kommunekode": "0101", "kommune": "København", "year": 2018, "cash_balance": 8100000,
         "net_operating_result": 230000, "capital_expenditure": 120000,
         "population": 624000, "new_businesses": 4300, "employees": 310000,
         "grants": 280000, "liquidity_per_capita": 12.98},
        {"kommunekode": "0101", "kommune": "København", "year": 2019, "cash_balance": 8350000,
         "net_operating_result": 260000, "capital_expenditure": 125000,
         "population": 632000, "new_businesses": 4450, "employees": 315500,
         "grants": 282000, "liquidity_per_capita": 13.21},
        {"kommunekode": "0101", "kommune": "København", "year": 2020, "cash_balance": 8520000,
         "net_operating_result": 255000, "capital_expenditure": 130000,
         "population": 640000, "new_businesses": 4100, "employees": 312000,
         "grants": 284000, "liquidity_per_capita": 13.31},
        {"kommunekode": "0101", "kommune": "København", "year": 2021, "cash_balance": 8700000,
         "net_operating_result": 270000, "capital_expenditure": 128000,
         "population": 650000, "new_businesses": 4200, "employees": 318000,
         "grants": 285000, "liquidity_per_capita": 13.38},
        {"kommunekode": "0101", "kommune": "København", "year": 2022, "cash_balance": 8950000,
         "net_operating_result": 280000, "capital_expenditure": 135000,
         "population": 660000, "new_businesses": 4500, "employees": 320000,
         "grants": 288000, "liquidity_per_capita": 13.56},
        {"kommunekode": "0400", "kommune": "Frederiksberg", "year": 2018, "cash_balance": 1650000,
         "net_operating_result": 52000, "capital_expenditure": 35000,
         "population": 104000, "new_businesses": 760, "employees": 62000,
         "grants": 34000, "liquidity_per_capita": 15.87},
        {"kommunekode": "0400", "kommune": "Frederiksberg", "year": 2019, "cash_balance": 1680000,
         "net_operating_result": 55000, "capital_expenditure": 35500,
         "population": 105000, "new_businesses": 780, "employees": 62300,
         "grants": 34500, "liquidity_per_capita": 16.00},
        {"kommunekode": "0400", "kommune": "Frederiksberg", "year": 2020, "cash_balance": 1705000,
         "net_operating_result": 54000, "capital_expenditure": 36000,
         "population": 106000, "new_businesses": 750, "employees": 61900,
         "grants": 34800, "liquidity_per_capita": 16.09},
        {"kommunekode": "0400", "kommune": "Frederiksberg", "year": 2021, "cash_balance": 1730000,
         "net_operating_result": 56000, "capital_expenditure": 36200,
         "population": 107000, "new_businesses": 770, "employees": 62200,
         "grants": 35000, "liquidity_per_capita": 16.17},
        {"kommunekode": "0400", "kommune": "Frederiksberg", "year": 2022, "cash_balance": 1760000,
         "net_operating_result": 57500, "capital_expenditure": 36800,
         "population": 108000, "new_businesses": 790, "employees": 62500,
         "grants": 35500, "liquidity_per_capita": 16.30},
        {"kommunekode": "0760", "kommune": "Aarhus", "year": 2018, "cash_balance": 4350000,
         "net_operating_result": 125000, "capital_expenditure": 83000,
         "population": 341000, "new_businesses": 2300, "employees": 210000,
         "grants": 150000, "liquidity_per_capita": 12.76},
        {"kommunekode": "0760", "kommune": "Aarhus", "year": 2019, "cash_balance": 4400000,
         "net_operating_result": 132000, "capital_expenditure": 84500,
         "population": 345000, "new_businesses": 2350, "employees": 213000,
         "grants": 151500, "liquidity_per_capita": 12.75},
        {"kommunekode": "0760", "kommune": "Aarhus", "year": 2020, "cash_balance": 4475000,
         "net_operating_result": 128000, "capital_expenditure": 86000,
         "population": 349000, "new_businesses": 2200, "employees": 211500,
         "grants": 152500, "liquidity_per_capita": 12.82},
        {"kommunekode": "0760", "kommune": "Aarhus", "year": 2021, "cash_balance": 4550000,
         "net_operating_result": 134000, "capital_expenditure": 87000,
         "population": 353000, "new_businesses": 2260, "employees": 214000,
         "grants": 154000, "liquidity_per_capita": 12.89},
        {"kommunekode": "0760", "kommune": "Aarhus", "year": 2022, "cash_balance": 4620000,
         "net_operating_result": 138000, "capital_expenditure": 88000,
         "population": 358000, "new_businesses": 2320, "employees": 217000,
         "grants": 155500, "liquidity_per_capita": 12.91},
    ]
    df = pd.DataFrame.from_records(records)
    df["year"] = df["year"].astype(int)
    df.sort_values(["kommune", "year"], inplace=True)
    return df


def available_municipalities(df: pd.DataFrame) -> Iterable[str]:
    """Return a sorted iterable of municipality names present in the dataset."""

    column = "kommune" if "kommune" in df.columns else "kommunekode"
    return sorted(df[column].astype(str).unique())


def _ensure_municipality_column(df: pd.DataFrame) -> pd.DataFrame:
    if "kommune" in df.columns:
        return df

    df = df.copy()
    df["kommune"] = "Kommunekode " + df["kommunekode"].astype(str)
    return df


def filter_by_municipalities(df: pd.DataFrame, municipalities: Iterable[str]) -> pd.DataFrame:
    municipalities = list(municipalities)
    if not municipalities:
        return df
    return df[df["kommune"].isin(municipalities)].copy()


def calculate_aggregates(df: pd.DataFrame, group_by: Iterable[str] = ("year",)) -> pd.DataFrame:
    """Aggregate metrics such as cash balance and per capita liquidity."""

    group_cols = list(group_by)
    aggregated = (
        df.groupby(group_cols)
        .agg(
            cash_balance=("cash_balance", "sum"),
            net_operating_result=("net_operating_result", "sum"),
            capital_expenditure=("capital_expenditure", "sum"),
            grants=("grants", "sum"),
            population=("population", "sum"),
        )
        .reset_index()
    )
    aggregated["liquidity_per_capita"] = aggregated["cash_balance"] / aggregated["population"].replace(0, pd.NA)
    return aggregated
