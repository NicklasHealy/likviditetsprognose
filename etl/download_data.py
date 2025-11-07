"""Download and transform public datasets for the liquidity forecast app."""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_FILE = DATA_DIR / "liquidity_dataset.parquet"

STATBANK_BASE = "https://api.statbank.dk/v1/data"

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class TableConfig:
    table: str
    filters: Dict[str, Iterable[str]]
    filename: str


TABLES: tuple[TableConfig, ...] = (
    TableConfig(
        table="REGK4",
        filters={
            "KOMKODE": ["*"],
            "TID": ["2007-2024"],
        },
        filename="regk4.csv",
    ),
    TableConfig(
        table="REGK31",
        filters={
            "KOMKODE": ["*"],
            "TID": ["2007-2024"],
        },
        filename="regk31.csv",
    ),
    TableConfig(
        table="BUDK32",
        filters={
            "KOMKODE": ["*"],
            "TID": ["2007-2024"],
        },
        filename="budk32.csv",
    ),
    TableConfig(
        table="FOLK1A",
        filters={
            "OMRÅDE": ["*"],
            "ALDER": ["IALT"],
            "KØN": ["TOT"],
            "TID": ["2007-2024"],
        },
        filename="folk1a.csv",
    ),
    TableConfig(
        table="DEMO19",
        filters={
            "OMRÅDE": ["*"],
            "TID": ["2007-2024"],
        },
        filename="demo19.csv",
    ),
    TableConfig(
        table="EJDSK1",
        filters={
            "OMRÅDE": ["*"],
            "TID": ["2007-2024"],
        },
        filename="ejdsk1.csv",
    ),
    TableConfig(
        table="PSKAT",
        filters={
            "OMRÅDE": ["*"],
            "TID": ["2007-2024"],
        },
        filename="pskat.csv",
    ),
)


def download_table(config: TableConfig, target_dir: Path, overwrite: bool = False) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    output_path = target_dir / config.filename
    if output_path.exists() and not overwrite:
        logger.info("Skipping %s (already exists)", config.table)
        return output_path

    url = f"{STATBANK_BASE}/{config.table}/CSV"
    params = {key: ",".join(value) for key, value in config.filters.items()}
    logger.info("Downloading %s", url)
    response = requests.get(url, params=params, timeout=120)
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        logger.error("Failed to download %s: %s", config.table, exc)
        raise

    output_path.write_bytes(response.content)
    logger.info("Saved %s", output_path)
    return output_path


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    df.columns = [col.strip().lower() for col in df.columns]
    return df


def build_processed_dataset(raw_dir: Path, grants_file: Optional[Path] = None) -> pd.DataFrame:
    """Combine raw tables into a single modelling dataset."""

    regk4 = load_csv(raw_dir / "regk4.csv") if (raw_dir / "regk4.csv").exists() else pd.DataFrame()
    regk31 = load_csv(raw_dir / "regk31.csv") if (raw_dir / "regk31.csv").exists() else pd.DataFrame()
    budk32 = load_csv(raw_dir / "budk32.csv") if (raw_dir / "budk32.csv").exists() else pd.DataFrame()
    folk1a = load_csv(raw_dir / "folk1a.csv") if (raw_dir / "folk1a.csv").exists() else pd.DataFrame()
    demo19 = load_csv(raw_dir / "demo19.csv") if (raw_dir / "demo19.csv").exists() else pd.DataFrame()
    ejdsk1 = load_csv(raw_dir / "ejdsk1.csv") if (raw_dir / "ejdsk1.csv").exists() else pd.DataFrame()
    pskat = load_csv(raw_dir / "pskat.csv") if (raw_dir / "pskat.csv").exists() else pd.DataFrame()
    grants = load_grants(grants_file) if grants_file and grants_file.exists() else pd.DataFrame()

    dataset = prepare_cash_balance(regk4)
    dataset = dataset.merge(prepare_regk31(regk31), on=["kommunekode", "year"], how="left")
    dataset = dataset.merge(prepare_budget(budk32), on=["kommunekode", "year"], how="left")
    dataset = dataset.merge(prepare_population(folk1a), on=["kommunekode", "year"], how="left")
    dataset = dataset.merge(prepare_business_metrics(demo19), on=["kommunekode", "year"], how="left")
    dataset = dataset.merge(prepare_property_tax(ejdsk1), on=["kommunekode", "year"], how="left")
    dataset = dataset.merge(prepare_tax_rate(pskat), on=["kommunekode", "year"], how="left")
    dataset = dataset.merge(grants, on=["kommunekode", "year"], how="left")

    dataset["liquidity_per_capita"] = dataset["cash_balance"] / dataset["population"].replace(0, pd.NA)
    for column in [
        "net_operating_result",
        "capital_expenditure",
        "budget_balance",
        "property_tax",
        "tax_rate",
        "new_businesses",
        "employees",
        "population",
        "grants",
    ]:
        if column not in dataset.columns:
            dataset[column] = pd.NA
    dataset.sort_values(["kommunekode", "year"], inplace=True)
    return dataset


def prepare_cash_balance(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        logger.warning("REGK4 dataset is empty; returning synthetic placeholder")
        return pd.DataFrame(
            {
                "kommunekode": [],
                "year": [],
                "cash_balance": [],
            }
        )

    code_col = resolve_column(df, ["komkode", "kommunekode", "område"])
    time_col = resolve_column(df, ["tid", "år", "time"])
    value_col = resolve_column(df, ["indhold", "value", "beløb"])

    df = df[[code_col, time_col, value_col]].copy()
    df.rename(columns={code_col: "kommunekode", time_col: "year", value_col: "cash_balance"}, inplace=True)
    df["kommunekode"] = df["kommunekode"].astype(str).str.strip()
    df["cash_balance"] = pd.to_numeric(df["cash_balance"].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    df["year"] = df["year"].astype(str).str.extract(r"(\d{4})").astype(int)
    aggregated = df.groupby(["kommunekode", "year"], as_index=False).agg({"cash_balance": "sum"})
    return aggregated


def prepare_population(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        logger.warning("FOLK1A dataset is empty")
        return pd.DataFrame({"kommunekode": [], "year": [], "population": []})

    code_col = resolve_column(df, ["område", "komkode", "kommunekode"])
    time_col = resolve_column(df, ["tid", "år", "time"])
    value_col = resolve_column(df, ["indhold", "value", "antal"])

    df = df[[code_col, time_col, value_col]].copy()
    df.rename(columns={code_col: "kommunekode", time_col: "year", value_col: "population"}, inplace=True)
    df["kommunekode"] = df["kommunekode"].astype(str).str.strip()
    df["population"] = pd.to_numeric(df["population"].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    df["year"] = df["year"].astype(str).str.extract(r"(\d{4})").astype(int)
    return df.groupby(["kommunekode", "year"], as_index=False).agg({"population": "sum"})


def prepare_business_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        logger.warning("DEMO19 dataset is empty")
        return pd.DataFrame({"kommunekode": [], "year": [], "new_businesses": [], "employees": []})

    code_col = resolve_column(df, ["område", "komkode", "kommunekode"])
    time_col = resolve_column(df, ["tid", "år", "time"])
    value_col = resolve_column(df, ["indhold", "value"])

    # The DEMO19 table contains multiple indicators; identify them via a column containing the indicator id.
    indicator_col = resolve_column(df, ["erhverv", "enhed", "indholdstype", "erhvervsindikator"], optional=True)

    df = df.copy()
    df.rename(columns={code_col: "kommunekode", time_col: "year", value_col: "value"}, inplace=True)
    df["kommunekode"] = df["kommunekode"].astype(str).str.strip()
    df["value"] = pd.to_numeric(df["value"].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    df["year"] = df["year"].astype(str).str.extract(r"(\d{4})").astype(int)

    if indicator_col:
        indicator_map = {
            "Nystartede virksomheder": "new_businesses",
            "Nye virksomheder": "new_businesses",
            "Fuldtidsbeskæftigede": "employees",
            "Fuldtidsansatte": "employees",
        }
        df[indicator_col] = df[indicator_col].astype(str)
        df["metric"] = df[indicator_col].map(indicator_map)
        df = df.dropna(subset=["metric"])
    else:
        df["metric"] = "new_businesses"

    pivot = df.pivot_table(
        index=["kommunekode", "year"],
        columns="metric",
        values="value",
        aggfunc="sum",
    ).reset_index()
    pivot.columns.name = None
    return pivot


def prepare_regk31(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        logger.warning("REGK31 dataset is empty")
        return pd.DataFrame({"kommunekode": [], "year": [], "net_operating_result": [], "capital_expenditure": []})

    code_col = resolve_column(df, ["komkode", "kommunekode", "område"])
    time_col = resolve_column(df, ["tid", "år", "time"])
    value_col = resolve_column(df, ["indhold", "value", "beløb"])
    dranst_col = resolve_column(df, ["dranst", "art", "funktionstype"], optional=True)

    df = df.copy()
    df.rename(columns={code_col: "kommunekode", time_col: "year", value_col: "value"}, inplace=True)
    df["year"] = df["year"].astype(str).str.extract(r"(\d{4})").astype(int)

    if dranst_col:
        df[dranst_col] = df[dranst_col].astype(str)
        operating_mask = df[dranst_col].str.startswith("2") | df[dranst_col].str.contains("drift", case=False, na=False)
        capital_mask = df[dranst_col].str.startswith("4") | df[dranst_col].str.contains("anlæg", case=False, na=False)
        operating = (
            df[operating_mask]
            .groupby(["kommunekode", "year"], as_index=False)["value"]
            .sum()
            .rename(columns={"value": "net_operating_result"})
        )
        capital = (
            df[capital_mask]
            .groupby(["kommunekode", "year"], as_index=False)["value"]
            .sum()
            .rename(columns={"value": "capital_expenditure"})
        )
        merged = operating.merge(capital, on=["kommunekode", "year"], how="outer")
    else:
        aggregated = (
            df.groupby(["kommunekode", "year"], as_index=False)["value"].sum()
            .rename(columns={"value": "net_operating_result"})
        )
        aggregated["capital_expenditure"] = pd.NA
        merged = aggregated

    return merged


def prepare_budget(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        logger.warning("BUDK32 dataset is empty")
        return pd.DataFrame({"kommunekode": [], "year": [], "budget_balance": []})

    code_col = resolve_column(df, ["komkode", "kommunekode", "område"])
    time_col = resolve_column(df, ["tid", "år", "time"])
    value_col = resolve_column(df, ["indhold", "value", "beløb"])

    df = df[[code_col, time_col, value_col]].copy()
    df.rename(columns={code_col: "kommunekode", time_col: "year", value_col: "budget_balance"}, inplace=True)
    df["kommunekode"] = df["kommunekode"].astype(str).str.strip()
    df["budget_balance"] = pd.to_numeric(df["budget_balance"].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    df["year"] = df["year"].astype(str).str.extract(r"(\d{4})").astype(int)
    return df.groupby(["kommunekode", "year"], as_index=False)["budget_balance"].sum()


def prepare_property_tax(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        logger.warning("EJDSK1 dataset is empty")
        return pd.DataFrame({"kommunekode": [], "year": [], "property_tax": []})

    code_col = resolve_column(df, ["område", "komkode", "kommunekode"])
    time_col = resolve_column(df, ["tid", "år", "time"])
    value_col = resolve_column(df, ["indhold", "value", "beløb"])

    df = df[[code_col, time_col, value_col]].copy()
    df.rename(columns={code_col: "kommunekode", time_col: "year", value_col: "property_tax"}, inplace=True)
    df["kommunekode"] = df["kommunekode"].astype(str).str.strip()
    df["property_tax"] = pd.to_numeric(df["property_tax"].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    df["year"] = df["year"].astype(str).str.extract(r"(\d{4})").astype(int)
    return df.groupby(["kommunekode", "year"], as_index=False)["property_tax"].sum()


def prepare_tax_rate(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        logger.warning("PSKAT dataset is empty")
        return pd.DataFrame({"kommunekode": [], "year": [], "tax_rate": []})

    code_col = resolve_column(df, ["område", "komkode", "kommunekode"])
    time_col = resolve_column(df, ["tid", "år", "time"])
    value_col = resolve_column(df, ["indhold", "value", "pct", "procent"])

    df = df[[code_col, time_col, value_col]].copy()
    df.rename(columns={code_col: "kommunekode", time_col: "year", value_col: "tax_rate"}, inplace=True)
    df["kommunekode"] = df["kommunekode"].astype(str).str.strip()
    df["tax_rate"] = pd.to_numeric(df["tax_rate"].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    df["year"] = df["year"].astype(str).str.extract(r"(\d{4})").astype(int)
    return df.groupby(["kommunekode", "year"], as_index=False)["tax_rate"].mean()


def load_grants(path: Path) -> pd.DataFrame:
    logger.info("Loading grants from %s", path)
    df = pd.read_excel(path)
    df.columns = [col.strip().lower() for col in df.columns]
    code_col = resolve_column(df, ["kommunekode", "komkode", "kode"])
    year_col = resolve_column(df, ["år", "year"])
    value_col = resolve_column(df, ["tilskud", "bloktilskud", "beløb", "amount"])

    df = df[[code_col, year_col, value_col]].copy()
    df.rename(columns={code_col: "kommunekode", year_col: "year", value_col: "grants"}, inplace=True)
    df["year"] = df["year"].astype(int)
    return df


def resolve_column(df: pd.DataFrame, candidates: Iterable[str], optional: bool = False) -> Optional[str]:
    lower_cols = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate in lower_cols:
            return lower_cols[candidate]
    if optional:
        return None
    raise KeyError(f"None of the columns {candidates} were found in dataframe")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Re-download all tables even if they exist")
    parser.add_argument("--grants-file", type=Path, help="Path to the manually downloaded grants Excel file", default=None)
    parser.add_argument("--no-transform", action="store_true", help="Skip building the processed dataset")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    for table in TABLES:
        try:
            download_table(table, RAW_DIR, overwrite=args.force)
        except Exception:
            logger.exception("Unable to download %s", table.table)

    if args.no_transform:
        return 0

    try:
        dataset = build_processed_dataset(RAW_DIR, args.grants_file)
    except Exception:
        logger.exception("Failed to build processed dataset")
        return 1

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(PROCESSED_FILE, index=False)
    logger.info("Saved processed dataset to %s", PROCESSED_FILE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
