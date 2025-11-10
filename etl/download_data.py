"""Download and transform public datasets for the liquidity forecast app."""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_FILE = DATA_DIR / "liquidity_dataset.parquet"

STATBANK_BASE = "https://api.statbank.dk/v1/data"
STATBANK_META_BASE = "https://api.statbank.dk/v1/tableinfo"

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class TableConfig:
    table: str
    filters: Dict[str, Iterable[str]]
    filename: str


def get_last_five_years() -> str:
    # Return a year range for the last 5 full years, e.g. "2020-2024"
    current_year = datetime.now().year - 1
    start_year = current_year - 4
    return f"{start_year}-{current_year}"


def fetch_tableinfo(table: str, lang: str = "da", timeout: int = 60) -> Optional[dict]:
    """Fetch StatBank table metadata to validate variable codes and allowed values.

    Returns None if metadata request fails; caller should fall back gracefully.
    """
    url = f"{STATBANK_META_BASE}/{table}?contentType=JSON&lang={lang}"
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        logger.warning("Could not fetch tableinfo for %s: %s", table, exc)
        return None


TABLES: tuple[TableConfig, ...] = (
    TableConfig(
        table="REGK4",
        filters={
            "OMRÅDE": ["*"],
            "FUNKTION": ["*"],
            "Tid": ["2024%2C2023%2C2022%2C2021%2C2020"],
        },
        filename="regk4.csv",
    ),
    TableConfig(
        table="REGK31",
        filters={
            "OMRÅDE": ["*"],
            "Tid": [get_last_five_years()],
        },
        filename="regk31.csv",
    ),
    TableConfig(
        table="BUDK32",
        filters={
            "OMRÅDE": ["*"],
            "Tid": [get_last_five_years()],
        },
        filename="budk32.csv",
    ),
    TableConfig(
        table="FOLK1A",
        filters={
            "OMRÅDE": ["*"],
            "ALDER": ["IALT"],
            "KØN": ["TOT"],
            "Tid": [get_last_five_years()],
        },
        filename="folk1a.csv",
    ),
    TableConfig(
        table="DEMO19",
        filters={
            "OMRÅDE": ["*"],
            "Tid": [get_last_five_years()],
        },
        filename="demo19.csv",
    ),
    TableConfig(
        table="EJDSK1",
        filters={
            "OMRÅDE": ["*"],
            "Tid": [get_last_five_years()],
        },
        filename="ejdsk1.csv",
    ),
    TableConfig(
        table="PSKAT",
        filters={
            "OMRÅDE": ["*"],
            "Tid": [get_last_five_years()],
        },
        filename="pskat.csv",
    ),
)


def default_tables() -> tuple[TableConfig, ...]:
    """Clean, human-readable defaults for StatBank downloads.

    The downloader still normalizes/augments variables at runtime based on metadata.
    """
    return (
        TableConfig(
            table="REGK4",
            filters={
                "OMRÅDE": ["*"],
                "FUNKTION": ["*"],
                "TID": [get_last_five_years()],
            },
            filename="regk4.csv",
        ),
        TableConfig(
            table="REGK31",
            filters={
                "OMRÅDE": ["*"],
                "TID": [get_last_five_years()],
            },
            filename="regk31.csv",
        ),
        TableConfig(
            table="BUDK32",
            filters={
                "OMRÅDE": ["*"],
                "TID": [get_last_five_years()],
            },
            filename="budk32.csv",
        ),
        TableConfig(
            table="FOLK1A",
            filters={
                "OMRÅDE": ["*"],
                "ALDER": ["IALT"],
                "KØN": ["TOT"],
                "TID": [get_last_five_years()],
            },
            filename="folk1a.csv",
        ),
        TableConfig(
            table="DEMO19",
            filters={
                "OMRÅDE": ["*"],
                "TID": [get_last_five_years()],
            },
            filename="demo19.csv",
        ),
        TableConfig(
            table="EJDSK1",
            filters={
                "OMRÅDE": ["*"],
                "TID": [get_last_five_years()],
            },
            filename="ejdsk1.csv",
        ),
        TableConfig(
            table="PSKAT",
            filters={
                "OMRÅDE": ["*"],
                "TID": [get_last_five_years()],
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

    # Inject required variables per table when missing
    merged_filters: Dict[str, Iterable[str]] = dict(config.filters)
    if config.table in ("REGK31", "BUDK32") and "PRISENHED" not in {str(k).upper(): v for k, v in merged_filters.items()}:
        merged_filters["PRISENHED"] = ["*"]
    if config.table == "DEMO19" and "REGI07A" not in {str(k).upper(): v for k, v in merged_filters.items()}:
        merged_filters["REGI07A"] = ["*"]
    if config.table == "PSKAT" and "SKATPCT" not in {str(k).upper(): v for k, v in merged_filters.items()}:
        merged_filters["SKATPCT"] = ["*"]
    if config.table == "DEMO19" and "MÆNGDE4" not in {str(k).upper(): v for k, v in merged_filters.items()}:
        merged_filters["MÆNGDE4"] = ["*"]

    # Build canonical StatBank variable codes using safe Unicode escapes
    def canon_var_code(code: str) -> str:
        c = str(code).strip().upper()
        if c.startswith("OMR"):
            return "OMR\u00C5DE"
        if c.startswith("TID") or c == "TIME":
            return "TID"
        if c.startswith("K"):
            return "K\u00D8N"
        if "ALDER" in c:
            return "ALDER"
        if "FUNKTION" in c:
            return "FUNKTION"
        return str(code)
    # Build GET query params with proper joining and range for TID
    # Optionally adapt variable codes and time values using table metadata
    tableinfo = fetch_tableinfo(config.table)
    available_ids = set()
    time_var_id = "TID"
    id_by_lower = {}
    if tableinfo and isinstance(tableinfo, dict) and "variables" in tableinfo:
        for var in tableinfo["variables"]:
            vid = var.get("id")
            if not isinstance(vid, str):
                continue
            available_ids.add(vid)
            id_by_lower[vid.lower()] = vid
            if var.get("time") is True:
                time_var_id = vid

    def resolve_code(canon: str) -> str:
        # If metadata present and exact match exists, use it
        if canon in available_ids:
            return canon
        # Fallbacks by pattern
        up = canon.upper()
        patterns = []
        if up.startswith("OMR"):
            patterns = ["OMR", "KOM", "REG", "GEO"]
        elif up.startswith("K\u00D8N") or up.startswith("K"):
            patterns = ["K\u00D8N", "KON", "SEX"]
        elif up.startswith("ALDER"):
            patterns = ["ALDER", "ALD"]
        elif up.startswith("FUNKTION"):
            patterns = ["FUNK", "ART", "DRANST", "FUNKTION"]
        elif up.startswith("TID"):
            patterns = ["TID", "KVART", "M\u00C5N"]
        for p in patterns:
            for aid in available_ids:
                if aid.upper().startswith(p):
                    return aid
        return canon  # last resort

    params: Dict[str, str] = {"timeOrder": "Ascending"}
    for code, values in merged_filters.items():
        canon = canon_var_code(code)
        resolved = resolve_code(canon) if available_ids else canon
        flat: list[str] = []
        for v in values:
            s = str(v).replace("%2C", ",")
            parts = [p.strip() for p in s.split(",") if p.strip()]
            flat.extend(parts)

        if resolved == time_var_id:
            # Map desired years to actual time member values if metadata available
            if tableinfo and isinstance(tableinfo, dict):
                try:
                    tvar = next(v for v in tableinfo["variables"] if v.get("id") == time_var_id)
                    raw_vals = tvar.get("values", [])
                    if raw_vals and isinstance(raw_vals[0], dict):
                        allowed_vals = [str(x.get("id")) for x in raw_vals if isinstance(x, dict) and x.get("id") is not None]
                    else:
                        allowed_vals = [str(x) for x in raw_vals]
                except StopIteration:
                    allowed_vals = []
                years: set[str] = set()
                for p in flat:
                    if "-" in p and len(p) == 9:
                        try:
                            start, end = p.split("-")
                            sy, ey = int(start), int(end)
                            years.update(str(y) for y in range(sy, ey + 1))
                        except Exception:
                            pass
                    else:
                        if p.isdigit() and len(p) == 4:
                            years.add(p)
                if years and allowed_vals:
                    selected = [val for val in allowed_vals if any(val.startswith(y) for y in years)]
                    params[resolved] = ",".join(selected) if selected else ",".join(allowed_vals)
                else:
                    params[resolved] = ",".join(flat)
            else:
                # No metadata; keep expanded or raw
                params[resolved] = ",".join(flat)
        else:
            params[resolved] = ",".join(flat)

    logger.info("Downloading %s with params %s", config.table, params)
    response = requests.get(url, params=params, headers={"Accept": "text/csv"}, timeout=120)
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        logger.error("Failed to download %s: %s\nResponse: %s", config.table, exc, response.text[:500])
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

    dataset = prepare_cash_balance(regk4)
    dataset = dataset.merge(prepare_regk31(regk31), on=["kommunekode", "year"], how="left")
    dataset = dataset.merge(prepare_budget(budk32), on=["kommunekode", "year"], how="left")
    dataset = dataset.merge(prepare_population(folk1a), on=["kommunekode", "year"], how="left")
    dataset = dataset.merge(prepare_business_metrics(demo19), on=["kommunekode", "year"], how="left")
    dataset = dataset.merge(prepare_property_tax(ejdsk1), on=["kommunekode", "year"], how="left")
    dataset = dataset.merge(prepare_tax_rate(pskat), on=["kommunekode", "year"], how="left")

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

    code_col = resolve_column(df, ["OMR", "kommunekode", "område"])
    time_col = resolve_column(df, ["tid", "år", "time"])
    value_col = resolve_column(df, ["indhold", "value", "bel�b"])

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

    code_col = resolve_column(df, ["område", "OMR", "kommunekode"])
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

    try:
        code_col = resolve_column(df, ["regi07a", "omr", "kommunekode", "kommune", "region", "geo"])
    except KeyError:
        code_col = resolve_column(df, ["omr", "kommunekode"])
    time_col = resolve_column(df, ["tid", "time"])
    value_col = resolve_column(df, ["indhold", "value"])

    # Identify indicator column (e.g., M�NGDE4) if present
    indicator_col = resolve_column(df, ["m\u00C6ngde4", "erhverv", "enhed", "indholdstype", "erhvervsindikator"], optional=True)
    df = df.copy()
    df.rename(columns={code_col: "kommunekode", time_col: "year", value_col: "value"}, inplace=True)
    df["kommunekode"] = df["kommunekode"].astype(str).str.strip()
    df["value"] = pd.to_numeric(df["value"].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    df["year"] = df["year"].astype(str).str.extract(r"(\d{4})").astype(int)

    if not indicator_col:
        # Fallback: choose the first non key/value column as indicator
        non_keys = [c for c in df.columns if c not in {"kommunekode", "year", "value"}]
        if non_keys:
            indicator_col = non_keys[0]
        else:
            return pd.DataFrame({"kommunekode": [], "year": [], "new_businesses": [], "employees": []})

    df[indicator_col] = df[indicator_col].astype(str).str.lower()
    def _metric_from_text(s: str) -> str | None:
        if "fuldtid" in s:
            return "employees"
        if "nye" in s or "nystart" in s:
            return "new_businesses"
        return None
    df["metric"] = df[indicator_col].map(_metric_from_text)
    df = df.dropna(subset=["metric"])

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

    code_col = resolve_column(df, ["OMR", "kommunekode", "område"])
    time_col = resolve_column(df, ["tid", "år", "time"])
    value_col = resolve_column(df, ["indhold", "value", "bel�b"])
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
    try:
        code_col = resolve_column(df, ["område","omr","kommunekode","kommune","regi07a","region","geo"])
    except KeyError:
        logger.warning("BUDK32: could not resolve geography column; returning empty dataset")
        return pd.DataFrame({"kommunekode": [], "year": [], "budget_balance": []})


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

    code_col = resolve_column(df, ["område", "OMR", "kommunekode"])
    time_col = resolve_column(df, ["tid", "år", "time"])
    value_col = resolve_column(df, ["indhold", "value", "bel�b"])

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

    code_col = resolve_column(df, ["område", "OMR", "kommunekode"])
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
    code_col = resolve_column(df, ["kommunekode", "OMR", "kode"])
    year_col = resolve_column(df, ["år", "year"])
    value_col = resolve_column(df, ["tilskud", "bloktilskud", "bel�b", "amount"])

    df = df[[code_col, year_col, value_col]].copy()
    df.rename(columns={code_col: "kommunekode", year_col: "year", value_col: "grants"}, inplace=True)
    df["year"] = df["year"].astype(int)
    return df


def resolve_column(df: pd.DataFrame, candidates: Iterable[str], optional: bool = False) -> Optional[str]:
    lower_cols = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        cand = str(candidate).lower()
        if cand in lower_cols:
            return lower_cols[cand]
    # Heuristic fallbacks for common StatBank column names if direct match fails
    cand_text = " ".join([str(c).lower() for c in candidates])
    if ("omr" in cand_text or "kommune" in cand_text) and "område" in lower_cols:
        return lower_cols["område"]
    if ("tid" in cand_text or "år" in cand_text or "time" in cand_text) and "tid" in lower_cols:
        return lower_cols["tid"]
    if ("bel" in cand_text or "indhold" in cand_text or "value" in cand_text):
        for key in ("indhold", "bel�b", "value"):
            if key in lower_cols:
                return lower_cols[key]
    if optional:
        return None
    raise KeyError(f"None of the columns {candidates} were found in dataframe")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Re-download all tables even if they exist")
    parser.add_argument("--no-transform", action="store_true", help="Skip building the processed dataset")
    return parser.parse_args(argv)

def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    # Use cleaned default_tables() for clarity; legacy TABLES kept for reference
    for table in default_tables():
        try:
            download_table(table, RAW_DIR, overwrite=args.force)
        except Exception:
            logger.exception("Unable to download %s", table.table)

    if args.no_transform:
        return 0

    try:
        dataset = build_processed_dataset(RAW_DIR)
    except Exception:
        logger.exception("Failed to build processed dataset")
        return 1

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(PROCESSED_FILE, index=False)
    logger.info("Saved processed dataset to %s", PROCESSED_FILE)
    return 0


if __name__ == "__main__":
    sys.exit(main())


