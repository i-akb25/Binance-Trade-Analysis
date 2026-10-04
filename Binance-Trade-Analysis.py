"""Rank Binance account histories with transparent, reproducible metrics.

This module analyses the supplied account-level ``Trade_History`` JSON records.
It does not place trades, predict prices, or provide investment advice.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


LOGGER = logging.getLogger("binance_trade_analysis")
REQUIRED_SOURCE_COLUMNS = {"Port_IDs", "Trade_History"}
REQUIRED_TRADE_COLUMNS = {"realizedProfit", "quantity"}
TIMESTAMP_COLUMNS = ("time", "timestamp", "updateTime")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyse historical Binance trade records and rank accounts.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).with_name("TRADES_CopyTr_90D_ROI.csv"),
        help="Input CSV containing Port_IDs and Trade_History columns.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis-output"),
        help="Directory for final_metrics.csv and top_accounts.csv.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of ranked accounts to write to top_accounts.csv.",
    )
    return parser.parse_args()


def load_source(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Input dataset was not found: {path}")

    frame = pd.read_csv(path, usecols=lambda column: column in REQUIRED_SOURCE_COLUMNS)
    missing = REQUIRED_SOURCE_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Input dataset is missing columns: {', '.join(sorted(missing))}")
    if frame.empty:
        raise ValueError("Input dataset contains no account records")
    return frame


def _parse_history(value: Any, port_id: Any, row_number: int) -> list[dict[str, Any]]:
    if pd.isna(value):
        return []
    if not isinstance(value, str):
        LOGGER.warning("Skipping non-text Trade_History at source row %s", row_number)
        return []

    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        LOGGER.warning("Skipping malformed Trade_History at source row %s", row_number)
        return []

    if not isinstance(decoded, list):
        LOGGER.warning("Skipping non-list Trade_History at source row %s", row_number)
        return []

    records: list[dict[str, Any]] = []
    for trade in decoded:
        if not isinstance(trade, dict):
            continue
        records.append({**trade, "Port_IDs": port_id})
    return records


def expand_trade_history(source: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row_number, row in source.reset_index(drop=True).iterrows():
        records.extend(
            _parse_history(row["Trade_History"], row["Port_IDs"], row_number + 2)
        )

    if not records:
        raise ValueError("No valid trades were found in Trade_History")

    trades = pd.DataFrame.from_records(records)
    missing = REQUIRED_TRADE_COLUMNS.difference(trades.columns)
    if missing:
        raise ValueError(
            f"Expanded trade history is missing columns: {', '.join(sorted(missing))}"
        )

    trades["realizedProfit"] = pd.to_numeric(
        trades["realizedProfit"], errors="coerce"
    )
    trades["quantity"] = pd.to_numeric(trades["quantity"], errors="coerce")
    if "price" in trades:
        trades["price"] = pd.to_numeric(trades["price"], errors="coerce")
    if "quoteQty" in trades:
        trades["quoteQty"] = pd.to_numeric(trades["quoteQty"], errors="coerce")

    invalid = trades["realizedProfit"].isna() | trades["quantity"].isna()
    if invalid.any():
        LOGGER.warning("Dropping %s trades with invalid numeric fields", int(invalid.sum()))
        trades = trades.loc[~invalid].copy()
    if trades.empty:
        raise ValueError("No valid numeric trade records remain after cleaning")

    timestamp_column = next((name for name in TIMESTAMP_COLUMNS if name in trades), None)
    if timestamp_column:
        numeric_time = pd.to_numeric(trades[timestamp_column], errors="coerce")
        unit = "ms" if numeric_time.dropna().median() > 10_000_000_000 else "s"
        trades["_event_time"] = pd.to_datetime(
            numeric_time, unit=unit, utc=True, errors="coerce"
        )
        trades = trades.sort_values(
            ["Port_IDs", "_event_time"], kind="stable", na_position="last"
        )
    else:
        LOGGER.warning(
            "No timestamp column was found; drawdown follows source-record order"
        )

    if "quoteQty" in trades and trades["quoteQty"].notna().any():
        trades["_notional"] = trades["quoteQty"].abs()
    elif "price" in trades and trades["price"].notna().any():
        trades["_notional"] = trades["quantity"].abs() * trades["price"].abs()
    else:
        LOGGER.warning(
            "Neither quoteQty nor price is available; return-on-notional will be unavailable"
        )
        trades["_notional"] = np.nan

    return trades


def maximum_drawdown(group: pd.DataFrame) -> float:
    cumulative_pnl = group["realizedProfit"].cumsum()
    drawdown = cumulative_pnl - cumulative_pnl.cummax()
    return float(drawdown.min()) if not drawdown.empty else 0.0


def trade_sharpe(group: pd.DataFrame) -> float:
    pnl = group["realizedProfit"]
    standard_deviation = pnl.std(ddof=1)
    if len(pnl) < 2 or pd.isna(standard_deviation) or standard_deviation == 0:
        return 0.0
    return float((pnl.mean() / standard_deviation) * np.sqrt(len(pnl)))


def percentile_score(series: pd.Series, *, higher_is_better: bool = True) -> pd.Series:
    values = series.replace([np.inf, -np.inf], np.nan)
    if values.notna().sum() <= 1:
        return pd.Series(50.0, index=series.index)
    ranked = values.rank(pct=True, method="average", ascending=higher_is_better) * 100
    return ranked.fillna(0.0)


def calculate_metrics(trades: pd.DataFrame) -> pd.DataFrame:
    grouped = trades.groupby("Port_IDs", sort=False, dropna=False)
    metrics = grouped.agg(
        Total_PnL=("realizedProfit", "sum"),
        Total_Trades=("realizedProfit", "size"),
        Win_Trades=("realizedProfit", lambda values: int((values > 0).sum())),
        Loss_Trades=("realizedProfit", lambda values: int((values < 0).sum())),
        Break_Even_Trades=("realizedProfit", lambda values: int((values == 0).sum())),
        Notional_Turnover=("_notional", "sum"),
    ).reset_index()

    metrics["Return_on_Notional_Pct"] = np.where(
        metrics["Notional_Turnover"] > 0,
        metrics["Total_PnL"] / metrics["Notional_Turnover"] * 100,
        np.nan,
    )
    metrics["Win_Rate_Pct"] = np.where(
        metrics["Total_Trades"] > 0,
        metrics["Win_Trades"] / metrics["Total_Trades"] * 100,
        0.0,
    )

    risk = grouped.apply(
        lambda group: pd.Series(
            {
                "Maximum_Drawdown": maximum_drawdown(group),
                "Trade_Sharpe": trade_sharpe(group),
            }
        ),
        include_groups=False,
    ).reset_index()
    metrics = metrics.merge(risk, on="Port_IDs", how="left", validate="one_to_one")

    metrics["Score_Return"] = percentile_score(metrics["Return_on_Notional_Pct"])
    metrics["Score_PnL"] = percentile_score(metrics["Total_PnL"])
    metrics["Score_Sharpe"] = percentile_score(metrics["Trade_Sharpe"])
    metrics["Score_Drawdown"] = percentile_score(metrics["Maximum_Drawdown"])
    metrics["Score"] = (
        metrics["Score_Return"] * 0.40
        + metrics["Score_PnL"] * 0.30
        + metrics["Score_Sharpe"] * 0.20
        + metrics["Score_Drawdown"] * 0.10
    )

    return metrics.sort_values(
        ["Score", "Total_PnL", "Port_IDs"],
        ascending=[False, False, True],
        kind="stable",
    ).reset_index(drop=True)


def write_results(metrics: pd.DataFrame, output_directory: Path, top_count: int) -> None:
    if top_count < 1:
        raise ValueError("--top must be at least 1")
    output_directory.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_directory / "final_metrics.csv", index=False)
    metrics.head(top_count).to_csv(output_directory / "top_accounts.csv", index=False)


def main() -> int:
    arguments = parse_arguments()
    try:
        source = load_source(arguments.input.resolve())
        trades = expand_trade_history(source)
        metrics = calculate_metrics(trades)
        write_results(metrics, arguments.output_dir.resolve(), arguments.top)
    except (FileNotFoundError, ValueError, OSError, pd.errors.ParserError) as error:
        LOGGER.error("Analysis failed: %s", error)
        return 1

    LOGGER.info(
        "Analysed %s valid trades across %s accounts. Results: %s",
        len(trades),
        len(metrics),
        arguments.output_dir.resolve(),
    )
    print(metrics.head(arguments.top).to_string(index=False))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raise SystemExit(main())
