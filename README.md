# Binance Trade Analysis

This repository contains a reproducible Python workflow for comparing historical Binance account records over the supplied 90-day dataset.

It analyses completed trade history. It does not place trades, predict prices, connect to a Binance account or provide investment advice.

## What it reports

- Total realised profit and loss
- Total, winning, losing and break-even trades
- Win rate
- Notional turnover
- Return on notional
- Maximum drawdown of cumulative realised PnL
- A trade-level Sharpe proxy
- A percentile-based comparison score

The dataset does not provide a reliable invested-capital series. The script therefore reports **return on notional**, not true portfolio ROI. The Sharpe value is based on trade-level realised PnL rather than periodic portfolio returns, so it is a comparison aid rather than a conventional portfolio Sharpe ratio.

## Ranking method

Each account metric is converted to a cross-sectional percentile before weighting. This prevents a large raw PnL value from overwhelming metrics that use different units.

| Component | Weight |
| --- | ---: |
| Return on notional | 40% |
| Total realised PnL | 30% |
| Trade-level Sharpe proxy | 20% |
| Maximum drawdown | 10% |

Higher maximum-drawdown values are better because a drawdown closer to zero represents a smaller loss from the running peak.

## Requirements

- Python 3.10 or newer
- pandas
- NumPy

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run

The default command expects `TRADES_CopyTr_90D_ROI.csv` beside the script:

```bash
python Binance-Trade-Analysis.py
```

Or provide explicit paths:

```bash
python Binance-Trade-Analysis.py \
  --input path/to/trades.csv \
  --output-dir analysis-output \
  --top 20
```

The command writes:

- `analysis-output/final_metrics.csv`
- `analysis-output/top_accounts.csv`

## Input contract

The source CSV must include:

- `Port_IDs`
- `Trade_History`, containing a JSON array for each account

Each trade record must include `realizedProfit` and `quantity`. `quoteQty` is preferred for notional turnover. If it is absent, the script uses `abs(quantity × price)`. A `time`, `timestamp` or `updateTime` field is used to sort trades before calculating drawdown.

## Limits

- The result depends on the supplied dataset and its definitions.
- Fees, funding, leverage and unrealised PnL may not be fully represented.
- Return on notional is not the same as return on invested capital.
- The trade-level Sharpe proxy is not a periodic portfolio Sharpe ratio.
- Rankings are descriptive and do not establish strategy quality or future performance.
