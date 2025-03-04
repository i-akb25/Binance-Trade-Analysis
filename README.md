# Binance Trade Data Analysis Assignment

## Overview
This project involves analyzing historical trade data from Binance accounts over 90 days. The objective is to calculate key financial metrics, rank accounts based on performance, and provide insights.

## Task Details
- **Dataset**: Contains trade history with details like timestamps, assets, trade side (BUY/SELL), price, and more.
- **Objective**: Analyze the dataset, calculate financial metrics, rank accounts, and provide a top 20 list.

## Metrics Calculated
1. **ROI (Return on Investment)**: Measures profitability relative to investment.
2. **PnL (Profit and Loss)**: Sum of realized profits/losses for each account.
3. **Sharpe Ratio**: Risk-adjusted return metric.
4. **MDD (Maximum Drawdown)**: Maximum loss from a peak.
5. **Win Rate**: Percentage of profitable trades.
6. **Win Positions**: Number of profitable trades.
7. **Total Positions**: Total number of trades executed.

## Steps to Complete the Task
### 1. Data Exploration and Cleaning
- Loaded and inspected the dataset.
- Handled missing values and structured trade history.

### 2. Feature Engineering
- Created derived features such as cumulative PnL, drawdowns, and return distributions.
- Applied transformations to enhance insights.

### 3. Ranking Algorithm
- Developed a weighted scoring system:
  - **40%** ROI
  - **30%** PnL
  - **20%** Sharpe Ratio
  - **10%** Win Rate
- Sorted accounts based on final scores.

## Deliverables
- **Python Analysis Script**: Jupyter Notebook containing full analysis.

## Assumptions & Considerations
- **Risk-Free Rate Assumed as Zero**: Used in Sharpe Ratio calculation.
- **Trade Data Integrity**: Assumed valid and accurate.
- **No Leverage Considered**: PnL and ROI calculated based on provided data.


