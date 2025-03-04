import pandas as pd
import json
import logging
import os
import sys
import numpy as np

# Enable logging
logging.basicConfig(level=logging.INFO)

# Load dataset
file_path = os.path.join(os.getcwd(), 'TRADES_CopyTr_90D_ROI.csv')

# Read CSV file
try:
    df = pd.read_csv(file_path)
    print("✅ Dataset Loaded Successfully!")
    print("Columns in Dataset:", df.columns)  # Debug: Check column names
except Exception as e:
    logging.error(f"❌ Failed to load dataset: {e}")
    sys.exit(1)

# Debugging: Check Port_IDs and Trade_History
if 'Port_IDs' not in df.columns or 'Trade_History' not in df.columns:
    logging.error("❌ Port_IDs or Trade_History column is missing in the dataset!")
    sys.exit(1)

print("🔹 Port_IDs Sample:\n", df['Port_IDs'].head())
print("🔹 Trade_History Sample:\n", df['Trade_History'].head())
print(f"🔹 Count of Null Port_IDs: {df['Port_IDs'].isnull().sum()}")

# Function to expand Trade History
def expand_trade_history(df):
    expanded_data = []
    for _, row in df.iterrows():
        port_id = row['Port_IDs']
        trade_history = row['Trade_History']

        if pd.notna(trade_history) and isinstance(trade_history, str):
            try:
                trade_history = json.loads(trade_history)  # Parse JSON string
                if isinstance(trade_history, list):  # Ensure it's a list
                    for trade in trade_history:
                        trade['Port_IDs'] = port_id  # Add Port_IDs to each trade
                        expanded_data.append(trade)
            except json.JSONDecodeError as e:
                logging.warning(f"⚠️ Error parsing Trade_History for Port_ID {port_id}: {e}")

    return pd.DataFrame(expanded_data)

# Expand trade history
trade_df = expand_trade_history(df)

# Validate expanded data
if trade_df.empty or 'Port_IDs' not in trade_df.columns or trade_df['Port_IDs'].isnull().any():
    logging.error("❌ Port_IDs column is missing, contains null values, or trade data is empty.")
    print(trade_df.head())  # Debugging output
    sys.exit(1)

print("✅ Trade history expanded successfully!")

# Convert realizedProfit column to numeric
trade_df['realizedProfit'] = pd.to_numeric(trade_df['realizedProfit'], errors='coerce').fillna(0)
trade_df['quantity'] = pd.to_numeric(trade_df['quantity'], errors='coerce').fillna(0)

# Compute Financial Metrics for each Port_ID
metrics = trade_df.groupby('Port_IDs').agg(
    Total_PnL=('realizedProfit', 'sum'),
    Total_Trades=('realizedProfit', 'count'),
    Win_Trades=('realizedProfit', lambda x: (x > 0).sum()),
    Loss_Trades=('realizedProfit', lambda x: (x <= 0).sum()),
    Total_Investment=('quantity', 'sum')
).reset_index()

# ROI Calculation
metrics['ROI'] = np.where(metrics['Total_Investment'] > 0, 
                          (metrics['Total_PnL'] / metrics['Total_Investment']) * 100, 0)

# Win Rate Calculation
metrics['Win_Rate'] = np.where(metrics['Total_Trades'] > 0, 
                               (metrics['Win_Trades'] / metrics['Total_Trades']) * 100, 0)

# Maximum Drawdown Calculation
def calculate_mdd(df):
    df = df.copy()  # Avoid modifying original DataFrame
    df['Cumulative PnL'] = df['realizedProfit'].cumsum()
    df['Max Cumulative PnL'] = df['Cumulative PnL'].cummax()
    df['Drawdown'] = df['Cumulative PnL'] - df['Max Cumulative PnL']
    return df['Drawdown'].min()

mdd_values = trade_df.groupby('Port_IDs').apply(calculate_mdd)
metrics['MDD'] = mdd_values.values

# Sharpe Ratio Calculation
def calculate_sharpe(df, risk_free_rate=0):
    returns = df['realizedProfit']
    return 0 if returns.std() == 0 else (returns.mean() - risk_free_rate) / returns.std()

sharpe_values = trade_df.groupby('Port_IDs').apply(calculate_sharpe)
metrics['Sharpe_Ratio'] = sharpe_values.values

# Ranking Accounts Based on Metrics
metrics['Score'] = (
    metrics['ROI'] * 0.4 +
    metrics['Total_PnL'] * 0.3 +
    metrics['Sharpe_Ratio'] * 0.2 +
    metrics['Win_Rate'] * 0.1
)

metrics = metrics.sort_values(by='Score', ascending=False)

# Save Results
top_20_accounts = metrics.head(20)
top_20_accounts.to_csv('top_20_accounts.csv', index=False)
metrics.to_csv('final_metrics.csv', index=False)

# Display Top 20 Accounts
print("✅ Analysis Completed! Top 20 Accounts:")
print(top_20_accounts)
