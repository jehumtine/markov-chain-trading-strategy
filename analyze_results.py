import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load CSV
df = pd.read_csv('full_hyperopt_results.csv')

# Rename your actual columns if needed
sharpe_col = 'sharpe'
winrate_col = 'win_rate'
profit_col = 'total_profit'

# Check columns exist
for col in [sharpe_col, winrate_col, profit_col]:
    if col not in df.columns:
        raise ValueError(f"Missing column: {col}")

# Normalize the key columns using MinMaxScaler
scaler = MinMaxScaler()
df[['norm_sharpe', 'norm_winrate', 'norm_profit']] = scaler.fit_transform(
    df[[sharpe_col, winrate_col, profit_col]]
)

# Weights: adjust these if you want to prioritize one more
w_sharpe = 0.4
w_winrate = 0.3
w_profit = 0.3

# Compute composite score
df['score'] = (
    w_sharpe * df['norm_sharpe'] +
    w_winrate * df['norm_winrate'] +
    w_profit * df['norm_profit']
)

# Sort by composite score
top_n = 10
sorted_df = df.sort_values(by='score', ascending=False)

# Show top results
print(f"Top {top_n} results ranked by composite score:")
print(sorted_df.head(top_n)[[sharpe_col, winrate_col, profit_col, 'score']])
