import os
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from data_processor import merge_4h_state, classify_state
from backtester import backtest
import seaborn as sns
import pandas_ta as ta

def rolling_window(a, window):
    """
    Creates a rolling window view of a NumPy array.
    This is a highly efficient way to generate sequences.
    """
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def extract_useful_sequences(df, params:dict):
    """
    Extracts sequences from the composite, 1H, and 4H states using
    efficient NumPy vectorization.
    """
    sequence_length= params['sequence_length']
    # 1. Define a helper function to perform the vectorized extraction for a given state type.
    def get_sequences_vectorized(state_series, next_state_series):
        # Convert pandas Series to NumPy arrays for speed
        states = state_series.to_numpy()
        next_states = next_state_series.to_numpy()

        num_pairs = len(states) - sequence_length
        if num_pairs <= 0:  # Add a check for cases where the dataframe is too short
            return defaultdict(int)

        # Generate all sequences at once using the rolling window view
        sequences = rolling_window(states, sequence_length)

        # Trim the sequences and next_states to be the same length (num_pairs)
        trimmed_sequences = sequences[:num_pairs]
        next_states_for_sequences = next_states[sequence_length:]

        # The next state for each sequence is the element immediately following the sequence
        next_states_for_sequences = next_states[sequence_length:]

        # Combine sequences and their next states for unique counting
        # We need a unique way to represent the sequence tuples for numpy
        # This can be done by converting them to void type.
        void_sequences = np.ascontiguousarray(trimmed_sequences).view(
            np.dtype((np.void, trimmed_sequences.dtype.itemsize * trimmed_sequences.shape[1]))
        ).flatten()
        hashable_sequences = [seq.tobytes() for seq in void_sequences]
        pair_counts = Counter(zip(hashable_sequences, next_states_for_sequences))
        # Reconstruct the transitions dictionary from the Counter's items
        transitions = defaultdict(int)
        for (bytes_seq, next_s), count in pair_counts.items():
            # Use np.frombuffer() which is designed to read bytes objects.
            original_sequence = tuple(np.frombuffer(bytes_seq, dtype=states.dtype))
            transitions[(original_sequence, int(next_s))] = count

        return transitions

    # 2. Apply the vectorized function to each state type
    #composite_transitions = get_sequences_vectorized(df['composite_state'], df['state_1h'])
    #h1_transitions = get_sequences_vectorized(df['state_1h'], df['state_1h'])
    h4_transitions = get_sequences_vectorized(df['state_4h'], df['state_1h'])

    # 3. The rest of the logic for processing transitions remains the same
    #composite_useful_sequences = process_transitions_to_sequences(composite_transitions)
    #h1_useful_sequences = process_transitions_to_sequences(h1_transitions)
    h4_useful_sequences = process_transitions_to_sequences(h4_transitions, params)

    return h4_useful_sequences

def process_transitions_to_sequences(transitions, params:dict):
    """
    Helper function to process transitions and extract useful sequences
    """
    # Normalize counts to probabilities
    transition_probabilities = defaultdict(dict)
    for (sequence, next_state), count in transitions.items():
        # Get total transitions for the sequence over all possible next states
        total_transitions = sum(transitions.get((sequence, s), 0)
                                for s in {s for (_, s) in transitions if _ == sequence})
        transition_probabilities[sequence][next_state] = count / total_transitions if total_transitions else 0

    # Extract sequences with strong predictive power
    useful_sequences = {}
    for sequence, probs in transition_probabilities.items():
        for next_state, prob in probs.items():
            if params['min_signal_probability'] < prob < 1:  # High probability threshold
                if sequence not in useful_sequences:
                    useful_sequences[sequence] = {}
                useful_sequences[sequence][next_state] = prob
    return useful_sequences

def visualize_trade_sources(backtest_result, save_path):
    """
    Create visualizations for trade sources
    """
    trade_sources = backtest_result['trade_sources']

    # Create figure for entry sources
    plt.figure(figsize=(15, 10))

    # Entry sources pie chart
    plt.subplot(2, 2, 1)
    entry_labels = list(trade_sources['entry'].keys())
    entry_values = list(trade_sources['entry'].values())
    if sum(entry_values) > 0:  # Only plot if there are trades
        plt.pie(entry_values, labels=entry_labels, autopct='%1.1f%%',
                shadow=True, startangle=90)
        plt.axis('equal')
        plt.title('Trade Entry Sources')
    else:
        plt.text(0.5, 0.5, 'No trade entries', horizontalalignment='center',
                 verticalalignment='center')

    # Exit sources pie chart
    plt.subplot(2, 2, 2)
    exit_labels = list(trade_sources['exit'].keys())
    exit_values = list(trade_sources['exit'].values())
    if sum(exit_values) > 0:  # Only plot if there are trades
        plt.pie(exit_values, labels=exit_labels, autopct='%1.1f%%',
                shadow=True, startangle=90)
        plt.axis('equal')
        plt.title('Trade Exit Sources')
    else:
        plt.text(0.5, 0.5, 'No trade exits', horizontalalignment='center',
                 verticalalignment='center')

    # Bar chart for entry source performance
    plt.subplot(2, 2, 3)

    # Prepare data for performance comparison
    sources = ['composite', '1h', '4h']
    trade_counts = []
    win_rates = []
    avg_profits = []

    for source in sources:
        source_trades = [t for t in backtest_result['all_trades'] if t['entry_source'] == source]

        if len(source_trades) > 0:
            trade_counts.append(len(source_trades))
            winning_trades = len([t for t in source_trades if t['profit'] > 0])
            win_rate = winning_trades / len(source_trades) if len(source_trades) > 0 else 0
            win_rates.append(win_rate * 100)  # Convert to percentage

            avg_profit = sum(t['profit'] for t in source_trades) / len(source_trades) if len(source_trades) > 0 else 0
            avg_profits.append(avg_profit)
        else:
            trade_counts.append(0)
            win_rates.append(0)
            avg_profits.append(0)

    x = np.arange(len(sources))
    width = 0.35

    ax1 = plt.gca()
    ax1.bar(x - width / 2, trade_counts, width, label='Trade Count')
    ax1.set_ylabel('Number of Trades')
    ax1.set_title('Entry Source Performance')
    ax1.set_xticks(x)
    ax1.set_xticklabels(sources)

    ax2 = ax1.twinx()
    ax2.bar(x + width / 2, win_rates, width, color='green', label='Win Rate %')
    ax2.set_ylabel('Win Rate %')

    # Add a legend
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper right')

    # Average profit by signal source
    plt.subplot(2, 2, 4)
    colors = ['green' if x > 0 else 'red' for x in avg_profits]
    plt.bar(sources, avg_profits, color=colors)
    plt.ylabel('Average Profit per Trade ($)')
    plt.title('Average Profit by Entry Source')

    # Add profit values on top of bars
    for i, v in enumerate(avg_profits):
        plt.text(i, v + (5 if v >= 0 else -15), f'${v:.2f}',
                 ha='center', va='bottom' if v >= 0 else 'top')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def visualize_trade_performance_by_source(backtest_result, save_path):
    """
    Create visualizations comparing detailed performance metrics by trade source
    """
    # Extract trade analysis data by source
    composite_metrics = backtest_result['trade_analysis']['composite']
    h1_metrics = backtest_result['trade_analysis']['1h']
    h4_metrics = backtest_result['trade_analysis']['4h']

    # Prepare data for bar charts
    sources = ['Composite', '1H', '4H']

    # Create a more detailed figure
    plt.figure(figsize=(15, 15))

    # 1. Number of Trades
    plt.subplot(3, 2, 1)
    trade_counts = [
        composite_metrics['total_trades'],
        h1_metrics['total_trades'],
        h4_metrics['total_trades']
    ]
    plt.bar(sources, trade_counts)
    plt.title('Number of Trades by Source')
    plt.ylabel('Count')

    # 2. Win Rate
    plt.subplot(3, 2, 2)
    win_rates = [
        composite_metrics['percent_profitable'],
        h1_metrics['percent_profitable'],
        h4_metrics['percent_profitable']
    ]
    plt.bar(sources, win_rates, color='lightgreen')
    plt.title('Win Rate by Source')
    plt.ylabel('Win Rate %')
    plt.axhline(y=50, color='r', linestyle='--')  # Reference line at 50%

    # 3. Average Win vs Loss
    plt.subplot(3, 2, 3)
    avg_wins = [
        composite_metrics['avg_win'] if composite_metrics['winning_trades'] > 0 else 0,
        h1_metrics['avg_win'] if h1_metrics['winning_trades'] > 0 else 0,
        h4_metrics['avg_win'] if h4_metrics['winning_trades'] > 0 else 0
    ]

    avg_losses = [
        -composite_metrics['avg_loss'] if composite_metrics['losing_trades'] > 0 else 0,
        -h1_metrics['avg_loss'] if h1_metrics['losing_trades'] > 0 else 0,
        -h4_metrics['avg_loss'] if h4_metrics['losing_trades'] > 0 else 0
    ]

    x = np.arange(len(sources))
    width = 0.35

    plt.bar(x - width / 2, avg_wins, width, label='Avg Win', color='green')
    plt.bar(x + width / 2, avg_losses, width, label='Avg Loss', color='red')
    plt.title('Average Win vs. Loss by Source')
    plt.ylabel('Amount ($)')
    plt.xticks(x, sources)
    plt.legend()

    # 4. Win/Loss Ratio
    plt.subplot(3, 2, 4)
    win_loss_ratios = [
        composite_metrics['ratio_avg_win_loss'] if not np.isinf(composite_metrics['ratio_avg_win_loss']) else 0,
        h1_metrics['ratio_avg_win_loss'] if not np.isinf(h1_metrics['ratio_avg_win_loss']) else 0,
        h4_metrics['ratio_avg_win_loss'] if not np.isinf(h4_metrics['ratio_avg_win_loss']) else 0
    ]
    plt.bar(sources, win_loss_ratios, color='orange')
    plt.title('Win/Loss Ratio by Source')
    plt.ylabel('Ratio')
    plt.axhline(y=1, color='r', linestyle='--')  # Reference line at 1.0

    # 5. Largest Win and Loss
    plt.subplot(3, 2, 5)
    largest_wins = [
        composite_metrics['largest_win'],
        h1_metrics['largest_win'],
        h4_metrics['largest_win']
    ]

    largest_losses = [
        composite_metrics['largest_loss'],
        h1_metrics['largest_loss'],
        h4_metrics['largest_loss']
    ]

    plt.bar(x - width / 2, largest_wins, width, label='Largest Win', color='darkgreen')
    plt.bar(x + width / 2, largest_losses, width, label='Largest Loss', color='darkred')
    plt.title('Largest Win/Loss by Source')
    plt.ylabel('Amount ($)')
    plt.xticks(x, sources)
    plt.legend()

    # 6. Trade Duration Analysis
    plt.subplot(3, 2, 6)

    # Group trades by source
    composite_trades = [t for t in backtest_result['all_trades'] if t['entry_source'] == 'composite']
    h1_trades = [t for t in backtest_result['all_trades'] if t['entry_source'] == '1h']
    h4_trades = [t for t in backtest_result['all_trades'] if t['entry_source'] == '4h']

    # Calculate average durations
    avg_durations = [
        np.mean([t['duration'] for t in composite_trades]) if composite_trades else 0,
        np.mean([t['duration'] for t in h1_trades]) if h1_trades else 0,
        np.mean([t['duration'] for t in h4_trades]) if h4_trades else 0
    ]

    plt.bar(sources, avg_durations, color='purple')
    plt.title('Average Trade Duration by Source')
    plt.ylabel('Hours')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main(df, start_year, end_year, save_dir, params:dict):
    os.makedirs(save_dir, exist_ok=True)

    # Ensure date column is datetime
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # Calculate hourly price changes and define thresholds for 1-hour bars
    df['diff_close'] = df['close'].diff()
    df.dropna(inplace=True)
    thresholds_1h = {
        '25th': df['diff_close'].quantile(0.25),
        '0': 0,
        '75th': df['diff_close'].quantile(0.75)
    }
    df['volume_sma'] = df['volume'].rolling(window=24).mean().fillna(method='bfill')
    df, df_4h = merge_4h_state(df,params)


    # Now perform backtesting for each month
    main_results = []
    initial_balance = 100000
    final_balance = 0
    total_profit = 0
    sharpe_ratios = []
    sortino_ratios = []
    profit_factors = []
    win_rates = []
    max_drawdowns = []
    atr_period = params['atr_period']
    atr_col_name = f'ATRr_{atr_period}'
    df.ta.atr(length=atr_period, append=True)
    trend_period = params['trend_period_sma']
    sma_col_name = f'SMA_{trend_period}'
    df[sma_col_name] = df['close'].rolling(window=trend_period).mean()
    df['trend'] = np.where(df['close'] > df[sma_col_name], 'Uptrend', 'Downtrend')
    all_period_trades = []
    aggregated_trade_sources = {
        "entry": {"1h": 0, "4h": 0, "composite": 0},
        "exit": {"1h": 0, "4h": 0, "composite": 0, "stop_loss": 0, "take_profit": 0, "end_of_period": 0}
    }
    master_trade_list = []

    for year in range(start_year, end_year):
        for month in range(1, 13):
            train_start = pd.Timestamp(year=year - 4, month=month, day=1, tz='UTC')
            train_end = pd.Timestamp(year=year, month=month, day=1, tz='UTC') - pd.Timedelta(days=1)
            test_start = pd.Timestamp(year=year, month=month, day=1, tz='UTC')
            test_end = pd.Timestamp(year=year, month=month, day=1, tz='UTC') + pd.DateOffset(months=1) - pd.Timedelta(
                days=1)

            test_df = df[(df['date'] >= test_start) & (df['date'] <= test_end)]
            train_df = df[(df['date'] >= train_start) & (df['date'] <= train_end)]
            if len(train_df) == 0 or len(test_df) == 0:
                continue

            # Recalculate thresholds on training data for consistency
            train_df['diff_close'] = train_df['close'].diff()
            thresholds = {
                '25th': train_df['diff_close'].quantile(0.25),
                '0': 0,
                '75th': train_df['diff_close'].quantile(0.75)
            }
            df['volume_ratio'] = df['volume'] / df['volume'].rolling(24).mean().fillna(df['volume'])
            df['volume_sma'] = df['volume'].rolling(window=24).mean().fillna(method='bfill')
            # Update states on training data
            train_df['state_1h'] = train_df.apply(
                lambda row: classify_state(
                    row['diff_close'], row['volume'], row['volume_sma'], row[atr_col_name],
                    price_state_threshold=params['price_state_threshold'],
                    volume_high_multiplier=params['volume_high_multiplier'],
                    volume_low_multiplier=params['volume_low_multiplier']
                ), axis=1
            )

            # Update composite state on training data
            num_1h_states = 4
            train_df['composite_state'] = train_df['state_1h'] * num_1h_states + train_df['state_4h']

            # Extract useful sequences from the training data for all timeframes
            h4_useful_sequences = extract_useful_sequences(train_df, params)

            # Prepare the test data similarly
            test_df['diff_close'] = test_df['close'].diff()
            test_df['volume_ratio'] = test_df['volume'] / test_df['volume'].rolling(24).mean().fillna(test_df['volume'])
            df['volume_sma'] = df['volume'].rolling(window=24).mean().fillna(method='bfill')
            test_df = df[(df['date'] >= test_start) & (df['date'] <= test_end)].copy()
            test_df['composite_state'] = test_df['state_4h'] * num_1h_states + test_df['state_4h']
            print(f"--- Debugging {year}-{month:02d} ---")
            # print(f"Found {len(composite_useful_sequences)} useful composite sequences.")
            # print(f"Found {len(h1_useful_sequences)} useful 1H sequences.")
            print(f"Found {len(h4_useful_sequences)} useful 4H sequences.")
            print("--- End Debugging ---")
            # Run backtesting with the combined signals
            backtest_result = backtest(
                df=test_df,
                h4_useful_sequences=h4_useful_sequences,
                sequence_length=params['sequence_length'],
                max_positions=1,
                initial_balance=100000,
                risk_free_rate=0.02 / 12,
                commission_rate=0.001, slippage_pct=0.0005,
                sl_multiplier = params['sl_multiplier'],
                tp_multiplier=params['tp_multiplier'],
                atr_col_name=atr_col_name
            )
            trades_with_period_info = backtest_result['all_trades']
            for trade in trades_with_period_info:
                trade['year'] = year
                trade['month'] = month

            master_trade_list.extend(trades_with_period_info)

            all_period_trades.extend(backtest_result['all_trades'])
            for source, count in backtest_result['trade_sources']['entry'].items():
                aggregated_trade_sources['entry'][source] += count

            for source, count in backtest_result['trade_sources']['exit'].items():
                if source in aggregated_trade_sources['exit']:
                    aggregated_trade_sources['exit'][source] += count
                else:
                    aggregated_trade_sources['exit'][source] = count

            # Store metrics for overall summaries
            sharpe_ratios.append(backtest_result['sharpe_ratio'])
            sortino_ratios.append(backtest_result['sortino_ratio'])
            profit_factors.append(backtest_result['profit_factor'])
            win_rates.append(backtest_result['win_rate'])
            max_drawdowns.append(backtest_result['max_drawdown'])
            total_profit+=backtest_result['total_profit']
            visualize_trade_sources(backtest_result, f"{save_dir}/trade_sources_{year}_{month:02d}.png")
            visualize_trade_performance_by_source(backtest_result,f"{save_dir}/trade_perf_by_source_{year}_{month:02d}.png")

            # Save metrics to files
            with open(f"{save_dir}/metrics_{year}_{month:02d}.txt", "w") as f:
                f.write(f"Backtest Metrics for {year}-{month:02d}:\n")
                f.write(f"Sharpe Ratio: {backtest_result['sharpe_ratio']:.4f}\n")
                f.write(f"Sortino Ratio: {backtest_result['sortino_ratio']:.4f}\n")
                f.write(f"Profit Factor: {backtest_result['profit_factor']:.4f}\n")
                f.write(f"Win Rate: {backtest_result['win_rate']:.2%}\n")
                f.write(f"Expectancy: {backtest_result['expectancy']:.4f}\n")
                f.write(f"Max Drawdown: {backtest_result['max_drawdown']:.2%}\n")
                f.write(f"Total Trades: {backtest_result['total_trades']}\n")
                f.write(f"Avg Holding Period: {backtest_result['avg_holding_period']:.2f} hours\n")
                f.write(f"Final Balance: ${backtest_result['final_balance']:.2f}\n")
                f.write(f"Total Profit: ${backtest_result['total_profit']:.2f}\n")

                ta_all = backtest_result['trade_analysis']['all']
                ta_long = backtest_result['trade_analysis']['long']
                ta_short = backtest_result['trade_analysis']['short']

                f.write("\nTrade Analysis (All):\n")
                f.write(f"  Total Trades: {ta_all['total_trades']}\n")
                f.write(f"  Winning Trades: {ta_all['winning_trades']}\n")
                f.write(f"  Losing Trades: {ta_all['losing_trades']}\n")
                f.write(f"  Percent Profitable: {ta_all['percent_profitable']:.2f}%\n")
                f.write(f"  Avg Winning Trade: ${ta_all['avg_win']:.2f}\n")
                f.write(f"  Avg Losing Trade: ${ta_all['avg_loss']:.2f}\n")
                f.write(f"  Ratio Avg Win/Loss: {ta_all['ratio_avg_win_loss']:.2f}\n")
                f.write(f"  Largest Winning Trade: ${ta_all['largest_win']:.2f} ({ta_all['largest_win_pct']:.2f}%)\n")
                f.write(f"  Largest Losing Trade: ${ta_all['largest_loss']:.2f} ({ta_all['largest_loss_pct']:.2f}%)\n")

                f.write("\nTrade Analysis (Long):\n")
                f.write(f"  Total Trades: {ta_long['total_trades']}\n")
                f.write(f"  Winning Trades: {ta_long['winning_trades']}\n")
                f.write(f"  Losing Trades: {ta_long['losing_trades']}\n")
                f.write(f"  Percent Profitable: {ta_long['percent_profitable']:.2f}%\n")
                f.write(f"  Avg Winning Trade: ${ta_long['avg_win']:.2f}\n")
                f.write(f"  Avg Losing Trade: ${ta_long['avg_loss']:.2f}\n")
                f.write(f"  Ratio Avg Win/Loss: {ta_long['ratio_avg_win_loss']:.2f}\n")
                f.write(f"  Largest Winning Trade: ${ta_long['largest_win']:.2f} ({ta_long['largest_win_pct']:.2f}%)\n")
                f.write(
                    f"  Largest Losing Trade: ${ta_long['largest_loss']:.2f} ({ta_long['largest_loss_pct']:.2f}%)\n")

                f.write("\nTrade Analysis (Short):\n")
                f.write(f"  Total Trades: {ta_short['total_trades']}\n")
                f.write(f"  Winning Trades: {ta_short['winning_trades']}\n")
                f.write(f"  Losing Trades: {ta_short['losing_trades']}\n")
                f.write(f"  Percent Profitable: {ta_short['percent_profitable']:.2f}%\n")
                f.write(f"  Avg Winning Trade: ${ta_short['avg_win']:.2f}\n")
                f.write(f"  Avg Losing Trade: ${ta_short['avg_loss']:.2f}\n")
                f.write(f"  Ratio Avg Win/Loss: {ta_short['ratio_avg_win_loss']:.2f}\n")
                f.write(
                    f"  Largest Winning Trade: ${ta_short['largest_win']:.2f} ({ta_short['largest_win_pct']:.2f}%)\n")
                f.write(
                    f"  Largest Losing Trade: ${ta_short['largest_loss']:.2f} ({ta_short['largest_loss_pct']:.2f}%)\n")
            main_results.append({
                "year": year,
                "month": month,
                "h4_useful_sequences": h4_useful_sequences,
                "backtest_result": backtest_result
            })

            # Create enhanced visualization with performance metrics
            plt.figure(figsize=(15, 10))

            # Price and signals subplot
            plt.subplot(2, 1, 1)
            plt.plot(test_df['date'], test_df['close'], label="Price", color="blue", alpha=0.6)
            plt.scatter(backtest_result['buy_times'], backtest_result['buy_prices'],
                        color="green", marker="^", label="Buy Signal", alpha=0.8)
            plt.scatter(backtest_result['sell_times'], backtest_result['sell_prices'],
                        color="red", marker="v", label="Sell Signal", alpha=0.8)
            plt.title(f"Backtest Results: {year}-{month:02d}")
            plt.ylabel("Price")
            plt.legend()
            plt.grid(alpha=0.3)

            # Portfolio equity curve
            plt.subplot(2, 1, 2)
            plt.plot(range(len(backtest_result['portfolio_history'])),
                     backtest_result['portfolio_history'], label="Equity Curve", color="green")
            plt.axhline(y=100000, color='gray', linestyle='--', label="Initial Balance")
            plt.title(f"Portfolio Performance (PF: {backtest_result['profit_factor']:.2f}, " +
                      f"Win Rate: {backtest_result['win_rate']:.2%}, " +
                      f"Max DD: {backtest_result['max_drawdown']:.2%})")
            plt.xlabel("Time")
            plt.ylabel("Portfolio Value ($)")
            plt.legend()
            plt.grid(alpha=0.3)

            plt.tight_layout()
            plt.savefig(f"{save_dir}/backtest_{year}_{month:02d}.png")
            plt.close()

            # Save portfolio history
            pd.DataFrame({'portfolio_value': backtest_result['portfolio_history']}).to_csv(
                f"{save_dir}/portfolio_{year}_{month:02d}.csv", index=False)

            # Save useful sequences
            with open(f"{save_dir}/useful_sequences_{year}_{month:02d}.txt", "w") as f:
                # f.write("==== COMPOSITE SEQUENCES ====\n")
                # for sequence, transitions in composite_useful_sequences.items():
                #     f.write(f"{sequence}: {transitions}\n")
                # f.write("\n==== 1H SEQUENCES ====\n")
                # for sequence, transitions in h1_useful_sequences.items():
                #     f.write(f"{sequence}: {transitions}\n")
                f.write("\n==== 4H SEQUENCES ====\n")
                for sequence, transitions in h4_useful_sequences.items():
                    f.write(f"{sequence}: {transitions}\n")
    if master_trade_list:
        all_trades_df = pd.DataFrame(master_trade_list)

        # Save the comprehensive trade log to a CSV for easy access later
        all_trades_df.to_csv(f"{save_dir}/master_trade_log.csv", index=False)

        print("\n--- Master Trade Analysis ---")
        print(f"Total Trades Executed Across All Periods: {len(all_trades_df)}")
        print(all_trades_df.head())

        # Now you can perform powerful, high-level analysis...
        perform_master_analysis(all_trades_df)
    # Calculate and write overall average metrics
    avg_sharpe_ratio = np.mean([sr for sr in sharpe_ratios if not np.isnan(sr)])
    avg_sortino_ratio = np.mean([sr for sr in sortino_ratios if not np.isnan(sr)])
    avg_profit_factor = np.mean([pf for pf in profit_factors if not np.isnan(pf) and pf != float('inf')])
    avg_win_rate = np.mean([wr for wr in win_rates if not np.isnan(wr)])
    avg_max_drawdown = np.mean([md for md in max_drawdowns if not np.isnan(md)])

    with open(f"{save_dir}/average_metrics.txt", "w") as f:
        f.write(f"Average Sharpe Ratio: {avg_sharpe_ratio:.4f}\n")
        f.write(f"Average Sortino Ratio: {avg_sortino_ratio:.4f}\n")
        f.write(f"Average Profit Factor: {avg_profit_factor:.4f}\n")
        f.write(f"Average Win Rate: {avg_win_rate:.2%}\n")
        f.write(f"Average Max Drawdown: {avg_max_drawdown:.2%}\n")

    final_metrics = {'profit_factor': avg_profit_factor, 'total_profit': total_profit, 'sharpe_ratio': avg_sharpe_ratio,'win_rate':avg_win_rate, 'sortino_ratio':avg_sortino_ratio}
    return final_metrics



def perform_master_analysis(df):
    """
    Performs high-level analysis on the master trade DataFrame.
    """
    print("\n--- Analysis by Exit Source ---")
    # Group by the reason the trade was closed and see the average profit for each
    exit_analysis = df.groupby('exit_source')['profit'].agg(['mean', 'sum', 'count'])
    print(exit_analysis)

    print("\n--- Analysis by Entry Source ---")
    entry_analysis = df.groupby('entry_source')['profit'].agg(['mean', 'sum', 'count'])
    print(entry_analysis)

    print("\n--- Worst Performing Months ---")
    monthly_profit = df.groupby(['year', 'month'])['profit'].sum().sort_values().head(5)
    print(monthly_profit)

    # Example of plotting the relationship between trade duration and profit
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.scatter(df['duration'], df['profit'], alpha=0.5)
    plt.title('Trade Duration vs. Profit/Loss')
    plt.xlabel('Duration (Hours)')
    plt.ylabel('Profit/Loss ($)')
    plt.axhline(0, color='red', linestyle='--')
    plt.grid(True)
    plt.savefig("duration_vs_profit.png")
    plt.show()


def run_parameter_optimization(df, start_year, end_year, save_dir):
    """
    Runs the backtest across a grid of SL and TP parameters to find the optimal set.
    """
    # 1. Define the parameter ranges to test
    sl_multipliers = [1.5, 1.75, 2.0, 2.25, 2.5]
    tp_multipliers = [2.5]

    optimization_results = []
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # Calculate hourly price changes and define thresholds for 1-hour bars
    df['diff_close'] = df['close'].diff()
    df.dropna(inplace=True)
    thresholds_1h = {
        '25th': df['diff_close'].quantile(0.25),
        '0': 0,
        '75th': df['diff_close'].quantile(0.75)
    }
    df['volume_sma'] = df['volume'].rolling(window=24).mean().fillna(method='bfill')
    # Apply enhanced state classification
    df['state_1h'] = df.apply(
        lambda x: classify_state(x['diff_close'], x['volume'], x['volume_sma'], x['ATRr_14']), axis=1
    )

    df, df_4h = merge_4h_state(df)

    # Create a composite state from the 1-hour and 4-hour states
    df['composite_state'] = list(zip(df['state_1h'], df['state_4h']))

    # Now perform backtesting for each month
    main_results = []
    sharpe_ratios = []
    sortino_ratios = []
    profit_factors = []
    win_rates = []
    max_drawdowns = []

    all_period_trades = []
    aggregated_trade_sources = {
        "entry": {"1h": 0, "4h": 0, "composite": 0},
        "exit": {"1h": 0, "4h": 0, "composite": 0, "stop_loss": 0, "take_profit": 0, "end_of_period": 0}
    }
    master_trade_list = []

    # 2. Create the nested loops for the grid search
    for sl in sl_multipliers:
        for tp in tp_multipliers:
            print(f"--- Testing Parameters: SL Multiplier = {sl}, TP Multiplier = {tp} ---")

            # For each parameter set, you run the entire monthly backtesting process
            master_trade_list = []  # Reset for each run
            for year in range(start_year, end_year):
                for month in range(1, 13):
                    train_start = pd.Timestamp(year=year - 2, month=month, day=1, tz='UTC')
                    train_end = pd.Timestamp(year=year, month=month, day=1, tz='UTC') - pd.Timedelta(days=1)
                    test_start = pd.Timestamp(year=year, month=month, day=1, tz='UTC')
                    test_end = pd.Timestamp(year=year, month=month, day=1, tz='UTC') + pd.DateOffset(
                        months=1) - pd.Timedelta(
                        days=1)

                    test_df = df[(df['date'] >= test_start) & (df['date'] <= test_end)]

                    train_df = df[(df['date'] >= train_start) & (df['date'] <= train_end)]
                    if len(train_df) == 0 or len(test_df) == 0:
                        continue

                    # Recalculate thresholds on training data for consistency
                    train_df['diff_close'] = train_df['close'].diff()
                    thresholds = {
                        '25th': train_df['diff_close'].quantile(0.25),
                        '0': 0,
                        '75th': train_df['diff_close'].quantile(0.75)
                    }
                    df['volume_ratio'] = df['volume'] / df['volume'].rolling(24).mean().fillna(df['volume'])
                    df['volume_sma'] = df['volume'].rolling(window=24).mean().fillna(method='bfill')
                    # Update states on training data
                    train_df['state_1h'] = train_df.apply(
                        lambda x: classify_state(x['diff_close'], x['volume'], x['volume_sma'], x['ATRr_14']), axis=1
                    )

                    # Update composite state on training data
                    num_1h_states = 4
                    train_df['composite_state'] = train_df['state_1h'] * num_1h_states + train_df['state_4h']

                    # Extract useful sequences from the training data for all timeframes
                    h4_useful_sequences = extract_useful_sequences(
                        train_df)

                    # Prepare the test data similarly
                    test_df['diff_close'] = test_df['close'].diff()
                    test_df['volume_ratio'] = test_df['volume'] / test_df['volume'].rolling(24).mean().fillna(
                        test_df['volume'])
                    df['volume_sma'] = df['volume'].rolling(window=24).mean().fillna(method='bfill')
                    test_df['state_1h'] = test_df.apply(
                        lambda x: classify_state(x['diff_close'], x['volume'], x['volume_sma'], x['ATRr_14']), axis=1
                    )
                    test_df['composite_state'] = test_df['state_1h'] * num_1h_states + test_df['state_4h']
                    print(f"--- Debugging {year}-{month:02d} ---")
                    #print(f"Found {len(composite_useful_sequences)} useful composite sequences.")
                    #print(f"Found {len(h1_useful_sequences)} useful 1H sequences.")
                    print(f"Found {len(h4_useful_sequences)} useful 4H sequences.")
                    print("--- End Debugging ---")
                    backtest_result = backtest(
                        df=test_df,
                        # composite_useful_sequences=composite_use
                        h4_useful_sequences=h4_useful_sequences,
                        sequence_length=6,
                        max_positions=1,
                        initial_balance=100000,
                        risk_free_rate=0.02 / 12,
                        commission_rate=0.001, slippage_pct=0.0005,
                        sl_multiplier=sl,
                        tp_multiplier=tp
                    )

                    master_trade_list.extend(backtest_result['all_trades'])

            # After testing all months, calculate overall metrics for this parameter set
            if not master_trade_list:
                continue

            all_trades_df = pd.DataFrame(master_trade_list)
            total_profit = all_trades_df['profit'].sum()
            sl_trades = all_trades_df[all_trades_df['exit_source'] == 'stop_loss']
            tp_trades = all_trades_df[all_trades_df['exit_source'] == 'take_profit']
            # We need the portfolio history from each run to calculate the overall Sharpe Ratio.
            # This requires a deeper refactor. A simpler proxy is to analyze the final trade stats.
            # Let's use Profit Factor as our metric for this example, as it's easier to calculate.
            gross_profit = all_trades_df[all_trades_df['profit'] > 0]['profit'].sum()
            gross_loss = abs(all_trades_df[all_trades_df['profit'] < 0]['profit'].sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            mean_sl_loss = sl_trades['profit'].mean() if not sl_trades.empty else 0
            win_loss_ratio = len(tp_trades) / len(sl_trades) if not sl_trades.empty else float('inf')
            mean_profit_per_trade = all_trades_df['profit'].mean()
            total_profit = all_trades_df['profit'].sum()

            optimization_results.append({
                'sl_multiplier': sl,
                'mean_sl_loss': mean_sl_loss,
                'win_loss_ratio': win_loss_ratio,
                'tp_multiplier': tp,
                'mean_profit_per_trade': mean_profit_per_trade,
                'total_profit': total_profit,
                'profit_factor': profit_factor,
                'total_trades': len(all_trades_df)
            })

    # 3. Analyze the results
    if not optimization_results:
        print("No trades were executed during optimization.")
        return

    results_df = pd.DataFrame(optimization_results)

    # Save the results to a CSV
    results_df.to_csv(f"{save_dir}/optimization_results_with_signals.csv", index=False)
    print("\n--- Optimization Complete ---")
    print(results_df.sort_values(by='profit_factor', ascending=False))

    # 4. Visualize the results as a heatmap
    try:
        pivot_table = results_df.pivot(index='sl_multiplier', columns='tp_multiplier', values='profit_factor')
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="viridis")
        plt.title('Profit Factor Heatmap (SL vs. TP Multipliers)')
        plt.xlabel('Take-Profit Multiplier')
        plt.ylabel('Stop-Loss Multiplier')
        plt.savefig(f"{save_dir}/optimization_heatmap.png")
        plt.show()
    except Exception as e:
        print(f"Could not generate heatmap: {e}")


if __name__ == "__main__":

    default_params = {
        'atr_period': 18,
        'price_state_threshold': 0.774325295833127,
        'volume_high_multiplier': 2.96381982151409,
        'volume_low_multiplier': 0.76691565071116,
        'sequence_length': 5,
        'min_signal_probability': 0.610321553188911,
        'sl_multiplier': 3.2935543511297,
        'tp_multiplier': 2.04204584013095,
        'trend_period_sma': 240
    }
    dataframe = pd.read_feather('ETH_USDT-1h.feather')
    dataframe['date'] = pd.to_datetime(dataframe['date'])

    results = main(dataframe, start_year=2019, end_year=2024, save_dir="backtest_results_btc_2019_2024",params=default_params)
    #run_parameter_optimization(dataframe, start_year=2019, end_year=2024, save_dir="optimization_results_sol")