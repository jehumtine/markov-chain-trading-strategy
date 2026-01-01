import os
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


def parse_metrics_file(filepath):
    """Parse a metrics file and extract all trade metrics."""
    with open(filepath, 'r') as file:
        content = file.read()

    # Extract the basic metrics using regex
    sharpe_match = re.search(r'Sharpe Ratio: ([0-9.-]+)', content)
    sortino_match = re.search(r'Sortino Ratio: ([0-9.-]+)', content)
    profit_factor_match = re.search(r'Profit Factor: ([0-9.]+)', content)
    win_rate_match = re.search(r'Win Rate: ([0-9.]+)%', content)
    expectancy_match = re.search(r'Expectancy: ([0-9.-]+)', content)
    drawdown_match = re.search(r'Max Drawdown: ([0-9.]+)%', content)
    total_trades_match = re.search(r'Total Trades: ([0-9]+)', content)
    holding_period_match = re.search(r'Avg Holding Period: ([0-9.]+) hours', content)
    final_balance_match = re.search(r'Final Balance: \$([0-9.]+)', content)
    total_profit_match = re.search(r'Total Profit: \$([0-9.-]+)', content)

    # Extract trade analysis metrics - All trades
    all_total_match = re.search(r'Trade Analysis \(All\):\s+Total Trades: ([0-9]+)', content)
    all_winning_match = re.search(r'Trade Analysis \(All\):\s+.*\s+Winning Trades: ([0-9]+)', content)
    all_losing_match = re.search(r'Trade Analysis \(All\):\s+.*\s+Losing Trades: ([0-9]+)', content)
    all_profitable_match = re.search(r'Trade Analysis \(All\):\s+.*\s+Percent Profitable: ([0-9.]+)%', content)
    all_avg_win_match = re.search(r'Trade Analysis \(All\):\s+.*\s+Avg Winning Trade: \$([0-9.]+)', content)
    all_avg_loss_match = re.search(r'Trade Analysis \(All\):\s+.*\s+Avg Losing Trade: \$([0-9.]+)', content)
    all_win_loss_ratio_match = re.search(r'Trade Analysis \(All\):\s+.*\s+Ratio Avg Win/Loss: ([0-9.]+)', content)
    all_largest_win_match = re.search(r'Trade Analysis \(All\):\s+.*\s+Largest Winning Trade: \$([0-9.]+)', content)
    all_largest_loss_match = re.search(r'Trade Analysis \(All\):\s+.*\s+Largest Losing Trade: \$\-([0-9.]+)', content)

    # Check if we have the basic data
    if not all([total_trades_match, final_balance_match, total_profit_match]):
        return None

    # Create metrics dictionary
    metrics = {
        'sharpe_ratio': float(sharpe_match.group(1)) if sharpe_match else None,
        'sortino_ratio': float(sortino_match.group(1)) if sortino_match else None,
        'profit_factor': float(profit_factor_match.group(1)) if profit_factor_match else None,
        'win_rate': float(win_rate_match.group(1)) if win_rate_match else None,
        'expectancy': float(expectancy_match.group(1)) if expectancy_match else None,
        'max_drawdown': float(drawdown_match.group(1)) if drawdown_match else None,
        'total_trades': int(total_trades_match.group(1)) if total_trades_match else 0,
        'holding_period': float(holding_period_match.group(1)) if holding_period_match else None,
        'final_balance': float(final_balance_match.group(1)) if final_balance_match else None,
        'total_profit': float(total_profit_match.group(1)) if total_profit_match else None,
        'initial_balance': 100000.0,  # Assuming fixed initial balance

        # All trades metrics
        'all_total': int(all_total_match.group(1)) if all_total_match else 0,
        'all_winning': int(all_winning_match.group(1)) if all_winning_match else 0,
        'all_losing': int(all_losing_match.group(1)) if all_losing_match else 0,
        'all_profitable_pct': float(all_profitable_match.group(1)) if all_profitable_match else 0,
        'all_avg_win': float(all_avg_win_match.group(1)) if all_avg_win_match else 0,
        'all_avg_loss': float(all_avg_loss_match.group(1)) if all_avg_loss_match else 0,
        'all_win_loss_ratio': float(all_win_loss_ratio_match.group(1)) if all_win_loss_ratio_match else 0,
        'all_largest_win': float(all_largest_win_match.group(1)) if all_largest_win_match else 0,
        'all_largest_loss': float(all_largest_loss_match.group(1)) if all_largest_loss_match else 0
    }

    # Calculate percentage profit for this individual month
    metrics['percentage_profit'] = (metrics['total_profit'] / metrics['initial_balance']) * 100

    return metrics


def analyze_trade_metrics(directory):
    """Analyze trade metrics from files."""
    # Dictionary to store yearly results
    yearly_results = defaultdict(lambda: {
        'months': 0,
        'total_trades': 0,
        'winning_trades': 0,
        'losing_trades': 0,
        'net_profit': 0.0,
        'month_data': [],
        'cumulative_profit': 0.0,
        'initial_investment': 100000.0
    })

    # Find and sort all metrics files
    metric_files = []
    for filename in os.listdir(directory):
        if filename.startswith('metrics_') and filename.endswith('.txt'):
            match = re.match(r'metrics_(\d{4})_(\d{2})\.txt', filename)
            if match:
                year, month = match.groups()
                filepath = os.path.join(directory, filename)
                metric_files.append((year, month, filepath))

    # Sort files by year and month
    metric_files.sort(key=lambda x: (x[0], x[1]))

    # Prepare arrays for time series plots
    dates = []
    win_rates = []
    profit_factors = []
    sharpe_ratios = []
    sortino_ratios = []
    cumulative_profits = []
    cumulative_trades = []
    cumulative_win_trades = []
    cumulative_lose_trades = []

    # Process files in chronological order
    cumulative_balance = 100000.0  # Start with initial investment
    cumulative_profit = 0.0
    all_cumulative_trades = 0
    all_cumulative_win_trades = 0
    all_cumulative_lose_trades = 0

    for year, month, filepath in metric_files:
        # Parse the metrics file
        metrics = parse_metrics_file(filepath)
        if metrics:
            # Calculate cumulative profit
            cumulative_profit += metrics['total_profit']
            cumulative_balance = 100000.0 + cumulative_profit

            # Calculate cumulative percentage profit
            cumulative_percentage = (cumulative_profit / 100000.0) * 100

            # Update trade counts
            all_cumulative_trades += metrics['total_trades']
            all_cumulative_win_trades += metrics['all_winning']
            all_cumulative_lose_trades += metrics['all_losing']

            # Update yearly statistics
            yearly_results[year]['months'] += 1
            yearly_results[year]['total_trades'] += metrics['total_trades']
            yearly_results[year]['winning_trades'] += metrics['all_winning']
            yearly_results[year]['losing_trades'] += (metrics["total_trades"]-metrics["all_winning"])
            yearly_results[year]['net_profit'] += metrics['total_profit']
            yearly_results[year]['cumulative_profit'] = cumulative_profit

            yearly_results[year]['month_data'].append({
                'month': month,
                'profit': metrics['total_profit'],
                'percentage_profit': metrics['percentage_profit'],
                'win_rate': metrics['win_rate'],
                'total_trades': metrics['total_trades'],
                'winning_trades': metrics['all_winning'],
                'losing_trades': metrics['total_trades']-metrics['all_winning'],
                'sharpe_ratio': metrics['sharpe_ratio'],
                'sortino_ratio': metrics['sortino_ratio'],
                'profit_factor': metrics['profit_factor'],
                'cumulative_profit': cumulative_profit,
                'cumulative_balance': cumulative_balance,
                'cumulative_percentage': cumulative_percentage
            })

            # Append data for time series plots
            date_obj = datetime.strptime(f"{year}-{month}-01", "%Y-%m-%d")
            dates.append(date_obj)
            win_rates.append(metrics['win_rate'])
            profit_factors.append(metrics['profit_factor'])
            sharpe_ratios.append(metrics['sharpe_ratio'])
            sortino_ratios.append(metrics['sortino_ratio'])
            cumulative_profits.append(cumulative_profit)
            cumulative_trades.append(all_cumulative_trades)
            cumulative_win_trades.append(all_cumulative_win_trades)
            cumulative_lose_trades.append(all_cumulative_lose_trades)

    # Print the results
    print("\n===== COMPREHENSIVE TRADE METRICS ANALYSIS =====\n")

    for year, data in sorted(yearly_results.items()):
        # Calculate yearly win rate
        yearly_win_rate = (data['winning_trades'] / data['total_trades'] * 100) if data['total_trades'] > 0 else 0

        # Calculate yearly percentage profit
        yearly_percentage = (data['net_profit'] / 100000.0) * 100

        print(f"Year: {year}")
        print(f"  Months Analyzed: {data['months']}")
        print(f"  Total Trades: {data['total_trades']}")
        print(f"  Winning Trades: {data['winning_trades']} ({yearly_win_rate:.2f}%)")
        print(f"  Losing Trades: {data['losing_trades']} ({100 - yearly_win_rate:.2f}%)")
        print(f"  Win/Loss Ratio: {data['winning_trades'] / max(1, data['losing_trades']):.2f}")
        print(f"  Net Profit for Year: ${data['net_profit']:.2f} ({yearly_percentage:.2f}%)")
        print(f"  Cumulative Profit (End of Year): ${data['cumulative_profit']:.2f}")
        print(f"  Year Status: {'PROFITABLE' if data['net_profit'] > 0 else 'UNPROFITABLE'}")
        print("\n" + "-" * 50 + "\n")

    # Print overall summary
    total_years = len(yearly_results)
    profitable_years = sum(1 for year, data in yearly_results.items() if data['net_profit'] > 0)
    unprofitable_years = total_years - profitable_years

    total_months = sum(data['months'] for data in yearly_results.values())
    total_trades = sum(data['total_trades'] for data in yearly_results.values())
    total_winning_trades = sum(data['winning_trades'] for data in yearly_results.values())
    total_losing_trades = total_trades-total_winning_trades
    overall_win_rate = (total_winning_trades / total_trades * 100) if total_trades > 0 else 0

    print("Overall Trading Summary:")
    print(f"  Total Years Analyzed: {total_years}")
    print(f"  Total Months Analyzed: {total_months}")
    print(f"  Profitable Years: {profitable_years}")
    print(f"  Unprofitable Years: {unprofitable_years}")
    print(f"  Total Trades: {total_trades}")
    print(f"  Winning Trades: {total_winning_trades} ({overall_win_rate:.2f}%)")
    print(f"  Losing Trades: {total_losing_trades} ({100 - overall_win_rate:.2f}%)")
    print(f"  Win/Loss Ratio: {total_winning_trades / max(1, total_losing_trades):.2f}")
    print(f"  Initial Investment: $100,000.00")
    print(f"  Final Balance: ${cumulative_balance:.2f}")
    print(f"  Total Cumulative Profit: ${cumulative_profit:.2f} ({cumulative_percentage:.2f}%)")
    print(f"  Average Monthly Return: ${cumulative_profit / max(1, total_months):.2f}")
    print(f"  Average Profit per Trade: ${cumulative_profit / max(1, total_trades):.2f}")

    # Calculate averages for key metrics
    avg_win_rate = np.mean([r for r in win_rates if r is not None]) if win_rates else 0
    avg_profit_factor = np.mean([f for f in profit_factors if f is not None]) if profit_factors else 0
    avg_sharpe = np.mean([s for s in sharpe_ratios if s is not None]) if sharpe_ratios else 0
    avg_sortino = np.mean([s for s in sortino_ratios if s is not None]) if sortino_ratios else 0

    print("\nAverage Performance Metrics:")
    print(f"  Average Win Rate: {avg_win_rate:.2f}%")
    print(f"  Average Profit Factor: {avg_profit_factor:.2f}")
    print(f"  Average Sharpe Ratio: {avg_sharpe:.2f}")
    print(f"  Average Sortino Ratio: {avg_sortino:.2f}")

    # Create visualizations
    # 1. Win Rate and Profit Factor Over Time
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 1, 1)
    plt.plot(dates, win_rates, 'b-', marker='o', label='Win Rate (%)')
    plt.axhline(y=avg_win_rate, color='b', linestyle='--', alpha=0.7, label=f'Avg Win Rate: {avg_win_rate:.2f}%')
    plt.title('Monthly Win Rate')
    plt.xlabel('Date')
    plt.ylabel('Win Rate (%)')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(dates, profit_factors, 'g-', marker='o', label='Profit Factor')
    plt.axhline(y=avg_profit_factor, color='g', linestyle='--', alpha=0.7,
                label=f'Avg Profit Factor: {avg_profit_factor:.2f}')
    plt.title('Monthly Profit Factor')
    plt.xlabel('Date')
    plt.ylabel('Profit Factor')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig('win_rate_profit_factor.png')

    # 2. Sharpe and Sortino Ratios Over Time
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 1, 1)
    plt.plot(dates, sharpe_ratios, 'b-', marker='o', label='Sharpe Ratio')
    plt.axhline(y=avg_sharpe, color='b', linestyle='--', alpha=0.7, label=f'Avg Sharpe: {avg_sharpe:.2f}')
    plt.title('Monthly Sharpe Ratio')
    plt.xlabel('Date')
    plt.ylabel('Sharpe Ratio')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(dates, sortino_ratios, 'g-', marker='o', label='Sortino Ratio')
    plt.axhline(y=avg_sortino, color='g', linestyle='--', alpha=0.7, label=f'Avg Sortino: {avg_sortino:.2f}')
    plt.title('Monthly Sortino Ratio')
    plt.xlabel('Date')
    plt.ylabel('Sortino Ratio')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig('sharpe_sortino_ratios.png')

    # 3. Cumulative Profit Over Time
    plt.figure(figsize=(12, 6))
    plt.plot(dates, cumulative_profits, 'r-', marker='o')
    plt.title('Cumulative Profit Over Time')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Profit ($)')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.tight_layout()
    plt.savefig('cumulative_profit.png')

    # 4. Cumulative Trades Over Time
    plt.figure(figsize=(12, 6))
    plt.plot(dates, cumulative_trades, 'b-', marker='o', label='Total Trades')
    plt.plot(dates, cumulative_win_trades, 'g-', marker='s', label='Winning Trades')
    plt.plot(dates, cumulative_lose_trades, 'r-', marker='^', label='Losing Trades')
    plt.title('Cumulative Trade Count Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Trades')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('cumulative_trades.png')

    print("\nVisualization graphs have been saved:")
    print("  - win_rate_profit_factor.png")
    print("  - sharpe_sortino_ratios.png")
    print("  - cumulative_profit.png")
    print("  - cumulative_trades.png")


if __name__ == "__main__":
    # Replace with your directory path
    metrics_directory = "backtest_results_sol_2019_2024"

    if os.path.exists(metrics_directory):
        analyze_trade_metrics(metrics_directory)
    else:
        print(f"Error: Directory '{metrics_directory}' not found.")
        print("Please update the 'metrics_directory' variable with the correct path to your metrics files.")