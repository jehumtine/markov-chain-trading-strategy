import numpy as np
import pandas as pd
from collections import defaultdict

from data_processor import calculate_adaptive_thresholds

NUM_VOLUME_STATES = 3


def calculate_dynamic_exits(entry_price, atr_at_entry, sl_multiplier=2.0, tp_multiplier=3.0):
    """
        Calculates dynamic stop-loss and take-profit levels based on ATR.

        Args:
            entry_price (float): The price at which the position was entered.
            atr_at_entry (float): The ATR value at the time of entry.
            sl_multiplier (float): The multiplier for the stop-loss.
            tp_multiplier (float): The multiplier for the take-profit.
        Returns:
            tuple: A tuple containing the (stop_loss_price, take_profit_price).
        """
    if atr_at_entry <= 0:
        # Fallback to a fixed percentage if ATR is zero or negative
        stop_loss_price = entry_price * 0.95
        take_profit_price = entry_price * 1.05
        return stop_loss_price, take_profit_price

    stop_loss_price = entry_price - (atr_at_entry * sl_multiplier)
    take_profit_price = entry_price + (atr_at_entry * tp_multiplier)

    return stop_loss_price, take_profit_price

def interpret_state_direction(state):
    """
    Decodes a composite state back into its core price direction.
    Returns 'BULLISH', 'BEARISH', or 'NEUTRAL'.
    """
    # This is the reverse calculation of:
    # final_state = price_state * NUM_VOLUME_STATES + volume_state
    price_state = state // NUM_VOLUME_STATES  # Integer division gives the original price state
    if price_state in (0, 1):  # 0=Strong Uptrend, 1=Mild Uptrend
        return 'BULLISH'
    elif price_state in (2, 3):  # 2=Mild Downtrend, 3=Strong Downtrend
        return 'BEARISH'
    else:
        return 'NEUTRAL'  # Fallback for any unexpected state

def evaluate_trading_signals(h4_signals,composite_signals=None, h1_signals=None,  h1_sequence=None, h4_sequence=None, composite_sequence=None):
    """
        Evaluate signals based on presence in any sequence (H1, H4, Composite)
        Returns buy_signal and sell_signal booleans
        """
    buy_signal = False
    sell_signal = False
    signal_source = None
    # # Check composite sequences
    # if composite_sequence in composite_signals:
    #     for next_state in composite_signals[composite_sequence]:
    #         direction = interpret_state_direction(next_state)
    #         if direction == 'BULLISH':
    #             buy_signal = True
    #             signal_source = "composite"
    #         elif direction == 'BEARISH':
    #             sell_signal = True
    #             signal_source = "composite"
    #
    # Check 1H sequences
    # if h1_sequence in h1_signals:
    #     for next_state in h1_signals[h1_sequence]:
    #         direction = interpret_state_direction(next_state)  # Decode the state
    #         if direction == 'BULLISH':
    #             buy_signal = True
    #             signal_source = "1h"
    #         elif direction == 'BEARISH':
    #             sell_signal = True
    #             signal_source = "1h"

    # Check 4H sequences
    if h4_sequence in h4_signals:
        for next_state in h4_signals[h4_sequence]:
            direction = interpret_state_direction(next_state)  # Decode the state
            if direction == 'BULLISH':
                buy_signal = True
                signal_source = "4h"
            elif direction == 'BEARISH':
                sell_signal = True
                signal_source = "4h"

    return buy_signal, sell_signal, signal_source

def backtest(df, h4_useful_sequences,
             sequence_length: int,
             sl_multiplier: float,
             tp_multiplier: float,
             atr_col_name="",
             initial_balance=100000, max_positions=3, risk_free_rate=0.0,
             commission_rate=0.001, slippage_pct=0.0005,
             ):
    positions = []
    current_balance = initial_balance
    portfolio_history = []
    portfolio_returns = []
    buy_times, buy_prices = [], []
    sell_times, sell_prices = [], []
    all_trades = []  # Track all trades with details
    trade_durations = []

    trade_sources = {
        "entry":{"1h":0,"4h":0, "composite":0},
        "exit":{"1h":0,"4h":0, "composite":0, "stop_loss":0,"take_profit":0},
    }

    stop_loss = 0.025
    for i in range(len(df) - sequence_length):
        current_composite_sequence = tuple(df['composite_state'].iloc[i:i + sequence_length])
        current_h1_sequence = tuple(df['state_4h'].iloc[i:i + sequence_length])
        current_h4_sequence = tuple(df['state_4h'].iloc[i:i + sequence_length])
        current_price = df['close'].iloc[i + sequence_length]
        current_date = df['date'].iloc[i + sequence_length]

        recent_df = df.iloc[max(0, i - 24):i + sequence_length]
        price_increment_threshold, stop_loss_threshold = calculate_adaptive_thresholds(recent_df)
        take_profit = 0.05
        current_atr = df[atr_col_name].iloc[i + sequence_length] # Get the ATR for the current candle
        current_high = df['high'].iloc[i + sequence_length]
        current_low = df['low'].iloc[i + sequence_length]

        buy_signal, sell_signal,signal_source = evaluate_trading_signals(
            #composite_useful_sequences,
            #h1_useful_sequences,
            h4_signals=h4_useful_sequences,
            h1_sequence=current_h1_sequence,
            h4_sequence=current_h4_sequence,
            composite_sequence=current_composite_sequence,
        )

        positions_to_remove = []
        for idx, position in enumerate(positions):
            close_return = (current_price / position['buy_price'] - 1)
            low_return = (current_low / position['buy_price'] - 1)
            high_return = (current_high / position['buy_price'] - 1)
            exit_source = None
            exit_price = 0
            # Determine exit reason
            if current_low <= position['stop_loss_price']:
                exit_source = "stop_loss"
                exit_price = position['stop_loss_price']
            elif current_high >= position['take_profit_price']:
                exit_source = "take_profit"
                exit_price = position['take_profit_price']
            elif sell_signal and current_price > position['buy_price']:
                exit_source = signal_source
                exit_price = current_price
            if exit_source:
                sell_price = current_price
                slipped_exit_price = exit_price * (1 - slippage_pct)
                sale_value = position['quantity'] * slipped_exit_price
                commission_fee = sale_value * commission_rate
                current_balance += sale_value - commission_fee
                trade_pl = (sale_value) - (position['buy_price'] * position['quantity']) - position['entry_commission']
                return_percent = (slipped_exit_price / position['buy_price'] - 1) * 100
                trade_duration = (current_date - position['buy_date']).total_seconds() / 3600

                # Track exit signal source
                if exit_source in ["1h", "4h", "composite"]:
                    trade_sources["exit"][exit_source] += 1
                elif exit_source == "stop_loss":
                    trade_sources["exit"]["stop_loss"] += 1
                elif exit_source == "take_profit":
                    trade_sources["exit"]["take_profit"] += 1

                all_trades.append({
                    'direction': 'long',
                    'profit': trade_pl,
                    'return_pct': return_percent,
                    'duration': trade_duration,
                    'entry_price': position['buy_price'],
                    'exit_price': slipped_exit_price,
                    'entry_time': position['buy_date'],
                    'exit_time': current_date,
                    'entry_source': position['entry_source'],
                    'exit_source': exit_source
                })
                trade_durations.append(trade_duration)
                positions_to_remove.append(idx)

        for idx in sorted(positions_to_remove, reverse=True):
            del positions[idx]

        if buy_signal:
            trend_now = df['trend'].iloc[i + sequence_length]
            print(f"DEBUG at {current_date}: Buy signal found. Market Trend is: {trend_now}.")
            if trend_now != 'Uptrend':
                print("--> RESULT: Trade ignored due to trend filter.")

        if buy_signal and len(positions) < max_positions and current_balance > 0:
            print(f"BUY SIGNAL FOUND at {current_date} from source: {signal_source}")
            position_size = (current_balance / (max_positions - len(positions))) * 0.99
            slipped_buy_price = current_price * (1 + slippage_pct)
            if slipped_buy_price <= 0:
                continue
            position_quantity = position_size / slipped_buy_price
            position_cost = position_quantity * slipped_buy_price
            commission_fee = position_cost * commission_rate
            # Track entry signal source
            trade_sources["entry"][signal_source] += 1

            if position_cost + commission_fee > current_balance:
                continue
            stop_loss_price, take_profit_price = calculate_dynamic_exits(
                entry_price=slipped_buy_price,
                atr_at_entry=current_atr,
                sl_multiplier=sl_multiplier,
                tp_multiplier=tp_multiplier,
            )
            print(
                f"New Position at {current_date}: Entry={slipped_buy_price:.2f}, SL={stop_loss_price:.2f}, TP={take_profit_price:.2f}, ATR={current_atr:.2f}")
            positions.append({
                'buy_price': slipped_buy_price,
                'quantity': position_quantity,
                'buy_date': current_date,
                'entry_source': signal_source,  # Store the source of the entry signal
                'entry_commission': commission_fee,
                'stop_loss_price': stop_loss_price,
                'take_profit_price': take_profit_price
            })
            current_balance -= (position_cost + commission_fee)
            buy_times.append(current_date)
            buy_prices.append(slipped_buy_price)

        portfolio_value = current_balance + sum(position['quantity'] * current_price for position in positions)
        portfolio_history.append(portfolio_value)

    for position in positions:
        # Use the last price of the dataframe as the exit price
        exit_price = df['close'].iloc[-1]

        # --- Apply the same robust accounting as mid-loop exits ---
        slipped_exit_price = exit_price * (1 - slippage_pct)  # Apply slippage
        sale_value = position['quantity'] * slipped_exit_price
        exit_commission_fee = sale_value * commission_rate

        # Update balance correctly
        current_balance += sale_value - exit_commission_fee

        # Calculate P&L correctly, including the stored entry commission
        trade_pl = (sale_value) - (position['buy_price'] * position['quantity']) - position['entry_commission']

        return_percent = (slipped_exit_price / position['buy_price'] - 1) * 100
        trade_duration = (df['date'].iloc[-1] - position['buy_date']).total_seconds() / 3600

        all_trades.append({
            'direction': 'long',
            'profit': trade_pl,
            'return_pct': return_percent,
            'duration': trade_duration,
            'entry_price': position['buy_price'],
            'exit_price': slipped_exit_price,
            'entry_time': position['buy_date'],
            'exit_time': df['date'].iloc[-1],
            'entry_source': position['entry_source'],
            'exit_source': 'end_of_period'
        })
        trade_durations.append(trade_duration)

    final_balance = current_balance
    total_profit = final_balance - initial_balance

    def calculate_trade_metrics(trades):
        if not trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'percent_profitable': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'ratio_avg_win_loss': 0.0,
                'largest_win': 0.0,
                'largest_win_pct': 0.0,
                'largest_loss': 0.0,
                'largest_loss_pct': 0.0,
            }
        winning = [t for t in trades if t['profit'] > 0]
        losing = [t for t in trades if t['profit'] <= 0]
        total = len(trades)
        percent_profitable = (len(winning) / total) * 100 if total > 0 else 0.0
        avg_win = np.mean([t['profit'] for t in winning]) if winning else 0.0
        avg_loss = np.mean([abs(t['profit']) for t in losing]) if losing else 0.0
        ratio_avg_win_loss = avg_win / avg_loss if avg_loss != 0 else float('inf') if avg_win > 0 else 0.0
        largest_win = max([t['profit'] for t in winning], default=0.0)
        largest_win_pct = max([t['return_pct'] for t in winning], default=0.0)
        largest_loss = min([t['profit'] for t in losing], default=0.0)
        largest_loss_pct = min([t['return_pct'] for t in losing], default=0.0)
        return {
            'total_trades': total,
            'winning_trades': len(winning),
            'losing_trades': len(losing),
            'percent_profitable': percent_profitable,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'ratio_avg_win_loss': ratio_avg_win_loss,
            'largest_win': largest_win,
            'largest_win_pct': largest_win_pct,
            'largest_loss': largest_loss,
            'largest_loss_pct': largest_loss_pct,
        }

    # Calculate metrics by signal source
    composite_trades = [t for t in all_trades if t['entry_source'] == 'composite']
    h1_trades = [t for t in all_trades if t['entry_source'] == '1h']
    h4_trades = [t for t in all_trades if t['entry_source'] == '4h']

    composite_metrics = calculate_trade_metrics(composite_trades)
    h1_metrics = calculate_trade_metrics(h1_trades)
    h4_metrics = calculate_trade_metrics(h4_trades)

    all_metrics = calculate_trade_metrics(all_trades)
    long_metrics = calculate_trade_metrics([t for t in all_trades if t['direction'] == 'long'])
    short_metrics = calculate_trade_metrics([t for t in all_trades if t['direction'] == 'short'])

    # Calculate win rate
    total_trades = all_metrics['total_trades']
    win_rate = all_metrics['winning_trades'] / total_trades if total_trades > 0 else 0
    gross_profit = sum(t['profit'] for t in all_trades if t['profit'] > 0)
    gross_loss = abs(sum(t['profit'] for t in all_trades if t['profit'] <= 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    expectancy = (gross_profit - gross_loss) / total_trades if total_trades > 0 else 0
    avg_holding_period = np.mean(trade_durations) if trade_durations else 0

    cumulative_returns = np.array(portfolio_history) / initial_balance
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = (peak - cumulative_returns) / peak
    max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0

    sharpe_ratio = calculate_sharpe_ratio([(p / initial_balance - 1) for p in portfolio_history], risk_free_rate)
    sortino_ratio = calculate_sortino_ratio([(p / initial_balance - 1) for p in portfolio_history], risk_free_rate)

    return {
        "final_balance": final_balance,
        "total_profit": total_profit,
        "portfolio_history": portfolio_history,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "profit_factor": profit_factor,
        "win_rate": win_rate,
        "expectancy": expectancy,
        "max_drawdown": max_drawdown,
        "total_trades": total_trades,
        "avg_holding_period": avg_holding_period,
        "buy_times": buy_times,
        "buy_prices": buy_prices,
        "sell_times": sell_times,
        "sell_prices": sell_prices,
        "trade_analysis": {
            "all": all_metrics,
            "long": long_metrics,
            "short": short_metrics,
            "composite": composite_metrics,
            "1h": h1_metrics,
            "4h": h4_metrics
        },
        "trade_sources": trade_sources,
        "all_trades": all_trades  # Include the full trade list for detailed analysis
    }

def calculate_sharpe_ratio(returns, risk_free_rate=2.0):
    if len(returns) == 0:
        return np.nan
    avg_return = np.mean(returns)
    std_return = np.std(returns)
    return (avg_return - risk_free_rate) / std_return * np.sqrt(12) if std_return > 0 else 0

def calculate_sortino_ratio(returns, risk_free_rate=2.0):
    if len(returns) == 0:
        return np.nan
    avg_return = np.mean(returns)
    downside_returns = [r for r in returns if r < risk_free_rate]
    if len(downside_returns) == 0:
        downside_returns = returns
    downside_std = np.std(downside_returns)
    return (avg_return - risk_free_rate) / downside_std * np.sqrt(12) if downside_std > 0 else 0