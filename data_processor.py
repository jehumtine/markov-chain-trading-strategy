import pandas as pd
import numpy as np

# --- All functions related to data manipulation and feature creation go here ---

def classify_state(diff_close, volume, volume_sma, atr,
                   price_state_threshold: float,
                   volume_high_multiplier: float,
                   volume_low_multiplier: float):
    """
    Enhanced state classification that incorporates volume.
    """
    normalized_diff = diff_close / atr if atr > 0 else 0
    if normalized_diff > price_state_threshold:
        price_state = 0
    elif normalized_diff > 0:
        price_state = 1
    elif normalized_diff > -price_state_threshold: # Symmetrical threshold
        price_state = 2
    else:
        price_state = 3

    if volume > volume_sma * volume_high_multiplier: # High volume
        volume_state = 0
    elif volume < volume_sma * volume_low_multiplier: # Low volume
        volume_state = 1
    else:
        volume_state = 2 # Average Volume

    num_volume_states = 3
    final_state = price_state * num_volume_states + volume_state
    return final_state

def determine_higher_timeframe_trend(df_4h, lookback=24):
    """
    Determine if we're in a bullish or bearish trend on higher timeframe
    """
    if len(df_4h) < lookback:
        return "NEUTRAL"  # Not enough data

    recent_data = df_4h.tail(lookback)

    # Simple moving average approach
    sma_short = recent_data['close'].rolling(6).mean().fillna(recent_data['close'])
    sma_long = recent_data['close'].rolling(12).mean().fillna(recent_data['close'])

    # Last values
    latest_short = sma_short.iloc[-1]
    latest_long = sma_long.iloc[-1]

    if latest_short > latest_long * 1.01:  # 1% buffer
        return "BULLISH"
    elif latest_short < latest_long * 0.99:  # 1% buffer
        return "BEARISH"
    else:
        return "NEUTRAL"


def merge_4h_state(df, params: dict):
    """
    Resamples to 4h and calculates state, now using dynamic parameters.
    """
    df = df.sort_values('date').reset_index(drop=True)
    df_temp = df.set_index('date').resample('4h').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).reset_index()
    df_temp.dropna(subset=['close'], inplace=True)

    # Use the atr_period from params and get the dynamic column name
    atr_period = params['atr_period']
    atr_col_name = f'ATRr_{atr_period}'
    df_temp.ta.atr(length=atr_period, append=True)

    df_temp['diff_close'] = df_temp['close'].diff()
    df_temp['volume_sma'] = df_temp['volume'].rolling(window=24).mean()
    df_temp.fillna(method='bfill', inplace=True)
    df_temp.dropna(inplace=True)

    # Call classify_state with all necessary parameters
    df_temp['state_4h'] = df_temp.apply(
        lambda row: classify_state(
            row['diff_close'], row['volume'], row['volume_sma'], row[atr_col_name], # Use dynamic ATR name
            price_state_threshold=params['price_state_threshold'],
            volume_high_multiplier=params['volume_high_multiplier'],
            volume_low_multiplier=params['volume_low_multiplier']
        ), axis=1
    )
    # Merge back to the main dataframe
    df = pd.merge_asof(
        df.sort_values('date'),
        df_temp[['date', 'state_4h']].sort_values('date'),
        on='date',
        direction='backward'
    )
    return df, df_temp

def calculate_adaptive_thresholds(df, volatility_window=24):
    """
        Calculate adaptive thresholds based on recent volatility
        """
    recent_volatility = df['diff_close'].rolling(volatility_window).std().fillna(df['diff_close'].std())
    avg_volatility = recent_volatility.mean()
    current_volatility = recent_volatility.iloc[-1]

    # Avoid division by zero
    if avg_volatility == 0:
        volatility_ratio = 1.0
    else:
        volatility_ratio = current_volatility / avg_volatility

    # Base thresholds
    base_increment = 0.025
    base_stop_loss = 0.05

    # Adjusted thresholds (with limits to prevent extreme values)
    volatility_ratio = max(0.5, min(2.0, volatility_ratio))  # Cap between 0.5 and 2.0
    adjusted_increment = base_increment * volatility_ratio
    adjusted_stop_loss = base_stop_loss * volatility_ratio

    return adjusted_increment, adjusted_stop_loss