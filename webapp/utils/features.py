"""
Features - Tạo technical indicators cho dữ liệu cổ phiếu
"""

import numpy as np
import pandas as pd
import pandas_ta as ta


# Giải thích ý nghĩa tài chính của từng feature
FEATURE_EXPLANATIONS = {
    'return_1d': 'Momentum 1 ngày',
    'return_3d': 'Momentum 3 ngày',
    'return_5d': 'Momentum tuần',
    'return_10d': 'Momentum 2 tuần',
    'return_20d': 'Momentum tháng',
    'return_60d': 'Momentum quý',
    'roc_5': 'Tốc độ thay đổi giá 5d',
    'roc_10': 'Tốc độ thay đổi 10d',
    'roc_20': 'Tốc độ thay đổi 20d',
    'rsi_14': 'RSI 14 — quá mua/bán',
    'rsi_7': 'RSI 7 — ngắn hạn',
    'macd': 'MACD — xu hướng momentum',
    'macd_signal': 'MACD Signal',
    'macd_hist': 'MACD Histogram — sức mạnh xu hướng',
    'stoch_k': 'Stochastic %K — dao động giá',
    'stoch_d': 'Stochastic %D — tín hiệu dao động',
    'willr_14': 'Williams %R — quá mua/bán',
    'cci_20': 'CCI — chỉ số kênh hàng hóa',
    'atr_ratio': 'ATR Ratio — biến động tương đối',
    'bb_pct_b': 'Bollinger %B — vị trí trong dải',
    'bb_width': 'Bollinger Width — độ rộng dải',
    'std_ratio': 'Std Ratio — biến động giá',
    'vol_ratio_5_20': 'Volume Ratio — thay đổi khối lượng',
    'vol_change_1d': 'Volume Change 1d',
    'vol_change_5d': 'Volume Change 5d',
    'obv_norm': 'OBV — dòng tiền tích lũy',
    'hl_ratio': 'High-Low Ratio — biên độ trong ngày',
    'co_ratio': 'Close-Open Ratio — xu hướng trong ngày',
    'gap': 'Gap — khoảng cách mở cửa',
    'upper_shadow': 'Bóng trên nến — áp lực bán',
    'lower_shadow': 'Bóng dưới nến — áp lực mua',
    'close_to_sma_5': 'Giá/SMA5',
    'close_to_sma_10': 'Giá/SMA10',
    'close_to_sma_20': 'Giá/SMA20',
    'close_to_sma_50': 'Giá/SMA50 — xu hướng dài hạn',
}


def create_features(group):
    """
    Tạo ~35 technical indicators cho 1 mã cổ phiếu.
    Input: DataFrame có cột Open, High, Low, Close, Volume
    Output: DataFrame với features mới
    """
    df = group.copy().reset_index(drop=True)
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = df[col].astype(float)

    close, high, low = df['Close'], df['High'], df['Low']
    volume, open_price = df['Volume'], df['Open']

    # MOMENTUM
    for p in [1, 3, 5, 10, 20, 60]:
        df[f'return_{p}d'] = close.pct_change(p)
    for p in [5, 10, 20]:
        df[f'roc_{p}'] = ta.roc(close, length=p)

    # TREND
    for p in [5, 10, 20, 50]:
        sma = ta.sma(close, length=p)
        df[f'close_to_sma_{p}'] = close / sma

    macd_df = ta.macd(close, fast=12, slow=26, signal=9)
    if macd_df is not None:
        df['macd'] = macd_df.iloc[:, 0]
        df['macd_hist'] = macd_df.iloc[:, 1]
        df['macd_signal'] = macd_df.iloc[:, 2]

    # OSCILLATOR
    df['rsi_14'] = ta.rsi(close, length=14)
    df['rsi_7'] = ta.rsi(close, length=7)

    stoch_df = ta.stoch(high, low, close, k=14, d=3)
    if stoch_df is not None:
        df['stoch_k'] = stoch_df.iloc[:, 0]
        df['stoch_d'] = stoch_df.iloc[:, 1]

    df['willr_14'] = ta.willr(high, low, close, length=14)
    df['cci_20'] = ta.cci(high, low, close, length=20)

    # VOLATILITY
    atr = ta.atr(high, low, close, length=14)
    df['atr_ratio'] = atr / close

    bb_df = ta.bbands(close, length=20, std=2)
    if bb_df is not None:
        df['bb_pct_b'] = (close - bb_df.iloc[:, 0]) / (bb_df.iloc[:, 2] - bb_df.iloc[:, 0])
        df['bb_width'] = (bb_df.iloc[:, 2] - bb_df.iloc[:, 0]) / bb_df.iloc[:, 1]

    df['std_ratio'] = close.rolling(20).std() / close

    # VOLUME
    vol_sma_5 = ta.sma(volume, length=5)
    vol_sma_20 = ta.sma(volume, length=20)
    df['vol_ratio_5_20'] = vol_sma_5 / vol_sma_20
    df['vol_change_1d'] = volume.pct_change(1)
    df['vol_change_5d'] = volume.pct_change(5)
    obv = ta.obv(close, volume)
    if obv is not None:
        df['obv_norm'] = obv / vol_sma_20

    # PRICE PATTERN
    df['hl_ratio'] = (high - low) / close
    df['co_ratio'] = (close - open_price) / open_price
    df['gap'] = (open_price - close.shift(1)) / close.shift(1)
    df['upper_shadow'] = (high - np.maximum(close, open_price)) / close
    df['lower_shadow'] = (np.minimum(close, open_price) - low) / close

    return df


def prepare_features(df_symbol, scaler, feature_cols):
    """
    Tạo features + scale cho 1 mã, trả về DataFrame có cả features lẫn metadata.
    """
    df_feat = create_features(df_symbol)
    df_feat = df_feat.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    if len(df_feat) == 0:
        return None, None

    X = df_feat[feature_cols].values.astype(float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X_scaled = scaler.transform(X)

    return df_feat, X_scaled
