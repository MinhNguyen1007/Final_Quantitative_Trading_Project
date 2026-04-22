"""
Data Loader - Đọc và xử lý dữ liệu VN30
"""

import os
import pandas as pd
from config import VN30_SYMBOLS


def load_all_stocks(data_dir, symbols=VN30_SYMBOLS):
    """Đọc tất cả file CSV VN30 và gộp thành 1 DataFrame."""
    all_dfs = []
    for symbol in symbols:
        fp = os.path.join(data_dir, f"{symbol}.csv")
        if os.path.exists(fp):
            df = pd.read_csv(fp)
            all_dfs.append(df)

    if not all_dfs:
        return None

    combined = pd.concat(all_dfs, ignore_index=True)
    combined['TradingDate'] = pd.to_datetime(combined['TradingDate'], format='%d/%m/%Y')
    combined = combined.drop(columns=['Time', 'Value'], errors='ignore')
    combined = combined.sort_values(['Symbol', 'TradingDate']).reset_index(drop=True)
    return combined


def load_single_stock(data_dir, symbol):
    """Đọc dữ liệu 1 mã cổ phiếu."""
    fp = os.path.join(data_dir, f"{symbol}.csv")
    if not os.path.exists(fp):
        return None
    df = pd.read_csv(fp)
    df['TradingDate'] = pd.to_datetime(df['TradingDate'], format='%d/%m/%Y')
    df = df.drop(columns=['Time', 'Value'], errors='ignore')
    df = df.sort_values('TradingDate').reset_index(drop=True)
    return df
