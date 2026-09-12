import pandas as pd
import ta
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class FeatureEngineer:
    """Class for feature engineering on stock data."""

    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add SMA, EMA, RSI, MACD, and Bollinger Bands."""
        df = df.copy()
        
        # Moving Averages
        df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['EMA_20'] = ta.trend.ema_indicator(df['Close'], window=20)
        
        # RSI
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
        
        # MACD
        df['MACD'] = ta.trend.macd_diff(df['Close'])
        
        # Bollinger Bands
        indicator_bb = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=2)
        df['BB_High'] = indicator_bb.bollinger_hband()
        df['BB_Low'] = indicator_bb.bollinger_lband()
        
        return df.dropna()

    def scale_data(self, df: pd.DataFrame, target_col: str = 'Close'):
        """Scale data using MinMaxScaler."""
        scaled_data = self.scaler.fit_transform(df)
        return pd.DataFrame(scaled_data, columns=df.columns, index=df.index), self.scaler

    def create_sequences(self, data: np.ndarray, seq_length: int):
        """Create sequences for LSTM."""
        X, y = [], []
        for i in range(seq_length, len(data)):
            X.append(data[i-seq_length:i])
            y.append(data[i, 0]) # Assuming target is the first column
        return np.array(X), np.array(y)
