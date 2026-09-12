# Stock Price Predictor LSTM

![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)

An advanced deep learning framework for predicting stock prices using Long Short-Term Memory (LSTM) and Bidirectional LSTM networks. Includes automated feature engineering of technical indicators and a beautiful Streamlit dashboard.

## ⚠️ Financial Disclaimer
**This project is for educational and research purposes only.** The predictions made by this model should not be considered financial advice. Stock markets are highly volatile, and predicting exact prices is inherently uncertain.

## Features
- **Deep Learning Models:** Standard LSTM and Bi-LSTM implementations.
- **Technical Indicators:** Automatically calculates SMA, EMA, RSI, MACD, and Bollinger Bands as features.
- **Walk-forward Validation:** Time-series aware data splitting.
- **Interactive UI:** A Streamlit dashboard for real-time training and visualization.
- **Data Pipeline:** Cached Yahoo Finance data fetching.

## Architecture
Data Fetcher -> Feature Engineer (Tech Indicators) -> Sequence Generator -> LSTM/Bi-LSTM Network -> Evaluation/Visualization

## Installation
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Interactive Demo
Use the sidebar in the Streamlit app to select a ticker, adjust the sequence length, and pick a model architecture. View real-time training and interactive Plotly charts showing the difference between actual and predicted stock values.
