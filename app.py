import streamlit as st
import pandas as pd
import numpy as np
from data.data_fetcher import DataFetcher
from data.feature_engineering import FeatureEngineer
from models.lstm_model import create_lstm_model
from models.bilstm_model import create_bilstm_model
from training.trainer import ModelTrainer
from visualization.charts import plot_predictions
from evaluation.metrics import calculate_metrics

st.set_page_config(page_title="Stock Price Predictor", layout="wide")

st.title("📈 Stock Price Predictor using LSTM / Bi-LSTM")
st.markdown("Forecast stock prices with deep learning and technical indicators.")

# Sidebar
st.sidebar.header("Parameters")
ticker = st.sidebar.text_input("Stock Ticker", "AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2018-01-01"))
model_type = st.sidebar.selectbox("Model", ["LSTM", "Bi-LSTM"])
seq_length = st.sidebar.slider("Sequence Length", 10, 100, 60)
epochs = st.sidebar.slider("Epochs", 10, 100, 20)

if st.sidebar.button("Train & Predict"):
    with st.spinner("Fetching data..."):
        fetcher = DataFetcher()
        try:
            df = fetcher.fetch_data(ticker, start_date.strftime("%Y-%m-%d"))
            st.success(f"Data loaded for {ticker}")
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            st.stop()
            
    with st.spinner("Engineering features..."):
        fe = FeatureEngineer()
        df = fe.add_technical_indicators(df)
        scaled_df, scaler = fe.scale_data(df)
        X, y = fe.create_sequences(scaled_df.values, seq_length)
        
        # Train/Test Split
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
    with st.spinner(f"Training {model_type} model..."):
        input_shape = (X_train.shape[1], X_train.shape[2])
        model = create_lstm_model(input_shape) if model_type == "LSTM" else create_bilstm_model(input_shape)
        
        trainer = ModelTrainer(model, f"{ticker}_{model_type}.h5")
        trainer.train(X_train, y_train, X_test, y_test, epochs=epochs)
        
    with st.spinner("Generating predictions..."):
        predictions = model.predict(X_test)
        
        # Inverse transform
        # We need to create a dummy array with the same shape as scaled_df to inverse transform correctly
        dummy = np.zeros((len(predictions), scaled_df.shape[1]))
        dummy[:, 0] = predictions.flatten()
        pred_inv = scaler.inverse_transform(dummy)[:, 0]
        
        dummy_test = np.zeros((len(y_test), scaled_df.shape[1]))
        dummy_test[:, 0] = y_test
        y_test_inv = scaler.inverse_transform(dummy_test)[:, 0]
        
        # Metrics
        metrics = calculate_metrics(y_test_inv, pred_inv)
        st.subheader("Model Performance")
        col1, col2, col3 = st.columns(3)
        col1.metric("RMSE", f"{metrics['RMSE']:.2f}")
        col2.metric("MAE", f"{metrics['MAE']:.2f}")
        col3.metric("R² Score", f"{metrics['R2']:.4f}")
        
        # Plot
        dates = df.index[split+seq_length:]
        fig = plot_predictions(dates, y_test_inv, pred_inv, f"{ticker} Price Prediction")
        st.plotly_chart(fig, use_container_width=True)
