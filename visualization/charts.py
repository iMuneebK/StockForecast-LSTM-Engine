import plotly.graph_objs as go
import pandas as pd

def plot_predictions(dates, y_true, y_pred, title="Stock Price Prediction"):
    """Plot actual vs predicted prices using Plotly."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=y_true, mode='lines', name='Actual Price', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=dates, y=y_pred, mode='lines', name='Predicted Price', line=dict(color='red')))
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_dark",
        hovermode="x unified"
    )
    return fig
