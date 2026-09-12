from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout

def create_bilstm_model(input_shape):
    """Create a Bidirectional LSTM model."""
    model = Sequential([
        Bidirectional(LSTM(units=50, return_sequences=True), input_shape=input_shape),
        Dropout(0.2),
        Bidirectional(LSTM(units=50, return_sequences=False)),
        Dropout(0.2),
        Dense(units=25),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
