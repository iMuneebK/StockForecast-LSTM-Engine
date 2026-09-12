import os

# Application Settings
APP_NAME = "Stock Price Predictor LSTM"
VERSION = "1.0.0"

# Data Settings
DEFAULT_TICKER = "AAPL"
START_DATE_DEFAULT = "2018-01-01"

# Model Settings
SEQUENCE_LENGTH = 60
BATCH_SIZE = 32
EPOCHS = 50

# Path Settings
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
DATA_DIR = os.path.join(BASE_DIR, "data", "cache")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
