import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

from src.preprocessing import preprocess_data

# -------------------------------------------------
# STREAMLIT PAGE CONFIG
# -------------------------------------------------

st.set_page_config(
    page_title="Crypto Price Prediction",
    layout="wide"
)

st.title("📈 Bitcoin Price Prediction using LSTM")
st.markdown("Predicting cryptocurrency prices using deep learning (LSTM)")

# -------------------------------------------------
# LOAD & PREPROCESS DATA
# -------------------------------------------------

@st.cache_data
def load_data():
    df = preprocess_data()
    df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")
    return df

df = load_data()
close_prices = df[["Close"]].values

# -------------------------------------------------
# SCALE DATA
# -------------------------------------------------

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)

# -------------------------------------------------
# CREATE SEQUENCES
# -------------------------------------------------

lookback = 60
X, y = [], []

for i in range(lookback, len(scaled_data)):
    X.append(scaled_data[i - lookback:i, 0])
    y.append(scaled_data[i, 0])

X, y = np.array(X), np.array(y)
X = X.reshape((X.shape[0], X.shape[1], 1))

# -------------------------------------------------
# TRAIN–TEST SPLIT
# -------------------------------------------------

train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# -------------------------------------------------
# BUILD & TRAIN MODEL (CACHE)
# -------------------------------------------------

@st.cache_resource
def train_lstm(X_train, y_train):
    model = Sequential([
        Input(shape=(lookback, 1)),
        LSTM(50, return_sequences=True),
        LSTM(50),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
    return model

model = train_lstm(X_train, y_train)

# -------------------------------------------------
# TEST SET PREDICTION
# -------------------------------------------------

predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions.reshape(-1, 1))

test_dates = df["Date"].iloc[train_size + lookback:]

# -------------------------------------------------
# FUTURE 30-DAY FORECAST
# -------------------------------------------------

future_days = 30
last_60_days = scaled_data[-lookback:]
current_input = last_60_days.reshape(1, lookback, 1)

future_predictions = []

for _ in range(future_days):
    next_price = model.predict(current_input, verbose=0)
    future_predictions.append(next_price[0, 0])

    current_input = np.append(
        current_input[:, 1:, :],
        [[[next_price[0, 0]]]],
        axis=1
    )

future_predictions = scaler.inverse_transform(
    np.array(future_predictions).reshape(-1, 1)
)

future_dates = pd.date_range(
    start=df["Date"].iloc[-1] + pd.Timedelta(days=1),
    periods=future_days
)

# -------------------------------------------------
# PLOTS
# -------------------------------------------------

st.subheader("📊 Historical Bitcoin Prices")
fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(df["Date"], df["Close"])
ax1.set_xlabel("Date")
ax1.set_ylabel("Price (USD)")
ax1.grid(True)
st.pyplot(fig1)

st.subheader("🔍 LSTM Prediction (Test Data)")
fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(df["Date"], df["Close"], label="Actual", alpha=0.6)
ax2.plot(test_dates, predictions, label="Prediction", color="red")
ax2.legend()
ax2.grid(True)
st.pyplot(fig2)

st.subheader("🔮 Next 30 Days Forecast")
fig3, ax3 = plt.subplots(figsize=(10, 4))
ax3.plot(df["Date"], df["Close"], label="Historical")
ax3.plot(future_dates, future_predictions, label="30-Day Forecast", color="green")
ax3.legend()
ax3.grid(True)
st.pyplot(fig3)

# -------------------------------------------------
# FOOTER
# -------------------------------------------------

st.markdown("---")
st.markdown("**Project:** Time Series Analysis with Cryptocurrency  \n**Model:** LSTM Neural Network")
