import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from preprocessing import preprocess_data

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------

df = preprocess_data()

# Convert Date back to datetime for plotting
df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")

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

# Reshape for LSTM: (samples, timesteps, features)
X = X.reshape((X.shape[0], X.shape[1], 1))

# -------------------------------------------------
# TRAIN–TEST SPLIT
# -------------------------------------------------

train_size = int(len(X) * 0.8)

X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# -------------------------------------------------
# BUILD LSTM MODEL
# -------------------------------------------------

model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(lookback, 1)),
    LSTM(50),
    Dense(1)
])

model.compile(optimizer="adam", loss="mean_squared_error")

# -------------------------------------------------
# TRAIN MODEL
# -------------------------------------------------

model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

print("✅ LSTM model trained successfully")

# -------------------------------------------------
# PREDICTIONS
# -------------------------------------------------

predictions = model.predict(X_test)

# Inverse scale predictions & actual values
predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Align dates
test_dates = df["Date"].iloc[train_size + lookback:]

# -------------------------------------------------
# PLOT RESULTS
# -------------------------------------------------

plt.figure(figsize=(12, 6))

plt.plot(df["Date"], df["Close"], label="Actual Price", alpha=0.6)
plt.plot(test_dates, predictions, label="LSTM Prediction", color="red")

plt.title("Bitcoin Price Prediction using LSTM")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
