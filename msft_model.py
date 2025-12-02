import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

ticker = "MSFT"
print("Downloading", ticker)

df = yf.download(ticker, start="2010-01-01", end="2025-12-31")

# 🔥 FIX MULTI-INDEX COLUMNS
df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

df = df[["Open", "High", "Low", "Close", "Volume"]]
print("Downloaded:", df.shape)

# ---------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------
def add_features(df):

    df["return_1d"] = df["Close"].pct_change()
    df["return_5d"] = df["Close"].pct_change(5)
    df["return_10d"] = df["Close"].pct_change(10)

    df["ma10"] = df["Close"].rolling(10).mean()
    df["ma20"] = df["Close"].rolling(20).mean()
    df["ma50"] = df["Close"].rolling(50).mean()
    df["ma100"] = df["Close"].rolling(100).mean()

    # RSI
    delta = df["Close"].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    rs = up.rolling(14).mean() / down.rolling(14).mean()
    df["rsi"] = 100 - 100 / (1 + rs)

    # MACD
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    # Bollinger Bands (NOW FIXED)
    window = 20
    bb_mid = df["Close"].rolling(window).mean()        # <-- Series
    bb_std = df["Close"].rolling(window).std()

    df["bb_middle"] = bb_mid
    df["bb_upper"] = bb_mid + 2 * bb_std
    df["bb_lower"] = bb_mid - 2 * bb_std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / bb_mid

    df["vol_change"] = df["Volume"].pct_change()
    df["vol_ma20"] = df["Volume"].rolling(20).mean()

    return df

df = add_features(df)
df.dropna(inplace=True)
print("After Feature Engineering:", df.shape)

# TARGET
df["target"] = np.where(df["Close"].shift(-1) > df["Close"], 1, 0)

features = [c for c in df.columns if c not in ["target"]]
X = df[features]
y = df["target"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, shuffle=False
)

model = RandomForestClassifier(
    n_estimators=400,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

acc = model.score(X_test, y_test)
print("--------------------------------------")
print("Accuracy:", round(acc * 100, 2), "%")
print("--------------------------------------")

last = X_scaled[-1].reshape(1, -1)
pred = model.predict(last)[0]

if pred == 1:
    print("Prediction for tomorrow: UP 📈")
else:
    print("Prediction for tomorrow: DOWN 📉")
