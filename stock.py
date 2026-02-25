import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Stock Direction Prediction", layout="centered")
st.title("📈 Stock Market 5-Day Direction Prediction")
st.caption("Predict price direction 5 trading days ahead using ML")

# =====================================================
# 1. MULTI-STOCK SELECTION
# =====================================================
st.sidebar.subheader("Select Stock")
stock_choice = st.sidebar.selectbox(
    "Stock Symbol",
    ["AAPL", "MSFT", "SPY"]  # Add CSV files in ./data folder
)

data_file = f"data/{stock_choice}.csv"

# =====================================================
# 2. LOAD DATA
# =====================================================
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df.columns = [c.lower().replace("/", "_").replace(" ", "_") for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    for col in ["close_last", "open", "high", "low"]:
        df[col] = df[col].str.replace("$", "", regex=False).astype(float)
    return df

df = load_data(data_file)

# =====================================================
# 3. FEATURE ENGINEERING
# =====================================================
df["return"] = df["close_last"].pct_change()

# Moving averages
df["sma_5"] = df["close_last"].rolling(5).mean()
df["sma_10"] = df["close_last"].rolling(10).mean()
df["sma_20"] = df["close_last"].rolling(20).mean()

# Volatility
df["volatility_10"] = df["return"].rolling(10).std()

# RSI
delta = df["close_last"].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()
rs = avg_gain / avg_loss
df["rsi"] = 100 - (100 / (1 + rs))

# MACD
ema_12 = df["close_last"].ewm(span=12, adjust=False).mean()
ema_26 = df["close_last"].ewm(span=26, adjust=False).mean()
df["macd"] = ema_12 - ema_26
df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

# Lag features
df["return_lag_1"] = df["return"].shift(1)
df["return_lag_2"] = df["return"].shift(2)

# Target: 5-day ahead direction
df["target"] = (df["close_last"].shift(-5) > df["close_last"]).astype(int)

df = df.dropna().reset_index(drop=True)

# Features
features = [
    "return", "sma_5", "sma_10", "sma_20",
    "volatility_10", "rsi",
    "macd", "macd_signal",
    "return_lag_1", "return_lag_2"
]

X = df[features]
y = df["target"]

# Train-test split (time-based)
split = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# =====================================================
# 4. MODEL TRAINING
# =====================================================
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

rf_model = RandomForestClassifier(
    n_estimators=500,
    max_depth=6,
    min_samples_leaf=30,
    class_weight="balanced",
    random_state=42
)
rf_model.fit(X_train, y_train)

# =====================================================
# 5. NEXT 5-DAY PREDICTION
# =====================================================
latest_features = df.iloc[-1][features].values.reshape(1, -1)

log_prob = log_model.predict_proba(latest_features)[0][1]
rf_prob = rf_model.predict_proba(latest_features)[0][1]

log_direction = "UP 📈" if log_prob > 0.5 else "DOWN 📉"
rf_direction = "UP 📈" if rf_prob > 0.55 else "DOWN 📉"

st.subheader("🔮 5-Day Ahead Prediction")

col1, col2 = st.columns(2)
with col1:
    st.metric(
        label="Logistic Regression",
        value=log_direction,
        delta=f"{log_prob:.2%} confidence"
    )
with col2:
    st.metric(
        label="Random Forest",
        value=rf_direction,
        delta=f"{rf_prob:.2%} confidence"
    )

# Trading signal
st.subheader("📌 Suggested Signal")
if rf_prob > 0.62:
    st.success("BUY signal (higher confidence)")
elif rf_prob < 0.38:
    st.error("SELL signal (higher confidence)")
else:
    st.warning("HOLD (low confidence)")

st.caption(
    "Signals are probabilistic. Confidence thresholds reduce false signals due to short-term noise."
)

# =====================================================
# 6. FEATURE IMPORTANCE
# =====================================================
st.subheader("🧠 Feature Importance (Random Forest)")
importances = pd.Series(rf_model.feature_importances_, index=features).sort_values()
fig, ax = plt.subplots()
importances.plot(kind="barh", ax=ax)
st.pyplot(fig)

# =====================================================
# 7. BACKTESTING STRATEGY
# =====================================================
st.subheader("📈 Backtesting Strategy")

df["future_return_5d"] = df["close_last"].shift(-5) / df["close_last"] - 1
rf_probs_all = rf_model.predict_proba(X)[:,1]
signal = np.where(rf_probs_all > 0.62, 1, np.where(rf_probs_all < 0.38, -1, 0))
df["strategy_return"] = signal * df["future_return_5d"]
df_bt = df.dropna(subset=["strategy_return"])
df_bt["cumulative_strategy"] = (1 + df_bt["strategy_return"]).cumprod()
df_bt["cumulative_hold"] = (1 + df_bt["future_return_5d"]).cumprod()

fig_bt, ax_bt = plt.subplots()
ax_bt.plot(df_bt["date"], df_bt["cumulative_strategy"], label="Strategy")
ax_bt.plot(df_bt["date"], df_bt["cumulative_hold"], label="Buy & Hold", alpha=0.7)
ax_bt.set_ylabel("Cumulative Return")
ax_bt.legend()
st.pyplot(fig_bt)

# =====================================================
# 8. DATA PREVIEW
# =====================================================
with st.expander("📄 View latest data"):
    st.dataframe(df.tail(10))