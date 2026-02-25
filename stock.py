import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Stock Direction Prediction", layout="centered")
st.title("📈 Stock Market 5-Day Direction Prediction")
st.caption("Predict price direction 5 trading days ahead using ML")

# =====================================================
# 0. USER INSTRUCTIONS
# =====================================================
st.markdown(
    """
    **ℹ️ Instructions for Uploading Your Own CSV:**  
    1. Go to [Nasdaq Historical Quotes](https://www.nasdaq.com/market-activity/stocks).  
    2. Search for the stock you want (e.g., AAPL, MSFT, SPY).  
    3. Click **Historical Data** → **Download Data** → CSV.  
    4. Ensure the CSV contains at least the following columns:  
       `Date, Close/Last, Open, High, Low, Volume`.  
    5. Upload your CSV in the sidebar.  

    ⚠️ **Important:** Only Nasdaq format CSV is supported. Column headers must match exactly, and prices can include `$` or be plain numbers.
    """
)

# =====================================================
# 1. STOCK SELECTION / CSV UPLOAD
# =====================================================
st.sidebar.subheader("Select or Upload Stock Data")
stock_choice = st.sidebar.selectbox(
    "Preset Stock",
    ["AAPL", "MSFT", "SPY"]
)
uploaded_file = st.sidebar.file_uploader("Or upload your own CSV (Nasdaq format)", type="csv")

show_price_prediction = st.sidebar.checkbox("Show predicted price for next 5 days")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    data_file = f"data/{stock_choice}.csv"
    df = pd.read_csv(data_file)

# =====================================================
# 2. LOAD & CLEAN DATA
# =====================================================
@st.cache_data
def clean_data(df):
    df.columns = [c.lower().replace("/", "_").replace(" ", "_") for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    # Ensure numeric conversion works for $ and plain numbers
    for col in ["close_last", "open", "high", "low"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace("$", "", regex=False)
                .str.replace(",", "", regex=False)
                .astype(float)
            )
        else:
            st.error(f"Missing column: {col}. CSV must be from Nasdaq format.")
            st.stop()
    return df

df = clean_data(df)

# =====================================================
# 3. FEATURE ENGINEERING
# =====================================================
df["return"] = df["close_last"].pct_change()
df["sma_5"] = df["close_last"].rolling(5).mean()
df["sma_10"] = df["close_last"].rolling(10).mean()
df["sma_20"] = df["close_last"].rolling(20).mean()
df["volatility_10"] = df["return"].rolling(10).std()

delta = df["close_last"].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()
rs = avg_gain / avg_loss
df["rsi"] = 100 - (100 / (1 + rs))

ema_12 = df["close_last"].ewm(span=12, adjust=False).mean()
ema_26 = df["close_last"].ewm(span=26, adjust=False).mean()
df["macd"] = ema_12 - ema_26
df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

df["return_lag_1"] = df["return"].shift(1)
df["return_lag_2"] = df["return"].shift(2)

# Classification target (direction)
df["target"] = (df["close_last"].shift(-5) > df["close_last"]).astype(int)
# Regression target (price)
df["target_price_5d"] = df["close_last"].shift(-5)

df = df.dropna().reset_index(drop=True)

features = [
    "return", "sma_5", "sma_10", "sma_20",
    "volatility_10", "rsi",
    "macd", "macd_signal",
    "return_lag_1", "return_lag_2"
]

X = df[features]
y = df["target"]
y_price = df["target_price_5d"]

split = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]
y_train_price, y_test_price = y_price.iloc[:split], y_price.iloc[split:]

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

# Regression model
rf_reg = RandomForestRegressor(n_estimators=500, max_depth=6, random_state=42)
rf_reg.fit(X_train, y_train_price)

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
    st.metric("Logistic Regression", log_direction, delta=f"{log_prob:.2%} confidence")
with col2:
    st.metric("Random Forest", rf_direction, delta=f"{rf_prob:.2%} confidence")

# Optional predicted price
if show_price_prediction:
    next_price_pred = rf_reg.predict(latest_features)[0]
    st.metric("Predicted Close Price (5-day ahead)", f"${next_price_pred:.2f}")

# Trading signal
st.subheader("📌 Suggested Signal")
if rf_prob > 0.62:
    st.success("BUY signal (higher confidence)")
elif rf_prob < 0.38:
    st.error("SELL signal (higher confidence)")
else:
    st.warning("HOLD (low confidence)")

st.caption("Signals are probabilistic. Confidence thresholds reduce false signals due to short-term noise.")

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