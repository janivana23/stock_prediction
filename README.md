# 📈 Stock Market 5-Day Direction Prediction

### Predict the direction of a stock price over the next 5 trading days using machine learning, with feature importance, backtesting, and confidence-based trading signals.

## 🔹 Demo Screenshots
1. Upload Stock CSV or Select Preset Stock

2. 5-Day Direction Prediction & Suggested Signal

3. Feature Importance Visualization

4. Backtesting Strategy vs Buy & Hold

## ⚡ Features

### Stock Selection / CSV Upload

- Preset stocks: AAPL, MSFT, SPY

- User-uploaded Nasdaq-format CSV with columns:
 - Date, Close/Last, Open, High, Low, Volume

- Supports $ or plain number prices

### Machine Learning Models

- Logistic Regression → predicts UP/DOWN direction with confidence

- Random Forest Classifier → predicts UP/DOWN direction, confidence-based thresholds

### Trading Signal

- Displays BUY / SELL / HOLD based on classifier confidence

- Reduces false signals from low-confidence predictions

### Feature Importance

- Displays contribution of features used by Random Forest

- Features: returns, SMA (5,10,20), volatility, RSI, MACD, lagged returns

### Backtesting Strategy

- Compares cumulative returns of strategy vs buy & hold

- Strategy uses classifier signals to generate returns

### Data Preview

- Inspect the latest stock data used for predictions


## 📝 How to Use

1. Select a stock from the sidebar or upload a Nasdaq-format CSV.

2. View 5-day prediction: UP 📈 / DOWN 📉 with confidence percentage.

3. Check suggested trading signal: BUY / SELL / HOLD.

4. Analyze feature importance to understand which features drive predictions.

5. View backtesting chart comparing your strategy vs buy & hold.

6. Inspect the latest stock data in the expandable table at the bottom.


## ⚠️ Notes

The app uses historical stock features (moving averages, volatility, RSI, MACD, lagged returns) for predictions.

Predictions are probabilistic; use caution and do not consider as financial advice.

Only Nasdaq CSV format is supported. Column headers must match exactly.


## 🌟 Optional Improvements

Add 5-day predicted price (aligned with classifier direction).

Scale backtesting returns using predicted magnitude, not just direction.

Implement user-configurable prediction horizons (e.g., 1-day, 10-day).


## 🔗 References

Nasdaq Historical Quotes

Scikit-Learn Documentation

Streamlit Documentation