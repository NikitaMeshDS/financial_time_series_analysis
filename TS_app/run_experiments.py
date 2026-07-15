"""
run_experiments.py — run this file. It computes, for all 3 series, the three
things the poster's checklist says are missing: a naive baseline, a fair
walk-forward protocol, and fit/predict timing.

SETUP (once):
    export TWELVEDATA_API_KEY="your_new_key"     # after rotating the old one
    cd TS_app
    pip install -r requirements.txt
    pip install statsmodels   # if not already covered by requirements.txt

RUN:
    python run_experiments.py

Place this file inside TS_app/ (next to arima_pipeline.py, ssa_pipeline.py,
financial_pipeline.py, improvements.py) before running.

Output: prints a results table to the console. Copy that table back to me
(Claude) and I'll update the poster's numbers and tick off the checklist.
"""

import os
import time
import numpy as np

import arima_pipeline as ap
import ssa_pipeline as sp
import financial_pipeline as fp
from improvements import naive_forecast, evaluate, timed, walk_forward_arima, walk_forward_ssa

API_KEY = os.environ.get("TWELVEDATA_API_KEY")
if not API_KEY:
    raise RuntimeError("export TWELVEDATA_API_KEY=... before running this script")

OUTPUTSIZE = "5000"
results = {}


def split(df, test_frac=0.1):
    n_test = int(len(df) * test_frac)
    return df["close"][:-n_test], df["close"][-n_test:]


# ---------------------------------------------------------------------------
# 1) USD/RUB, hourly — SSA+AR
# ---------------------------------------------------------------------------
print("\n=== USD/RUB (1h) — SSA+AR ===")
df = sp.get_data("USD/RUB", "1h", OUTPUTSIZE, API_KEY).dropna()
train, test = split(df)

naive_preds = naive_forecast(test.values)
results["USD/RUB naive"] = evaluate(test.values, naive_preds, label="Naive")

(coef_unused), fit_time = timed(sp.ssa_decompose, train.values, 120)  # rough fit-time proxy
preds, wf_time = walk_forward_ssa(train.values, test.values, window=120, r=20, refit_every=10)
results["USD/RUB walk-forward SSA"] = evaluate(test.values, preds, label=f"Walk-forward SSA ({wf_time:.1f}s)")


# ---------------------------------------------------------------------------
# 2) AAPL, daily — ARIMA
# ---------------------------------------------------------------------------
print("\n=== AAPL (1day) — ARIMA ===")
df = ap.get_data("AAPL", "1day", OUTPUTSIZE, API_KEY)
df["close"] = df["close"].replace([np.inf, -np.inf], np.nan).dropna()
train, test = split(df)

naive_preds = naive_forecast(test.values)
results["AAPL naive"] = evaluate(test.values, naive_preds, label="Naive")

best_order, order_search_time = timed(ap.find_best_arima_params, train)
print(f"ARIMA order search took {order_search_time:.1f}s -> order={best_order}")
preds, wf_time = walk_forward_arima(train.values, test.values, order=best_order, refit_every=10)
results["AAPL walk-forward ARIMA"] = evaluate(test.values, preds, label=f"Walk-forward ARIMA ({wf_time:.1f}s)")


# ---------------------------------------------------------------------------
# 3) EUR/USD, hourly — LSTM (already one-step-ahead by construction; we only
#    need naive baseline + fit/predict timing, per improvements.py docstring)
# ---------------------------------------------------------------------------
print("\n=== EUR/USD (1h) — LSTM timing ===")
df = fp.get_data("EUR/USD", "1h", OUTPUTSIZE, API_KEY)
df = fp.add_technical_indicators(df).dropna()
features = ["close", "SMA_20", "SMA_50", "RSI", "MACD"]

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[features])
X, y = fp.prepare_sequences(scaled, 60)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)

naive_preds = naive_forecast(y_test[:, 0])
results["EUR/USD naive (scaled)"] = evaluate(y_test[:, 0], naive_preds, label="Naive (scaled units)")

model, build_time = timed(fp.create_lstm_model, 60, len(features))
_, fit_time = timed(model.fit, X_train, y_train[:, 0], epochs=50, batch_size=32, validation_split=0.1, verbose=0)
preds, predict_time = timed(model.predict, X_test, verbose=0)
print(f"LSTM build: {build_time:.2f}s | fit: {fit_time:.1f}s | predict: {predict_time:.2f}s")


# ---------------------------------------------------------------------------
print("\n\n===== SUMMARY (copy everything below back to Claude) =====")
for k, v in results.items():
    print(f"{k}: {v}")
print(f"AAPL ARIMA order search: {order_search_time:.1f}s, order={best_order}")
print(f"EUR/USD LSTM: build {build_time:.2f}s, fit {fit_time:.1f}s, predict {predict_time:.2f}s")
