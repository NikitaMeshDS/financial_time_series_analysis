"""
improvements.py — drop-in additions to strengthen the existing pipelines
(financial_pipeline.py, arima_pipeline.py, ssa_pipeline.py) before the
SMILES poster session.

What this gives you, in priority order:
  1. A naive baseline (walk-forward persistence) — run this FIRST. If your
     ARIMA/SSA/NeuralProphet/LSTM/LightGBM models can't beat it, that's an
     important (and honest) result in itself.
  2. A timing decorator — wrap any model's fit/predict call to get the
     numbers your task description promised but the report never measured.
  3. A walk-forward (rolling, one-step-ahead) evaluator for ARIMA and SSA,
     so they are judged on the same kind of forecast as LightGBM/LSTM
     instead of "one straight line over the whole test set".

How to use in your Colab notebook:

    from improvements import (
        naive_forecast, evaluate, timed,
        walk_forward_arima, walk_forward_ssa,
    )

    # 1) naive baseline — always compute this for every series
    naive_preds = naive_forecast(test_data.values)
    print("Naive baseline:", evaluate(test_data.values, naive_preds))

    # 2) timing any existing model call
    best_order, fit_seconds = timed(find_best_arima_params, train_data)

    # 3) fair walk-forward ARIMA / SSA
    arima_preds, arima_seconds = walk_forward_arima(train_data.values, test_data.values, order=best_order)
    ssa_preds, ssa_seconds = walk_forward_ssa(train_data.values, test_data.values, window=120, r=20, refit_every=10)

Note: walk-forward SSA recomputes the SVD periodically (refit_every) instead
of on every single step, to keep runtime reasonable on ~500 test points.
Increase refit_every for speed, decrease it for accuracy.
"""

import time
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


# ---------------------------------------------------------------------------
# 1) Naive baseline
# ---------------------------------------------------------------------------
def naive_forecast(test):
    """Walk-forward persistence: predict tomorrow = today's REAL value.

    This is the standard baseline for one-step-ahead financial forecasting.
    test[0] has no "yesterday" inside the test set, so it is left as NaN —
    drop it (and the matching actual) before scoring, see `evaluate`.
    """
    test = np.asarray(test, dtype=float)
    preds = np.empty_like(test)
    preds[0] = np.nan
    preds[1:] = test[:-1]
    return preds


def seasonal_naive_forecast(test, season_length):
    """Predict value = value `season_length` steps ago (e.g. 24 for hourly
    data with a daily cycle). Useful as a second baseline if you suspect
    intraday seasonality in USD/RUB."""
    test = np.asarray(test, dtype=float)
    preds = np.full_like(test, np.nan)
    preds[season_length:] = test[:-season_length]
    return preds


def evaluate(actual, predicted, label=None):
    """Same MSE / MAE / MAPE you already report, with NaN-safe handling so
    it works directly on naive_forecast's output."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    actual, predicted = actual[mask], predicted[mask]

    mse = mean_squared_error(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    result = {"MSE": mse, "MAE": mae, "MAPE": mape, "n": len(actual)}
    if label:
        print(f"{label}: MSE={mse:.2f}  MAE={mae:.2f}  MAPE={mape:.2f}%  (n={len(actual)})")
    return result


# ---------------------------------------------------------------------------
# 2) Timing
# ---------------------------------------------------------------------------
def timed(func, *args, **kwargs):
    """Run func(*args, **kwargs), return (result, elapsed_seconds).

    Example:
        model, fit_time = timed(lambda: create_lstm_model(60, 5))
        preds, predict_time = timed(model.predict, X_test)
    """
    t0 = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    return result, elapsed


# ---------------------------------------------------------------------------
# 3) Walk-forward (rolling, one-step-ahead) evaluation
# ---------------------------------------------------------------------------
def walk_forward_arima(train, test, order, refit_every=1):
    """Rolling one-step-ahead ARIMA forecast, fed the REAL previous value at
    every step (like LightGBM/LSTM effectively are) instead of forecasting
    the whole test horizon from a single fit.

    refit_every=1 re-estimates parameters every step (slow, most correct).
    A larger value (e.g. 10-20) updates the model state every step but only
    re-optimizes parameters periodically — much faster, small accuracy cost.
    """
    from statsmodels.tsa.arima.model import ARIMA

    train = np.asarray(train, dtype=float)
    test = np.asarray(test, dtype=float)

    t0 = time.perf_counter()
    fitted = ARIMA(train, order=order).fit()
    preds = np.empty(len(test))

    for i, true_val in enumerate(test):
        preds[i] = fitted.forecast(1)[0]
        refit = (i % refit_every == 0)
        # `.append` extends the model with the newly observed true value;
        # refit=False just updates the state (fast), refit=True re-estimates
        # parameters (slower, more accurate over long horizons).
        fitted = fitted.append([true_val], refit=refit)

    elapsed = time.perf_counter() - t0
    return preds, elapsed


def walk_forward_ssa(train, test, window, r, refit_every=10):
    """Rolling one-step-ahead SSA+AR forecast. Recomputes the SVD/AR fit
    every `refit_every` steps (SVD on a few thousand points is not free),
    but always feeds the model the REAL observed value at each step rather
    than its own previous forecast — this is what removes the "collapses
    to a straight line" failure mode you saw with the full-horizon version.
    """
    from numpy.linalg import svd
    from scipy.linalg import lstsq

    def fit_ar(history):
        N = len(history)
        K = N - window + 1
        X = np.column_stack([history[i:i + K] for i in range(window)])
        U, s, Vt = svd(X, full_matrices=False)
        r_eff = min(r, window)
        X_elem = np.zeros((U.shape[0], Vt.shape[1]))
        for i in range(r_eff):
            X_elem += s[i] * np.outer(U[:, i], Vt[i, :])
        L, K2 = X_elem.shape
        Nr = L + K2 - 1
        recon = np.zeros(Nr)
        counts = np.zeros(Nr)
        for i in range(L):
            for j in range(K2):
                recon[i + j] += X_elem[i, j]
                counts[i + j] += 1
        recon = recon / counts
        Xr = np.array([recon[i:i + window] for i in range(len(recon) - window)])
        yr = recon[window:]
        coef, _, _, _ = lstsq(Xr, yr)
        return coef, recon[-window:]

    train = list(np.asarray(train, dtype=float))
    test = np.asarray(test, dtype=float)

    t0 = time.perf_counter()
    coef, last_vals = fit_ar(np.array(train))
    preds = np.empty(len(test))

    for i, true_val in enumerate(test):
        next_val = float(np.dot(coef, last_vals[-window:]))
        preds[i] = next_val
        train.append(true_val)
        last_vals = np.append(last_vals, true_val)
        if (i + 1) % refit_every == 0:
            coef, last_vals = fit_ar(np.array(train))

    elapsed = time.perf_counter() - t0
    return preds, elapsed


if __name__ == "__main__":
    # Tiny smoke test on synthetic data so you can sanity-check the code
    # before pointing it at real API data.
    rng = np.random.default_rng(0)
    series = np.cumsum(rng.normal(0, 1, 600)) + 100
    train, test = series[:500], series[500:]

    naive_preds = naive_forecast(test)
    evaluate(test, naive_preds, label="Naive (smoke test)")

    arima_preds, secs = walk_forward_arima(train, test, order=(1, 1, 1), refit_every=5)
    evaluate(test, arima_preds, label=f"Walk-forward ARIMA (smoke test, {secs:.1f}s)")
