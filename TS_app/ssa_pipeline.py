import pandas as pd
import numpy as np
import requests
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from numpy.linalg import svd
from scipy.linalg import lstsq

def get_data(symbol, interval, outputsize, api_key):
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": api_key
    }
    url = "https://api.twelvedata.com/time_series"
    data = requests.get(url, params=params).json()
    return preprocess_data(data, interval, outputsize)

def preprocess_data(data, interval):
    df = pd.DataFrame(data["values"])
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
    numeric_columns = ['close', 'high', 'low', 'open', 'volume']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df.drop(columns=["high", "low", "open", "volume"], inplace=True, errors='ignore')
    freq_map = {"1day": "D", "1h": "H", "5min": "5T"}
    full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq_map[interval])
    df = df.reindex(full_index)
    df['close'] = df['close'].interpolate(method='time').ffill().bfill()
    return df

def ssa_decompose(ts, window):
    N = len(ts)
    K = N - window + 1
    X = np.column_stack([ts[i:i+K] for i in range(window)])
    U, s, Vt = svd(X, full_matrices=False)
    return U, s, Vt

def ssa_reconstruct(U, s, Vt, r):
    X_elem = np.zeros((U.shape[0], Vt.shape[1]))
    for i in range(r):
        X_elem += s[i] * np.outer(U[:, i], Vt[i, :])

    L, K = X_elem.shape
    N = L + K - 1
    recon = np.zeros(N)
    counts = np.zeros(N)
    for i in range(L):
        for j in range(K):
            recon[i + j] += X_elem[i, j]
            counts[i + j] += 1
    return recon / counts

def ssa_forecast(ts, window, r, n_forecast):

    U, s, Vt = ssa_decompose(ts, window)
    r = min(20, window) 
    recon = ssa_reconstruct(U, s, Vt, r)

    p = window

    X = np.array([recon[i:i+p] for i in range(len(recon) - p)])
    y = recon[p:]
    coef, _, _, _ = lstsq(X, y)
    
    forecast = []
    last_vals = list(recon[-p:])
    for _ in range(n_forecast):
        next_val = np.dot(coef, last_vals[-p:])
        forecast.append(next_val)
        last_vals.append(next_val)
    return forecast

def plot_results(actual, predicted, title):
    plt.figure(figsize=(12, 6))
    plt.plot(actual, label='Фактические значения')
    plt.plot(predicted, label='Прогноз')
    plt.title(title)
    plt.xlabel('Время')
    plt.ylabel('Цена')
    plt.legend()
    plt.show()

def main():
    symbol = "USD/RUB"
    interval = "1h"
    outputsize = "5000"
    api_key = "5326f043daec4329a7041b579b9aaa53"

    df = get_data(symbol, interval, outputsize, api_key)
    df = df.dropna()
    series = df['close'].values

    test_size = int(len(series) * 0.1)
    train, test = series[:-test_size], series[-test_size:]

    window = 120
    r = 20
    forecast = ssa_forecast(train, window, r, len(test))

    mse = mean_squared_error(test, forecast)
    mae = mean_absolute_error(test, forecast)
    mape = np.mean(np.abs((test - forecast) / test)) * 100

    print(f'Среднеквадратичная ошибка (MSE): {mse:.2f}')
    print(f'Средняя абсолютная ошибка (MAE): {mae:.2f}')
    print(f'Средняя абсолютная процентная ошибка (MAPE): {mape:.2f}%')

    plot_results(test, forecast, f'Прогноз цены {symbol} (SSA+AR)')

    U, s, Vt = ssa_decompose(train, window)
    recon = ssa_reconstruct(U, s, Vt, r)
    plt.figure(figsize=(12, 6))
    plt.plot(train, label='Исходный train')
    plt.plot(recon, label='Восстановленный SSA')
    plt.legend()
    plt.title('Сравнение исходного и восстановленного ряда (train)')
    plt.show()

if __name__ == "__main__":
    main()