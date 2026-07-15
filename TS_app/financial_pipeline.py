import os
import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

def get_data(symbol, interval, outputsize, api_key):
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": api_key
    }
    url = "https://api.twelvedata.com/time_series"
    data = requests.get(url, params=params).json()
    return preprocess_data(data, interval)

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
    
    freq_map = {
        "1day": "D",
        "1h": "h",
        "5min": "5min"
    }
    
    full_index = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq=freq_map[interval]
    )
    
    df = df.reindex(full_index)
    df['close'] = df['close'].interpolate(method='time').ffill().bfill()
    
    return df

def add_technical_indicators(df):
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    return df

def prepare_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

def create_lstm_model(seq_length, n_features):
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(seq_length, n_features), return_sequences=True),
        Dropout(0.2),
        LSTM(50, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

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
    symbol = "USD/EUR"
    interval = "1h"
    outputsize = "5000"
    api_key = os.environ.get("TWELVEDATA_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Set the TWELVEDATA_API_KEY environment variable before running. "
            "Never hardcode API keys in source files that go to a public repo."
        )

    df = get_data(symbol, interval, outputsize, api_key)
    df = add_technical_indicators(df)
    
    df = df.dropna()
    
    features = ['close', 'SMA_20', 'SMA_50', 'RSI', 'MACD']
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[features])
    
    seq_length = 60
    X, y = prepare_sequences(scaled_data, seq_length)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)

    model = create_lstm_model(seq_length, len(features))
    history = model.fit(
        X_train, y_train[:, 0],
        epochs=50,
        batch_size=32,
        validation_split=0.1,
        verbose=1
    )
    
    predictions = model.predict(X_test)
    
    predictions_full = np.zeros((len(predictions), len(features)))
    predictions_full[:, 0] = predictions.reshape(-1)

    
    y_test_full = np.zeros((len(y_test), len(features)))
    y_test_full[:, 0] = y_test[:, 0]
    
    predictions_original = scaler.inverse_transform(predictions_full)[:, 0]
    actual_original = scaler.inverse_transform(y_test_full)[:, 0]
    
    print(f'Среднеквадратичная ошибка (MSE): {mean_squared_error(actual_original, predictions_original):.2f}')
    print(f'Средняя абсолютная ошибка (MAE): {mean_absolute_error(actual_original, predictions_original):.2f}')
    print(f'Средняя абсолютная процентная ошибка (MAPE): {np.mean(np.abs((actual_original - predictions_original) / actual_original)) * 100:.2f}%')

    plot_results(actual_original, predictions_original, f'Прогноз цены {symbol} (LSTM)')

if __name__ == "__main__":
    main() 