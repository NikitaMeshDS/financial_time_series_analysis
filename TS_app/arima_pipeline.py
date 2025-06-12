import pandas as pd
import numpy as np
import requests
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

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

def preprocess_data(data, interval, data_size):
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
        "1h": "H",
        "5min": "5T"
    }
    
    full_index = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq=freq_map[interval]
    )
    
    df = df.reindex(full_index)
    df['close'] = df['close'].interpolate(method='time').ffill().bfill()
    
    return df

def check_stationarity(timeseries):
    if timeseries.isna().any() or np.isinf(timeseries).any():
        print("Обнаружены пропуски в данных")
        timeseries = timeseries.replace([np.inf, -np.inf], np.nan)
        timeseries = timeseries.dropna()
    
    if len(timeseries) < 2:
        return False
    
    try:
        timeseries_array = timeseries.values
        result = adfuller(timeseries_array)
        print('Тест Дики-Фуллера:')
        print(f'ADF статистика: {result[0]}')
        print(f'p-значение: {result[1]}')
        print('Критические значения:')
        for key, value in result[4].items():
            print(f'\t{key}: {value}')
        
        return result[1] < 0.05
    except Exception as e:
        print(f"Ошибка при проверке стационарности: {str(e)}")
        return False

def find_best_arima_params(data, max_p=5, max_d=2, max_q=5):
    best_aic = float('inf')
    best_order = None
    
    total_combinations = (max_p + 1) * (max_d + 1) * (max_q + 1)
    current_combination = 0
    
    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                current_combination += 1
                print(f"\rПрогресс: {current_combination}/{total_combinations} комбинаций", end="")
                
                try:
                    model = ARIMA(data, order=(p, d, q))
                    results = model.fit()
                    if results.aic < best_aic:
                        best_aic = results.aic
                        best_order = (p, d, q)
                        print(f"\nлучшие параметры: {best_order} (AIC: {best_aic:.2f})")
                except:
                    continue
    
    print(f"\n\nЛучшие параметры ARIMA: {best_order}")
    return best_order

def plot_results(actual, predicted, title):
    
    plt.figure(figsize=(12, 6))
    plt.plot(actual, label='Фактические значения')
    plt.plot(predicted, label='Прогноз')
    plt.title(title)
    plt.xlabel('Время')
    plt.ylabel('Цена')
    plt.legend()
    plt.show()

def evaluate_model(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    print(f'Среднеквадратичная ошибка (MSE): {mse:.2f}')
    print(f'Средняя абсолютная ошибка (MAE): {mae:.2f}')
    print(f'Средняя абсолютная процентная ошибка (MAPE): {mape:.2f}%')

def main():

    symbol = "USD/RUB"
    interval = "1h"
    outputsize = "5000"
    api_key = "5326f043daec4329a7041b579b9aaa53"
    
    df = get_data(symbol, interval, outputsize, api_key)
    
    df['close'] = df['close'].replace([np.inf, -np.inf], np.nan)
    df['close'] = df['close'].dropna()
    
    print(f"Размер датасета после очистки: {len(df)}")
    
    print("\nПроверка стационарности исходного ряда:")
    is_stationary = check_stationarity(df['close'])
    print(f"Ряд {'стационарен' if is_stationary else 'не стационарен'}")
    
    if not is_stationary:
        diff_df = pd.DataFrame(index=df.index)
        diff_df['close_diff'] = df['close'].diff()
        
        diff_df['close_diff'] = diff_df['close_diff'].replace([np.inf, -np.inf], np.nan)
        diff_df['close_diff'] = diff_df['close_diff'].dropna()
        
        print(f"Размер датасета после взятия разностей: {len(diff_df)}")
        
        if len(diff_df) > 0:
            print("\nПроверка стационарности после взятия разностей:")
            is_stationary = check_stationarity(diff_df['close_diff'])
            print(f"Ряд {'стационарен' if is_stationary else 'не стационарен'}")
        else:
            print("Недостаточно данных после взятия разностей")
            is_stationary = False
    
    train_size = int(len(df) * 0.9)
    train_data = df['close'][:train_size]
    test_data = df['close'][train_size:]
    
    print(f"\nРазмер обучающей выборки: {len(train_data)}")
    print(f"Размер тестовой выборки: {len(test_data)}")
    
    print("\nПоиск оптимальных параметров:")
    best_order = find_best_arima_params(train_data)
    
    model = ARIMA(train_data, order=best_order)
    model_fit = model.fit()

    print("\nРезультаты обучения модели:")
    print(model_fit.summary())

    forecast_steps = len(test_data)
    forecast = model_fit.forecast(steps=forecast_steps)
    
    print("\nОценка качества модели:")
    evaluate_model(test_data, forecast)
    
    plot_results(test_data, forecast, f'Прогноз цены {symbol} (ARIMA)')

if __name__ == "__main__":
    main() 