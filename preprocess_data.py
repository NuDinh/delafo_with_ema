import pandas as pd
import numpy as np
import pickle
from dateutil.relativedelta import relativedelta
from sklearn.model_selection import train_test_split

def calculateEma(series, period, keep_length= True):
    ema = []
    num_ticker = series.shape[1]
    empty = [0 for _ in range(num_ticker)]
    if keep_length:
        ema = [empty for _ in range(period - 1)]
    # print(ema)

    #get n sma first and calculate the next n period ema
    sma = sum(series[:period]) / period
    multiplier = 2 / float(1 + period)
    ema.append(sma)
    j = len(ema)

    #EMA(current) = ( (Price(current) - EMA(prev) ) x Multiplier) + EMA(prev)
    ema.append(((series[period] - sma) * multiplier) + sma)

    #now calculate the rest of the values
    for i in series[period+1:]:
        tmp = ((i - ema[j]) * multiplier) + ema[j]
        j = j + 1
        ema.append(tmp)
    return np.asarray(ema, dtype= np.float32)

def prepair_data(path, window_x, window_y, periods, data_from, data_to):
    data = pd.read_csv(path)
    original_dataset = data.copy()
    original_dataset['date'] = pd.to_datetime(original_dataset.date)
    filter_dataset = original_dataset.loc[(pd.DatetimeIndex(original_dataset['date']).year >= data_from) \
                                        & (pd.DatetimeIndex(original_dataset['date']).year <= data_to)]
    filter_dataset['dow'] = filter_dataset.date.apply(lambda x: x.dayofweek)
    filter_dataset_w = filter_dataset.loc[(filter_dataset['dow'] <= 4)&(filter_dataset['dow']>=0)]
    filter_dataset_w = filter_dataset_w.drop(['dow'], axis = 1)

    filter_dataset_w['date'] = filter_dataset_w.date.dt.date
    filter_dataset_w = filter_dataset_w.loc[filter_dataset_w.volume >= 0]

    filter_dataset_w = filter_dataset_w.pivot_table(index = 'date', columns= 'ticker')

    columns = filter_dataset_w.columns

    # remove ticket having ratio of null > 0.05
    for col in columns:
        if filter_dataset_w[col].isnull().sum()/filter_dataset_w.shape[0] > 0.05:
            filter_dataset_w = filter_dataset_w.drop([col], axis = 1)

    # check return greater than 7%
    closes = filter_dataset_w.close
    daily_return = ((closes.shift(-1) - closes)/ closes).shift(1)
    remain_daily_return = daily_return.drop([tick for tick in daily_return.columns if (abs(daily_return[tick] > 0.07)).any()], axis = 1)

    df_final = filter_dataset_w.iloc[:,filter_dataset_w.columns.get_level_values(1).isin(remain_daily_return.columns)]


    # removing data do not transaction in 3 months ago

    last_date = max(df_final.close.reset_index().date)
    last_date_minus_3_months = last_date + relativedelta(months=-3)
    proc_dt_minus_3_months_str = last_date_minus_3_months.strftime('%Y-%m-%d')

    df_3m = df_final.close.reset_index()
    df_3m['date'] = pd.to_datetime(df_3m['date'])

    df_3_tmp = df_3m.loc[df_3m.date >= proc_dt_minus_3_months_str]

    # remove tiker disapper in 3 months ago
    remain_ticker_3 = df_3_tmp.drop([col for col in df_3_tmp.columns if df_3_tmp[col].isnull().any()], axis =1)
    df_3_3 = df_final.iloc[:, df_final.columns.get_level_values(1).isin(remain_ticker_3.columns)]

    # replace missing data in volume => replace by 0
    df_3_3.volume = df_3_3.volume.fillna(0)
    # df_3_3.close = df_3_3.close.interpolate(method = 'linear', limit_area= 'inside', limit_direction='both', axis = 0)
    df_3_3.close = df_3_3.close.ffill()

    # split data into train & test set, which output equal daily_return
    close = df_3_3.close
    daily_return = ((close.shift(-1) - close)/close).shift(1)
    daily_return = daily_return.fillna(0)

    tickers = df_3_3.close.columns
    X = df_3_3.values.reshape(df_3_3.shape[0],2,-1)
    y = daily_return.values

    X[np.isnan(X)] = 0.0
    y[np.isnan(y)] = -1e2
    close = X[:,0,:]

    if isinstance(periods, int):
        periods = [periods]
    max_period = max(periods)
    if max_period != 0:
        for period in periods:
            ema  = calculateEma(close, period)
        # ema34 = calculateEma(close, 34)
        # ema89 = calculateEma(close, 89)
        # ema100 = calculateEma(close, 100)
        # ema200 = calculateEma(close, 200)
            X = np.concatenate((X, ema[:, np.newaxis, :]), axis=1)
    
    X = X[max_period:]
    y = y[max_period:]
    # X1 = rolling_array(X[window_x:],stepsize=1,window=window_y)

    X = rolling_array(X[:-window_y],stepsize=1,window=window_x)
    y = rolling_array(y[window_x:],stepsize=1,window=window_y)
    X = np.moveaxis(X,-1,1)
    # X1 = np.moveaxis(X1,-1,1)
    y = np.swapaxes(y,1,2)

    return X,y,tickers

def rolling_array(a, stepsize=1, window=60):
    n = a.shape[0]
    return np.stack((a[i:i + window:stepsize] for i in range(0,n - window + 1)),axis=0)
