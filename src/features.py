import numpy as np

def create_features(df):

    df['hour'] = df.index.hour
    df['day'] = df.index.dayofweek

    # safer lag (only if enough data)
    if len(df) > 2:
        df['lag_1'] = df['Energy'].shift(1)
        df['rolling_mean'] = df['Energy'].rolling(2).mean()
    else:
        df['lag_1'] = df['Energy']
        df['rolling_mean'] = df['Energy']

    # ❗ REMOVE lag_24 for now
    # df['lag_24'] = df['Energy'].shift(24)

    df = df.dropna()

    return df