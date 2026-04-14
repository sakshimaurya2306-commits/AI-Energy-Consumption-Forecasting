import pandas as pd

def load_data(path):
    df = pd.read_csv(path, parse_dates=['Datetime'])
    df.set_index('Datetime', inplace=True)

    df = df.resample('h').mean()
    df = df.ffill()

    return df