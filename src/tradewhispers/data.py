import pandas as pd
import numpy as np

def forward_returns(prices, news, horizons=(1, 2, 3)):
    fx = pd.read_csv(prices).rename(columns={'date': 'timestamp'})
    fx['timestamp'] = pd.to_datetime(fx['timestamp'])
    fx = fx[['timestamp', 'mid_price']]
    fx['day'] = fx['timestamp'].dt.date
    day_end = fx.groupby('day')['timestamp'].transform('max')

    for h in horizons:
        future = fx['mid_price'].shift(-h)
        valid = (fx['timestamp'] + pd.Timedelta(minutes=h) <= day_end)
        fx[f't+{h}'] = np.where(valid, future - fx['mid_price'], np.nan)

    news = pd.read_csv(news).rename(columns={'date': 'timestamp'})
    news['timestamp'] = pd.to_datetime(news['timestamp'])
    merged = pd.merge(news, fx.drop(columns=['day']), on='timestamp', how='left')
    merged = merged.dropna(subset=[f't+{h}' for h in horizons], how='all')
    return merged


def evaluate_predictions(df, threshold=None):
    return_cols = [c for c in df.columns if c.startswith("t+")]
    for col in return_cols:
        if threshold is None:
            df[col] = df[col].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        else:
            df[col] = df[col].apply(
                lambda x: 0 if abs(x) <= threshold else (1 if x > 0 else -1)
            )
    return df