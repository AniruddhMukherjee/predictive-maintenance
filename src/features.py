import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Sensors that are constant aross all engines in FD001 - carry 0 information
# Indentified during EDA removing them reduces noise
USELESS_SENSORS = ['sensor1', 'sensor10', 'sensor19', 'sensor18',
                    'sensor16', 'sensor5', 'sensor6']

# operational settings close to 0
USELESS_SETTINGS = ['sensor3']

def drop_useless_columns(df):
    """Remove sensors and settings that have near 0 variance(useless for prediction)"""
    cols_to_drop = USELESS_SENSORS + USELESS_SETTINGS
    df = df.drop(columns=cols_to_drop)
    return df

def add_rolling_features(df, window=10):
    """
    for each sensor, compute the rolling mean and std for each cycle.
    group by engine_id so as to not leak data across diff engines

    Rolling mean - smoothes noise, shows trend direction
    rolling std - shows how unstable the sensor is getting
    """
    # Identify sensor columns (everything except metadata)
    sensor_cols = [c for c in df.columns if c.startswith('sensor')]

    # groupby engines so that rolling windows don't bleed across engines
    grouped = df.groupby('engine_id')

    for col in sensor_cols:
        #rolling mean - fill NaN at start of window with first valid value
        df[f'{col}_mean{window}'] = grouped[col].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        #rolling std - how much the sensor fluctuating
        df[f'{col}_std{window}'] = grouped[col].transform(
            lambda x: x.rolling(window, min_periods=1).std().fillna(0)
        )

    return df

def add_lag_features(df, lags=[1,3,5]):
    """
    Lag features: what the sensor was reading N cycles ago?
    Helps the model understand rate of change wo needing sequence input
    Useful espically for XGBoost which can't handle sequences natively.
    """

    sensor_cols = [c for c in df.columns if c.startswith('sensor') and
                   '_mean' not in c and '_std' not in c] # only original sensors
    
    grouped = df.groupby('engine_id')

    for col in sensor_cols:
        for lag in lags:
            #shift(lag) gives value from lag cycles ago
            # fill_value=0 for the first few cycles where lag doesnt exist
            df[f'{col}_lag{lag}'] = grouped[col].transform(
                lambda x: x.shift(lag, fill_value=0)
            )

    return df

def clip_rul(df,max_rul=125):
    """
    clip RUL at maximum value (linear target)
    Engines are healthy for most of their life - don't care about predicting
    RUL = 300 / RUL = 280, engine fine either way
    We care about the ldegredation cycle (last ~125 cycles)
    """

    df['RUL'] = df['RUL'].clip(upper=max_rul)
    return df

def normalize_features(train_df, test_df):
    """
    MinMax normalize all column features to [0,1]
    Fit scaler on TRAIN only, then apply to both train and test.
    """
    # All cols except metadata and target
    exclude = ['engine_id', 'cycle', 'RUL']
    feature_cols = [c for c in train_df.columns if c not in exclude]

    scaler = MinMaxScaler()

    # fit on train, tranform on both
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])

    return train_df, test_df, scaler

def run_feature_pipeline(train_df, test_df):
    """
    Master function - runs all feature engineering steps in order
    This will process the train and test dataframes
    """

    #Step 1: drop useless sensors
    train_df = drop_useless_columns(train_df)
    test_df = drop_useless_columns(test_df)
    print(f"After dropping useless cos: {train_df.shape}")

    #Step 2: clip RUL to max 125 (piecewise linear target)
    train_df = clip_rul(train_df)
    print(f"RUL clipped at 125. Max RUL: {train_df['RUL'].max()}")

    #Step 3: rolling features
    train_df = add_rolling_features(train_df)
    test_df = add_rolling_features(test_df)
    print(f"After rolling features: {train_df.shape}")

    #Step 4: lag features
    train_df = add_lag_features(train_df)
    test_df = add_lag_features(test_df)
    print(f"After lag features: {train_df.shape}")

    #Step 5: normalize
    train_df, test_df, scaler = normalize_features(train_df, test_df)
    print("Normalization done.")

    print(f"\n Final train shape: {train_df.shape}")
    print(f"\n Final test shape: {test_df.shape}")

    return train_df, test_df, scaler

if __name__ == '__main__':
    # test pipeline end to end
    from data_loader import load_data, add_rul

    train, test, rul = load_data()
    train = add_rul(train)

    train_processed, test_processed, scaler = run_feature_pipeline(train, test)
    print("\nSample of processed training data:")
    print(train_processed.head())
