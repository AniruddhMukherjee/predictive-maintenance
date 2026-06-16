import xgboost as xgb
import numpy as np
import pandas as pd
import joblib
import os

def build_xgb_model():
    """
    XGBoost regressor for RUL prediction.

    n_estimators     - number of trees to build
    max_depth        - how deep each tree can go (controls overfitting)
    learning_rate    - how much each tree contributes (lower = more robust)
    subsample        - fraction of data used per tree (prevents overfitting)
    colsample_bytree - fraction of features used per tree
    random_state     - reproductibility
    """
    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1      # use all CPU cores
    )
    return model

def get_features_and_target(df):
    """
    Split dataframe into features (X) and target (y).
    Drop metadata columns taht shouldn't be model inputs.
    """
    drop_cols = ['engine_id', 'cycle', 'RUL']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df['RUL'] if 'RUL' in df.columns else None
    return X, y

def train(train_df):
    """Train XGBoost on processed training data."""
    X_train, y_train = get_features_and_target(train_df)

    model = build_xgb_model()
    print(f"Training XGBoost on {X_train.shape[0]} samples, {X_train.shape[1]} features...")

    model.fit(X_train, y_train)
    print("training compelete.")
    return model

def predict_test(model, test_df, rul_df):
    """
    For test data, CMAPSS only gives us the last cycle of each engine.
    So we take the last row per engine and predict RUL from there
    Then compare against ground truth RUL from rul_df.
    """
    # Take last cycle for each engine (that's what we predict on)
    last_cycles = test_df.groupby('engine_id').last().reset_index()

    X_test, _ = get_features_and_target(last_cycles)

    predictions = model.predict(X_test)

    #clip predictions - RUL cant be -ve or above 125
    predictions = np.clip(predictions, 0, 125)

    return predictions, rul_df['RUL'].values

def save_model(model, path='models/xgboost_rul.pkl'):
    """Save trained model"""
    os.makedirs(os.path.dirname(path), exist_ok = True)
    joblib.dump(model, path)
    print(f"Model saved to {path}")

def load_model(path='models/xgboost_rul.pkl'):
    """Load saved model from disk."""
    return joblib.load(path)