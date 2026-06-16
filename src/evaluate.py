import numpy as np

def rmse(y_true, y_pred):
    """Root mean Squared error - regression metric"""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mae(y_true, y_pred):
    """Mean absolute error - prediction error"""
    return np.mean(np.abs(y_true - y_pred))

def nasa_score(y_true, y_pred):
    """
    NANA's custom scoring for RUL prediction.
    fine LATE predictions more than EARLY predictions

    predicting late is BAD
    predicting early is unnescessary

    d = predicted - actual (+ve = predicted late, -ve = predicted early)
    Score = sum(exp(-d/13) - 1) for d < 0 (early)
            sum(exp(d/10) - 1) for d >= 0 (late)

    Lower the score the better
    """

    d = y_pred - y_true
    score = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) -1)
    return np.sum(score)

def evaluate(y_true, y_pred, model_name="Model"):
    """Print all metrics for model"""
    r = rmse(y_true, y_pred)
    m = mae(y_true, y_pred)
    s = nasa_score(y_pred, y_true)

    print(f"   {model_name} Results: ")
    print(f" \nRMSE:       {r: .2f} cycles")
    print(f" \nMAE:        {m: .2f} cycles")
    print(f" \nNASA Score: {s: .2f} (lower is better)")

    return {'rmse': r, 'mae': m, 'nasa_score': s}