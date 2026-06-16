import sys 
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_data, add_rul
from features import run_feature_pipeline
from models.xgboost_model import train, predict_test, save_model
from evaluate import evaluate

def main():
    print(" predictive maintenance -- XGBoost Baseline ")

    #Step 1: Load data
    print("data...")
    train_df, test_df, rul_df = load_data()
    train_df = add_rul(train_df)

    #step2: feature Engineering
    train_df, test_df, scaler = run_feature_pipeline(train_df, test_df)

    #step3: train
    model = train(train_df)

    #step4: predict and evaluate
    predictions, actuals = predict_test(model, test_df, rul_df)
    evaluate(actuals, predictions, model_name="XGBoost Baseline")

    #step 5: Save model
    save_model(model)

if __name__ == '__main__':
    main()