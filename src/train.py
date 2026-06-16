import sys 
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_data, add_rul
from features import run_feature_pipeline
# from models.xgboost_model import train, predict_test, save_model
from models.xgboost_model import train as xgb_train, predict_test as xgb_predict, save_model as xgb_save
from models.lstm_model import train as lstm_train, predict_test as lstm_predict, save_model as lstm_save
from evaluate import evaluate

def main():
    
    print(" predictive maintenance -- XGBoost Baseline ")

    ## load and engineer data
    
    #Step 1: Load data
    print("data...")
    train_df, test_df, rul_df = load_data()
    train_df = add_rul(train_df)

    #step2: feature Engineering
    train_df, test_df, scaler = run_feature_pipeline(train_df, test_df)

    # ONLY XGBOOST
    # #step3: train
    # model = train(train_df)

    # #step4: predict and evaluate
    # predictions, actuals = predict_test(model, test_df, rul_df)
    # evaluate(actuals, predictions, model_name="XGBoost Baseline")

    # #step 5: Save model
    # save_model(model)

    actuals = rul_df['RUL'].values

    # XGBoost
    print("\n XGBoost")
    xgb_model = xgb_train(train_df)
    xgb_preds, _= xgb_predict(xgb_model, test_df, rul_df)
    evaluate(actuals, xgb_preds, model_name="XGBoost Baseline")
    xgb_save(xgb_model)

    # LSTM
    print("\n LSTM")
    lstm_model = lstm_train(train_df)
    lstm_preds = lstm_predict(lstm_model, test_df)
    evaluate(actuals, lstm_preds, model_name="LSTM")
    lstm_save(lstm_model)

if __name__ == '__main__':
    main()