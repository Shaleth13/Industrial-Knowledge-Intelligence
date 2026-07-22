import joblib
import pandas as pd


class Predictor:
    def __init__(self, model_path="models/xgboost_model.pkl"):
        self.model = joblib.load(model_path)

    def predict(self, test):
        X = test.drop(columns=["engine_id", "cycle"])
        predictions = self.model.predict(X)
        result = pd.DataFrame({
            "engine_id": test["engine_id"],
            "cycle": test["cycle"],
            "Predicted_RUL": predictions})
        return result

    def latest_prediction(self, predictions):
        latest = (predictions.groupby("engine_id").tail(1).sort_values("engine_id").reset_index(drop=True))
        return latest

#USAGE

from dataset import CMAPSSData
from predictor import Predictor

data = CMAPSSData(
    "data/train_FD001.txt",
    "data/test_FD001.txt",
    "data/RUL_FD001.txt")

train, test, rul = data.load()
predictor = Predictor()
predictions = predictor.predict(test)
latest = predictor.latest_prediction(predictions)
print(latest.head())
