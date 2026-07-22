import joblib
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


class RULModel:

    def __init__(self):
        self.model = XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42)

    def split_data(self, train):
        X = train.drop(columns=["engine_id", "cycle", "RUL"])
        y = train["RUL"]
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train(self, train):
        X_train, X_test, y_train, y_test = self.split_data(train)
        self.model.fit(X_train, y_train)
        pred = self.model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        r2 = r2_score(y_test, pred)
        print(f"RMSE : {rmse:.2f}")
        print(f"R²    : {r2:.3f}")
        return self.model

    def save(self, path="models/xgboost_model.pkl"):
        joblib.dump(self.model, path)
        print("Model saved.")

//USAGE
from dataset import CMAPSSData
from model import RULModel

data = CMAPSSData(
    "data/train_FD001.txt",
    "data/test_FD001.txt",
    "data/RUL_FD001.txt")

train, test, rul = data.load()
model = RULModel()
model.train(train)
model.save()
