import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class CMAPSSData:

    def __init__(self, train_file, test_file, rul_file):
        self.train_file = train_file
        self.test_file = test_file
        self.rul_file = rul_file
        self.scaler = MinMaxScaler()
        self.cols = (["engine_id", "cycle", "setting1", "setting2", "setting3"] + [f"sensor{i}" for i in range(1, 22)])

    def read_files(self):
        train = pd.read_csv(self.train_file, sep=r"\s+", header=None)
        test = pd.read_csv(self.test_file, sep=r"\s+", header=None)
        rul = pd.read_csv(self.rul_file, header=None, names=["RUL"])
        train = train.dropna(axis=1)
        test = test.dropna(axis=1)
        train.columns = self.cols
        test.columns = self.cols
      
        return train, test, rul

    def add_rul(self, train):

        last_cycle = (train.groupby("engine_id")["cycle"].max().rename("max_cycle"))
        train = train.merge(last_cycle, on="engine_id")
        train["RUL"] = train["max_cycle"] - train["cycle"]
        train.drop(columns="max_cycle", inplace=True)

        return train

    def scale(self, train, test):
        features = [col for col in train.columns if col not in ["engine_id", "cycle", "RUL"]]
        train[features] = self.scaler.fit_transform(train[features])
        test[features] = self.scaler.transform(test[features])
        return train, test

    def load(self):
        train, test, rul = self.read_files()
        train = self.add_rul(train)
        train, test = self.scale(train, test)
        return train, test, rul
