
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from src.config import SEQUENCE_LENGTH, TEST_RATIO

class TimeSeriesWindowGenerator:
    def __init__(self, df):
        self.df = df
        self.scaler = MinMaxScaler()
        self.sequence_length = SEQUENCE_LENGTH

    def preprocess(self):
        scaled_values = self.scaler.fit_transform(self.df)
        return scaled_values

    def split_sequences(self, scaled_data):
        X, y = [], []
        for i in range(len(scaled_data) - self.sequence_length):
            seq_x = scaled_data[i:i + self.sequence_length]
            seq_y = scaled_data[i + self.sequence_length]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)

    def train_test_split(self, X, y):
        split_index = int(len(X) * (1 - TEST_RATIO))
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        return X_train, y_train, X_test, y_test

    def generate(self):
        scaled = self.preprocess()
        X, y = self.split_sequences(scaled)
        return self.train_test_split(X, y)
