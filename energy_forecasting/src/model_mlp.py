import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from src.config import EPOCHS, BATCH_SIZE

def train_mlp(X_train, y_train, X_test, y_test):
    # LSTM 3D input -> MLP 2D input (flatten)
    X_train_flat = X_train.reshape((X_train.shape[0], -1))
    X_test_flat = X_test.reshape((X_test.shape[0], -1))

    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(X_train_flat.shape[1],)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')

    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    print("MLP modeli eğitiliyor...")
    model.fit(
        X_train_flat, y_train,
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=1
    )

    y_pred = model.predict(X_test_flat)
    return model, y_pred
