from src.preprocess import load_and_preprocess
from src.window_generator import TimeSeriesWindowGenerator
from src.model_lstm import train_lstm
from src.model_mlp import train_mlp
from src.evaluate import evaluate_model

def main():
    print(" Veri yükleniyor...")
    df = load_and_preprocess()
    print(df.head())

    print("\n Sekanslar oluşturuluyor...")
    window = TimeSeriesWindowGenerator(df)
    X_train, y_train, X_test, y_test = window.generate()

    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

    print("\n --- LSTM Modeli ---")
    lstm_model, lstm_pred = train_lstm(X_train, y_train, X_test, y_test)
    evaluate_model(y_test, lstm_pred, save_path="results/lstm_plot.png")
    lstm_model.save("models/lstm_model.keras")

    print("\n --- MLP Modeli ---")
    mlp_model, mlp_pred = train_mlp(X_train, y_train, X_test, y_test)
    evaluate_model(y_test, mlp_pred, save_path="results/mlp_plot.png")
    mlp_model.save("models/mlp_model.keras")


if __name__ == "__main__":
    main()
