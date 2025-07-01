import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_model(y_test, y_pred, save_path="results/plots.png"):
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")

    plt.figure(figsize=(12, 6))
    plt.plot(y_test[:100], label="Gerçek", linewidth=2)
    plt.plot(y_pred[:100], label="Tahmin", linewidth=2)
    plt.title("Gerçek vs Tahmin (ilk 100 örnek)")
    plt.xlabel("Zaman")
    plt.ylabel("Enerji Tüketimi (scaled)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
