
from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from src.preprocess import load_and_preprocess
from src.window_generator import TimeSeriesWindowGenerator

app = Flask(__name__)

model = tf.keras.models.load_model("models/lstm_model.keras")

df = load_and_preprocess()
window = TimeSeriesWindowGenerator(df)
scaled = window.preprocess()
latest_seq = scaled[-window.sequence_length:]
latest_seq = np.expand_dims(latest_seq, axis=0)  

@app.route("/")
def index():
    return "Energy Forecast API"

@app.route("/predict", methods=["GET"])
def predict():
    prediction = model.predict(latest_seq)
    result = float(prediction[0][0])
    return jsonify({
        "predicted_energy_usage": result
    })

if __name__ == "__main__":
    app.run(debug=True)
