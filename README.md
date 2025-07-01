# PowerCast: Energy Forecasting with LSTM and MLP

PowerCast is an end-to-end time series forecasting project that predicts hourly household energy consumption.  
It uses deep learning models (LSTM and MLP), compares their performance, and exposes a REST API for predictions.

## Project Overview

Dataset: UCI Individual Household Electric Power Consumption  
URL: https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption

This project includes:

- Data preprocessing and hourly resampling
- Sequence generation for time series modeling
- LSTM and MLP model training
- Evaluation with MAE and RMSE
- Visual comparison of predictions
- Flask-based REST API for inference

## Project Structure

energy_forecasting/
│
├── data/ # Raw dataset (not tracked by git)
│ └── household_power_consumption.txt
├── models/ # Trained model files (.keras)
├── results/ # Evaluation plots (PNG)
├── src/ # Source modules
│ ├── config.py
│ ├── preprocess.py
│ ├── window_generator.py
│ ├── model_lstm.py
│ ├── model_mlp.py
│ ├── evaluate.py
│ └── init.py
├── app/
│ └── flask_api.py # REST API endpoint
├── main.py # Entry point for training/evaluation
├── requirements.txt
├── .gitignore
└── README.md


## How to Run the Project

### 1. Download the Dataset

Due to GitHub file size restrictions, the dataset is not included in the repository.

You must manually download it from the [UCI repository](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption)  
and place it in the following path:

energy_forecasting/data/household_power_consumption.txt


### 2. Install Dependencies

Run the following command to install all required packages:

pip install -r requirements.txt


### 3. Train and Evaluate the Models

Execute the pipeline using:

python main.py


This will:
- Preprocess the dataset
- Generate time series sequences
- Train both LSTM and MLP models
- Evaluate and compare their predictions
- Save the models and output plots

## REST API (Flask)

To start the Flask API server and get predictions:

python app/flask_api.py


Then access:

http://127.0.0.1:5000/predict


This will return the next-hour energy consumption prediction using the trained LSTM model.

## Output Plots

Prediction graphs are saved in the `results/` folder.  
They show the first 100 prediction points vs actual usage for both models.

## License

MIT License © 2025 Moongazing

## Author

This project is developed by [Moongazing](https://github.com/Moongazing)  
Repository: https://github.com/Moongazing/PowerCast
