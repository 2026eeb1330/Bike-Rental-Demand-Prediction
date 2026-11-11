# Predicting Bike Rental Demand Based on Weather and Time

## Objective
Predict the number of bike rentals using weather conditions and date/time information.

## Dataset
`train.csv` and `test.csv` (from https://www.kaggle.com/competitions/bike-sharing-demand/data) are already in the `data` folder.

## Machine Learning algorithm
This project uses **Multiple Linear Regression**, as covered in the supervised-learning coursework.

The model predicts a numerical target (`count`) from multiple input features.

## Features used
season, holiday, workingday, weather, temp, atemp, humidity, windspeed, year, month, day, hour and weekday.

`casual` and `registered` are excluded because their sum equals `count`; they would reveal the target and are not available in Kaggle's test data.

## Project structure
```text
Bike-Rental-Demand-Prediction/
├── data/
│   ├── train.csv
│   └── test.csv
├── results/
├── src/
│   ├── data_preprocessing.py
│   ├── exploratory_data_analysis.py
│   ├── model_training.py
│   └── prediction.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Install libraries
Run in terminal from the project root:
```bash
pip install -r requirements.txt
```

## Run the files
Run these commands from the project root, in this order:
```bash
python src/exploratory_data_analysis.py
python src/model_training.py
python src/prediction.py
```

The scripts generate charts in `results/` and create `results/submission.csv`. `prediction.py` trains the same Linear Regression model again (same 80/20 split and `random_state=42`) and then predicts the Kaggle test data.

## Evaluation
The model is evaluated using:
- MAE: Mean Absolute Error
- MSE: Mean Squared Error
- RMSE: Root Mean Squared Error
- R²: Coefficient of determination

## Team division
- Member 1: Data loading, inspection and preprocessing.
- Member 2: Exploratory data analysis and graphs.
- Member 3: Linear Regression training, predictions and evaluation.
- Member 4: Prediction on unseen Kaggle test data and integration.

Add the actual names of all four members and their contributions before final submission.
