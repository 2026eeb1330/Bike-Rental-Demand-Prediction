import pandas as pd
import numpy as np

def load_data(filepath):
    df = pd.read_csv(filepath)
    print("Dataset shape:", df.shape)
    print("\nFirst five rows:")
    print(df.head())
    print("\nDataset information:")
    df.info()
    print("\nMissing values:")
    print(df.isnull().sum())
    return df

def preprocess_data(df):
   
    df = df.drop_duplicates().copy()

    df["datetime"] = pd.to_datetime(df["datetime"])
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["day"] = df["datetime"].dt.day
    df["hour"] = df["datetime"].dt.hour
    df["weekday"] = df["datetime"].dt.weekday

    target_column = "count"
    feature_columns = [
        "season", "holiday", "workingday", "weather",
        "temp", "atemp", "humidity", "windspeed",
        "year", "month", "day", "hour", "weekday"
    ]

   
    for column in feature_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
        df[column] = df[column].fillna(df[column].median())


    df[target_column] = pd.to_numeric(df[target_column], errors="coerce")
    df = df.dropna(subset=[target_column])

    X = df[feature_columns]
    y = df[target_column]

    print("\nTarget column:", target_column)
    print("Feature columns:", feature_columns)
    print("Shape of X:", X.shape)
    print("Shape of y:", y.shape)
    return X, y, feature_columns

if __name__ == "__main__":
    df = load_data("data/train.csv")
    X, y, features = preprocess_data(df)
