from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

print("Training dataset shape (rows, columns):", train_df.shape)
print("Test dataset shape (rows, columns):", test_df.shape)

print("\nFirst five rows:")
print(train_df.head())

print("\nDataset information:")
train_df.info()

print("\nMissing values in each column:")
print(train_df.isnull().sum())

train_df["datetime"] = pd.to_datetime(train_df["datetime"])
test_df["datetime"] = pd.to_datetime(test_df["datetime"])

for data in [train_df, test_df]:
    data["year"] = data["datetime"].dt.year
    data["month"] = data["datetime"].dt.month
    data["day"] = data["datetime"].dt.day
    data["hour"] = data["datetime"].dt.hour
    data["weekday"] = data["datetime"].dt.day_of_week

target_column = "count"
feature_columns = [
    "season", "holiday", "workingday", "weather",
    "temp", "atemp", "humidity", "windspeed",
    "year", "month", "day", "hour", "weekday"
]

print("\nTarget:", target_column)
print("Features:", feature_columns)

category_columns = ["season", "weather", "hour", "month", "weekday"]

X = pd.get_dummies(train_df[feature_columns],
                   columns=category_columns,
                   drop_first=True).astype(float)
y = train_df[target_column]

X_test_final = pd.get_dummies(test_df[feature_columns],
                              columns=category_columns,
                              drop_first=True).astype(float)
X_test_final = X_test_final.reindex(columns=X.columns, fill_value=0)

print("\nShape of X:", X.shape)
print("Shape of y:", y.shape)
print("Shape of test features:", X_test_final.shape)

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42
)

y_train_log = np.log1p(y_train)
y_valid_log = np.log1p(y_valid)

print("\nTraining examples:", len(X_train))
print("Validation examples:", len(X_valid))

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)

print("Feature scaling completed.")

lin_model = LinearRegression()
lin_model.fit(X_train, y_train)

lin_pred = lin_model.predict(X_valid)
lin_pred[lin_pred < 0] = 0

print("\n--- Linear Regression (baseline) ---")
print(f"MAE  : {mean_absolute_error(y_valid, lin_pred):.3f}")
print(f"RMSE : {np.sqrt(mean_squared_error(y_valid, lin_pred)):.3f}")
print(f"R2   : {r2_score(y_valid, lin_pred):.3f}")

rmsle_list = []

for k in range(1, 21):
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train_scaled, y_train_log)
    pred_k_log = knn.predict(X_valid_scaled)
    pred_k = np.expm1(pred_k_log)
    pred_k[pred_k < 0] = 0
    rmsle_val = np.sqrt(mean_squared_error(y_valid_log, np.log1p(pred_k)))
    rmsle_list.append(rmsle_val)

best_k = int(np.argmin(rmsle_list)) + 1
print("\nBest K (lowest validation RMSLE):", best_k)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 21), rmsle_list, color="blue", linestyle="dashed",
         marker="o", markerfacecolor="red", markersize=8)
plt.title("RMSLE vs. K Value")
plt.xlabel("K (Number of Neighbors)")
plt.ylabel("RMSLE")
plt.xticks(range(1, 21))
plt.grid(True)
plt.show()

knn_model = KNeighborsRegressor(n_neighbors=best_k)
knn_model.fit(X_train_scaled, y_train_log)
print("KNN model training completed!")

y_pred_log = knn_model.predict(X_valid_scaled)
y_pred = np.expm1(y_pred_log)
y_pred[y_pred < 0] = 0

comparison = pd.DataFrame({
    "Actual Count": y_valid.to_numpy(),
    "Predicted Count": y_pred,
    "Error (Actual - Predicted)": y_valid.to_numpy() - y_pred
})
print("\nFirst 10 validation predictions:")
print(comparison.head(10))

mae = mean_absolute_error(y_valid, y_pred)
mse = mean_squared_error(y_valid, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_valid, y_pred)
rmsle = np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_valid)) ** 2))

print(f"\n--- KNN Regressor (K={best_k}) ---")
print(f"MAE   : {mae:,.3f}")
print(f"MSE   : {mse:,.3f}")
print(f"RMSE  : {rmse:,.3f}")
print(f"R2    : {r2:.3f}")
print(f"RMSLE : {rmsle:.3f}")

plt.figure(figsize=(7, 6))
plt.scatter(y_valid, y_pred, alpha=0.5, edgecolor="black")

low = min(float(np.min(y_valid)), float(np.min(y_pred)))
high = max(float(np.max(y_valid)), float(np.max(y_pred)))
plt.plot([low, high], [low, high], linestyle="--", label="Perfect prediction")

plt.xlabel("Actual rental count")
plt.ylabel("Predicted rental count")
plt.title("Actual vs Predicted Bike Rentals")
plt.legend()
plt.grid(alpha=0.25)
plt.show()

final_scaler = StandardScaler()
X_scaled = final_scaler.fit_transform(X)
X_test_scaled = final_scaler.transform(X_test_final)

y_log = np.log1p(y)
final_model = KNeighborsRegressor(n_neighbors=best_k)
final_model.fit(X_scaled, y_log)
print("\nFinal model trained on the full training data.")

test_predictions_log = final_model.predict(X_test_scaled)
predictions = np.expm1(test_predictions_log)
predictions[predictions < 0] = 0

Path("results").mkdir(parents=True, exist_ok=True)

submission = pd.DataFrame({
    "datetime": test_df["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S"),
    "count": np.round(predictions).astype(int)
})
submission.to_csv("results/submission.csv", index=False)

print("\nPredictions generated successfully!")
print(submission.head())
print("Submission shape:", submission.shape)
print("\nSubmission saved as results/submission.csv")

