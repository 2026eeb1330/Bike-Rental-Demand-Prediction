# MEMBER 3: LINEAR REGRESSION MODEL TRAINING AND EVALUATION

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from data_preprocessing import load_data, preprocess_data

# Read and prepare the training dataset
df = load_data("data/train.csv")
X, y, feature_columns = preprocess_data(df)

# Split data: 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42
)

print("Training examples:", len(X_train))
print("Testing examples:", len(X_test))

# Create and train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)
print("\nModel training completed!")

# Make predictions on test data
y_pred = model.predict(X_test)

comparison = pd.DataFrame({
    "Actual Count": y_test.to_numpy(),
    "Predicted Count": y_pred,
    "Error": y_test.to_numpy() - y_pred
})
print("\nActual vs Predicted values:")
print(comparison.head(10))

# Calculate regression metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"MAE  : {mae:.3f}")
print(f"MSE  : {mse:.3f}")
print(f"RMSE : {rmse:.3f}")
print(f"R2   : {r2:.3f}")

# Plot actual versus predicted values
plt.figure(figsize=(7, 6))
plt.scatter(y_test, y_pred, alpha=0.5, edgecolor="black")
low = min(float(y_test.min()), float(np.min(y_pred)))
high = max(float(y_test.max()), float(np.max(y_pred)))
plt.plot([low, high], [low, high], linestyle="--", label="Perfect prediction")
plt.xlabel("Actual Bike Rentals")
plt.ylabel("Predicted Bike Rentals")
plt.title("Actual vs Predicted Bike Rental Demand")
plt.legend()
plt.grid(alpha=0.25)
plt.savefig("results/actual_vs_predicted.png")
plt.show()
