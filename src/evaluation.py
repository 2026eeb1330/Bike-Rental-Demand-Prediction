from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# Evaluate Linear Regression
lr_r2 = r2_score(y_test, lr_pred)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
lr_mae = mean_absolute_error(y_test, lr_pred)

print("\nLinear Regression Results")
print("R2 Score:", round(lr_r2, 4))
print("RMSE:", round(lr_rmse, 2))
print("MAE:", round(lr_mae, 2))

# Evaluate Decision Tree
dt_r2 = r2_score(y_test, dt_pred)
dt_rmse = np.sqrt(mean_squared_error(y_test, dt_pred))
dt_mae = mean_absolute_error(y_test, dt_pred)

print("\nDecision Tree Results")
print("R2 Score:", round(dt_r2, 4))
print("RMSE:", round(dt_rmse, 2))
print("MAE:", round(dt_mae, 2))

# Compare models
print("\nModel Comparison")

if dt_r2 > lr_r2:
    print("Decision Tree performed better")
elif lr_r2 > dt_r2:
    print("Linear Regression performed better")
else:
    print("Both models performed equally")

# Actual vs Predicted graph
plt.figure(figsize=(8, 6))

plt.scatter(y_test, dt_pred, alpha=0.3, color='teal')

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--'
)

plt.xlabel("Actual Rentals")
plt.ylabel("Predicted Rentals")
plt.title("Actual vs Predicted Bike Rentals")

plt.tight_layout()
plt.savefig("outputs/predicted_vs_actual.png")
plt.close()

print("Graph saved successfully")
