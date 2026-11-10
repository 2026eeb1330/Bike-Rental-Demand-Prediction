# MEMBER 4: PREDICTION ON UNSEEN KAGGLE TEST DATA
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from data_preprocessing import load_data, preprocess_data

# Read and prepare the training dataset (same steps as model_training.py)
df = load_data("data/train.csv")
X, y, feature_columns = preprocess_data(df)

# Same 80/20 split and random_state as training, so the model is identical
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42
)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)
print("\nModel training completed!")

# Read Kaggle test dataset
test_df = pd.read_csv("data/test.csv")
print("Test dataset shape:", test_df.shape)
print(test_df.head())

# Create the same date/time features used during training
test_df["datetime"] = pd.to_datetime(test_df["datetime"])
test_df["year"] = test_df["datetime"].dt.year
test_df["month"] = test_df["datetime"].dt.month
test_df["day"] = test_df["datetime"].dt.day
test_df["hour"] = test_df["datetime"].dt.hour
test_df["weekday"] = test_df["datetime"].dt.weekday

# Select features in exactly the same order as training
X_test = test_df[feature_columns]

# Make predictions
predictions = model.predict(X_test)

# Rental counts cannot be negative
predictions[predictions < 0] = 0

# Create Kaggle submission file
submission = pd.DataFrame({
    "datetime": test_df["datetime"],
    "count": predictions
})
submission["count"] = submission["count"].round().astype(int)
submission.to_csv("results/submission.csv", index=False)

print("\nPredictions generated successfully!")
print(submission.head())
print("\nSubmission saved as results/submission.csv")
