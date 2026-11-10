
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from data_preprocessing import load_data, preprocess_data


df = load_data("data/train.csv")
X, y, feature_columns = preprocess_data(df)

# Same 80/20 split and random_state as training, so the model is identical
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42
)


model = LinearRegression()
model.fit(X_train, y_train)
print("\nModel training completed!")


test_df = pd.read_csv("data/test.csv")
print("Test dataset shape:", test_df.shape)
print(test_df.head())


test_df["datetime"] = pd.to_datetime(test_df["datetime"])
test_df["year"] = test_df["datetime"].dt.year
test_df["month"] = test_df["datetime"].dt.month
test_df["day"] = test_df["datetime"].dt.day
test_df["hour"] = test_df["datetime"].dt.hour
test_df["weekday"] = test_df["datetime"].dt.weekday


X_test = test_df[feature_columns]


predictions = model.predict(X_test)


predictions[predictions < 0] = 0


submission = pd.DataFrame({
    "datetime": test_df["datetime"],
    "count": predictions
})
submission["count"] = submission["count"].round().astype(int)
submission.to_csv("results/submission.csv", index=False)

print("\nPredictions generated successfully!")
print(submission.head())
print("\nSubmission saved as results/submission.csv")
