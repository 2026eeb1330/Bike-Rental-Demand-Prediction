
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv("data/train.csv")

print("Dataset dimensions:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())
print("\nFirst five rows:")
print(df.head())
print("\nLast five rows:")
print(df.tail())
print("\nData types and missing-value overview:")
df.info()
print("\nSummary of numeric columns:")
print(df.describe())
print("\nMissing values in each column:")
print(df.isnull().sum())


df["datetime"] = pd.to_datetime(df["datetime"])
df["hour"] = df["datetime"].dt.hour
df["month"] = df["datetime"].dt.month


plt.figure(figsize=(8, 4))
plt.hist(df["count"], bins=20, edgecolor="black")
plt.title("Distribution of Bike Rental Count")
plt.xlabel("Number of Bike Rentals")
plt.ylabel("Frequency")
plt.grid(alpha=0.25)
#plt.savefig("results/count_histogram.png")
plt.show()


plt.figure(figsize=(8, 5))
plt.scatter(df["temp"], df["count"], alpha=0.5)
plt.title("Bike Rental Count vs Temperature")
plt.xlabel("Temperature")
plt.ylabel("Bike Rental Count")
plt.grid(alpha=0.25)
#plt.savefig("results/temperature_vs_count.png")
plt.show()


hourly_average = df.groupby("hour")["count"].mean()
plt.figure(figsize=(8, 5))
plt.bar(hourly_average.index, hourly_average.values)
plt.title("Average Bike Rentals by Hour")
plt.xlabel("Hour of the Day")
plt.ylabel("Average Bike Rentals")
plt.grid(alpha=0.25)
#plt.savefig("results/average_rentals_by_hour.png")
plt.show()


plt.figure(figsize=(8, 5))
plt.scatter(df["humidity"], df["count"], alpha=0.5)
plt.title("Bike Rental Count vs Humidity")
plt.xlabel("Humidity")
plt.ylabel("Bike Rental Count")
plt.grid(alpha=0.25)
#plt.savefig("results/humidity_vs_count.png")
plt.show()
