import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from datetime import datetime, timedelta

# Sample user grocery consumption data (you can replace this with real data)
data = {
    "date": pd.date_range(start="2024-01-01", periods=30, freq="D"),
    "item": ["Milk"] * 30,
    "quantity_used": np.random.randint(1, 3, size=30)  # Simulating daily consumption
}

df = pd.DataFrame(data)

# Aggregate data to get total consumption per day
df = df.groupby("date").sum().reset_index()
df.rename(columns={"date": "ds", "quantity_used": "y"}, inplace=True)

# Train the Prophet model
model = Prophet()
model.fit(df)

# Predict the next 15 days
future = model.make_future_dataframe(periods=15)
forecast = model.predict(future)

# Visualize predictions
plt.figure(figsize=(10, 5))
plt.plot(df["ds"], df["y"], label="Actual Consumption", marker="o")
plt.plot(forecast["ds"], forecast["yhat"], label="Predicted Consumption", linestyle="dashed")
plt.axhline(y=0.5, linestyle="--", label="Restock Threshold")
plt.xlabel("Date")
plt.ylabel("Quantity Used")
plt.legend()
plt.title("Predictive Buying Recommendation for Milk")
plt.show()

# Find the next restock date
threshold = 0.5  # When predicted quantity reaches below 0.5, recommend buying
next_restock = forecast[forecast["yhat"] < threshold]["ds"].min()

if next_restock:
    print(f"⚠️ Recommended restock date: {next_restock.date()}")
else:
    print("No restock needed in the next 15 days.")
