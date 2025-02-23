import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError

# Load the trained model
autoencoder = load_model("autoencoder_model.h5", custom_objects={"mse": MeanSquaredError()})
print("Model loaded successfully!")

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Load the dataset
df = pd.read_csv("dataset_final.csv")  # Ensure the dataset is in the same directory

# Convert Unix timestamp to datetime format
df["Time"] = pd.to_datetime(df["Time"], unit="s")

# Normalize sensor values (excluding time column)
scaler = MinMaxScaler()
df.iloc[:, 1:] = scaler.fit_transform(df.iloc[:, 1:])

# Prepare the dataset (exclude the Time column)
X = df.iloc[:, 1:].values  # Convert to NumPy array

# Split into training (80%) and testing (20%) sets
_, X_test = train_test_split(X, test_size=0.2, random_state=42)  # No need for training set

# Reconstruct the test data using the trained autoencoder
X_test_pred = autoencoder.predict(X_test)

# Calculate the reconstruction error
reconstruction_error = np.mean(np.abs(X_test - X_test_pred), axis=1)

# Set the anomaly threshold (mean + 3 standard deviations)
threshold = np.mean(reconstruction_error) + 3 * np.std(reconstruction_error)

# Classify anomalies (1 = anomaly, 0 = normal)
anomalies = reconstruction_error > threshold

# Print anomaly counts
print(f"Total anomalies detected: {np.sum(anomalies)}")
print(f"Anomaly threshold: {threshold}")

# Plot reconstruction error distribution
plt.figure(figsize=(10, 5))
plt.hist(reconstruction_error, bins=50, alpha=0.7, label="Reconstruction Error")
plt.axvline(threshold, color="red", linestyle="dashed", linewidth=2, label="Anomaly Threshold")
plt.xlabel("Reconstruction Error")
plt.ylabel("Frequency")
plt.legend()
plt.title("Reconstruction Error Distribution")
plt.show()

# Add anomaly labels to the test dataset
df_test = pd.DataFrame(X_test, columns=["Temperature", "Humidity", "Air Quality", "Light", "Loudness"])
df_test["Reconstruction Error"] = reconstruction_error
df_test["Anomaly"] = anomalies.astype(int)  # Convert Boolean to 0/1

# Show some detected anomalies
anomalous_data = df_test[df_test["Anomaly"] == 1]
print("Anomalous Data Samples:")
print(anomalous_data.head())

# Plot a sensor value (e.g., Humidity) over time with anomalies highlighted
anomaly_indices = df_test[df_test["Anomaly"] == 1].index

# Plot anomalies for all sensor features
features = ["Temperature", "Humidity", "Air Quality", "Light", "Loudness"]

for feature in features:
    plt.figure(figsize=(12, 6))
    plt.plot(df_test.index, df_test[feature], label=feature, color="blue")
    
    # Highlight anomalies in red
    anomaly_indices = df_test[df_test["Anomaly"] == 1].index
    plt.scatter(anomaly_indices, df_test[feature].iloc[anomaly_indices], color="red", label="Anomalies", marker="o")

    plt.xlabel("Time Index")
    plt.ylabel(feature)
    plt.title(f"{feature} Sensor Data with Anomalies")
    plt.legend()
    plt.show()

