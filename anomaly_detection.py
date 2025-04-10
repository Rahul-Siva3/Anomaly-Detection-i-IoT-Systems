import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

# Load the trained autoencoder model
autoencoder = tf.keras.models.load_model("autoencoder_model.keras")
print("Model loaded successfully!")

# Load datasets
df1 = pd.read_csv("dataset_final.csv")
df2 = pd.read_csv("second_dataset.csv")
    
df = pd.concat([df1, df2], ignore_index=True)

# Convert Unix timestamp to datetime format
df["Time"] = pd.to_datetime(df["Time"], unit="s")

# Normalize sensor values (excluding time column)
scaler = MinMaxScaler()
df.iloc[:, 1:] = scaler.fit_transform(df.iloc[:, 1:])

# Prepare the dataset (exclude the Time column)
X = df.iloc[:, 1:].values

# Reconstruct the data using the trained autoencoder
X_pred = autoencoder.predict(X)

# Calculate the reconstruction error
reconstruction_error = np.mean(np.abs(X - X_pred), axis=1)

# Calculate mean and standard deviation of reconstruction error
mean_reconstruction_error = np.mean(reconstruction_error)
std_reconstruction_error = np.std(reconstruction_error)

# Set the anomaly threshold (mean + 3 standard deviations)
threshold = mean_reconstruction_error + 3 * std_reconstruction_error

# Classify anomalies (1 = anomaly, 0 = normal)
anomalies_autoencoder = reconstruction_error > threshold

detection_rate_autoencoder = np.sum(anomalies_autoencoder) / len(anomalies_autoencoder)

# Train Isolation Forest on loaded data
iso_forest = IsolationForest(contamination=0.05, random_state=42)
iso_forest.fit(X)
anomalies_iforest = iso_forest.predict(X)
anomalies_iforest = anomalies_iforest == -1

detection_rate_iforest = np.sum(anomalies_iforest) / len(anomalies_iforest)

# Train One-Class SVM
oc_svm = OneClassSVM(nu=0.05, kernel="rbf")
oc_svm.fit(X)
anomalies_ocsvm = oc_svm.predict(X)
anomalies_ocsvm = anomalies_ocsvm == -1

detection_rate_ocsvm = np.sum(anomalies_ocsvm) / len(anomalies_ocsvm)

# Combine anomaly detections (Majority Voting)
anomalies_combined = (anomalies_autoencoder.astype(int) + anomalies_iforest.astype(int) + anomalies_ocsvm.astype(int)) >= 2

detection_rate_combined = np.sum(anomalies_combined) / len(anomalies_combined)

# Print statistics
print(f"Mean Reconstruction Error: {mean_reconstruction_error}")
print(f"Standard Deviation of Reconstruction Error: {std_reconstruction_error}")
print(f"Anomaly Threshold (Autoencoder): {threshold}")
print(f"Detection Rate (Autoencoder): {detection_rate_autoencoder:.4f}")
print(f"Detection Rate (Isolation Forest): {detection_rate_iforest:.4f}")
print(f"Detection Rate (One-Class SVM): {detection_rate_ocsvm:.4f}")
print(f"Detection Rate (Majority Voting): {detection_rate_combined:.4f}")

# Print anomaly counts
print(f"Total anomalies detected: {np.sum(anomalies_combined)}")

# Plot reconstruction error distribution
plt.figure(figsize=(10, 5))
plt.hist(reconstruction_error, bins=50, alpha=0.7, label="Reconstruction Error")
plt.axvline(threshold, color="red", linestyle="dashed", linewidth=2, label="Anomaly Threshold")
plt.xlabel("Reconstruction Error")
plt.ylabel("Frequency")
plt.legend()
plt.title("Reconstruction Error Distribution")
plt.show()

# Add anomaly labels to the dataset
df_results = pd.DataFrame(X, columns=["Temperature", "Humidity", "Air Quality", "Light", "Loudness"])
df_results["Reconstruction Error"] = reconstruction_error
df_results["Anomaly"] = anomalies_combined.astype(int)

# Show some detected anomalies
anomalous_data = df_results[df_results["Anomaly"] == 1]
print("Anomalous Data Samples:")
print(anomalous_data.head())

# Plot anomalies for all sensor features
features = ["Temperature", "Humidity", "Air Quality", "Light", "Loudness"]

for feature in features:
    plt.figure(figsize=(12, 6))
    plt.plot(df_results.index, df_results[feature], label=feature, color="blue")
    
    # Highlight anomalies in red
    anomaly_indices = df_results[df_results["Anomaly"] == 1].index
    plt.scatter(anomaly_indices, df_results[feature].iloc[anomaly_indices], color="red", label="Anomalies", marker="o")

    plt.xlabel("Time Index")
    plt.ylabel(feature)
    plt.title(f"{feature} Sensor Data with Anomalies")
    plt.legend()
    plt.show()
