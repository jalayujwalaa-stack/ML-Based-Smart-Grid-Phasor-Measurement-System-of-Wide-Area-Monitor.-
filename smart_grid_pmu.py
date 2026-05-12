import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Generate Smart Grid PMU Data
# -----------------------------

np.random.seed(42)

samples = 500

# Normal grid conditions
voltage = np.random.normal(230, 5, samples)
current = np.random.normal(10, 1, samples)
frequency = np.random.normal(50, 0.2, samples)
phase_angle = np.random.normal(30, 2, samples)

# Create dataframe
pmu_data = pd.DataFrame({
    'Voltage': voltage,
    'Current': current,
    'Frequency': frequency,
    'Phase_Angle': phase_angle
})

# -----------------------------
# Introduce anomalies
# -----------------------------

anomaly_indices = np.random.choice(samples, 20, replace=False)

pmu_data.loc[anomaly_indices, 'Voltage'] += np.random.normal(40, 10, 20)
pmu_data.loc[anomaly_indices, 'Frequency'] += np.random.normal(3, 1, 20)

# -----------------------------
# Feature Scaling
# -----------------------------

scaler = StandardScaler()
scaled_data = scaler.fit_transform(pmu_data)

# -----------------------------
# Machine Learning Model
# Isolation Forest for anomaly detection
# -----------------------------

model = IsolationForest(contamination=0.04, random_state=42)
model.fit(scaled_data)

# Predictions
predictions = model.predict(scaled_data)

# Convert:
# -1 = anomaly
#  1 = normal

pmu_data['Anomaly'] = predictions

# -----------------------------
# Display Results
# -----------------------------

print("\nDetected Anomalies:\n")
print(pmu_data[pmu_data['Anomaly'] == -1])

# -----------------------------
# Visualization
# -----------------------------

plt.figure(figsize=(12,6))

plt.plot(pmu_data['Voltage'], label='Voltage')

anomalies = pmu_data[pmu_data['Anomaly'] == -1]

plt.scatter(
    anomalies.index,
    anomalies['Voltage'],
    color='red',
    label='Anomaly'
)

plt.title('Smart Grid PMU Wide Area Monitoring')
plt.xlabel('Time Sample')
plt.ylabel('Voltage')
plt.legend()
plt.grid(True)
plt.show()
