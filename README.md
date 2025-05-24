# IBM-Data-Science-Certification-Projects
Network Latency Prediction for Satellite Networks

# Network Latency Prediction for Satellite Networks

This project simulates satellite network conditions to predict latency using machine learning.

## Dataset
- Bandwidth (64, 128, 256 kbps)
- Number of network hops
- Time of day (0–23 hours)
- Simulated latency (ms)

## Model
- Random Forest Regressor
- MAE and R² used for performance evaluation

## Usage
Run the notebook to:
1. Simulate data
2. Train and test the model
3. Visualize results

## Output
- Actual vs. Predicted Latency scatter plot

## Author
Carlos A. Barajas

# Network Latency Prediction for Satellite Communications

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Simulate dataset
np.random.seed(42)
data_size = 300
bandwidth_options = [64, 128, 256]

data = pd.DataFrame({
    'bandwidth_kbps': np.random.choice(bandwidth_options, data_size),
    'network_hops': np.random.randint(1, 10, size=data_size),
    'time_of_day': np.random.randint(0, 24, size=data_size)
})

# Simulate latency based on some influence of bandwidth and hops
data['latency_ms'] = (
    500 +
    (10 * data['network_hops']) +
    (-1.5 * data['bandwidth_kbps']) +
    (np.random.normal(0, 20, data_size))
).round(2)

# Prepare data
X = data[['bandwidth_kbps', 'network_hops', 'time_of_day']]
y = data['latency_ms']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae:.2f} ms")
print(f"R^2 Score: {r2:.2f}")

# Visualization
plt.figure(figsize=(10,6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Latency (ms)")
plt.ylabel("Predicted Latency (ms)")
plt.title("Actual vs Predicted Latency")
plt.grid(True)
plt.tight_layout()
plt.show()

# Export dataset for GitHub if needed
data.to_csv("simulated_satellite_latency.csv", index=False)
