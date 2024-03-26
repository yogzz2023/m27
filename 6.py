import numpy as np
import matplotlib.pyplot as plt

# Kalman Filter functions
def predict(x, P, F, Q):
    x_pred = np.dot(F, x)
    P_pred = np.dot(np.dot(F, P), F.T) + Q
    return x_pred, P_pred

def update(x_pred, P_pred, z, H, R):
    y = z - np.dot(H, x_pred)
    S = np.dot(np.dot(H, P_pred), H.T) + R
    K = np.dot(np.dot(P_pred, H.T), np.linalg.inv(S))
    x_updated = x_pred + np.dot(K, y)
    P_updated = P_pred - np.dot(np.dot(K, H), P_pred)
    return x_updated, P_updated, K, y, S

# Joint Probabilistic Data Association (JPDA)
def measurement_log_likelihood(z, x_pred, P_pred, H, R):
    innovation = z - np.dot(H, x_pred)
    S = np.dot(np.dot(H, P_pred), H.T) + R
    log_det_S = np.log(np.linalg.det(S))
    log_likelihood = -0.5 * (len(z) * np.log(2 * np.pi) + log_det_S + np.dot(innovation.T, np.dot(np.linalg.inv(S), innovation)))
    return log_likelihood

def association_probabilities(z, x_pred, P_pred, H, R, measurements):
    log_likelihoods = []
    for measurement in measurements:
        log_likelihood = measurement_log_likelihood(measurement, x_pred, P_pred, H, R)
        log_likelihoods.append(log_likelihood)
    max_log_likelihood = max(log_likelihoods)
    exp_log_likelihoods = np.exp(log_likelihoods - max_log_likelihood) # Subtract max for numerical stability
    association_probs = exp_log_likelihoods / np.sum(exp_log_likelihoods)
    return association_probs

# Constants
dt = 1.0  # Time step
F = np.array([[1, dt, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, dt],
              [0, 0, 0, 1]])  # State transition matrix
H = np.eye(4)  # Measurement matrix
Q = np.eye(4) * 0.01  # Process noise covariance
R = np.eye(4) * 0.1  # Measurement noise covariance

# Initial state
x = np.array([0, 10, 0, 5])  # [position_x, velocity_x, position_y, velocity_y]
P = np.eye(4)  # Initial state covariance

# Measurements
measurements = [
    np.array([94779.54, 217.0574, 2.7189, 21486.916]),
    np.array([27197.81, 153.2595, 1.2913, 21487.193]),
    np.array([85839.11, 226.6049, 5.0573, 21487.252])
]

# Store predicted and measured positions for plotting
predicted_positions = []
measured_positions = []

# Kalman Filter and JPDA
for z in measurements:
    # Predict step
    x_pred, P_pred = predict(x, P, F, Q)
    predicted_positions.append(x_pred[:2])  # Store only positions (x, y)

    # Measurement update step
    association_probs = association_probabilities(z, x_pred, P_pred, H, R, measurements)
    # Update state and covariance using the first measurement
    x, P, _, _, _ = update(x_pred, P_pred, z, H, R)

    # Store measured positions for comparison
    measured_positions.append(z[:2])  # Store only positions (x, y)

# Convert predicted and measured positions to numpy arrays for easy plotting
predicted_positions = np.array(predicted_positions)
measured_positions = np.array(measured_positions)

# Plotting
plt.figure(figsize=(10, 6))

# Plot predicted and measured positions
plt.plot(predicted_positions[:, 0], predicted_positions[:, 1], marker='o', linestyle='-', color='blue', label='Predicted Positions')
plt.plot(measured_positions[:, 0], measured_positions[:, 1], marker='x', linestyle='--', color='red', label='Measured Positions')

# Annotate the points with their probabilities
for i in range(len(measurements)):
    plt.text(measured_positions[i, 0], measured_positions[i, 1], f'{association_probabilities(measurements[i], x, P, H, R, measurements)[i]:.2f}')

plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Predicted vs Measured Positions')
plt.legend()
plt.grid(True)
plt.show()
