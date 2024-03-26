import numpy as np

# Given inputs
x_measurement = 94779.54
y_measurement = 217.0574
z_measurement = 2.7189

# Assume initial velocity
vx = 0.1
vy = 0.2
vz = 0.3

# Initial state vector [x, y, z, vx, vy, vz]
X = np.array([x_measurement, y_measurement, z_measurement, vx, vy, vz])

# State transition matrix
F = np.eye(6)  # Assuming no change in state for now

# Initial covariance matrix
P = np.eye(6)  # Assuming identity matrix for simplicity

# Measurement noise covariance matrix
R = np.eye(3)  # Assuming identity matrix for simplicity

# Measurement model matrix
H = np.array([[1, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0]])

# Perform Prediction Step
X_predicted = np.dot(F, X)
P_predicted = np.dot(np.dot(F, P), F.T)

# Perform Measurement Update Step
# Compute Innovation
innovation = np.array([x_measurement, y_measurement, z_measurement]) - np.dot(H, X_predicted)

# Compute Kalman Gain
K = np.dot(np.dot(P_predicted, H.T), np.linalg.inv(np.dot(np.dot(H, P_predicted), H.T) + R))

# Update State
X_updated = X_predicted + np.dot(K, innovation)

# Update Covariance
P_updated = np.dot((np.eye(6) - np.dot(K, H)), P_predicted)

# Print updated state vector and covariance matrix
print("Updated State Vector:")
print(X_updated)
print("\nUpdated Covariance Matrix:")
print(P_updated)
