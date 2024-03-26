import numpy as np

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
def measurement_likelihood(z, x_pred, P_pred, H, R):
    innovation = z - np.dot(H, x_pred)
    S = np.dot(np.dot(H, P_pred), H.T) + R
    det_S = np.linalg.det(S)
    if det_S == 0:
        likelihood = 0
    else:
        likelihood = (1 / np.sqrt((2 * np.pi) ** len(z) * det_S)) * \
                     np.exp(-0.5 * np.dot(innovation.T, np.dot(np.linalg.inv(S), innovation)))
    return likelihood

def association_probabilities(z, x_pred, P_pred, H, R, measurements):
    association_probs = []
    for measurement in measurements:
        likelihood = measurement_likelihood(measurement, x_pred, P_pred, H, R)
        association_probs.append(likelihood)
    return association_probs / np.sum(association_probs)

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
    np.array([10, 0, 10, 1]),
    np.array([10, 0, 10, 2]),
    np.array([10, 0, 10, 3])
]

# Kalman Filter and JPDA
for z in measurements:
    # Predict step
    x_pred, P_pred = predict(x, P, F, Q)

    # Measurement update step
    association_probs = association_probabilities(z, x_pred, P_pred, H, R, measurements)
    for i in range(len(association_probs)):
        print("Measurement", i+1, "Probability:", association_probs[i])
