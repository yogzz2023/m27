import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

# Define the cluttered environment
class ClutteredEnvironment:
    def __init__(self, width, height, num_clutter):
        self.width = width
        self.height = height
        self.num_clutter = num_clutter
    
    def generate_clutter(self):
        clutter_x = np.random.uniform(0, self.width, self.num_clutter)
        clutter_y = np.random.uniform(0, self.height, self.num_clutter)
        return clutter_x, clutter_y

# Define the target class
class Target:
    def __init__(self, id, position, velocity, classification):
        self.id = id
        self.position = position
        self.velocity = velocity
        self.classification = classification

# Define the particle filter (Kalman filter)
class KalmanFilter:
    def __init__(self, num_particles):
        self.num_particles = num_particles
    
    def predict(self, particles, process_noise):
        # Implement particle prediction step
        for i in range(self.num_particles):
            particles[i][0] += particles[i][2] + np.random.normal(0, process_noise)
            particles[i][1] += particles[i][3] + np.random.normal(0, process_noise)
        return particles
    
    def update(self, particles, measurements, measurement_noise):
        # Implement particle update step
        weights = np.zeros(self.num_particles)
        for i in range(self.num_particles):
            for j in range(len(measurements)):
                dist = np.linalg.norm(particles[i][:2] - measurements[j][:2])
                weights[i] += multivariate_normal.pdf(dist, cov=measurement_noise)
        weights /= np.sum(weights)
        return weights

# Classification algorithm for targets
def classify_targets(targets):
    for target in targets:
        if target.position[0] < 50 and target.position[1] < 50:
            target.classification = 'class1'
        elif target.position[0] >= 50 and target.position[1] < 50:
            target.classification = 'class2'
        else:
            target.classification = 'class3'

# Main function to run the simulation
def main():
    # Simulation parameters
    width = 100
    height = 100
    num_targets = 1
    num_clutter = 20
    num_particles = 100
    process_noise = 1
    measurement_noise = 1
    
    # Create cluttered environment
    env = ClutteredEnvironment(width, height, num_clutter)
    clutter_x, clutter_y = env.generate_clutter()
    
    # Initialize Kalman filter
    kalman_filter = KalmanFilter(num_particles)
    
    # Main simulation loop
    for t in range(1):
        # Generate initial targets
        targets = []
        for i in range(num_targets):
            id = i
            position = np.array([94779.54, 217.0574])  # Measurement values (x, y)
            velocity = np.random.uniform(-1, 1, size=3)  # Sample velocities (vx, vy, vz)
            classification = None  # Classification will be assigned later
            target = Target(id, position, velocity, classification)
            targets.append(target)
        
        # Predict target motion
        for target in targets:
            target.position += target.velocity
        
        # Generate measurements (radar, sensor, etc.)
        measurements = np.array([np.concatenate((target.position, [target.id])) 
                                  for target in targets])
        
        # Incorporate clutter measurements
        clutter_measurements = np.array([np.array([clutter_x[i], clutter_y[i], -1]) 
                                          for i in range(num_clutter)])
        measurements = np.concatenate((measurements, clutter_measurements), axis=0)
        
        # Perform target classification
        classify_targets(targets)
        
        # Update Kalman filter
        particles = np.random.uniform(0, 100, size=(num_particles, 4))  # (x, y, vx, vy)
        particles = kalman_filter.predict(particles, process_noise)
        weights = kalman_filter.update(particles, measurements, measurement_noise)
        
        # Resample particles
        indices = np.random.choice(np.arange(num_particles), size=num_particles, replace=True, p=weights)
        particles = particles[indices]
        
        # Estimate target states
        estimated_positions = np.mean(particles[:, :2], axis=0)
        
        # Visualization or further processing
        plt.figure(figsize=(8, 6))
        plt.scatter(particles[:, 0], particles[:, 1], color='blue', alpha=0.3, label='Particles')
        plt.scatter(estimated_positions[0], estimated_positions[1], color='red', marker='x', label='Estimated Position')
        for target in targets:
            if target.classification == 'class1':
                plt.scatter(target.position[0], target.position[1], color='green', marker='o', label='Class 1 Target')
            elif target.classification == 'class2':
                plt.scatter(target.position[0], target.position[1], color='blue', marker='o', label='Class 2 Target')
            else:
                plt.scatter(target.position[0], target.position[1], color='purple', marker='o', label='Class 3 Target')
        plt.scatter(clutter_x, clutter_y, color='gray', alpha=0.3, label='Clutter')
        plt.xlim(0, width)
        plt.ylim(0, height)
        plt.title("Kalman Filter Tracking")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)
        plt.legend()
        plt.show()
        
        # Pause for visualization (optional)
        plt.pause(0.1)
    
    # End of simulation

if __name__ == "__main__":
    main()
