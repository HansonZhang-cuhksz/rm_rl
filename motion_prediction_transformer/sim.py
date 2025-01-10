import numpy as np

# Simulation parameters
r = 1.0  # radius in meters
omega_center = 0.5 * np.pi  # angular velocity around the center in rad/s
omega_self = np.pi  # angular velocity around its own axis in rad/s
sample_rate = 100  # sample rate in Hz
num_points = 1000000  # number of data points
dt = 1.0 / sample_rate  # time step

# Initialize arrays to store the data
data = np.zeros((num_points, 8))

# Generate data points
for i in range(num_points):
    t = i * dt
    x = r * np.cos(omega_center * t)
    y = r * np.sin(omega_center * t)
    z = 0  # Assuming the robot moves in a 2D plane
    yaw = omega_self * t
    
    vx = -r * omega_center * np.sin(omega_center * t)
    vy = r * omega_center * np.cos(omega_center * t)
    vz = 0  # Assuming the robot moves in a 2D plane
    v_yaw = omega_self
    
    data[i] = [x, y, z, yaw, vx, vy, vz, v_yaw]

# Save the data to a file
np.savetxt('robot_motion_data.csv', data, delimiter=',', header='x,y,z,yaw,vx,vy,vz,v_yaw', comments='')

print("Data generation complete. Data saved to 'robot_motion_data.csv'.")