import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

CSV_FILE = 'imu_data.csv'

def read_imu_data(csv_file):
    A = pd.read_csv(csv_file)  # read csv file with raw data

    imu_data = {
        't': A['t'].values.tolist(),   # time stamp
        'Ax': A['Ax'].values.tolist(), # data from the accelerometers 
        'Ay': A['Ay'].values.tolist(),
        'Az': A['Az'].values.tolist(),
        'Gx': A['Gx'].values.tolist(), # data from the gyroscope
        'Gy': A['Gy'].values.tolist(),
        'Gz': A['Gz'].values.tolist()
    }

    return imu_data

imu_data = read_imu_data(CSV_FILE)

# Function to calculate phi
def calculate_phi(Ax, Ay, Az):
    phi_radians = math.atan(Ay / math.sqrt(Ax**2 + Az**2))
    phi_degrees = phi_radians * 180 / math.pi
    return phi_degrees

# Function to calculate theta
def calculate_theta(Ax, Ay, Az):
    theta_radians = math.atan(-Ax / math.sqrt(Ay**2 + Az**2))
    theta_degrees = theta_radians * 180 / math.pi
    return theta_degrees

# Calculate phi values for imu_data
def calculate_phi_for_imu_data(imu_data):
    phi_values = []
    for i in range(len(imu_data['Ax'])):
        Ax = imu_data['Ax'][i]
        Ay = imu_data['Ay'][i]
        Az = imu_data['Az'][i]
        phi = calculate_phi(Ax, Ay, Az)
        phi_values.append(phi)
    return phi_values

# Calculate theta values for imu_data
def calculate_theta_for_imu_data(imu_data):
    theta_values = []
    for i in range(len(imu_data['Ax'])):
        Ax = imu_data['Ax'][i]
        Ay = imu_data['Ay'][i]
        Az = imu_data['Az'][i]
        theta = calculate_theta(Ax, Ay, Az)
        theta_values.append(theta)
    return theta_values

# Moving average filter function
def moving_average(data, window_size):
    weights = np.repeat(1.0, window_size) / window_size
    return np.convolve(data, weights, 'valid')

# Calculate phi and theta values
phi_values = calculate_phi_for_imu_data(imu_data)
theta_values = calculate_theta_for_imu_data(imu_data)

# Apply moving average filter
window_size = 5
phi_filtered = moving_average(phi_values, window_size)
theta_filtered = moving_average(theta_values, window_size)

# Create four separate plots
plt.figure(figsize=(14, 10))

# Plot original phi values
plt.subplot(2, 2, 1)
plt.plot(phi_values, label='Phi (Original)')
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Original Phi Values')
plt.legend()
plt.grid(True)

# Plot filtered phi values
plt.subplot(2, 2, 2)
plt.plot(phi_filtered, label='Phi (Filtered)')
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Filtered Phi Values')
plt.legend()
plt.grid(True)

# Plot original theta values
plt.subplot(2, 2, 3)
plt.plot(theta_values, label='Theta (Original)')
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Original Theta Values')
plt.legend()
plt.grid(True)

# Plot filtered theta values
plt.subplot(2, 2, 4)
plt.plot(theta_filtered, label='Theta (Filtered)')
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Filtered Theta Values')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Function to demonstrate the Gyroscopic Drift
def gyro_drift():
    # Simulation parameters
    N = 1000  # samples
    Ts = 0.01 # sampling period
    noise_std = 0.5

    # Time vector
    t = np.arange(0,N) * Ts

    # True states
    theta = np.exp(-t) * np.sin(2 * np.pi * t)
    theta_dot = np.exp(-t) * (2 * np.pi * np.cos(2 * np.pi * t) - np.sin(2 * np.pi * t))

    # Noisy gyro measurements
    theta_dot_noise = theta_dot + noise_std * np.random.standard_normal(N)

    # Integrate gyro measurements to get state estimate
    # Considering the initial state value as zero
    # Using the Euler's method for numerical integration of the ODE

    theta_hat = np.zeros(N)
    
    for i in range(1,N):
        theta_hat[i] = theta_hat[i - 1] + theta_dot_noise[i-1] * Ts

    # Calculate the actual pitch angle
    actual_pitch_angle = np.cumsum(theta_dot) * Ts

    # Plotting
    plt.figure(figsize=(12,5))
    plt.plot(t, [a * 180.0 / np.pi for a in theta], label='True')
    plt.plot(t, [a * 180.0 / np.pi for a in theta_hat], label='Estimate')
    plt.plot(t, actual_pitch_angle * 180.0 / np.pi, label='Actual', linestyle='--')
    plt.xlabel('Time (s)')
    plt.ylabel('Theta (degrees)')
    plt.title('Gyro Drift')
    plt.legend(loc='upper right')
    plt.show()

# Calling the function
gyro_drift()