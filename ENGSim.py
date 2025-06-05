import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks, correlate

# Simulate ENG signals for 2 electrodes with known delay
def simulate_eng_data(fs, duration, conduction_velocity, distance, noise_level=0.05):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    n_samples = len(t)
    
    # Generate a Gaussian-shaped action potential
    spike = np.exp(-((t - 0.005) ** 2) / (2 * (0.0005) ** 2))
    
    # Compute delay in samples based on conduction velocity and electrode spacing
    delay_samples = int((distance / conduction_velocity) * fs)
    
    # Create ENG signal for channel 1 (proximal electrode)
    channel1 = np.zeros(n_samples)
    channel1[100:100+len(spike)] += spike
    
    # Create delayed ENG signal for channel 2 (distal electrode)
    channel2 = np.zeros(n_samples)
    channel2[100 + delay_samples:100 + delay_samples + len(spike)] += spike

    # Add Gaussian noise to both channels
    channel1 += np.random.normal(0, noise_level, n_samples)
    channel2 += np.random.normal(0, noise_level, n_samples)
    
    return np.vstack([channel1, channel2]), t

# Apply bandpass filter to isolate neural signal frequency range
def bandpass_filter(signal, fs, lowcut=300, highcut=3000, order=3):
    nyq = 0.5 * fs  # Nyquist frequency
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, signal)

# Delay-and-sum beamforming to align signals and enhance SNR
def beamform(data, fs, delays):
    aligned = []
    for i, delay in enumerate(delays):
        shift = int(delay * fs)  # Convert delay from seconds to samples
        shifted = np.roll(data[i], -shift)  # Time-align signal
        aligned.append(shifted)
    return np.mean(aligned, axis=0)  # Average aligned signals

# Cross-correlate signals to estimate latency between channels
def estimate_latency(signal1, signal2, fs):
    corr = correlate(signal2, signal1, mode='full')  # Cross-correlation
    lag = np.argmax(corr) - len(signal1)  # Compute lag in samples
    latency = lag / fs  # Convert lag to time (s)
    return latency

# ENG signal parameters
fs = 20000  # Sampling frequency in Hz
duration = 0.05  # Total signal duration in seconds
conduction_velocity = 30.0  # Nerve conduction velocity in m/s
electrode_spacing = 0.01  # Distance between electrodes in meters

# Simulate 2-channel ENG signal with noise and delay
data, t = simulate_eng_data(fs, duration, conduction_velocity, electrode_spacing)

# Bandpass filter to remove low-frequency drift and high-frequency noise
filtered = np.array([bandpass_filter(chan, fs) for chan in data])

# Apply beamforming using known inter-electrode delay
delay = electrode_spacing / conduction_velocity  # Delay in seconds
beamformed = beamform(filtered, fs, delays=[0, delay])

# Estimate latency using cross-correlation
latency = estimate_latency(filtered[0], filtered[1], fs)

# Plot filtered ENG signals from both channels
plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)
plt.title("Filtered ENG signals")
plt.plot(t, filtered[0], label="Channel 1")
plt.plot(t, filtered[1], label="Channel 2")
plt.legend()

# Plot beamformed output
plt.subplot(3, 1, 2)
plt.title("Beamformed Output (Delay-and-Sum)")
plt.plot(t, beamformed, color='purple')

# Plot latency estimate
plt.subplot(3, 1, 3)
plt.title("Estimated latency: {:.3f} ms".format(latency * 1000))
plt.xlabel("Time (s)")
plt.tight_layout()
plt.show()