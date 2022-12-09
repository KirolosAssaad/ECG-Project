from scipy import signal
import matplotlib.pyplot as plt
import numpy as np

# Create/view notch filter
samp_freq = 1000  # Sample frequency (Hz)
notch_freq = 60.0  # Frequency to be removed from signal (Hz)
quality_factor = 30.0  # Quality factor
b_notch, a_notch = signal.iirnotch(notch_freq, quality_factor, samp_freq)
freq, h = signal.freqz(b_notch, a_notch, fs=samp_freq)
plt.figure('filter')
plt.plot(freq, 20*np.log10(abs(h)))

# Create/view signal that is a mixture of two frequencies
f1 = 17
f2 = 60
t = np.linspace(0.0, 1, 1_000)
y_pure = np.sin(f1 * 2.0*np.pi*t) + np.sin(f2 * 2.0*np.pi*t)
plt.figure('result')
plt.subplot(211)
plt.plot(t, y_pure, color='r')

# apply notch filter to signal
y_notched = signal.filtfilt(b_notch, a_notch, y_pure)

# plot notch-filtered version of signal
plt.subplot(212)
plt.plot(t, y_notched, color='r')
