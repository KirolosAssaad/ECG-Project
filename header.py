import scipy.signal as signal
import numpy as np
import matplotlib.pyplot as plt


def read_file(fileName):
    infile = open(fileName, 'r')
    data = []
    line = infile.readline()
    while line != '':
        number = float(line[:-1:])
        data.append(number)
        line = infile.readline()
    infile.close()
    return data


def notch_filter(data, sampling_rate, freq, Q):

    # calculate the frequency
    w0 = freq/(sampling_rate/2)
    # calculate the biquad coefficients
    b, a = signal.iirnotch(w0, Q)
    # apply the filter
    output = signal.filtfilt(b, a, data)
    # return the filtered signal
    return output


def bandpass_filter(data, sampling_rate, lowcut, highcut, order=5):
    # calculate the nyquist frequency
    nyq = 0.5 * sampling_rate
    # calculate the low and high frequencies
    low = lowcut / nyq
    high = highcut / nyq
    # calculate the biquad coefficients
    b, a = signal.butter(order, [low, high], btype='band')
    # apply the filter
    output = signal.filtfilt(b, a, data)
    # return the filtered signal
    return output


def differentiate(data, sampling_rate):
    # calculate the time step
    dt = 1/sampling_rate
    # calculate the derivative
    derivative = [(data[i+1] - data[i])/dt for i in range(len(data)-1)]
    # return the derivative
    return derivative


def square(data):
    squared = [i**2 for i in data]
    return squared


def moving_average(data, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(data, window, 'same')


def get_threshold(data):
    mean = np.mean(data)
    return mean


def get_peaks(data, threshold, window_size=150):
    peaks = []
    for i in range(len(data)):
        if data[i] > threshold:
            peaks.append(i)

    max_peaks = []
    for i in range(len(peaks)):
        start = peaks[i] - window_size
        end = peaks[i] + window_size
        if start < 0:
            start = 0
        if end > len(data):
            end = len(data)
        max_peak = np.argmax(data[start:end]) + start
        max_peaks.append(max_peak)
    return max_peaks


def r_wave_detector(input_file, N):
    samplingRate = 256
    # read the file
    data = read_file(input_file)
    # notch filter
    notch_filtered = notch_filter(data, samplingRate, 50, 30)
    # bandpass filter
    bandpass_filtered = bandpass_filter(notch_filtered, samplingRate, 1, 50)
    # differentiate
    derivative = differentiate(bandpass_filtered, samplingRate)
    # square
    squared = square(derivative)
    # moving average
    moving_averaged = moving_average(squared, N)
    # get threshold
    threshold = get_threshold(moving_averaged)
    # get peaks
    peaks = get_peaks(moving_averaged, threshold)
    # convert to numpy array
    peaks = np.array(peaks)
    # remove repeated peaks
    peaks = np.unique(peaks)
    diff = np.diff(peaks/samplingRate)
    # return peaks
    return peaks, diff


def plot_after_moving_average(input_file, N, no_samples=0):
    samplingRate = 256
    # read the file
    data = read_file(input_file)
    # check if no_samples is given
    if no_samples != 0:
        data = data[:no_samples]
    else:
        pass

    notch_filtered = notch_filter(data, samplingRate, 50, 30)
    bandpass_filtered = bandpass_filter(notch_filtered, samplingRate, 1, 50)
    derivative = differentiate(bandpass_filtered, samplingRate)
    squared = square(derivative)
    moving_averaged = moving_average(squared, N)
    threshold = get_threshold(moving_averaged)
    peaks = get_peaks(moving_averaged, threshold)
    time = [i/samplingRate for i in range(len(moving_averaged))]

    # plotting the signal
    plt.plot(time, moving_averaged)
    plt.plot(time, [threshold for i in range(len(moving_averaged))])
    # plot the peaks
    plt.plot([time[i] for i in peaks], [moving_averaged[i]
             for i in peaks], '*', color='red')
    # label the axes
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    # show the plot
    plt.show()


def plot_intervals(input_file, N):
    _peaks, diff = r_wave_detector(input_file, N)
    # plot the intervals
    plt.plot(diff)
    plt.xlabel('RR Intervals')
    plt.ylabel('Interval (s)')
    plt.show()
