import scipy.signal as signal
import numpy as np
import matplotlib.pyplot as plt

sampling_rate = 256

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

def plot_signal(data, sampling_rate, title):
    time = [i/sampling_rate for i in range(len(data))]
    plt.plot(time, data)
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.show()


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
    return np.mean(data) + 0.5*np.std(data)


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


def r_wave_detector(input_file, N, end=0, start=0):
    samplingRate = 256
    # read the file
    data = read_file(input_file)
    # check if end is given
    if end != 0:
        data = data[:end]
    else:
        pass

    # check if start is given
    if start != 0:
        data = data[start:]
    else:
        pass

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
    diff = np.diff(peaks/samplingRate)*1000
    # return peaks
    return peaks, diff


def plot_after_moving_average(input_file, N, end=0, start=0):
    samplingRate = 256
    # read the file
    data = read_file(input_file)
    # check if end is given
    if end != 0:
        data = data[:end]
    else:
        end = len(data)
        pass

    # check if start is given
    if start != 0:
        data = data[start:]
    else:
        pass

    notch_filtered = notch_filter(data, samplingRate, 50, 30)
    bandpass_filtered = bandpass_filter(notch_filtered, samplingRate, 1, 50)
    derivative = differentiate(bandpass_filtered, samplingRate)
    squared = square(derivative)
    moving_averaged = moving_average(squared, N)
    threshold = get_threshold(moving_averaged)
    peaks = get_peaks(moving_averaged, threshold)
    # get missing peaks
    missing_peaks = get_missing_peaks(input_file, N)
    # get time relative to the start and end of data
    time = [i/samplingRate for i in range(start, end)]

    # plotting the signal
    plt.plot(time, data)

    for i in missing_peaks:
        if i > start/samplingRate and i < end/samplingRate:
            plt.axvline(i, color='purple', linestyle='--')

    # plot the peaks
    plt.plot([time[i] for i in peaks], [data[i] for i in peaks], '*', color='red')
    # label the axes
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    # show the plot
    plt.show()


def plot_intervals(input_file, N, end=0, start=0):

    _peaks, diff = r_wave_detector(input_file, N, end, start)
    # plot the intervals
    plt.plot(diff)
    plt.xlabel('RR Intervals')
    plt.ylabel('Interval [ms]')
    plt.show()


def get_missing_peaks(fileName, N):
    peaks, diff = r_wave_detector(fileName, N)
    peaks_timestamp = [i/sampling_rate for i in peaks]

    # get threshold using mean and std3000
    threshold = np.mean(diff) + 0.5*np.std(diff)

    # get index of peaks above threshold
    above_threshold = np.where(diff > threshold)[0]
    missing_peaks_time = np.array([], dtype=float)
    
    threshold = threshold/1000

    for i in above_threshold:
        peak_before_time = peaks_timestamp[i]
        peak_after_time = peaks_timestamp[i+1]
        timeDiff = peak_after_time - peak_before_time
        num_peaks = int(timeDiff/threshold)

        for j in range(num_peaks):
            if np.absolute(peak_before_time + threshold*(j+1) - peak_after_time) >= 0.5* threshold:
                missing_peaks_time = np.append(missing_peaks_time, peak_before_time + threshold*(j+1))

    return missing_peaks_time
    