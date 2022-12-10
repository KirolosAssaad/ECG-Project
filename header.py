import scipy.signal as signal
import numpy as np
import matplotlib.pyplot as plt



def read_file(fileName):

    # open file for reading
    infile = open(fileName, 'r')

    # create an empty list
    data = []

    # read the first line
    line = infile.readline()

    # loop through the file
    while line != '':

        # convert the line to a number
        number = float(line[:-1:])

        # append the number to the list
        data.append(number)

        # read the next line
        line = infile.readline()

    # close the file
    infile.close()

    # return the list
    return data


def plot_signal(data, sampling_rate):

    data = data[:1500]

    threshold = get_threshold(data)
    max_peaks = get_peaks(data, threshold)
    # unique_peaks = get_unique_peaks(max_peaks, sampling_rate)
    unique_peaks = max_peaks

    # create a list of time values
    time = [i/sampling_rate for i in range(len(data))]

    # plot the signal
    plt.plot(time, data)

    # plot the threshold
    plt.plot(time, [threshold for i in range(len(data))])
    # plot the peaks
    plt.plot([time[i] for i in unique_peaks], [data[i]
             for i in unique_peaks], '*', color='red')

    # label the axes
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    # show the plot
    plt.show()

# # read the file


def notch_filter(data, sampling_rate, freq, Q):

    # calculate the frequency
    w0 = freq/(sampling_rate/2)

    # calculate the biquad coefficients
    b, a = signal.iirnotch(w0, Q)

    # apply the filter
    output = signal.filtfilt(b, a, data)

    # return the filtered signal
    return output

# bandpass filter


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


# differentiate the signal
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

# get threshold value


def get_threshold(data):
    mean = np.mean(data)
    return mean

# get peaks using threshold


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


# r wave detector function  (function wrapper)

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

    peaks = np.array(peaks)

    # remove repeated peaks
    peaks = np.unique(peaks)

    diff = np.diff(peaks/samplingRate)
    # remove zeros
    diff = diff[diff != 0]

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
    # plot_signal(moving_averaged, samplingRate)

    # get threshold
    threshold = get_threshold(moving_averaged)

    # get peaks
    peaks = get_peaks(moving_averaged, threshold)

    # create a list of time values
    time = [i/samplingRate for i in range(len(moving_averaged))]

    # plot the signal
    plt.plot(time, moving_averaged)

    # plot the threshold
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