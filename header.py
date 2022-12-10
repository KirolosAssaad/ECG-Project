import scipy.signal as signal
import numpy as np
import matplotlib.pyplot as plt


samplingRate = 256
N = 15


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

# read the file


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
