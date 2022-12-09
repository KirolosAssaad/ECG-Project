import matplotlib.pyplot as plt
import scipy.signal as signal
import numpy as np

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

# function that takes a list of numbers and sampling rate and plots the signal


def plot_signal(data, sampling_rate):

    data = data[:1500]

    threshold = get_threshold(data)

    max_peaks = get_max_peaks(data, get_peaks(data, threshold), 150)

    # create a list of time values
    time = [i/sampling_rate for i in range(len(data))]

    # plot the signal
    plt.plot(time, data)

    # plot the threshold
    plt.plot(time, [threshold for i in range(len(data))])
    # plot the peaks
    plt.plot([time[i] for i in max_peaks], [data[i] for i in max_peaks], '*', color='red')

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
def get_peaks(data, threshold):
    peaks = []
    for i in range(len(data)):
        if data[i] > threshold:
            peaks.append(i)
    return peaks

# get maximum peaks in a window
def get_max_peaks(data, peaks, window_size):
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

if __name__ == '__main__':

    # read the file
    data = read_file('DataN.txt')
    # plot_signal(data, 256)

    notch_filtered = notch_filter(data, 256, 50, 30)
    # plot_signal(notch_filtered, 256)

    bandpass_filtered = bandpass_filter(notch_filtered, 256, 1, 50)
    # plot_signal(bandpass_filtered, 256)

    derivative = differentiate(bandpass_filtered, 256)
    # plot_signal(derivative, 256)

    squared = square(derivative)
    # plot_signal(squared, 256)

    moving_averaged = moving_average(squared, 15)
    plot_signal(moving_averaged, 256)

    threshold = get_threshold(moving_averaged)
    # print(threshold)

    peaks = get_peaks(moving_averaged, threshold)
