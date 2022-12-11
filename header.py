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


def plot_peaks(input_file, N, end=0, start=0, title=None, output_file=None):
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
    plt.plot([time[i] for i in peaks], [data[i]
             for i in peaks], '*', color='red')
    # label the axes
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    if title:
        plt.title(title)

    if output_file:
        plt.savefig(output_file)
    plt.close()


def plot_after_moving_average(input_file, N, showPeaks=False, end=0, start=0, title=None, output_file=None):
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
    time = [i/samplingRate for i in range(len(moving_averaged))]

    # plotting the signal
    plt.plot(time, moving_averaged)

    if showPeaks:
        # plot the peaks
        plt.plot([time[i] for i in peaks], [moving_averaged[i]
                 for i in peaks], '*', color='red')
        for i in missing_peaks:
            if i > start/samplingRate and i < end/samplingRate:
                plt.axvline(i, color='purple', linestyle='--')

    # label the axes
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    if title:
        plt.title(title)

    if output_file:
        plt.savefig(output_file)
    plt.close()


def plot_intervals(input_file, N, end=0, start=0, output_file=None):

    _peaks, diff = r_wave_detector(input_file, N, end, start)
    # plot the intervals
    plt.plot(diff)
    plt.xlabel('RR Intervals')
    plt.ylabel('Interval [ms]')
    plt.title('RR Intervals N = {}'.format(N))
    if output_file:
        plt.savefig(output_file)
    plt.close()


def get_missing_peaks(fileName, N):
    peaks, diff = r_wave_detector(fileName, N)
    peaks_timestamp = [i/sampling_rate for i in peaks]

    # get threshold using mean and std3000
    threshold = np.mean(diff) + 0.5*np.std(diff)

    # get index of peaks above threshold
    above_threshold = np.where(diff > threshold)[0]

    threshold = threshold/1000
    ppeaks = []

    for i in above_threshold:
        peak_before_time = peaks_timestamp[i]
        peak_after_time = peaks_timestamp[i+1]
        timeDiff = peak_after_time - peak_before_time
        num_peaks = int(timeDiff/threshold)

        for j in range(num_peaks):
            if np.absolute(peak_before_time + threshold*(j+1) - peak_after_time) >= 0.5 * threshold:
                ppeaks.append(peak_before_time + threshold*(j+1))
    return ppeaks


def plot_signal(data, sampling_rate, title, output_file=None):
    # get time
    time = [i/sampling_rate for i in range(len(data))]
    # plot the signal
    plt.plot(time, data)
    # label the axes
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    # set title
    plt.title(title)
    # save the plot as a png file
    if output_file is not None:
        plt.savefig(output_file)
    plt.close()


def part_1_a():
    N = 15
    # plotting the raw graph
    data = read_file('DataN.txt')
    data = data[0:1500]
    plot_signal(data, 256, "Signal After Filtering",
                "./part1_results/before_filtering.png")
    
    notched = notch_filter(data, 256, 50, 30)
    bandpass = bandpass_filter(notched, 256, 1, 50)
    plot_signal(bandpass, 256, "Signal After Filtering",
                "./part1_results/after_filtering.png")

    # plot_after_moving_average('DataN.txt', N, False, 1500, 0,
                            #   "Signal After Filtering", "./part1_results/after_filtering.png")


def part_1_b():
    N = 10
    plot_peaks('DataN.txt', N, 1500, 0, "Peaks on Raw Data",
               "./part1_results/n_10/peaks_raw_data.png")
    plot_after_moving_average('DataN.txt', N, True, 1500, 0,
                              "Peaks on Moving Average Filter", "./part1_results/n_10/peaks_moving_avg.png")
    plot_intervals('DataN.txt', N, start=0,
                   output_file="./part1_results/n_10/intervals.png")

    peaks, diff = r_wave_detector('DataN.txt', N, 1500, 0)
    np.savetxt("./part1_results/n_10/peaks_values.txt", peaks, fmt='%d')
    np.savetxt("./part1_results/n_10/intervals_values.txt", diff, fmt='%d')


def part_1_c():
    N = 15
    plot_peaks('DataN.txt', N, 1500, 0, "Peaks on Raw Data",
               "./part1_results/n_15/peaks_raw_data.png")
    plot_after_moving_average('DataN.txt', N, True, 1500, 0,
                              "Peaks on Moving Average Filter", "./part1_results/n_15/peaks_moving_avg.png")
    plot_intervals('DataN.txt', N, start=0,
                   output_file="./part1_results/n_15/intervals.png")

    peaks, diff = r_wave_detector('DataN.txt', N, 1500, 0)
    np.savetxt("./part1_results/n_15/peaks_values.txt", peaks, fmt='%d')
    np.savetxt("./part1_results/n_15/intervals_values.txt", diff, fmt='%d')



def part_1_d():
    N = 25
    plot_peaks('DataN.txt', N, 1500, 0, "Peaks on Raw Data",
               "./part1_results/n_25/peaks_raw_data.png")
    plot_after_moving_average('DataN.txt', N, True, 1500, 0,
                              "Peaks on Moving Average Filter", "./part1_results/n_25/peaks_moving_avg.png")
    plot_intervals('DataN.txt', N, start=0,
                   output_file="./part1_results/n_25/intervals.png")

    peaks, diff = r_wave_detector('DataN.txt', N, 1500, 0)
    np.savetxt("./part1_results/n_25/peaks_values.txt", peaks, fmt='%d')
    np.savetxt("./part1_results/n_25/intervals_values.txt", diff, fmt='%d')

def part2():
    N = 25
    missing_peaks = get_missing_peaks('Data2.txt', N)
    plot_after_moving_average('Data2.txt', N, True, 7500, 0,
                                "Peaks on Moving Average Filter", "./part2_results/peaks_moving_avg.png")
    np.savetxt("./part2_results/missing_peaks.txt", missing_peaks, fmt='%f')