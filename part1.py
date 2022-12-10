import matplotlib.pyplot as plt
from header import *

samplingRate = 256
N = 15



if __name__ == '__main__':

    # read the file
    data = read_file('DataN.txt')
    # plot_signal(data, samplingRate)

    notch_filtered = notch_filter(data, samplingRate, 50, 30)
    # plot_signal(notch_filtered, samplingRate)

    bandpass_filtered = bandpass_filter(notch_filtered, samplingRate, 1, 50)
    # plot_signal(bandpass_filtered, samplingRate)

    derivative = differentiate(bandpass_filtered, samplingRate)
    # plot_signal(derivative, samplingRate)

    squared = square(derivative)
    # plot_signal(squared, samplingRate)

    moving_averaged = moving_average(squared, N)
    plot_signal(moving_averaged, samplingRate)

    threshold = get_threshold(moving_averaged)

    peaks = get_peaks(moving_averaged, threshold)

    # from list to np
    peaks = np.array(peaks)
    diff = np.diff(peaks/samplingRate)
    # remove zeros
    diff = diff[diff != 0]
    # plot diff
    plt.plot(diff)
    plt.show()
