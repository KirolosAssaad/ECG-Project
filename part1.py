from header import plot_intervals, plot_after_moving_average, r_wave_detector

N = 15

if __name__ == '__main__':
    plot_after_moving_average('DataN.txt', N, start=55000)
    # plot_intervals('DataN.txt', N)
    # peaks, diff = r_wave_detector('DataN.txt', N)
    # print("Peaks: ", peaks)
    # print("\n\nRR intervals: ", diff)