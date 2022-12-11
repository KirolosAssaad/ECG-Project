from header import *

N = 15

if __name__ == '__main__':
    
    data = read_file('Data2.txt')
    data = data[0:1500]
    # plot_signal(data, 256, "raw data")
    plot_after_moving_average('Data2.txt', N)
    # plot_intervals('Data2.txt', N)
    # peaks, diff = r_wave_detector('Data2.txt', N)
    # print("Peaks: ", peaks)
    # print("\n\nRR intervals: ", diff)
