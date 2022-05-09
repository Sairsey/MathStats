# used octave.m to build A and B
import csv
import matplotlib.pyplot as plt


EPS = 1e-4

def load_csv(filename):
    data1 = [] 
    with open(filename) as csvfile:
        spamreader = csv.reader(csvfile)
        for row in spamreader:
            data1.append([float(row[0]), float(row[0])])
    return data1

def load_octave(filename):
    A = 0
    B = 0
    w = []
    with open(filename) as f:
        A, B = [float(t) for t in f.readline().split()]
        for line in f.readlines():
            w.append(float(line))
    return A, B, w

def plot_interval(y, x, color='b'):
    plt.vlines(x, y[0], y[1], color, lw=2)


if __name__ == "__main__":
    data1 = load_csv('data/Ch1_800nm_0.03.csv')
    data2 = load_csv('data/Ch2_800nm_0.03.csv')
    data1_A, data1_B, data1_w = load_octave('data/Ch1.txt')
    data2_A, data2_B, data2_w = load_octave('data/Ch2.txt')
    data1 = [[data1[i][0] - data1_w[i] * EPS, data1[i][1] + data1_w[i] * EPS] for i in range(len(data1))]
    data2 = [[data2[i][0] - data2_w[i] * EPS, data2[i][1] + data2_w[i] * EPS] for i in range(len(data2))]
    data_n = [t for t in range(1, len(data1) + 1)]
    # plot first
    for i in range(len(data1)):
       plot_interval(data1[i], data_n[i])

    # plot second
    for i in range(len(data2)):
       plot_interval(data2[i], data_n[i], 'r')
    
    plt.plot([data_n[0], data_n[-1]], [data_n[0] * data1_B + data1_A, data_n[-1] * data1_B + data1_A], color='green')
    plt.plot([data_n[0], data_n[-1]], [data_n[0] * data2_B + data2_A, data_n[-1] * data2_B + data2_A], color='green')
    plt.figure()
    data1_fixed = [[data1[i][0] - data1_B * data_n[i], data1[i][1] - data1_B * data_n[i]] for i in range(len(data1))]
    data2_fixed = [[data2[i][0] - data2_B * data_n[i], data2[i][1] - data2_B * data_n[i]] for i in range(len(data2))]
    # plot first
    for i in range(len(data1_fixed)):
       plot_interval(data1_fixed[i], data_n[i])

    plt.plot([data_n[0], data_n[-1]], [data1_A, data1_A], color='green')
    plt.figure()
    # plot second
    for i in range(len(data2_fixed)):
       plot_interval(data2_fixed[i], data_n[i], 'r')
    plt.plot([data_n[0], data_n[-1]], [data2_A, data2_A], color='green')
    plt.figure()

    R_interval = [0.001 * i + 1 for i in range(300)]
    Jaccars = []
    for R in R_interval:
        data1_new = [[data1_fixed[i][0] * R, data1_fixed[i][1] * R] for i in range(len(data1_fixed))]
        all_data = data1_new + data2_fixed
        min_inc = list(all_data[0])
        max_inc = list(all_data[0])
        for interval in all_data:
            min_inc[0] = max(min_inc[0], interval[0])
            min_inc[1] = min(min_inc[1], interval[1])
            max_inc[0] = min(max_inc[0], interval[0])
            max_inc[1] = max(max_inc[1], interval[1])
        JK = (min_inc[1] - min_inc[0]) / (max_inc[1] - max_inc[0])
        print(JK)
        Jaccars.append(JK)

    plt.plot(R_interval, Jaccars)
    plt.show()
    
      
    
