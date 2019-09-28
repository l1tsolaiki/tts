import matplotlib.pyplot as plt

from numpy import random


eps = [0.0001, 0.001, 0.02, 0.035, 0.05, 0.06, 0.065, 0.07, 0.075, 0.08, 0.085, 0.09, 0.095, 0.1, 0.11, 0.12,
       0.13, 0.14, 0.155, 0.17, 0.185, 0.2, 0.215, 0.23, 0.245, 0.27, 0.29, 0.31, 0.325, 0.34, 0.355, 0.37,
       0.39, 0.41, 0.44, 0.47, 0.50, 0.53, 0.56, 0.59, 0.62, 0.65, 0.68, 0.72, 0.77, 0.8]


def plot_size():
    file = open("size_stats.txt", "r", encoding="utf-8")
    eps = [0.05, 0.066, 0.08, 0.09, 0.1, 0.12, 0.14, 0.17, 0.2, 0.23, 0.26, 0.3, 0.34, 0.4, 0.48, 0.56, 0.66, 0.74, 0.8]
    size = [float(x) for x in file.readline().split()]
    plt.plot(eps, size, color="m")
    plt.xlabel("epsilon")
    plt.ylabel("times less elements stored")
    plt.title("graph shows how much less we need to store after rounding")
    plt.show()
    return 0


def plot_ttsvd():
    global eps
    file = open("ttsvd_data.txt", "r")
    elements_stored = [int(x) for x in file.readline().split()]
    plt.plot([0, 1], [1152000, 1152000], color="red", label="size of orig tensor")
    plt.plot(eps, elements_stored, color="grey", label="size of TT")
    plt.xlabel("epsilon")
    plt.ylabel("number of elements stored in TT format")
    plt.yscale("log")
    plt.title("decrease in elements stored in TT format with increase\n in eps for tensor of size 1152000")
    plt.legend()
    plt.figure(figsize=(7, 7))
    plt.show()
    return 0


def plot_sum_time():
    global eps
    file = open("sum_data.txt", "r")
    size = [int(x) for x in file.readline().split()][0]
    numpy_time = float(file.readline().split()[0])
    tt_times = [float(x) for x in file.readline().split()]
    plt.plot(eps, tt_times, label="TT-format time", c="green")
    plt.plot([0, 1], [numpy_time, numpy_time], label="numpy time", c="red")
    plt.title("time taken for performing summation of 2 tensors of sizes " + str(size))
    plt.xlabel("epsilon (for TT-format)")
    plt.ylabel("time taken")
    plt.yscale("log")
    plt.legend()
    plt.show()


def plot_ttsvd_time():
    global eps
    file = open("ttsvd_time_data.txt", "r")
    for i in range(5):
        size = int(file.readline().split()[0])
        tt_times = [float(x) for x in file.readline().split()]
        plt.plot(eps, tt_times, label="size=" + str(size), c=random.rand(3,))

    plt.title("time taken for converting tensor of different sizes into TT-format")
    plt.xlabel("epsilon (for TT-format)")
    plt.ylabel("time taken")
    plt.legend()
    plt.yscale("log")
    plt.show()


plot_ttsvd_time()
plot_size()
plot_ttsvd()
plot_sum_time()
