import matplotlib.pyplot as plt


def plot_size():
    file = open("size_stats.txt", "r", encoding="utf-8")
    eps = [0.05, 0.1, 0.12, 0.14, 0.17, 0.2, 0.25, 0.3, 0.37, 0.45, 0.6, 0.8]
    size = [float(x) for x in file.readline().split()]
    plt.plot(eps, size)
    plt.savefig('images/size.png')
    plt.xlabel("epsilon")
    plt.ylabel("times less elements stored")


plot_size()
