import random
from timeit import default_timer as timer

import numpy as np

from TT_class import TensorTrain

eps = [0.0001, 0.001, 0.02, 0.035, 0.05, 0.06, 0.065, 0.07, 0.075, 0.08, 0.085, 0.09, 0.095, 0.1, 0.11, 0.12, 0.13,
       0.14, 0.155, 0.17, 0.185, 0.2, 0.215, 0.23, 0.245, 0.27, 0.29, 0.31, 0.325, 0.34, 0.355, 0.37,
       0.39, 0.41, 0.44, 0.47, 0.50, 0.53, 0.56, 0.59, 0.62, 0.65, 0.68, 0.72, 0.77, 0.8]


def test_addition(*size, A_eps, B_eps):
    a = np.random.rand(*size)
    b = np.random.rand(*size)

    A = TensorTrain.construct_from_tensor(a, A_eps)
    B = TensorTrain.construct_from_tensor(b, B_eps)

    start = timer()
    c = a + b
    end = timer()
    numpy_time = end - start

    start = timer()
    C = TensorTrain.add(A, B)
    end = timer()
    TT_time = end - start

    return numpy_time, TT_time


def test_norm(*size, eps):
    a = np.random.rand(*size)
    A = TensorTrain.construct_from_tensor(a, eps)

    start = timer()
    a_norm = np.linalg.norm(a)
    end = timer()
    numpy_time = end - start

    start = timer()
    A_norm = A.norm()
    end = timer()
    TT_time = end - start

    return numpy_time, TT_time


def get_rounding_data(file=open("rounding_raw.txt", "w", encoding="utf-8")):

    '''
    generates a random tensor of dimension d with d_i randomly selected from {3, ..., 10}
    calculates its TT representation with high precision (0.0001) using TT-SVD
    then calculates rounded version of this representation with eps consequentially increasing by 0.1 each iteration
    finally outputs the number of elements required to store this representation and its accuracy, calculated
    as norm of (initial representation - rounded)
    '''

    data = []
    eps = [0.05, 0.066, 0.08, 0.09, 0.1, 0.12, 0.14, 0.17, 0.2, 0.23, 0.26, 0.3, 0.34, 0.4, 0.48, 0.56, 0.66, 0.74, 0.8]

    for j in range(100):
        shape = [random.randrange(4, 8) for x in range(random.randrange(3, 7))]
        tensor = np.random.rand(*shape)
        TT = TensorTrain.construct_from_tensor(tensor, eps=0.0001)

        for i in range(len(eps)):
            rounded = TT.round(eps[i])
            dif = TensorTrain.subtract(TT, rounded)
            data.append((rounded.size, eps[i], rounded.get_cores_size(), dif.norm()))
        print(j, "done")

    for entry in data:
        for i in range(4):
            file.write(str(entry[i]) + " ")
        file.write("\n")


def get_ttsvd_data():
    global eps

    file = open("ttsvd_data.txt", "w")
    tensor = np.random.rand(4, 8, 6, 5, 8, 10, 3, 5)
    for i in range(len(eps)):
        TT = TensorTrain.construct_from_tensor(tensor, eps[i])
        file.write(str(TT.get_cores_size()) + " ")
        print(TT.get_cores_size())
        print("eps=", eps[i], " done", sep="")


def get_sum_data():
    global eps

    shape = [random.randrange(8, 12) for x in range(random.randrange(5, 8))]
    a = np.random.rand(*shape)
    b = np.random.rand(*shape)

    start = timer()
    a + b
    end = timer()
    numpy_time = end - start

    tt_times = []
    for e in eps:
        print(e, "done")
        A = TensorTrain.construct_from_tensor(a, e)
        B = TensorTrain.construct_from_tensor(b, e)
        start = timer()
        TensorTrain.add(A, B)
        end = timer()
        tt_times.append(end - start)

    file = open("sum_data.txt", "w")
    file.write(str(a.size) + " " + str(b.size) + "\n")
    file.write(str(numpy_time) + "\n")
    for t in tt_times:
        file.write(str(t) + " ")


def get_ttsvd_time_data():
    file = open("ttsvd_time_data.txt", "w")
    for i in range(5):
        shape = [random.randrange(9, 10) for x in range(random.randrange(6, 7))]
        a = np.random.rand(*shape)

        tt_times = []
        print("size =", a.size)
        for e in eps:
            print(e, "done")
            start = timer()
            A = TensorTrain.construct_from_tensor(a, e)
            end = timer()
            tt_times.append(end - start)

        file.write(str(a.size) + "\n")
        for t in tt_times:
            file.write(str(t) + " ")
        file.write("\n")
        print(i, "DONE")


get_ttsvd_time_data()
