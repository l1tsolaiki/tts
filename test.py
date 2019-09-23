import random
from timeit import default_timer as timer

import numpy as np

from TT_class import TensorTrain


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
    eps = [0.05, 0.1, 0.12, 0.14, 0.17, 0.2, 0.25, 0.3, 0.37, 0.45, 0.6, 0.8]

    for j in range(500):
        shape = [random.randrange(3, 8) for x in range(random.randrange(3, 7))]
        tensor = np.random.rand(*shape)
        TT = TensorTrain.construct_from_tensor(tensor, eps=0.0001)

        for i in range(len(eps)):
            rounded = TT.round(eps[i])
            dif = TensorTrain.subtract(TT, rounded)
            data.append((rounded.size, eps[i], rounded.get_cores_size(), dif.norm()))
        print(j, "done")

    data.sort()
    for entry in data:
        for i in range(4):
            file.write(str(entry[i]) + " ")
        file.write("\n")


get_rounding_data()
