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
    numpy = end - start

    start = timer()
    C = TensorTrain.add(A, B)
    end = timer()
    TT = end - start

    return numpy, TT


def test_norm(*size, eps):
    a = np.random.rand(*size)
    A = TensorTrain.construct_from_tensor(a, eps)

    start = timer()
    a_norm = np.linalg.norm(a)
    end = timer()
    numpy = end - start

    start = timer()
    A_norm = A.norm()
    end = timer()
    TT = end - start

    return numpy, TT


def get_rounding_data(file=open("rounding_test.txt", "w", encoding="utf-8")):

    '''
    generates a random tensor of dimension d with d_i randomly selected from {3, ..., 10}
    calculates its TT representation with high precision (0.0001) using TT-SVD
    then calculates rounded version of this representation with eps consequentially increasing by 0.1 each iteration
    finally outputs the number of elements required to store this representation and its accuracy, calculated
    as norm of (initial representation - rounded)
    '''

    for d in range(3, 7):
        shape = [random.randrange(3, 11) for x in range(d)]
        tensor = np.random.rand(*shape)
        TT = TensorTrain.construct_from_tensor(tensor, eps=0.0001)
        file.write("d = " + str(d) + "\t" + "shape = " + str(shape) + "\n")

        for i in range(1, 10):
            eps = i / 10
            rounded = TT.round(eps)
            dif = TensorTrain.subtract(TT, rounded)
            print(TT.get_cores_size(), rounded.get_cores_size(), sep="\n")
            print("norm tt", dif.dot_prod(dif))
            print("norm np", np.linalg.norm(TT.recover_tensor() - rounded.recover_tensor()) ** 2)
            file.write(str(eps) + "\t" + str(rounded.get_cores_size()) + "\t" + str(dif.norm()) + "\n")


# f = open("output.txt", "w", encoding="utf-8")
# for i in range(100):
#     shape = [random.randrange(3, 11) for x in range(4)]
#     a = np.random.rand(*shape)
#     TT = TensorTrain.construct_from_tensor(a, 0.01)
#     f.write(str(np.linalg.norm(a) * np.linalg.norm(a)) + "\n")
#     f.write(str(TT.dot_prod(TT)) + "\n\n\n")
get_rounding_data()
# a = np.random.rand(9, 3, 4)
# A = TensorTrain.construct_from_tensor(a, eps=0.1)
# B = TensorTrain.construct_from_tensor(a, eps=0.1)
# a *= 3
# A = TensorTrain.add(TensorTrain.add(A, A), A)
#
# c = A.get_cores_size()
# print(*[x.shape for x in A.cores], sep="\n", end="\n\n=========\n")
# print(np.linalg.norm(A.recover_tensor() - a))
# A.round(0.01)
# print("\n", *[x.shape for x in A.cores], sep="\n")
#
# b = A.recover_tensor()
#
# print(A.get_cores_size())
# print(c)
# print(np.linalg.norm(a - b))