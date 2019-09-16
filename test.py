from timeit import default_timer as timer

import numpy as np

from TT_class import TensorTrain


def test(*size, eps):
    pass

    # test TT_SVD approximation with a random tensor and prescribed accuracy eps

    tensor = np.random.rand(*size)
    TT = TensorTrain()
    TT.construct_from_tensor(tensor, eps)

    print("A - B norm:\t", np.linalg.norm(tensor - TT.recover_tensor()))
    print("eps * norm(A):\t", eps * np.linalg.norm(tensor))


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


a = np.random.rand(20, 20, 20)
A = TensorTrain.construct_from_tensor(a, eps=0.0001)
A = TensorTrain.add(TensorTrain.add(A, A), A)
a *= 3
print(*[x.shape for x in A.cores], sep="\n", end="\n\n=========\n")
A.round(0.2)
print("\n", *[x.shape for x in A.cores], sep="\n")

# b = A.recover_tensor()
# print(a, b, sep="\n===================\n")
# print(a - b)
# print(np.linalg.norm(a - b))