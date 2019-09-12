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


# print(test_addition(9, 99, 10, 322, 2, A_eps=0.3, B_eps=0.3))
print(test_norm(9, 10, 20, 15, 2, eps=0.1))
