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


A = TensorTrain.construct_from_tensor(np.random.rand(3, 4, 5, 6), 0.01)
A.round(0.01)
print(1)