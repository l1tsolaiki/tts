import numpy as np

from TT_class import TensorTrain


def test(*size, eps):

    # test TT_SVD approximation with a random tensor and prescribed accuracy eps

    tensor = np.random.rand(*size)
    TT = TensorTrain()
    TT.construct_from_tensor(tensor, eps)

    print("A - B norm:\t", np.linalg.norm(tensor - TT.recover_tensor()))
    print("eps * norm(A):\t", eps * np.linalg.norm(tensor))


a = np.array([[[1, 2],
               [3, 4]],

              [[5, 6],
               [7, 8]]])

# eps = 0.4
#
# test(8, 9, 12, 53, 2, eps=eps)
# a = np.random.rand(2, 3, 4)
A = TensorTrain.construct_from_tensor(a, 0.01)
# A.test()
# print(1)
# print(A.dot_prod(A))
# print(np.linalg.norm(A.recover_tensor()) ** 2)
B = TensorTrain.construct_from_tensor(a, 0.01)
C = TensorTrain.addition(A, B)
print(C.recover_tensor())
print(1)
