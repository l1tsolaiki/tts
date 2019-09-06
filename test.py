import numpy as np

from TT_class import TensorTrain


def test(*size, eps, restore=False):


a = np.array([[[1, 2],
               [3, 4]],

              [[5, 6],
               [7, 8]]])

eps = 0.2

A = TensorTrain(a, eps)

# print(*A.cores, sep="\n\n\n")

# print(A.recover_tensor())
print(*A.cores, sep="\n\n\n")
print("A - B norm:", np.linalg.norm(a - A.recover_tensor()))
print("eps * A norm:", eps * np.linalg.norm(a))
