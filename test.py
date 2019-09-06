import numpy as np

from TT_class import TensorTrain

a = np.array([[[1, 2],
               [3, 4]],

              [[5, 6],
               [7, 8]]])

# cores = TT_SVD(a, 0.001)

approx = [[[0, 0],
           [0, 0]],

          [[0, 0],
           [0, 0]]]

for i in range(2):
    for j in range(2):
        for k in range(2):
            approx[i][j][k] = cores[0][:, i, :].dot(cores[1][:, j, :]).dot(cores[2][:, k, :])[0][0]

# print("tensor:", a, sep="\n", end="\n\nA")
#
# print("approximation:", approx, sep="\n")
#
# print(np.linalg.norm(a - approx))


# def recover_tensor(cores):
#     # todo use product from itertools to iterate over tuples of indices
#     # like this:
#     # dimensions = [4, 5, 2, 4, 7, 2]
#     # a = product(*[range(dimensions[k]) for k in range(len(dimensions))]) - iterator object
#     # then just do next(a) to get next index

eps = 0.2

A = TensorTrain(a, eps)

# print(*A.cores, sep="\n\n\n")

# print(A.recover_tensor())
print(*A.cores, sep="\n\n\n")
# print("A - B norm:", np.linalg.norm(a - A.recover_tensor()))
# print("eps * A norm:", eps * np.linalg.norm(a))
