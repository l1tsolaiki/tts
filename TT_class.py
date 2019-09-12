import itertools
import math
from functools import reduce

import numpy as np
from scipy import linalg


class TensorTrain:

    def __init__(self, cores, shape, size):
        self.shape = shape
        self.cores = cores
        self.size = size

    # converts given tensor to TT format upon initialization

    @staticmethod
    def construct_from_tensor(A, eps):  # A   - given tensor
        cores = []  # eps - prescribed accuracy
        shape = A.shape  # dimensions
        size = A.size  # number of elements

        delta = eps / (np.sqrt(len(A.shape) - 1)) * np.linalg.norm(A)  # truncation parameter
        C = np.copy(A)  # temporary tensor
        r_prev, r_cur = 1, 0  # TT ranks

        for k in range(1, len(A.shape)):
            C = np.reshape(C, (r_prev * A.shape[k - 1], C.size // (r_prev * A.shape[k - 1])))

            U, S, Vt = TensorTrain.__delta_svd(C, delta)
            r_cur = len(S)
            G_k = np.reshape(U, (r_prev, A.shape[k - 1], r_cur))

            cores.append(G_k)
            C = np.diag(S).dot(Vt)
            r_prev = r_cur

        cores.append(C.reshape(*C.shape, 1))  # adding G_d

        return TensorTrain(cores, shape, size)

    @staticmethod
    def construct_form_cores(cores, shape, size):
        cores = cores
        shape = shape
        size = size
        return TensorTrain(cores, shape, size)

    @staticmethod
    def __delta_svd(A, delta):  # linearly search for best rank
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        rank = len(S)

        while rank > 0 and np.linalg.norm(A - U[:, :rank].dot(np.diag(S[:rank])).dot(Vt[:rank, :])) <= delta:
            rank -= 1

        return U[:, :rank + 1], S[:rank + 1], Vt[:rank + 1, :]

    def calc_elem(self, index):
        res = self.cores[0][:, index[0], :]
        for j in range(1, len(self.shape)):
            res = res.dot(self.cores[j][:, index[j], :])

        return res[0][0]

    def recover_tensor(self):
        iter = itertools.product(*[range(self.shape[k]) for k in range(len(self.shape))])
        tensor = np.zeros(self.shape)

        for i in range(self.size):
            index = next(iter)
            tensor[index] = self.calc_elem(index)

        return tensor

    def dot_prod(self, other):
        v = np.kron(self.cores[0][:, 0, :], self.cores[0][:, 0, :])
        for i in range(1, self.shape[0]):
            v += np.kron(self.cores[0][:, i, :], self.cores[0][:, i, :])

        for k in range(1, len(self.shape)):
            p_k = []

            for i in range(self.shape[k]):
                p_k.append(v.dot(np.kron(self.cores[k][:, i, :], other.cores[k][:, i, :])))
            v = sum(p_k)

        return v[0][0]

    def norm(self):
        return math.sqrt(self.dot_prod(self))

    @staticmethod
    def add(A, B):
        C_cores = []
        c_1 = np.concatenate((A.cores[0], B.cores[0]), axis=2)
        C_cores.append(c_1)

        for i in range(1, len(A.shape) - 1):
            a_k = np.concatenate((A.cores[i], np.zeros([A.cores[i].shape[0], A.cores[i].shape[1], B.cores[i].shape[2]])), axis=2)
            b_k = np.concatenate((np.zeros([B.cores[i].shape[0], B.cores[i].shape[1], A.cores[i].shape[2]]), B.cores[i]), axis=2)

            c_k = np.concatenate((a_k, b_k), axis=0)
            C_cores.append(c_k)

        c_d = np.concatenate((A.cores[len(A.shape) - 1], B.cores[len(B.shape) - 1]), axis=0)
        C_cores.append(c_d)

        C = TensorTrain.construct_form_cores(C_cores,
                                             list(map(lambda x: x.shape[1], C_cores)),
                                             int(reduce(lambda x, y: x * y, [x.shape[1] for x in C_cores])))
        return C

    @staticmethod
    def numb_multiply(A, alpha):
        C_cores = A.cores()
        C_cores[0] *= alpha
        C = TensorTrain.construct_form_cores(C_cores, A.shape, A.size)

        return C

    def round(self, eps):
        delta = eps / math.sqrt(len(self.shape) - 1) * self.norm()

        for k in range(len(self.shape) - 1, 0, -1):
            G_k = np.reshape(self.cores[k], (self.cores[k].shape[0], self.cores[k].shape[1] * self.cores[k].shape[2]))
            R, self.cores[k] = linalg.rq(G_k)
            self.cores[k - 1] = np.tensordot(self.cores[k], R, axes=1)

        for k in range(len(self.shape) - 2):
            return 0
            self.cores[k], S, V = np.linalg.svd(np.reshape(self.cores[k], full_matrices=False))
            # S = [S[i] if S[i] > delta else 0 for i in range(S.shape[0])]
            s = []
            for i in range(S.shape[0]):
                b = S[i]
                if S[i] > delta:
                    s.append(S[i])
                else:
                    s.append(0)
            S = np.diag(S)
            self.cores[k + 1] = np.tensordot(self.cores[k + 1], (V.dot(S)).T, axes=([0], [0]))
        return 0
