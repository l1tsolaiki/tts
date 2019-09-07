import itertools
from functools import reduce

import numpy as np


class TensorTrain:

    # converts given tensor to TT format upon initialization

    def construct_from_tensor(self, A, eps):  # A   - given tensor
        self.set_cores([])  # eps - prescriped accuracy
        self.set_dims(A.shape)  # dimensions
        self.set_size(A.size)  # number of elements

        delta = eps / (np.sqrt(len(A.shape) - 1)) * np.linalg.norm(A)  # truncation parameter
        C = np.copy(A)  # temporary tensor
        r_prev, r_cur = 1, 0  # TT ranks

        for k in range(1, len(A.shape)):
            C = np.reshape(C, (r_prev * A.shape[k - 1], C.size // (r_prev * A.shape[k - 1])))

            U, S, Vt = self.__delta_svd(C, delta)
            r_cur = len(S)
            G_k = np.reshape(U, (r_prev, A.shape[k - 1], r_cur))

            self.cores.append(G_k)
            C = np.diag(S).dot(Vt)
            r_prev = r_cur

        self.cores.append(C.reshape(*C.shape, 1))  # adding G_d

    def construct_form_cores(self, cores, dims, size):
        self.set_cores(cores)
        self.set_dims(dims)
        self.set_size(size)

    @staticmethod
    def __delta_svd(A, delta):  # linearly search for best rank
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        rank = len(S)

        while rank > 0 and np.linalg.norm(A - U[:, :rank].dot(np.diag(S[:rank])).dot(Vt[:rank, :])) <= delta:
            rank -= 1

        return U[:, :rank + 1], S[:rank + 1], Vt[:rank + 1, :]

    def calc_elem(self, index):
        res = self.cores[0][:, index[0], :]
        for j in range(1, len(self.dims)):
            try:
                res = res.dot(self.cores[j][:, index[j], :])
            except Exception:
                print(index)

        return res[0][0]

    def recover_tensor(self):
        iter = itertools.product(*[range(self.dims[k]) for k in range(len(self.dims))])
        tensor = np.zeros(self.dims)

        for i in range(self.size):
            index = next(iter)
            tensor[index] = self.calc_elem(index)

        return tensor

    def dot_prod(self, other):
        v = np.kron(self.cores[0][:, 0, :], self.cores[0][:, 0, :])
        for i in range(1, self.dims[0]):
            v += np.kron(self.cores[0][:, i, :], self.cores[0][:, i, :])

        for k in range(1, len(self.dims)):
            p_k = []

            for i in range(self.dims[k]):
                p_k.append(v.dot(np.kron(self.cores[k][:, i, :], other.cores[k][:, i, :])))
            v = sum(p_k)

        return v[0][0]

    @staticmethod
    def addition(A, B):
        C_cores = []
        c_1 = np.concatenate((A.get_cores()[0], B.get_cores()[0]), axis=2)
        C_cores.append(c_1)

        for i in range(1, len(A.get_dims()) - 1):
            a_k = np.concatenate((A.get_cores()[i], np.zeros([B.get_dims()[0], A.get_dims()[1], A.get_dims()[2]])),
                                 axis=2)
            b_k = np.concatenate((np.zeros([A.get_dims()[0], B.get_dims()[1], B.get_dims()[2]]), B.get_cores()[i]),
                                 axis=2)
            c_k = np.concatenate((a_k, b_k), axis=0)
            C_cores.append(c_k)

        c_d = np.concatenate((A.get_cores()[len(A.get_dims()) - 1], B.get_cores()[len(B.get_dims()) - 1]), axis=0)
        C_cores.append(c_d)
        C = TensorTrain()
        a = list(map(lambda x: x.shape[1], C_cores))
        b = int(reduce(lambda x, y: x * y, [x.shape[1] for x in C_cores]))
        C.construct_form_cores(C_cores, list(map(lambda x: x.shape[1], C_cores)), int(reduce(lambda x, y: x * y, [x.shape[1] for x in C_cores])))
        return C

    @staticmethod
    def numb_multiply(A, alpha):
        C_cores = A.get_cores()
        C_cores[0] *= alpha
        C = TensorTrain()
        C.construct_form_cores(C_cores, A.get_dims(), A.get_size())

    # todo я бы на shape переименовал
    def get_dims(self):
        return self.dims

    def set_dims(self, shape):
        self.dims = shape

    def get_cores(self):
        return self.cores

    def set_cores(self, cores):
        self.cores = cores

    def get_size(self):
        return self.size

    def set_size(self, size):
        self.size = size

    def test(self):
        a = self.get_cores()
        a = []
