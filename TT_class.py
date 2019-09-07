import itertools

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

    @staticmethod
    def addition(A, B):
        C_cores = []
        # for i in range(1, len(A.shape) - 1):

        # C = TensorTrain()
        C.construct_form_cores(C_cores, A.get_dims, A.get_size)

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


A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
C = np.column_stack((A, B))
print(C)

D = np.array([0, 9, 3, 22])
F = np.row_stack((C, D))

Q = np.zeros([3, 4])
print(Q)

print(F)