import numpy as np


class TensorTrain:

    # converts given tensor to TT format upon initialization

    def __init__(self, A, eps):                                            # A   - given tensor
        self.cores = []                                                    # eps - prescriped accuracy

        delta = eps / (np.sqrt(len(A.shape) - 1)) * np.linalg.norm(A)      # truncation parameter
        C = np.copy(A)                                                     # temporary tensor
        r_prev, r_cur = 1, 0                                               # TT ranks

        for k in range(1, len(A.shape)):
            C = np.reshape(C, (r_prev * A.shape[k], C.size // (r_prev * A.shape[k])))

            U, S, Vt = self.__delta_svd(C, delta)
            r_cur = len(S)
            G_k = np.reshape(U, (r_prev, A.shape[k], r_cur))

            self.cores.append(G_k)
            C = np.diag(S).dot(Vt)
            r_prev = r_cur

        self.cores.append(C.reshape(*C.shape, 1))                          # adding G_d

    @staticmethod
    def __delta_svd(A, delta):                                             # linearly search for best rank
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        rank = len(S)

        while rank > 0 and np.linalg.norm(A - U[:, :rank].dot(np.diag(S[:rank])).dot(Vt[:rank, :])) <= delta:
            rank -= 1

        return U[:, :rank + 1], S[:rank + 1], Vt[:rank + 1, :]

