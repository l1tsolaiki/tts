import numpy as np


def delta_svd(A, delta):
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    rank = len(S)

    while rank > 0 and np.linalg.norm(A - U[:, :rank].dot(np.diag(S[:rank])).dot(Vt[:rank, :])) <= delta:
        rank -= 1

    return U[:, :rank + 1], S[:rank + 1], Vt[:rank + 1, :]


def TT_SVD(A, eps):
    delta = eps / (np.sqrt(len(A.shape) - 1)) * np.linalg.norm(A)
    C = np.copy(A)
    r_prev, r_cur = 1, 0
    cores = []

    for k in range(1, len(A.shape)):
        C = np.reshape(C, (r_prev * A.shape[k], C.size // (r_prev * A.shape[k])))
        U, S, Vt = delta_svd(C, delta)
        r_cur = len(S)
        G_k = np.reshape(U, (r_prev, A.shape[k], r_cur))

        cores.append(G_k)
        C = np.diag(S).dot(Vt)
        r_prev = r_cur

    cores.append(C.reshape(*C.shape, 1))
    return cores


a = np.array([[[1, 2],
               [3, 4]],

              [[5, 6],
               [7, 8]]])

cores = TT_SVD(a, 0.001)
print(*cores, sep="\n\n\n\n")
