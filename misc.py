import random

import numpy as np

from TT_class import TensorTrain


def get_rounding_data(d, file=open("rounding_processed.txt", "w", encoding="utf-8")):

    # demonstrates inconsistencies in TT norm

    file.write("we will generate a random tensor, calculate its TT representation and then its rounded version")
    file.write("\nsince rounded version is still quite precise (TT - rounded) will be a tensor of numbers close to 0")
    file.write("\nand for such tensors TT norm behaves inconsistently\n")
    for d in range(3, 7):
        shape = [random.randrange(3, 8) for x in range(d)]
        tensor = np.random.rand(*shape)
        TT = TensorTrain.construct_from_tensor(tensor, eps=0.0001)
        file.write(str(shape) + "\n")
        file.write("d = " + str(d) + ":\n")
        before = TT.get_cores_size()

        for i in range(1, 10):
            eps = i / 10
            rounded = TT.round(eps)
            dif = TensorTrain.subtract(TT, rounded)

            a = TT.recover_tensor()
            b = rounded.recover_tensor()
            print(TT.get_cores_size(), rounded.get_cores_size(), sep="\n")
            print("norm tt", dif.dot_prod(dif))
            print("norm np", np.linalg.norm(a - b) ** 2)

            file.write("eps = " + str(eps) + ":\n")
            file.write(str(before) + "\n")
            file.write("elements stored: " + str(rounded.get_cores_size()) + "\n")
            file.write("np dot prod: " + str(np.linalg.norm(dif.recover_tensor()) ** 2) + "\n")
            file.write("tt dot prod: " + str(dif.dot_prod(dif)) + "\n\n")
            # file.write(str(eps) + "\t" + str(rounded.get_cores_size()) + "\t" + str())


get_rounding_data(3)
