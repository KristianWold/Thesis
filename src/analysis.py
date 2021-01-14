import numpy as np
import qiskit as qk
from copy import deepcopy
from neuralnetwork import *


def fisher_information_matrix(network, x, y):
    fim_list = []
    for x_, y_ in tqdm(zip(x, y)):
        x_ = x_.reshape(1, -1)
        y_ = y_.reshape(1, -1)

        network.backward(x_, y_)
        weight_gradient_list = []

        for grad in network.weight_gradient_list:
            weight_gradient_list.append(grad.flatten())

        weight_gradient_list = np.concatenate(
            weight_gradient_list).reshape(-1, 1)

        fim_ = weight_gradient_list @ weight_gradient_list.T

        fim_list.append(fim_)

    fim = np.mean(np.array(fim_list), axis=0)

    return fim
