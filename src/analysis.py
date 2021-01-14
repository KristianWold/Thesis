import numpy as np
import qiskit as qk
from copy import deepcopy
from neuralnetwork import *
from tqdm.notebook import tqdm


def fisher_information_matrix(network, x, y):
    # extract weights from network
    weight_list = []
    for layer in network.layers:
        weight = layer.weight.flatten()
        weight_list.append(weight)

    weight_list = np.concatenate(weight_list).reshape(-1, 1)

    fim_list = []
    for x_, y_ in tqdm(zip(x, y), total=len(x)):
        x_ = x_.reshape(1, -1)
        y_ = y_.reshape(1, -1)

        # extract gradient from network
        network.backward(x_, y_)
        weight_gradient_list = []
        for grad in network.weight_gradient_list:
            weight_gradient_list.append(grad.flatten())

        weight_gradient_list = np.concatenate(
            weight_gradient_list).reshape(-1, 1)

        fim_ = weight_gradient_list @ weight_gradient_list.T

        fim_list.append(fim_)

    fim = np.mean(np.array(fim_list), axis=0)

    fr = weight_list.T @ fim @ weight_list

    return fim, fr
