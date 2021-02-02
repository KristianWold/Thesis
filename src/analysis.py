import numpy as np
import qiskit as qk
from copy import deepcopy
from tqdm.notebook import tqdm
from neuralnetwork import *
from utils import *


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


class FIM():
    def __init__(self, model):
        self.model = model
        self.fim = None

    def fit(self, x):
        n_samples = x.shape[0]

        self.model.backward(x, samplewise=True, include_loss=False)
        gradient = self.model.weight_gradient_list

        gradient_flattened = []
        for grad in gradient:
            gradient_flattened.append(grad.reshape(n_samples, -1))

        gradient_flattened = np.concatenate(gradient_flattened, axis=1)

        self.fim = 1 / n_samples * gradient_flattened.T @ gradient_flattened


def trajectory_length(x):
    diff = (x[1:] - x[:-1])
    diff = np.append(diff, (x[0] - x[-1]).reshape(1, -1), axis=0)
    accum = np.sum(diff**2, axis=1)
    accum = np.sum(np.sqrt(accum))
    return accum
