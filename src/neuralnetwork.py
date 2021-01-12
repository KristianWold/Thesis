import numpy as np
import qiskit as qk
from copy import deepcopy
from .optimizers import Adam, GD
from .layers import *


class NeuralNetwork():
    def __init__(self, layers, optimizer):
        self.layers = layers
        self.dim = []
        self.optimizer = optimizer

        for layer in self.layers:
            self.dim.append(layer.weight.shape)

        self.optimizer.initialize(self.dim)
        self.a = []
        self.weight_gradient_list = []

    def __call__(self, x):
        self.a = []
        self.a.append(x)
        for layer in self.layers:
            x = layer(x)
            self.a.append(x)

    def predict(self, x):
        self(x)
        return self.a[-1]

    def backward(self, x, y):
        self.weight_gradient_list = []

        self(x)
        y_pred = self.a[-1]
        delta = (y_pred - y)

        for i, layer in reversed(list(enumerate(self.layers))):
            weight_gradient, delta = layer.grad(self.a[i], delta)
            self.weight_gradient_list.append(weight_gradient)

        self.weight_gradient_list.reverse()

    def step(self):
        weight_gradient_modified = self.optimizer(self.weight_gradient_list)

        for layer, grad in zip(self.layers, weight_gradient_modified):
            layer.weight += -self.optimizer.lr * grad

    def deriv(self, x):
        self.weight_gradient_list = []

        self(x)
        delta = np.ones_like(x)

        for i, layer in reversed(list(enumerate(self.layers))):
            weight_gradient, delta = layer.grad(self.a[i], delta)

        return delta

    def set_shots(self, shots):
        for layer in self.layers:
            layer.shots = shots


def sequential(dim, backend, reps=1, shots=1000):
    layers = []
    for i in range(len(dim) - 2):
        in_dim = dim[i]
        out_dim = dim[i + 1]
        layer = QLayer(n_qubits=in_dim, n_features=in_dim, n_targets=out_dim, encoder=Encoder(
        ), ansatz=Ansatz(), reps=reps, scale=np.pi, backend=backend, shots=shots)
        layers.append(layer)

    layer = QLayer(n_qubits=dim[-2], n_features=dim[-2], n_targets=dim[-1], encoder=Encoder(
    ), ansatz=Ansatz(), reps=reps, scale=1, backend=backend, shots=shots)
    layers.append(layer)

    optimizer = Adam(lr=0.01)
    network = NeuralNetwork(layers, optimizer)

    return network
