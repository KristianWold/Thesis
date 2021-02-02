import numpy as np
import qiskit as qk
import pickle
from tqdm.notebook import tqdm

from optimizers import *
from layers import *
from utils import *


class NeuralNetwork():
    def __init__(self, layers=None, optimizer=None):
        self.layers = layers
        self.dim = []
        self.optimizer = optimizer

        if not self.layers == None:
            for layer in self.layers:
                self.dim.append(layer.weight.shape)

        if not self.optimizer == None:
            self.optimizer.initialize(self.dim)

        self.a = []
        self.weight_gradient_list = []

    def __call__(self, x, verbose=False):
        if verbose:
            dec = tqdm
        else:
            dec = identity

        self.a = []
        self.a.append(x)
        for layer in dec(self.layers):
            x = layer(x)
            self.a.append(x)

    def predict(self, x, verbose=False):
        self(x, verbose=verbose)
        return self.a[-1]

    def backward(self, x, y=None, samplewise=False, include_loss=True):
        n_samples = x.shape[0]
        self.weight_gradient_list = []

        self(x)
        y_pred = self.a[-1]

        if include_loss:
            delta = (y_pred - y)
        else:
            delta = np.ones((n_samples, 1))

        for i, layer in reversed(list(enumerate(self.layers))):
            weight_gradient, delta = layer.grad(
                self.a[i], delta, samplewise=samplewise)
            self.weight_gradient_list.append(weight_gradient)

        self.weight_gradient_list.reverse()

    def step(self):
        weight_gradient_modified = self.optimizer(self.weight_gradient_list)

        for layer, grad in zip(self.layers, weight_gradient_modified):
            layer.weight += -self.optimizer.lr * grad

    def train(self, x, y, epochs=100, verbose=False):
        if verbose:
            dec = tqdm
        else:
            dec = identity

        for i in dec(range(epochs)):
            self.backward(x, y)
            self.step()

            if verbose:
                y_pred = self.predict(x)
                loss = np.mean((y_pred - y)**2)
                print(loss)

    def deriv(self, x):
        self.weight_gradient_list = []

        self(x)
        delta = np.ones_like(x)

        for i, layer in reversed(list(enumerate(self.layers))):
            weight_gradient, delta = layer.grad(self.a[i], delta)

        return delta

    @property
    def weight(self):
        weight_list = []
        for layer in self.layers:
            weight_list.append(layer.weight)

        return weight_list

    def randomize_weight(self):
        for layer in self.layers:
            layer.randomize_weight()

    def set_shots(self, shots):
        for layer in self.layers:
            layer.shots = shots

    def save(self, filename):
        pickle.dump(self, open(filename, "wb"))

    def load(self, filename):
        self = pickle.load(open(filename, "rb"))


def sequential(dim, type, backend, reps=1, shots=1000):
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
