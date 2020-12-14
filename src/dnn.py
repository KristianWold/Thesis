import numpy as np
import qiskit as qk
import torch
import torch.nn as nn
from copy import deepcopy

class GD():
    def __init__(self, lr=0.01):
        self.lr = lr

    def initialize(self, dims):
        pass

    def __call__(self, weight_gradient_list):
        return weight_gradient_list


class Adam():
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.m = []
        self.v = []
        self.t = 0

    def initialize(self, dims):
        for dim in dims:
            self.m.append(np.zeros(dim))
            self.v.append(np.zeros(dim))

    def __call__(self, weight_gradient_list):
        self.t += 1
        weight_gradient_modified = []

        for grad, m_, v_ in zip(weight_gradient_list, self.m, self.v):
            m_ = self.beta1*m_ + (1 - self.beta1)*grad
            v_ = self.beta2*v_ + (1 - self.beta2)*grad**2

            m_hat = m_/(1 - self.beta1**self.t)
            v_hat = v_/(1 - self.beta2**self.t)
            grad_modified = m_hat/(np.sqrt(v_hat) + self.eps)
            weight_gradient_modified.append(grad_modified)

        return weight_gradient_modified


class Ansatz():
    def __call__(self, circuit, registers, weights):
        n = weights.shape[0]
        storage = registers[0]

        for i, w in enumerate(weights):
            circuit.ry(2*w, storage[i])

        for i in range(n - 1):
            circuit.cx(storage[i], storage[i + 1])

        return circuit


class Encoder():
    def __call__(self, circuit, registers, inputs):
        n = inputs.shape[0]
        storage = registers[0]
        n_qubits = storage.size

        for i, w in enumerate(inputs):
            circuit.ry(2*w, storage[i%n_qubits])

        for i in range(n_qubits - 1):
            circuit.cx(storage[i], storage[i + 1])

        return circuit


class QLayer():
    def __init__(self, n_qubits = None, n_inputs=None, n_outputs=None, reps=1, scale = 1, encoder=None, ansatz=None, backend=None, shots=1000):
        self.n_qubits = n_qubits
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.reps = reps
        self.scale = scale
        self.encoder = encoder
        self.ansatz = ansatz
        self.backend = backend
        self.shots = shots

        self.weights = np.random.uniform(0, np.pi, (reps*n_qubits, n_outputs))

    def __call__(self, inputs):
        outputs = []
        for i in range(self.n_outputs):
            storage = qk.QuantumRegister(self.n_qubits, name="storage")
            clas_reg = qk.ClassicalRegister(1, name="clas_reg")
            registers = [storage, clas_reg]
            circuit = qk.QuantumCircuit(*registers)

            self.encoder(circuit, registers, inputs)
            for j in range(self.reps):
                start = j*self.n_qubits
                end = (j+1)*self.n_qubits
                self.ansatz(circuit, registers, self.weights[start:end, i])

            circuit.measure(storage[-1], clas_reg)
            job = qk.execute(circuit, self.backend, shots=self.shots)
            result = job.result()
            counts = result.get_counts(circuit)
            if "1" in counts:
                outputs.append(counts["1"]/self.shots)
            else:
                outputs.append(0)

        return self.scale*np.array(outputs)

    def partial(self, inputs):
        inputs = deepcopy(inputs)
        self.weight_partial = np.zeros(self.weights.shape)
        self.input_partial = np.zeros((self.n_inputs, self.n_outputs))

        for i in range(self.reps*self.n_inputs):
            self.weights[i, :] += np.pi/4
            self.weight_partial[i, :] = 1/np.sqrt(2)*self(inputs)
            self.weights[i, :] += -np.pi/2
            self.weight_partial[i, :] += -1/np.sqrt(2)*self(inputs)
            self.weights[i, :] += np.pi/4

        for i in range(self.n_inputs):
            inputs[i] += np.pi/4
            self.input_partial[i, :] = 1/np.sqrt(2)*self(inputs)
            inputs[i] += -np.pi/2
            self.input_partial[i, :] += -1/np.sqrt(2)*self(inputs)
            inputs[i] += np.pi/4

        return self.weight_partial, self.input_partial


class CLayer(nn.Module):
    def __init__(self, n_inputs=None, n_outputs=None, scale = 1, activation = None):
        super(CLayer, self).__init__()

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.layer = nn.Linear(n_inputs,n_outputs)
        self.scale = scale
        self.activation = activation

    @property
    def weights(self):
        weights_ = torch.cat((self.layer.weight, self.layer.bias.reshape(-1,1)), 1)
        return(weights_.T)

    def update_weights(self,lr_grad):
        with torch.no_grad():
            self.layer.weight += lr_grad.T[:,:-1]
            self.layer.bias += lr_grad.T[:,-1]
        self.zero_grad()

    def __call__(self, inputs):
        input = deepcopy(inputs).reshape(1,-1)

        if not type(input) is np.ndarray:
            input = input.detach()
        self.input = torch.tensor(input, requires_grad=True, dtype=torch.float)
        self.output = self.layer(self.input).reshape(1,-1)
        self.output = self.activation(self.output)
        self.zero_grad()
        return deepcopy(self.output.detach().numpy())

    def partial(self, inputs):
        self(inputs)
        self.input_partial = torch.zeros((self.output.shape[1], self.input.shape[1]), requires_grad = False)
        self.weight_partial = torch.zeros((self.layer.weight.shape[0], self.layer.weight.shape[1] + 1), requires_grad=False)
        for j in range(self.output.shape[1]):
            self.output[0, j].backward(retain_graph = True)
            self.input_partial[j, :] = self.input.grad[0, :]
            self.weight_partial[j,:-1] = self.layer.weight.grad[j,:]
            self.weight_partial[j,-1] = self.layer.bias.grad[j]

        self.zero_grad()
        self.weight_partial = self.weight_partial.detach().numpy().T
        self.input_partial = self.input_partial.detach().numpy().T
        return self.weight_partial, self.input_partial


class QNN():
    def __init__(self, layers, optimizer):
        self.layers = layers
        self.dim = []
        self.optimizer = optimizer

        for layer in self.layers:
            self.dim.append(layer.weights.shape)

        self.optimizer.initialize(self.dim)
        self.a = 0
        self.weight_gradient_list = []

    def __call__(self, x):
        self.a = []
        self.a.append(x)
        for layer in self.layers:
            x = layer(x)
            self.a.append(x)

    def backward(self, X, Y):
        n = X.shape[0]
        weight_gradient_list = []
        for layer in self.layers:
            weight_gradient = np.zeros(layer.weights.shape)
            weight_gradient_list.append(weight_gradient)

        for x, y in zip(X,Y):
            self(x)
            y_pred = self.a[-1]
            delta = (y_pred - y).reshape(-1,1)

            for i, layer in reversed(list(enumerate(self.layers))):
                partial = layer.partial(self.a[i])
                weight_partial = partial[0]
                input_partial = partial[1]

                weight_gradient_list[i] += weight_partial*delta.T
                delta = input_partial@delta

        self.weight_gradient_list = [grad/n for grad in weight_gradient_list]

    def step(self):
        weight_gradient_modified = self.optimizer(self.weight_gradient_list)
        for layer, grad in zip(self.layers, weight_gradient_modified):
                if not type(layer.weights) is np.ndarray:
                    grad_ = torch.tensor(grad.copy(), requires_grad = False)
                    layer.update_weights(torch.nn.Parameter(-self.optimizer.lr*grad_))
                else:
                    layer.weights += -self.optimizer.lr*grad

    def set_shots(self, shots):
        for layer in self.layers:
            layer.shots = shots
