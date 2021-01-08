import numpy as np
import qiskit as qk
from copy import deepcopy
from .optimizers import Adam, GD


class Ansatz():
    def __call__(self, circuit, registers, weight):
        n = weight.shape[0]
        storage = registers[0]

        for i, w in enumerate(weight):
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


class Sigmoid():

    def __call__(self, x):
        x = 1/(1 + np.exp(-x))

        return x

    def derivative(self, x):
        x = x*(1 - x)

        return x


class Identity():

    def __call__(self, x):

        return x

    def derivative(self, x):

        return 1


class Dense():
    def __init__(self, n_inputs=None, n_outputs=None, scale=1, activation = None):

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.scale = scale
        self.activation = activation

        self.weight = np.random.normal(0, 1, (n_inputs+1, n_outputs))

    def __call__(self, inputs):
        x = inputs@self.weight[:-1] + self.weight[-1].reshape(1,-1)
        x = self.activation(x)

        return self.scale*x

    def grad(self, inputs, delta):
        output = self(inputs).T
        delta = self.activation.derivative(output)*delta

        weight_gradient = (delta@inputs).T
        bias_gradient = delta.T

        weight_gradient = np.concatenate((weight_gradient, bias_gradient), axis=0)

        delta = self.weight[:-1]@delta

        return weight_gradient, delta


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

        self.weight = np.random.uniform(0, np.pi, (reps*n_qubits, n_outputs))

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
                self.ansatz(circuit, registers, self.weight[start:end, i])

            circuit.measure(storage[-1], clas_reg)
            job = qk.execute(circuit, self.backend, shots=self.shots)
            result = job.result()
            counts = result.get_counts(circuit)
            if "1" in counts:
                outputs.append(counts["1"]/self.shots)
            else:
                outputs.append(0)

        return self.scale*np.array(outputs)

    def grad(self, inputs, delta):
        inputs = deepcopy(inputs)
        weight_partial = np.zeros(self.weight.shape)
        input_partial = np.zeros((self.n_inputs, self.n_outputs))

        for i in range(self.reps*self.n_inputs):
            self.weight[i, :] += np.pi/4
            weight_partial[i, :] = 1/np.sqrt(2)*self(inputs)
            self.weight[i, :] += -np.pi/2
            weight_partial[i, :] += -1/np.sqrt(2)*self(inputs)
            self.weight[i, :] += np.pi/4

        for i in range(self.n_inputs):
            inputs[i] += np.pi/4
            input_partial[i, :] = 1/np.sqrt(2)*self(inputs)
            inputs[i] += -np.pi/2
            input_partial[i, :] += -1/np.sqrt(2)*self(inputs)
            inputs[i] += np.pi/4

        weight_gradient = weight_partial*delta.T
        delta = input_partial@delta

        return weight_gradient, delta


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

    def backward(self, X, Y):
        n = X.shape[0]
        weight_gradient_list = []
        for layer in self.layers:
            weight_gradient = np.zeros(layer.weight.shape)
            weight_gradient_list.append(weight_gradient)

        for x, y in zip(X,Y):
            #x = x.reshape(1,-1)
            #y = y.reshape(1,-1)

            self(x)
            y_pred = self.a[-1]
            delta = (y_pred - y).reshape(-1,1)

            for i, layer in reversed(list(enumerate(self.layers))):
                weight_gradient, delta = layer.grad(self.a[i], delta)
                weight_gradient_list[i] = weight_gradient_list[i] + weight_gradient

        self.weight_gradient_list = [grad/n for grad in weight_gradient_list]

    def step(self):
        weight_gradient_modified = self.optimizer(self.weight_gradient_list)

        for layer, grad in zip(self.layers, weight_gradient_modified):
            layer.weight += -self.optimizer.lr*grad

    def set_shots(self, shots):
        for layer in self.layers:
            layer.shots = shots


def sequential(dim, backend, reps=1, shots=1000):
    layers = []
    for i in range(len(dim)-2):
        in_dim = dim[i]
        out_dim = dim[i+1]
        layer = QLayer(n_qubits=in_dim, n_inputs=in_dim, n_outputs=out_dim, encoder=Encoder(), ansatz=Ansatz(), reps=reps, scale=np.pi, backend=backend, shots=shots)
        layers.append(layer)

    layer = QLayer(n_qubits=dim[-2], n_inputs=dim[-2], n_outputs=dim[-1], encoder=Encoder(), ansatz=Ansatz(), reps=reps, scale=1, backend=backend, shots=shots)
    layers.append(layer)

    optimizer = Adam(lr=0.01)
    network = NeuralNetwork(layers, optimizer)

    return network
