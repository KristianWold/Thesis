import numpy as np
import qiskit as qk
from copy import deepcopy
from optimizers import Adam, GD


class Ansatz():
    def __call__(self, circuit, registers, weight):
        n = weight.shape[0]
        storage = registers[0]

        for i, w in enumerate(weight):
            circuit.ry(w, storage[i])

        for i in range(n - 1):
            circuit.cx(storage[i], storage[i + 1])

        return circuit


class Encoder():
    def __call__(self, circuit, registers, inputs):
        n = inputs.shape[0]
        storage = registers[0]
        n_qubits = storage.size

        for i, w in enumerate(inputs):
            circuit.ry(w, storage[i % n_qubits])

        for i in range(n_qubits - 1):
            circuit.cx(storage[i], storage[i + 1])

        return circuit


class Sigmoid():

    def __call__(self, x):
        x = 1 / (1 + np.exp(-x))

        return x

    def derivative(self, x):
        x = x * (1 - x)

        return x


class Tanh():

    def __call__(self, x):
        x = np.tanh(x)

        return x

    def derivative(self, x):
        x = 1 - x**2

        return x


class Identity():

    def __call__(self, x):

        return x

    def derivative(self, x):

        return 1


class Dense():
    def __init__(self, n_features=None, n_targets=None, scale=1, activation=None):

        self.n_features = n_features
        self.n_targets = n_targets
        self.scale = scale
        self.activation = activation

        std = 1 / np.sqrt(n_targets)
        self.weight = np.random.uniform(-std, std, (n_features + 1, n_targets))

    def __call__(self, inputs):
        x = inputs @ self.weight[:-1] + self.weight[-1].reshape(1, -1)
        x = self.activation(x)

        return self.scale * x

    def grad(self, inputs, delta):
        n_samples = inputs.shape[0]
        output = self(inputs)
        delta = self.activation.derivative(output) * delta

        weight_gradient = 1 / n_samples * inputs.T @ delta
        bias_gradient = np.mean(delta, axis=0, keepdims=True)

        weight_gradient = np.concatenate(
            (weight_gradient, bias_gradient), axis=0)

        delta = delta @ self.weight[:-1].T

        return weight_gradient, delta


class QLayer():
    def __init__(self, n_qubits=None, n_features=None, n_targets=None, reps=1, scale=1, encoder=None, ansatz=None, backend=None, shots=1000):
        self.n_qubits = n_qubits
        self.n_features = n_features
        self.n_targets = n_targets
        self.reps = reps
        self.scale = scale
        self.encoder = encoder
        self.ansatz = ansatz
        self.backend = backend
        self.shots = shots

        self.weight = np.random.uniform(
            0, 2 * np.pi, (reps * n_qubits, n_targets))

    def __call__(self, inputs):
        outputs = []
        circuit_list = []
        n_samples = inputs.shape[0]
        for x in inputs:
            for i in range(self.n_targets):
                storage = qk.QuantumRegister(self.n_qubits, name="storage")
                clas_reg = qk.ClassicalRegister(1, name="clas_reg")
                registers = [storage, clas_reg]
                circuit = qk.QuantumCircuit(*registers)

                self.encoder(circuit, registers, x)
                for j in range(self.reps):
                    start = j * self.n_qubits
                    end = (j + 1) * self.n_qubits
                    self.ansatz(circuit, registers, self.weight[start:end, i])

                circuit.measure(storage[-1], clas_reg)
                circuit_list.append(circuit)

        transpiled_list = qk.transpile(circuit_list, backend=self.backend)
        qobject_list = qk.assemble(transpiled_list,
                                   backend=self.backend,
                                   shots=self.shots,
                                   max_parallel_shots=1,
                                   max_parallel_experiments=0
                                   )
        job = self.backend.run(qobject_list)

        for circuit in circuit_list:
            counts = job.result().get_counts(circuit)
            if "1" in counts:
                outputs.append(counts["1"] / self.shots)
            else:
                outputs.append(0)

        outputs = np.array(outputs).reshape(n_samples, -1)

        return self.scale * np.array(outputs)

    def grad(self, inputs, delta, samplewise=False):
        inputs = deepcopy(inputs)
        n_samples = inputs.shape[0]
        weight_partial = np.zeros((n_samples, *self.weight.shape))
        input_partial = np.zeros((n_samples, self.n_features, self.n_targets))

        for i in range(self.reps * self.n_qubits):
            self.weight[i, :] += np.pi / 2
            weight_partial[:, i, :] = self(inputs)
            self.weight[i, :] += -np.pi
            weight_partial[:, i, :] += -self(inputs)
            self.weight[i, :] += np.pi / 2

        for i in range(self.n_features):
            inputs[:, i] += np.pi / 2
            input_partial[:, i, :] = self(inputs)
            inputs[:, i] += -np.pi
            input_partial[:, i, :] += -self(inputs)
            inputs[:, i] += np.pi / 2

        weight_gradient = weight_partial * delta.reshape(n_samples, 1, -1)
        if not samplewise:
            weight_gradient = np.mean(weight_gradient, axis=0)

        delta = np.einsum("ij,ikj->ik", delta, input_partial)

        return weight_gradient, delta
