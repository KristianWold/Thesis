import numpy as np
import qiskit as qk
from copy import deepcopy
from optimizers import Adam, GD
from data_encoders import *


class Ansatz():
    def __call__(self, circuit, data_register, weight):
        n_qubits = data_register.size

        for i, w in enumerate(weight):
            circuit.ry(w, data_register[i])

        for i in range(n_qubits - 1):
            circuit.cx(data_register[i], data_register[i + 1])

        return circuit


class SwapTest():
    def __call__(self, circuit, register_a, register_b, ancilla_swap):
        circuit.h(ancilla_swap)
        for r_a, r_b in zip(register_a, register_b):
            circuit.cswap(ancilla_swap, r_a, r_b)

        circuit.h(ancilla_swap)

        return circuit


class ParallelModel():
    def __init__(self, n_features=None, n_targets=None, reps=1,  backend=None, shots=1000):
        self.parallel_encoder = ParallelEncoder()
        self.encoder = Encoder()
        self.ansatz = Ansatz()
        self.swap_test = SwapTest()

        self.n_features = n_features
        self.n_targets = n_targets
        self.reps = reps

        self.theta = np.random.uniform(
            0, 2 * np.pi, self.n_features * self.reps)
        self.backend = backend
        self.shots = shots

    def predict(self, x):
        n_samples, _ = x.shape
        y_pred = []
        for i, x_ in enumerate(x):
            features = qk.QuantumRegister(self.n_features, name="features")
            classical = qk.ClassicalRegister(1)
            registers = [features, classical]
            circuit = qk.QuantumCircuit(*registers)

            circuit = self.encoder(circuit, features, x_)
            for i in range(self.reps):
                start = i * self.n_features
                end = (i + 1) * self.n_features
                circuit = self.ansatz(circuit, features, self.theta[start:end])

            circuit.measure(features[-1], classical)

            job = qk.execute(circuit, self.backend, shots=self.shots)
            counts = job.result().get_counts(circuit)
            if "0" in counts:
                y_pred.append(
                    [2 * np.arccos(np.sqrt(counts["0"] / self.shots))])
            else:
                y_pred.append([0])

        return np.array(y_pred)

    def loss(self, x, y):
        n_samples, n_features = x.shape
        _, n_targets = y.shape
        n_ancilla = np.log2(n_samples)

        features = qk.QuantumRegister(n_features, name="features")
        ancilla_features = qk.QuantumRegister(
            n_ancilla, name="ancilla_feature")

        targets = qk.QuantumRegister(n_targets, name="targets")
        ancilla_targets = qk.QuantumRegister(n_ancilla, name="ancilla_target")

        ancilla_swap = qk.QuantumRegister(1, name="swap")
        classical = qk.ClassicalRegister(1)

        registers = [features, ancilla_features, targets, ancilla_targets,
                     ancilla_swap, classical]

        circuit = qk.QuantumCircuit(*registers)

        circuit = self.parallel_encoder(circuit, features, ancilla_features, x)
        circuit = self.parallel_encoder(circuit, targets, ancilla_targets, y)

        for i in range(self.reps):
            start = i * self.n_features
            end = (i + 1) * self.n_features
            circuit = self.ansatz(circuit, features, self.theta[start:end])

        register_a = [features[-1]] + ancilla_features[:]
        register_b = targets[:] + ancilla_targets[:]
        circuit = self.swap_test(circuit, register_a, register_b, ancilla_swap)

        circuit.measure(ancilla_swap, classical)

        job = qk.execute(circuit, self.backend, shots=self.shots)
        counts = job.result().get_counts(circuit)
        if "0" in counts:
            loss = -counts["0"] / self.shots
        else:
            loss = 0

        return 2 * (loss + 1)

    def gradient(self, x, y):
        weight_gradient = np.zeros_like(self.theta)

        for i in range(len(self.theta)):
            self.theta[i] += np.pi / 2
            weight_gradient[i] += 1 / np.sqrt(2) * self.loss(x, y)
            self.theta[i] += -np.pi
            weight_gradient[i] += -1 / np.sqrt(2) * self.loss(x, y)
            self.theta[i] += np.pi / 2

        return weight_gradient
