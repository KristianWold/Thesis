import numpy as np
import qiskit as qk

class Ansatz():
    def __call__(self, x, circuit, registers, inverse = False):
        if not inverse:

    for theta_ in theta:
        for i, t in enumerate(theta_):
            circuit.ry(t, storage[i])

        for i in range(n-1):
            circuit.cx(storage[i], storage[i+1])

    return circuit


class Layer():
    def __init__(self, n_inputs, n_outputs):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.weights = np.random.normal(0, 1, (n_inputs, n_outputs))

    def __call__(self, x, shots):
        storage = qk.QuantumRegister(self.n_inputs, name="storage")
        ancilla = qk.QuantumRegister(1, name="ancilla")
        clas_reg = qk.ClassicalRegister(1, name="clas_reg")
        circuit = qk.QuantumCircuit(storage, ancillae, clas_reg)



class Network():
    def __init(layers):
        self.layers = layers

    def __call
