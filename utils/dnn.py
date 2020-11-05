import numpy as np
import qiskit as qk


class Ansatz():
    def __call__(self, circuit, registers, parameters, inverse=False):
        n = parameters.shape[0]
        storage = registers[0]

        if not inverse:
            for i, param in enumerate(parameters):
                circuit.ry(param, storage[i])

            for i in range(n - 1):
                circuit.cx(storage[i], storage[i + 1])

        else:
            for i in reversed(range(n - 1)):
                circuit.cx(storage[i], storage[i + 1])

            for i, param in enumerate(parameters):
                circuit.ry(-param, storage[i])

        return circuit


class Layer():
    def __init__(self, n_inputs=None, n_outputs=None, ansatz=None, backend=None, shots=1000):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.ansatz = ansatz
        self.backend = backend
        self.shots = shots

        self.parameters = np.random.uniform(0, 2 * np.pi, (n_inputs, n_outputs))

    def __call__(self, inputs):
        outputs = []

        for i in range(self.n_outputs):
            storage = qk.QuantumRegister(self.n_inputs, name="storage")
            ancilla = qk.QuantumRegister(1, name="ancilla")
            clas_reg = qk.ClassicalRegister(1, name="clas_reg")
            registers = [storage, ancilla, clas_reg]
            circuit = qk.QuantumCircuit(*registers)

            #self.ansatz(circuit, registers, inputs)
            self.ansatz(circuit, registers, self.parameters[:, i])
            circuit.mcrx(np.pi, storage, ancilla[0])
            circuit.measure(ancilla, clas_reg)

            job = qk.execute(circuit, self.backend, shots=self.shots)
            result = job.result()
            counts = result.get_counts(circuit)
            if "0" in counts:
                outputs.append(counts["0"]/self.shots)
            else:
                outputs.append(0)

        return np.array(outputs)

    def grad(self, inputs):
        self.gradient = np.zeros(self.parameters.shape)

        for i in range(self.n_inputs):
            self.parameters[i, :] += np.pi/2
            self.gradient[i, :] = 0.5*self(inputs)
            self.parameters[i, :] += -np.pi
            self.gradient[i, :] += -0.5*self(inputs)
            self.parameters[i, :] += np.pi/2

        return self.gradient

class QNN():
    def __init__(self, layers):
        self.layers = layers
        self.activations = 0

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def backprop(self):
        pass
