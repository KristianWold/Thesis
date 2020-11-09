import numpy as np
import qiskit as qk


class Ansatz():
    def __call__(self, circuit, registers, weights, inverse=False):
        n = weights.shape[0]
        storage = registers[0]

        if not inverse:
            for i, w in enumerate(weights):
                circuit.ry(2*w, storage[i])

            for i in range(n - 1):
                circuit.cx(storage[i], storage[i + 1])

        else:
            for i in reversed(range(n - 1)):
                circuit.cx(storage[i], storage[i + 1])

            for i, w in enumerate(weights):
                circuit.ry(-2*w, storage[i])

        return circuit


class Layer():
    def __init__(self, n_inputs=None, n_outputs=None, ansatz=None, backend=None, shots=1000):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.ansatz = ansatz
        self.backend = backend
        self.shots = shots

        self.weights = np.random.uniform(0, np.pi, (n_inputs + 1, n_outputs))

    def __call__(self, inputs):
        outputs = []

        for i in range(self.n_outputs):
            storage = qk.QuantumRegister(self.n_inputs, name="storage")
            ancilla = qk.QuantumRegister(1, name="ancilla")
            clas_reg = qk.ClassicalRegister(1, name="clas_reg")
            registers = [storage, ancilla, clas_reg]
            circuit = qk.QuantumCircuit(*registers)

            self.ansatz(circuit, registers, inputs)
            self.ansatz(circuit, registers, self.weights[:-1, i])
            circuit.ry(2*self.weights[-1, i], ancilla[0])
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
        inputs = np.copy(inputs)
        self.weights_gradient = np.zeros(self.weights.shape)
        self.input_gradient = np.zeros((self.n_inputs, self.n_outputs))

        for i in range(self.n_inputs + 1):
            self.weights[i, :] += np.pi/4
            self.weights_gradient[i, :] = 1/(np.sqrt(2))*self(inputs)
            self.weights[i, :] += -np.pi/2
            self.weights_gradient[i, :] += -1/(np.sqrt(2))*self(inputs)
            self.weights[i, :] += np.pi/4

        for i in range(self.n_inputs):
            inputs[i] += np.pi/4
            self.input_gradient[i, :] = 1/np.sqrt(2)*self(inputs)
            inputs[i] += -np.pi/2
            self.input_gradient[i, :] += -1/np.sqrt(2)*self(inputs)
            inputs[i] += np.pi/4

        return self.weights_gradient, self.input_gradient

class QNN():
    def __init__(self, layers):
        self.layers = layers
        self.activations = 0

    def __call__(self, x):
        self.a = []
        self.a.append(x)
        for layer in self.layers:
            x = layer(x)
            self.a.append(x)

        return x

    def backprop(self, x):
        self.weights_gradients = []
        self.input_gradients = []
        self(x)

        for i, layer in enumerate(self.layers):
            gradient = layer.grad(self.a[i])
            self.weights_gradients.append(gradient[0])
            self.input_gradients.append(gradient[1])
