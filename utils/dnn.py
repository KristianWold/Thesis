import numpy as np
import qiskit as qk


class Ansatz():
    def __call__(self, circuit, registers, parameters, inverse=False):
        n = parameters.shape[0]
        storage = registers[0]

        if not inverse:
            for param in parameters:
                circuit.ry(param, storage[i])

            for i in range(n - 1):
                circuit.cx(storage[i], storage[i + 1])

        else:
            for i in reversed(range(n - 1)):
                circuit.cx(storage[i], storage[i + 1])

            for param in reversed(parameters):
                circuit.ry(-param, storage[i])

        return circuit


class Layer():
    def __init__(self, n_inputs=None, n_outputs=None, ansatz=None, backend=None, shots=1000):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.ansatz = ansatz
        self.backend = backend
        self.shots = shots

        self.weights = np.random.uniform(0, 2 * np.pi, (n_inputs, n_outputs))

    def __call__(self, inputs):
        storage = qk.QuantumRegister(self.n_inputs, name="storage")
        ancilla = qk.QuantumRegister(1, name="ancilla")
        clas_reg = qk.ClassicalRegister(1, name="clas_reg")
        registers = [storage, ancilla, clas_reg]
        circuit = qk.QuantumCircuit(registers)

        backend = Aer.get_backend('qasm_simulator')
        outputs = []

        for i in range(n_outputs):
            parameters = self.weights[:, i]
            self.ansatz(circuit, registers, inputs, inverse=True)
            self.ansatz(circuit, registers, parameters, inverse=False)
            circuit.mcnot(storage, ancilla)
            circuit.measure(ancilla, clas_reg)

            job = qk.execute(circuit, backend, shots=shots)
            result = job.result()
            counts = result.get_counts(circuit)
            circuit.reset()


class Network():
    def __init(self, layers):
        self.layers = layers

    def __call__(self, x):
        for lay
