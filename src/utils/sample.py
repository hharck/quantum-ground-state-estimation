import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

SHOTS = 20_000


def sample(
    qc: QuantumCircuit, shots: int = SHOTS, sim: AerSimulator | None = None
) -> np.ndarray:
    """Run a circuit and return how often each outcome came up.

    Args:
        qc: circuit to run; it must measure into its classical bits.
        shots: how many runs.
        sim: simulator to use, e.g. one carrying a noise model. The default is a seeded
            `AerSimulator`, so repeated runs give the same numbers.

    Returns:
        An array of length 2**qc.num_clbits of frequencies summing to 1, indexed by the
        outcome read as an integer. The circuit is transpiled first, so custom gates are
        decomposed.
    """
    sim = sim or AerSimulator(seed_simulator=7)
    counts = sim.run(transpile(qc, sim), shots=shots).result().get_counts()
    p = np.zeros(2**qc.num_clbits)
    for bits, c in counts.items():
        p[int(bits, 2)] = c / shots
    return p
