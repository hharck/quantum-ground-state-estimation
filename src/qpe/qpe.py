import numpy as np
from enum import Enum
from functools import partial
from typing import Callable, List, Sequence
from qiskit import QuantumCircuit
from qiskit.circuit import ControlledGate, Gate
from qiskit.circuit.library import StatePreparation, UnitaryGate, PauliEvolutionGate
from qiskit.quantum_info import Operator, Statevector, random_unitary
from qiskit.synthesis.evolution import SuzukiTrotter

from .mps import dpss_gates
from .readout import Kick, Staircase, iqft_readout, semiclassical_readout, compact_readout

UnitaryLike = np.ndarray | QuantumCircuit | Gate | Operator
InitialStateLike = np.ndarray | QuantumCircuit | Statevector

class Prep(Enum):
    """How the counting register is prepared."""

    DEFAULT = "default"
    DPSS = "dpss"


class Readout(Enum):
    """Which IQFT reads the counting register out."""

    IQFT = "iqft"
    SEMICLASSICAL = "semiclassical"
    COMPACT = "compact"


STAIRCASES = {
    Prep.DEFAULT: None,  # plain |+> layer
    Prep.DPSS: partial(dpss_gates, delta=1e-3, bond_dim=4),
}


def build_qpe(
    prep: Prep,
    readout: Readout,
    U: UnitaryLike,
    initial_state: InitialStateLike | None,
    n: int = 5,
    k: int = 3,
) -> QuantumCircuit:
    """Build a QPE circuit from a counting-register preparation and an IQFT style.

    Args:
        prep: how the counting register is prepared.
        readout: which IQFT reads it out.
        U: the unitary whose eigenphase is wanted.
        initial_state: the target register's state, ideally an eigenvector of U, or None for |0...0>.
        n: digits of the estimate.
        k: counting qubits the compact readout recycles; ignored by the others.

    Returns:
        A circuit whose n classical bits hold the estimate j, phi ~ j / 2^n.
    """
    staircase = STAIRCASES[prep]
    match readout:
        case Readout.IQFT:
            return _qpe(U, n, initial_state, staircase)
        case Readout.SEMICLASSICAL:
            return _semiclassical_qpe(U, n, initial_state, staircase)
        case Readout.COMPACT:
            return _compact_qpe(U, k, n, initial_state, staircase)


def unitary_with_phases(
    phases: Sequence[float], seed: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Build a test unitary with the eigenphases you ask for.

    U = V diag(e^{2 pi i phase_k}) V^dagger for a random unitary V.

    Args:
        phases: the eigenphases, as fractions of a turn. There must be 2^m of them, which
            makes U act on m qubits.
        seed: seed for the random V, so a run can be reproduced.

    Returns:
        (U, V). Column k of V is the eigenvector whose eigenphase is phases[k], which is
        what to pass as `initial_state`.
    """
    V = random_unitary(len(phases), seed=seed).data
    return V @ np.diag(np.exp(2j * np.pi * np.asarray(phases))) @ V.conj().T, V


def exact(qc: QuantumCircuit) -> np.ndarray:
    """Compute a circuit's outcome probabilities exactly, without sampling.

    Args:
        qc: a circuit whose only measurements are at the end, e.g. from `build_qpe` with Readout.IQFT. It will not
            work on the semiclassical or compact readouts, which measure as they go.

    Returns:
        An array of length 2**qc.num_clbits, the probabilities of the measured bits.
    """
    return Statevector(qc.remove_final_measurements(inplace=False)).probabilities(
        range(qc.num_clbits)
    )


def ideal_distribution(phi: float, n: int) -> np.ndarray:
    """Compute QPE's outcome distribution in closed form, |sin(pi N d) / (N sin(pi d))|^2.

    With d = phi - j/N and N = 2^n. Every noiseless readout gives this, so it is the
    reference the sampled runs are compared against.

    Args:
        phi: the eigenphase, as a fraction of a turn.
        n: counting qubits.

    Returns:
        An array of length 2**n: the probability of each outcome j. When phi lands exactly
        on the grid, the distribution is 1 there and 0 elsewhere.
    """
    N = 2**n
    d = phi - np.arange(N) / N
    num, den = np.sin(np.pi * N * d), N * np.sin(np.pi * d)
    exact_hit = np.isclose(den, 0)
    return np.where(exact_hit, 1.0, (num / np.where(exact_hit, 1, den)) ** 2)


def _controlled_powers(U: PauliEvolutionGate, n: int) -> List[ControlledGate]:
    """Build the controlled powers of U that write the phase onto the counting register.

    Args:
        U: the unitary, as a matrix, gate or circuit.
        n: how many powers to build, i.e. counting qubits.

    Returns:
        [controlled-U^(2^q) for q < n], each named "cU^(2^q)" so a timing model can price
        them apart. Powers are taken as matrix powers here; on hardware U^(2^q) would cost
        2^q applications of U unless U has a cheaper form.
    """
    gates = []

    for q in range(n):
        power = 2**q

        gate = PauliEvolutionGate(
            U.operator,
            time=U.time * power,
            # U^(2^q) is 2^q Trotter steps; one step of length 2^q t has error growing with 2^q
            synthesis=SuzukiTrotter(order=U.synthesis.order, reps=U.synthesis.reps * power),
        ).control(1)

        gate.name = f"cU^{power}"
        gates.append(gate)

    return gates

def _kick_and_prep(
    U: UnitaryLike, n: int, initial_state: InitialStateLike | None,
) -> tuple[int, Kick, Callable[[QuantumCircuit], QuantumCircuit]]:
    """Make the pieces the three builders share: the target's width, its kick and its preparation.

    Args:
        U: the unitary being estimated.
        n: counting qubits, which sets how many powers are needed.
        initial_state: the target register's state vector, or None for |0...0>.

    Returns:
        (m, kick, prep), where m is the number of qubits U acts on, kick(qc, qubit, q)
        appends controlled-U^(2^q) from `qubit` onto the last m qubits, and prep(qc)
        returns a copy of qc with the target put into `initial_state` before anything else.
    """
    # gates and circuits know their width; Operator(U) would build U's dense 2^m x 2^m matrix just to count
    m = U.num_qubits if hasattr(U, "num_qubits") else Operator(U).num_qubits
    cU = _controlled_powers(U, n)

    def kick(qc: QuantumCircuit, qubit: int, q: int) -> None:
        """Append controlled-U^(2^q) from `qubit` onto the target, the last m qubits of qc."""
        qc.append(cU[q], [qubit, *qc.qubits[-m:]])

    def prep(qc: QuantumCircuit) -> QuantumCircuit:
        """Return qc with the target put into initial_state before anything else."""
        if initial_state is None:
            return qc
        return qc.compose(
            initial_state, qubits=qc.qubits[-m:], front=True
        )

    return m, kick, prep


def _qpe(
    U: UnitaryLike,
    n: int,
    initial_state: InitialStateLike | None = None,
    staircase: Staircase | None = None,
) -> QuantumCircuit:
    """Build textbook QPE: controlled powers, the coherent IQFT, then measure.

    Args:
        U: the unitary whose eigenphase is wanted.
        n: counting qubits, i.e. digits of the estimate.
        initial_state: the target register's state, ideally an eigenvector of U. None
            leaves it in |0...0>, which gives a mixture weighted by its overlaps.
        staircase: gates for the counting register, e.g.
            `partial(mps.dpss_gates, bond_dim=2)`. None gives the plain |+> layer.

    Returns:
        A circuit of n + m qubits and n classical bits, where m is U's width. Its
        measurements are all at the end, so `exact` works on it.
    """
    m, kick, prep = _kick_and_prep(U, n, initial_state)
    qc = QuantumCircuit(n + m, n)
    qc.compose(iqft_readout(n, kick, staircase, extra=m), inplace=True)
    qc.measure(range(n), range(n))
    return prep(qc)


def _semiclassical_qpe(
    U: UnitaryLike,
    n: int,
    initial_state: np.ndarray | None = None,
    staircase: Staircase | None = None,
) -> QuantumCircuit:
    """Build QPE with the semiclassical readout, which has no two-qubit gates in the IQFT.

    Args:
        U: the unitary whose eigenphase is wanted.
        n: counting qubits.
        initial_state: the target register's state, or None for |0...0>.
        staircase: gates for the counting register, or None for the plain |+> layer.

    Returns:
        A circuit of n + m qubits and n classical bits, measured as it goes, so it needs a
        simulator that supports mid-circuit measurement and feed-forward.
    """
    m, kick, prep = _kick_and_prep(U, n, initial_state)
    return prep(semiclassical_readout(n, kick, staircase, extra=m))


def _compact_qpe(
    U: UnitaryLike,
    qubits_to_use: int,
    digits_to_retrieve: int,
    initial_state: np.ndarray | None = None,
    staircase: Staircase | None = None,
) -> QuantumCircuit:
    """Build QPE on a recycled set of counting qubits; the target is shared throughout.

    With `qubits_to_use` = 1 this is Kitaev's iterative phase estimation: one counting
    qubit, measured and reset once per digit.

    Args:
        U: the unitary whose eigenphase is wanted.
        qubits_to_use: physical counting qubits.
        digits_to_retrieve: digits of the estimate.
        initial_state: the target register's state, or None for |0...0>. It is prepared
            once and survives every digit, since the target is never measured.
        staircase: gates for the counting register, or None for the plain |+> layer. Its
            widest gate sets the minimum qubits_to_use, e.g. 1 + log2(chi) for the DPSS gates.

    Returns:
        A circuit of qubits_to_use + m qubits and digits_to_retrieve classical bits.
    """
    m, kick, prep = _kick_and_prep(U, digits_to_retrieve, initial_state)
    return prep(
        compact_readout(qubits_to_use, digits_to_retrieve, kick, staircase, extra=m)
    )
