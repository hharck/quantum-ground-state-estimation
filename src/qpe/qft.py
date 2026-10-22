import numpy as np
from qiskit import QuantumCircuit
from typing import List


def qft(n: int) -> QuantumCircuit:
    """Build the quantum Fourier transform on n qubits.

    Each qubit gets an H, which writes the first digit of its phase, then controlled
    phases from the lower qubits, which append the finer digits. The product form comes
    out in reversed qubit order, so swaps at the end put it back.

    Args:
        n: number of qubits.

    Returns:
        An n-qubit circuit named "QFT", with barriers between the qubits' stages. It has
        no measurements, so it can be composed into a larger circuit.
    """
    qc = QuantumCircuit(n, name="QFT")
    for j in reversed(range(n)):  # MSB first
        qc.h(j)  # first digit of this qubit's phase
        for k in reversed(range(j)):  # lower bits append finer digits
            qc.cp(np.pi / 2 ** (j - k), k, j)  # controlled-R_{j-k+1}
        qc.barrier()
    for q in range(n // 2):  # fix the reversed output order
        qc.swap(q, n - 1 - q)
    return qc


def iqft(n: int) -> QuantumCircuit:
    """Build the inverse quantum Fourier transform on n qubits.

    The same circuit as `qft` read backwards, with every phase negated: undo the swaps,
    then for each qubit strip the digits already decoded before reading it out with an H.

    Args:
        n: number of qubits.

    Returns:
        An n-qubit circuit named "IQFT", with no measurements.
    """
    qc = QuantumCircuit(n, name="IQFT")
    for q in range(n // 2):  # undo the swaps first
        qc.swap(q, n - 1 - q)
    for j in range(n):  # LSB first: its bit decodes first
        qc.barrier()
        for k in range(j):  # strip already-decoded bits...
            qc.cp(-np.pi / 2 ** (j - k), k, j)  # controlled-R_{j-k+1}^dagger
        qc.h(j)  # ...then read this bit out
    return qc


def semiclassical_iqft(n: int) -> QuantumCircuit:
    """Build the IQFT for the case where its output is measured immediately.

    Griffiths & Niu (1996): a qubit is never touched after its H except as a control, so
    it can be measured right away, and a controlled phase is diagonal, so the phases it
    would apply become single-qubit phases conditioned on a bit already measured. No
    two-qubit gates are left, and the swaps become a relabelling.

    Args:
        n: number of digits, i.e. both qubits and classical bits.

    Returns:
        An n-qubit, n-bit circuit named "SC-IQFT". Digit j lands in classical bit j, read
        from qubit n-1-j. It contains mid-circuit measurements and feed-forward, so it
        needs a simulator that supports them.
    """
    qc = QuantumCircuit(n, n, name="SC-IQFT")
    for j in range(n):  # same order as iqft
        target = n - 1 - j  # the swaps become this relabelling
        for k in range(j):  # conditioned on already-measured bits
            with qc.if_test((qc.clbits[k], 1)):
                qc.p(-np.pi / 2 ** (j - k), target)
        qc.h(target)
        qc.measure(target, j)  # bit j of the result
    return qc


def compact_semiclassical_iqft(
    qubits_to_use: int, digits_to_retrieve: int
) -> List[QuantumCircuit]:
    """Build the semiclassical readout as one circuit per digit, so qubits can be recycled.

    Splitting after every measurement leaves room for the caller to prepare the next
    digit in between, which is what lets `digits_to_retrieve` digits come off
    `qubits_to_use` qubits. With `qubits_to_use` = 1 this is Kitaev's iterative phase
    estimation.

    Args:
        qubits_to_use: how many qubits are available; digit j is read from qubit
            `qubits_to_use - 1 - j % qubits_to_use`.
        digits_to_retrieve: how many digits to read in total.

    Returns:
        A list of `digits_to_retrieve` circuits, in the order they are applied, each of
        `qubits_to_use` qubits and `digits_to_retrieve` bits. Step j reads digit j and
        then resets its qubit when a later digit reuses it. The caller prepares digit j's
        qubit before step j: in QPE that means H and controlled-U^(2^q).
    """
    steps: List[QuantumCircuit] = []

    for j in range(digits_to_retrieve):
        qc = _read_digit(qubits_to_use, digits_to_retrieve, j)
        if j + qubits_to_use < digits_to_retrieve:
            qc.reset(qubits_to_use - 1 - j % qubits_to_use)
        steps.append(qc)

    return steps


def _read_digit(qubits_to_use: int, digits_to_measure: int, j: int) -> QuantumCircuit:
    """Build one step of the semiclassical readout: the corrections, H and measurement of digit j.

    Args:
        qubits_to_use: width of the circuit, i.e. how many qubits the readout recycles.
        digits_to_measure: total digits, which sets the classical register's width, since
            digit j's corrections may depend on any earlier bit.
        j: which digit to read, counted from the most significant.

    Returns:
        A circuit of `qubits_to_use` qubits and `digits_to_measure` bits, acting only on
        qubit `qubits_to_use - 1 - j % qubits_to_use`. The caller must have prepared digit
        j on that qubit. The result lands in classical bit j.
    """
    qc = QuantumCircuit(qubits_to_use, digits_to_measure, name=f"digit {j}")
    q = (
        qubits_to_use - 1 - j % qubits_to_use
    )  # the caller prepares digit j on this qubit
    for i in range(j):
        with qc.if_test((qc.clbits[i], 1)):
            qc.p(-np.pi / 2 ** (j - i), q)
    qc.h(q)
    qc.measure(q, j)
    return qc
