"""Circuits that read a phase out of the counting register.

Each one starts from a staircase: the gates the counting register is prepared with.
`plus_layer` is the default and `mps.dpss_gates` the alternative. `staircase_phase_state` builds what QPE's controlled-U stage leaves behind, and the
three readouts add the coherent, semiclassical or compact IQFT from qft.py on top.

The phase arrives either as a number, which becomes a stand-in P gate, or as a kick
function, which is what QPE passes to imprint it with controlled-U^(2^q).
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.circuit.library import HGate
from typing import Callable, List

from .qft import iqft, semiclassical_iqft, compact_semiclassical_iqft

Kick = Callable[[QuantumCircuit, int, int], object]
Staircase = Callable[[int], List[tuple[Gate, List[int]]]]


def plus_layer(n: int) -> List[tuple[Gate, List[int]]]:
    """The default staircase: an H on every counting qubit, i.e. the equal superposition.

    Args:
        n: counting qubits.

    Returns:
        (H, [q]) for every qubit, highest first, so the phases that follow also go in
        highest power first, the order the digits are read in.
    """
    return [(HGate(), [q]) for q in reversed(range(n))]


def staircase_phase_state(
    n: int,
    phase_or_kick: float | Kick,
    staircase: Staircase | None = None,
    extra: int = 0,
) -> QuantumCircuit:
    """Build the counting register QPE's controlled-U stage leaves behind.

    The staircase runs first, and each qubit gets its phase exactly once, right after the
    last staircase gate on it.

    Args:
        n: counting qubits, i.e. digits of the phase.
        phase_or_kick: a number phi, or a kick function; see `_as_kick`.
        staircase: a callable n -> [(gate, qubits)], e.g.
            `partial(mps.dpss_gates, bond_dim=2)`. None gives `plus_layer`.
        extra: qubits added after the counting ones, left for the kick to use. QPE passes
            the size of its target register here.

    Returns:
        A circuit of n + extra qubits and no classical bits.
    """
    kick = _as_kick(phase_or_kick)
    gates = (staircase or plus_layer)(n)
    phase_after = _last_gate(gates)
    qc = QuantumCircuit(n + extra)
    for q in reversed(range(n)):  # no staircase gate on q: nothing to wait for
        if q not in phase_after:
            kick(qc, q, q)
    for i, (gate, qubits) in enumerate(gates):
        qc.append(gate, qubits)
        for q in reversed(qubits):  # highest power first, the digit read first
            if phase_after[q] == i:
                kick(qc, q, q)
    return qc


def iqft_readout(
    n: int,
    phase_or_kick: float | Kick,
    staircase: Staircase | None = None,
    extra: int = 0,
) -> QuantumCircuit:
    """Build `staircase_phase_state` read out by the coherent IQFT.

    Args:
        n: counting qubits.
        phase_or_kick: a number phi, or a kick function; see `_as_kick`.
        staircase: gates for the counting register, or None for `plus_layer`.
        extra: target qubits after the counting ones.

    Returns:
        A circuit of n + extra qubits with no measurements, so `Statevector` can be used
        on it directly. Add `measure_all` to sample it.
    """
    return staircase_phase_state(n, phase_or_kick, staircase, extra).compose(
        iqft(n), qubits=range(n)
    )


def semiclassical_readout(
    n: int,
    phase_or_kick: float | Kick,
    staircase: Staircase | None = None,
    extra: int = 0,
) -> QuantumCircuit:
    """Build `staircase_phase_state` read out by the semiclassical IQFT.

    Args:
        n: counting qubits.
        phase_or_kick: a number phi, or a kick function; see `_as_kick`.
        staircase: gates for the counting register, or None for `plus_layer`.
        extra: target qubits after the counting ones.

    Returns:
        A circuit of n + extra qubits and n classical bits, measured as it goes. Digit j
        lands in bit j.
    """
    return staircase_phase_state(n, phase_or_kick, staircase, extra).compose(
        semiclassical_iqft(n), qubits=range(n)
    )


def compact_readout(
    qubits_to_use: int,
    digits_to_retrieve: int,
    phase_or_kick: float | Kick,
    staircase: Staircase | None = None,
    extra: int = 0,
) -> QuantumCircuit:
    """Build the staircase plus the semiclassical readout on a recycled set of qubits.

    The staircase is laid out on the counting qubits of `semiclassical_iqft`, where digit
    j sits on qubit digits_to_retrieve-1-j, while the compact readout reads digit j from
    qubit qubits_to_use-1 - j % qubits_to_use. So every gate is remapped, which permutes
    the order of its qubits unless qubits_to_use = digits_to_retrieve, and a qubit is
    recycled for digit j + qubits_to_use once digit j has been read.

    Rather than whole rounds, the circuit is assembled one digit at a time, because a gate
    can span digits that belong to different rounds. A qubit is read, reset and
    re-prepared just before a gate needs it for a new digit. Re-preparing means replaying
    that qubit's single-qubit staircase gate or, when its first gate is a wider one, just
    handing that gate the reset |0>. The DPSS MPS staircase walks down the register one
    qubit at a time, so it recycles on as few qubits as its widest gate, 1 + log2(chi).

    Args:
        qubits_to_use: physical counting qubits available.
        digits_to_retrieve: digits of the phase to read.
        phase_or_kick: a number phi, or a kick function; see `_as_kick`.
        staircase: gates for the counting register, or None for `plus_layer`.
        extra: target qubits after the counting ones.

    Returns:
        A circuit of qubits_to_use + extra qubits and digits_to_retrieve classical bits.
        Digit j lands in bit j.

    Raises:
        ValueError: if a multi-qubit staircase gate doesn't fit on the qubits available.
    """
    kick = _as_kick(phase_or_kick)
    gates = (staircase or plus_layer)(digits_to_retrieve)
    phase_after = _last_gate(gates)
    widest = max((len(qubits) for _, qubits in gates), default=1)
    if qubits_to_use < widest:
        raise ValueError(
            f"a {widest}-qubit staircase gate needs qubits_to_use >= {widest}, "
            f"got {qubits_to_use}"
        )
    first_index = {}  # counting qubit -> index of the first staircase gate on it
    for i, (_, qubits) in enumerate(gates):
        for q in qubits:
            first_index.setdefault(q, i)
    # a single-qubit first gate is the qubit's preparation, replayed on the recycled qubit;
    # a wider first gate, e.g. a DPSS MPS gate, prepares it itself from the reset |0>
    prep_index = {q: i for q, i in first_index.items() if len(gates[i][1]) == 1}
    phys = (
        lambda q: qubits_to_use - 1 - (digits_to_retrieve - 1 - q) % qubits_to_use
    )  # counting qubit q holds digit digits_to_retrieve-1-q
    steps = compact_semiclassical_iqft(
        qubits_to_use, digits_to_retrieve
    )  # one per digit, interleaved with the staircase below
    qc = QuantumCircuit(qubits_to_use + extra, digits_to_retrieve)
    live, phased, read = set(), set(), 0

    def imprint(q: int) -> None:
        """Put counting qubit q's phase on the physical qubit holding it."""
        kick(qc, phys(q), q)
        phased.add(q)

    def read_next() -> None:
        """Add the next digit's readout step, which also resets its qubit if it is reused."""
        nonlocal read
        assert (
            digits_to_retrieve - 1 - read in phased
        ), f"digit {read} read before its staircase finished"
        qc.compose(
            steps[read],
            qubits=range(qubits_to_use),
            clbits=range(digits_to_retrieve),
            inplace=True,
        )
        read += 1

    def prepare(q: int) -> None:  # move counting qubit q onto phys(q)
        """Make physical qubit phys(q) ready to carry counting qubit q's digit."""
        j = digits_to_retrieve - 1 - q
        while (
            read <= j - qubits_to_use
        ):  # phys(q) still holds digit j - qubits_to_use: read it first (its step resets it)
            read_next()
        live.add(q)
        if q not in prep_index:  # its first gate is a wider one, which takes the |0> as is
            return
        i = prep_index[q]
        qc.append(gates[i][0], [phys(q)])
        if phase_after[q] == i:  # nothing else on q: phase right away
            imprint(q)

    for i, (gate, qubits) in enumerate(gates):
        if prep_index.get(qubits[0]) == i and len(qubits) == 1:
            continue  # a preparation, replayed by prepare() when its turn comes
        for q in qubits:
            if q not in live:
                prepare(q)
        qc.append(gate, [phys(q) for q in qubits])
        for q in reversed(qubits):  # highest power first, the digit read first
            if phase_after[q] == i:  # gate i was q's last
                imprint(q)
    for q in range(digits_to_retrieve - 1, -1, -1):  # qubits nothing else touched
        if q not in live:
            prepare(q)
    while read < digits_to_retrieve:
        read_next()
    return qc


def _as_kick(phase_or_kick: float | Kick) -> Kick:
    """Normalise a phase argument into a kick function.

    Args:
        phase_or_kick: either a number phi, which becomes the stand-in
            P(2*pi*phi*2^q) on counting qubit q, or an existing kick(qc, qubit, q), e.g.
            QPE's controlled-U^(2^q) onto the `extra` target qubits.

    Returns:
        A callable kick(qc, qubit, q) that imprints 2^q * phi on `qubit`. A kick passed in
        is returned unchanged.
    """
    if callable(phase_or_kick):
        return phase_or_kick
    return lambda qc, qubit, q: qc.p(2 * np.pi * phase_or_kick * 2**q, qubit)


def _last_gate(gates: List[tuple[Gate, List[int]]]) -> dict[int, int]:
    """Find, for each qubit, the last staircase gate that touches it.

    That is the moment the qubit is free: its phase goes in right after, and from then on
    the readout can take it.

    Args:
        gates: a staircase's gates.

    Returns:
        {qubit: index in `gates` of the last gate on it}. Qubits no gate touches are left
        out. For `plus_layer` with n = 3: {2: 0, 1: 1, 0: 2}, each qubit's own H.
    """
    last = {}
    for i, (_, qubits) in enumerate(gates):
        for q in qubits:
            last[q] = i  # a later gate overwrites an earlier one
    return last
