"""Code taken from the paper repository, with additional function to convert to qiskit gates."""

import time
from typing import List, Sequence

import numpy as np
from numpy.typing import ArrayLike
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from scipy.signal.windows import dpss

# Standardised half-bandwidths (SciPy's `dpss` NW parameter) that place the DPSS
# at a given error probability delta, in the large-dimension limit.
NW_BY_DELTA: dict[float, float] = {
    1e-2: 1.1155843359412394,  # 99% confidence level
    1e-3: 1.5167560028538316,  # 99.9%
    1e-4: 1.9056832471873077,  # 99.99%
}


def half_width(nw: float, dim: int) -> float:
    """Confidence-interval half-width d for a DPSS built with SciPy parameter nw.

    Args:
        nw: SciPy's `dpss` half-bandwidth parameter, e.g. an entry of NW_BY_DELTA.
        dim: dimension the DPSS was built on, i.e. 2**n for n qubits.

    Returns:
        The half-width d in radians of phase, so the interval is phi +/- d / (2 pi) of a
        full turn. Since d scales as 1/dim, it is always about nw readout bins wide.
    """
    return nw * 2 * np.pi / dim


def half_bandwidth(d: float, dim: int) -> float:
    """Inverse of half_width: the SciPy dpss parameter giving half-width d.

    Args:
        d: wanted half-width, in radians of phase.
        dim: dimension the DPSS will be built on.

    Returns:
        The NW parameter to hand to `scipy.signal.windows.dpss`.
    """
    return d * dim / (2 * np.pi)


def state_to_mps_circuit(psi: ArrayLike, bond_dim: int) -> list[np.ndarray]:
    """Converts a state vector into the list of isometries preparing its MPS.

    The list is in circuit order, so it runs backwards relative to the paper's
    labels: the last element is M[1] (applied first, to the most significant
    qubits). Returns n - log2(bond_dim) operators, not n, because the tail
    operators are merged as in Section 4.

    Args:
        psi: state vector of length 2**n.
        bond_dim: bond dimension chi. Larger keeps more of the state, on wider operators:
            each one spans 1 + log2(chi) qubits.

    Returns:
        The isometries, each a square matrix of side 2 * chi or smaller. They are not
        unitary yet; `mps._mps_gates` completes and places them.
    """
    n = int(np.log2(len(psi)))
    vs = []  # stores mps site tensors
    bs = []  # stores isometries
    bond = 1
    # Sweep right to left, peeling off one site per truncated SVD.
    for i in range(n - 1):
        psi = np.reshape(psi, (-1, 2 * bond))
        u, s, vd = _split(psi, bond_dim)
        vs.append(vd.swapaxes(0, -1))  # v = v^dagger^dagger
        bs.append(_mps_site_to_isometry(vs[-1]))
        psi = u * s[None, None, :]
        bond = psi.shape[-1]
    # handle last site
    psi = np.reshape(psi, (1, -1))
    _, _, vd = np.linalg.svd(psi, full_matrices=False)
    vs.append(vd.reshape(-1, 2, 1))
    bs.append(_mps_site_to_isometry(vd.reshape(1, 2, -1).swapaxes(0, -1)))
    # Match qubit endian ordering. Each site is independent of the others.
    bs = [_move_physical_index(b, v) for b, v in zip(bs, vs)]
    # Join the last three operators at the end of the circuit, as in Section 4.
    for _ in range(min(int(np.log2(bond_dim)), n // 2)):
        bs[1] = np.kron(np.eye(2), bs[0]) @ bs[1]
        bs = bs[1:]
    return bs


def mps_circuit_to_state(us: list[np.ndarray], n: int) -> np.ndarray:
    """Reconstructs the state vector from the list of isometries. Eq. (52).

    This is the reference a circuit built from the same isometries must reproduce.

    Args:
        us: the isometries from `state_to_mps_circuit`, in its order.
        n: number of qubits, so the state has 2**n entries.

    Returns:
        The state vector the isometries prepare from |0...0>, indexed most significant bit
        first.
    """
    psi = np.zeros(2**n)
    psi[0] = 1
    for i, u in enumerate(reversed(us)):
        # Operator i acts on the 1 + log2(chi) qubits starting at position i.
        psi = u @ psi.reshape(2**i, len(u), -1)
    return psi.flatten()


def prob(psi: np.ndarray, d: float) -> float:
    """Confidence level of Eq. (22): the probability the estimate lands within d of the phase.

    Evaluated as a convolution with the first column of the Toeplitz matrix S(d) (Eq. 23),
    so it costs O(D log D) via FFT rather than the O(D^2) of the definition.

    Args:
        psi: the control state; it is normalised first, so it need not arrive normalised.
        d: half-width in radians of phase, e.g. from `half_width`.

    Returns:
        The confidence level, between 0 and 1. For a DPSS built with the matching entry of
        NW_BY_DELTA this comes out at 1 - delta.
    """
    dim = len(psi)
    psi = psi / np.linalg.norm(psi)
    psi_padded = np.pad(psi, (0, dim - 1))
    t = np.sinc(np.arange(-dim + 1, dim) * d / np.pi)
    conv = np.fft.ifft(np.fft.fft(psi_padded) * np.fft.fft(t))
    p = np.dot(psi, conv[dim - 1 :]).real
    p *= d / np.pi
    return p


def prob_direct(psi: np.ndarray, d: float) -> float:
    """The O(D**2) double sum of Eq. (22), written out literally.

    Same output as `prob`, but much slower for psi with large dimension. It is the
    reference `prob` is checked against.

    Args:
        psi: the control state; normalised first, as in `prob`.
        d: half-width in radians of phase.

    Returns:
        The confidence level, between 0 and 1.
    """
    dim = len(psi)
    psi = psi / np.linalg.norm(psi)
    p = 0
    for x in range(dim):
        for y in range(dim):
            p += psi[x] * psi[y] * np.sinc(d * (x - y) / np.pi)
    p *= d / np.pi
    return p


def calculate_infidelities(
    n: int, chi: int, print_results: bool = False
) -> tuple[list[float], float, float]:
    """Infidelity 1 - F of the chi-bond MPS approximation on 2**n dimensions.

    Returns (one infidelity per entry of NW_BY_DELTA, mean DPSS build time,
    mean MPS build time). Feeds Table 1 and Appendix B.

    Args:
        n: qubits, so the state has 2**n entries.
        chi: bond dimension of the approximation.
        print_results: also print, per confidence level, the target against the confidence
            the state actually reaches, its infidelity and the two build times.

    Returns:
        (infidelities, average_dpss_time, average_mps_time). The infidelities are in the
        order of NW_BY_DELTA, and the times are in seconds, averaged over its entries. At
        chi = 8 and above the infidelities are at machine precision and can come out
        slightly negative.
    """
    infidelities = []
    average_dpss_time = 0
    average_mps_time = 0
    for delta, nw in NW_BY_DELTA.items():
        t1 = time.time()
        psi = dpss(
            2**n, nw
        )  # Generate the DPSS state vector for the given half-bandwidth parameter.
        psi /= np.linalg.norm(psi)  # Normalize the state vector.
        t2 = time.time()
        circ = state_to_mps_circuit(
            psi, chi
        )  # Convert state vector to MPS with the specified bond dimension.
        t3 = time.time()
        psi_mps = mps_circuit_to_state(
            circ, n
        )  # Reconstruct the state vector from the MPS representation.
        infidelity = 1 - float(np.dot(psi, psi_mps) ** 2)
        infidelities.append(infidelity)
        dpss_time = t2 - t1
        mps_time = t3 - t2
        average_dpss_time += dpss_time / len(NW_BY_DELTA)
        average_mps_time += mps_time / len(NW_BY_DELTA)
        if print_results:
            ## Check to confirm that the confidence level is close to the target.
            d = half_width(nw, 2**n)
            print(
                f"target confidence level={1 - delta}, true confidence level={prob(psi, d)}, "
                f"infidelity={infidelity:.2e}",
                {d},
            )
            print(
                f"DPSS generation time: {dpss_time} seconds, MPS calculation time: {mps_time} seconds"
            )
    return infidelities, average_dpss_time, average_mps_time


def unitary_completion(m: np.ndarray) -> np.ndarray:
    """Turn an isometry into a unitary by filling its don't-care columns.

    Those columns come out zero, which is why the isometry is not a gate yet. The polar
    factor keeps the columns that carry the state and completes the rest orthonormally.

    Args:
        m: square matrix whose non-zero columns are orthonormal, as produced by
            `_mps_site_to_isometry`.

    Returns:
        A unitary of the same shape, equal to `m` on its non-zero columns.
    """
    w, _, vh = np.linalg.svd(m)
    return w @ vh


def mps_state_prep(
    psi: ArrayLike,
    bond_dim: int,
    qubits: Sequence[int] | None = None,
    name: str = "MPS",
) -> QuantumCircuit:
    """Build a circuit preparing an MPS approximation of a state from |0...0>.

    Args:
        psi: state vector of length 2**n. It is approximated, not prepared exactly,
            unless `bond_dim` is large enough for it.
        bond_dim: MPS bond dimension chi. Larger means a better approximation on wider
            gates: the circuit has n - log2(chi) of them, each on 1 + log2(chi) qubits.
        qubits: which qubits to prepare the state on, most significant first. The default
            uses every qubit in Qiskit's little-endian order, so the circuit's statevector
            comes out in the same order as `psi`.
        name: name given to the circuit, which is what the drawing shows.

    Returns:
        A circuit wide enough to contain `qubits`, holding one gate per isometry.

    Raises:
        ValueError: if `qubits` does not have one entry per qubit of `psi`.
    """
    psi = np.asarray(psi)
    n = int(np.log2(len(psi)))
    qubits = list(reversed(range(n))) if qubits is None else list(qubits)
    if len(qubits) != n:
        raise ValueError(f"psi needs {n} qubits, got {len(qubits)}")
    qc = QuantumCircuit(max(qubits) + 1, name=name)
    for gate, block in _mps_gates(state_to_mps_circuit(psi, bond_dim), qubits):
        qc.append(gate, block)
    return qc


def dpss_gates(
    n: int, delta: float = 1e-3, bond_dim: int = 4
) -> List[tuple[UnitaryGate, List[int]]]:
    """Staircase preparing the DPSS control state, in place of the usual |+> layer.

    This is a staircase in the sense of readout.py: a callable n -> [(gate, qubits)]. Bind
    `delta` and `bond_dim` with `functools.partial` and hand the result to the readouts
    there, or to the builders in qpe.py, to turn a readout into a confidence interval.

    Args:
        n: counting qubits the state spans.
        delta: error probability, so the confidence is 1 - delta.
        bond_dim: MPS bond dimension. 4 reproduces the window to machine precision; 2 is
            cheaper but leaves an infidelity around 1e-3, too coarse for delta = 1e-4.

    Returns:
        (gate, qubits it goes on) pairs in the order they are applied, on qubits n-1 down
        to 0. Each gate brings in fresh qubits below the ones it shares with the last, and
        qubit n-1 - i is finished after gate i, so the compact readout can recycle them on as
        few qubits as the widest gate, 1 + log2(bond_dim).

    Raises:
        KeyError: if `delta` is not one of the tabulated confidence levels.
    """
    psi = _dpss_state(n, delta)
    return _mps_gates(state_to_mps_circuit(psi, bond_dim), list(reversed(range(n))))


def dpss_control(n: int, delta: float = 1e-3, bond_dim: int = 4) -> QuantumCircuit:
    """The same DPSS preparation as `dpss_gates`, as a standalone circuit to draw or run.

    Args:
        n: counting qubits the state spans.
        delta: error probability, so the confidence is 1 - delta.
        bond_dim: MPS bond dimension.

    Returns:
        An n-qubit circuit preparing the state from |0...0>.
    """
    return mps_state_prep(_dpss_state(n, delta), bond_dim, name=f"DPSS {delta:g}")


def _mps_site_to_isometry(m: np.ndarray) -> np.ndarray:
    """Converts an MPS site tensor to an isometry for the circuit.

    Columns beyond the bond dimension are zero: they are the don't-care entries written as
    * in Eqs. (49)-(51). `mps.unitary_completion` is what fills them in later.

    Args:
        m: site tensor of shape (virtual in, 2, virtual out).

    Returns:
        A square matrix of side 2 * max(virt_in, virt_out // 2), holding the tensor in its
        top-left corner and zeros elsewhere.
    """
    virt_in, _, virt_out = m.shape
    m_bar = max(virt_in, virt_out // 2)
    padded = np.zeros((m_bar, 2, 2 * m_bar))
    padded[:virt_in, :2, :virt_out] = m
    # return padded square matrix
    u = padded.reshape(2 * m_bar, 2 * m_bar)
    return u


def _split(
    reshaped_psi: np.ndarray, bond_dim: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Performs an SVD on the reshaped state to retrieve MPS tensors. Eq. (17).

    Args:
        reshaped_psi: the state, reshaped so the rightmost site and its bond form the
            second axis.
        bond_dim: bond dimension chi to keep; the smaller singular values are dropped,
            which is where the approximation enters.

    Returns:
        (u, s, vd) truncated to chi bonds: `u` is the rest of the state with its physical
        index _split out, `s` its singular values, and `vd` the site tensor that was peeled
        off.
    """
    u, s, vd = np.linalg.svd(reshaped_psi, full_matrices=False)
    bonds = len(s)
    vd = vd.reshape(bonds, 2, -1)
    u = u.reshape((-1, 2, bonds))
    chi = np.min([bonds, bond_dim])  # keep only chi bonds
    u, s, vd = u[:, :, :chi], s[:chi], vd[:chi]
    return u, s, vd


def _move_physical_index(b: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Swaps s_k with alpha_k so the physical index lands on the least significant qubit.

    That turns V into M; see Fig. 2 in the paper. It is a permutation of the matrix's rows
    and columns, so it preserves orthonormality.

    Args:
        b: the isometry from `_mps_site_to_isometry`.
        v: the site tensor it was built from, which supplies the bond dimensions.

    Returns:
        The same isometry with its indices reordered, in `b`'s shape.
    """
    right_bond, left_bond = v.shape[0], v.shape[-1]
    temp = np.reshape(b, (right_bond, 2, -1, left_bond))
    temp = temp.swapaxes(0, 1)  # correct rows
    if left_bond != 1:
        temp = temp.swapaxes(2, 3)  # correct columns (except for last site)
    return np.reshape(temp, b.shape)


def _mps_gates(
    isometries: List[np.ndarray], qubits: Sequence[int]
) -> List[tuple[UnitaryGate, List[int]]]:
    """Turn MPS isometries into gates and work out which qubits each one acts on.

    Isometry i, counting from the end of the list as in `mps_circuit_to_state`, acts on
    the 1 + log2(chi) qubits starting at position i, so the gates walk down the register
    one qubit at a time.

    Args:
        isometries: the list from `state_to_mps_circuit`, in its own order.
        qubits: the qubits holding the state, most significant first. There must be one
            per qubit of the state, and they can be any qubits of a larger register.

    Returns:
        A list of (gate, qubits) pairs in the order they are applied. Each gate is a
        `UnitaryGate` labelled MPS0, MPS1, ..., and its qubit list is ordered least
        significant first, the way Qiskit's `append` expects.
    """
    placed = []
    for i, m in enumerate(reversed(isometries)):
        width = int(np.log2(len(m)))
        block = list(qubits[i : i + width])
        gate = UnitaryGate(unitary_completion(m), label=f"MPS{i}")
        placed.append(
            (gate, block[::-1])
        )  # Qiskit takes qubits least significant first
    return placed


def _dpss_state(n: int, delta: float = 1e-3) -> np.ndarray:
    """Build the DPSS window for a given confidence level, normalised.

    Args:
        n: counting qubits, so the window has 2**n entries.
        delta: error probability, so the confidence is 1 - delta. Must be a key of
            `NW_BY_DELTA`.

    Returns:
        The state vector, normalised to 1.

    Raises:
        KeyError: if `delta` is not one of the tabulated confidence levels.
    """
    psi = dpss(2**n, NW_BY_DELTA[delta])
    return psi / np.linalg.norm(psi)
