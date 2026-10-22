from typing import Literal

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import SparsePauliOp, Pauli, PauliList
from qiskit.quantum_info.states import Statevector, StabilizerState
from qiskit.circuit import Parameter, Gate
from dataclasses import dataclass, field
from qiskit.transpiler.passmanager import StagedPassManager
from qiskit.transpiler.layout import TranspileLayout
from qiskit.transpiler.passmanager import StagedPassManager



@dataclass(order=True)
class Krylov_sampler:
    H: list[SparsePauliOp]
    time_evolution_circuit: QuantumCircuit | Gate
    time_parameter: Parameter
    dt: float
    state_prep: QuantumCircuit
    use_barriers: bool
    _controlled_state_prep: QuantumCircuit | None = None
    _sampling_circuit: QuantumCircuit | None = None
    _h_at_t0: complex | None = None
    translated_circuit: QuantumCircuit | None = None
    H_flat: SparsePauliOp = None

    def __post_init__(self):
        assert isinstance(self.time_evolution_circuit, QuantumCircuit) or isinstance(self.time_evolution_circuit, Gate)
        assert isinstance(self.state_prep, QuantumCircuit)
        assert isinstance(self.H, list)
        assert len(self.H) > 0
        assert isinstance(self.H[0], SparsePauliOp)
        assert isinstance(self.H[0].num_qubits, int)
        assert self.time_evolution_circuit.num_qubits == self.H[0].num_qubits == self.state_prep.num_qubits
        self.H_flat = sum(h for h in self.H)
        assert isinstance(self.time_parameter, Parameter)

    def __setattr__(self, name, value):
        """
        Prevents mutation of assumed static properties
        """
        immutables = ["H", "time_evolution_circuit", "state_prep", "time_parameter", "dt", "use_barriers"]

        for im in immutables:
            if name == im and hasattr(self, im):
                raise AttributeError(f"{im} is immutable")
        super().__setattr__(name, value)

    @property
    def num_qubits(self) -> int:
        return self.time_evolution_circuit.num_qubits

    @property
    def controlled_state_prep(self) -> QuantumCircuit:
        if self._controlled_state_prep is not None: return self._controlled_state_prep
        self._controlled_state_prep = self.state_prep.control()
        return self._controlled_state_prep
    
    @property
    def sampling_circuit(self) -> QuantumCircuit:
        if self._sampling_circuit is not None: return self._sampling_circuit
    
        control_reg = QuantumRegister(1, name="control")
        data_reg = QuantumRegister(self.num_qubits, name="data")
        all_reg = list(control_reg)+list(data_reg)
        qc = QuantumCircuit(control_reg, data_reg)
        qc.h(control_reg)
        qc.compose(self.controlled_state_prep, all_reg, inplace=True)
        if self.use_barriers: qc.barrier()
        qc.append(self.time_evolution_circuit, qargs=data_reg)
        if self.use_barriers: qc.barrier()
        qc.x(control_reg)
        qc.compose(self.controlled_state_prep, all_reg, inplace=True)
        qc.x(control_reg)
        self._sampling_circuit = qc
        return self._sampling_circuit

    @property
    def S_at_t_0(self) -> complex:
        return complex(real=1.0)

    def H_at_t_0(self) -> complex:
        if self._h_at_t0 is not None: return self._h_at_t0

        qc_cliff = self.sampling_circuit.assign_parameters({self.time_parameter: 0})
        H_expval = 0

        for (pauli, coeff) in zip(self.H_flat.paulis, self.H_flat.coeffs):
            
            # Get expectation values from experiment
            label = pauli[::1].to_label()

            expval_real = StabilizerState(qc_cliff).expectation_value(Pauli(label + "X"))
            expval_imag = StabilizerState(qc_cliff).expectation_value(Pauli(label + "Y"))
            expval = expval_real + 1j * expval_imag

            H_expval += coeff * expval
        self._h_at_t0 = H_expval
        return self._h_at_t0

    def optimize_circuit(self, pm: StagedPassManager):
        assert self.translated_circuit == None, "Circuit can only be optimized once"
        self.translated_circuit = pm.run(self.sampling_circuit)

    def observables(self):
        assert self.translated_circuit is not None, "Run optimize_circuit before requsting observables"
        layout = self.translated_circuit.layout

        observables_S = [
            [SparsePauliOp("I" * (self.num_qubits) + "X").apply_layout(layout)],
            [SparsePauliOp("I" * (self.num_qubits) + "Y").apply_layout(layout)],
        ]

        observables_H = []
        for h in self.H:
            list_X: list[Pauli] = []
            list_Y: list[Pauli] = []

            for pauli in h.paulis:
                label = pauli[::1].to_label()
                list_X.append(Pauli(label + "X"))
                list_Y.append(Pauli(label + "Y"))
                
            observables_H.append([SparsePauliOp(list_X, coeffs=h.coeffs).apply_layout(layout)])
            observables_H.append([SparsePauliOp(list_Y, coeffs=h.coeffs).apply_layout(layout)])

        return observables_S + observables_H

    def make_pub(self, dim, shots=None):
        assert self.translated_circuit != None, "Run optimize_circuit before requsting pub"
        parameters = list(range(1, dim)) # Values of t, t=0 is handled classically, t<0 is handled by the hermitian structure of our matrices, so we only need to handle t \in {1, ..., N-1}

        params = np.vstack(parameters) * self.dt
        if shots:
            pub = (self.translated_circuit, self.observables(), params, shots)
        else:
            pub = (self.translated_circuit, self.observables(), params)
        return pub

    def E0(self, krylov_dim: int):
        """
        Computes the eigenvalues of e^{-i H a dt}|0> for a = 1,..., dim-1
        """
        assert isinstance(krylov_dim, int)
        # Eigenvalue for H\ket{0}
        beta = sum(c for p, c in self.H_flat.to_list() if ("X" not in p and "Y" not in p))
        pref = np.exp((-1j * beta * self.dt) * np.arange(1, krylov_dim, dtype=float))
        return pref

    # Based on IBM tutorial
    def assemble_S(self, evs, krylov_dim: int):
        assert len(evs.shape) == 2 and evs.shape[0] == len(self.H) * 2 + 2 and evs.shape[1] == krylov_dim - 1
        S_first_row  = np.zeros(krylov_dim, dtype=complex)
        S_first_row[0] = self.S_at_t_0
        prefactors = self.E0(krylov_dim)
        for i in range(krylov_dim - 1):
            # Get expectation values from experiment
            expval_real = evs[0][i]  # automatic extrapolated evs if ZNE is used
            expval_imag = evs[1][i]  # automatic extrapolated evs if ZNE is used

            # Get expectation values
            expval = expval_real + 1j * expval_imag
            S_first_row[i + 1] += prefactors[i] * expval

        S_circ = np.zeros((krylov_dim, krylov_dim), dtype=complex)
        for i in range(krylov_dim):
            for j in range(krylov_dim):
                if i >= j:
                    S_circ[j, i] = S_first_row[i - j]
                else:
                    S_circ[j, i] = np.conj(S_first_row[j - i])
        return S_circ

    # Based on IBM tutorial
    def assemble_H(self, evs, krylov_dim: int):
        assert len(evs.shape) == 2 and evs.shape[0] == len(self.H) * 2 + 2 and evs.shape[1] == krylov_dim - 1
        H_first_row  = np.zeros(krylov_dim, dtype=complex)
        H_first_row[0] = self.H_at_t_0()
        prefactors = self.E0(krylov_dim)

        for obs_idx, h in enumerate(self.H):
            # Add in ancilla-only measurements:
            for i in range(krylov_dim - 1):
                # Get expectation values from experiment
                expval_real = evs[2 + 2 * obs_idx][i]
                expval_imag = evs[2 + 2 * obs_idx + 1][i]

                # Get expectation values
                expval = expval_real + 1j * expval_imag
                H_first_row[i + 1] += prefactors[i] * expval

        H_eff_circ = np.zeros((krylov_dim, krylov_dim), dtype=complex)

        # Distribute entries from first row across matrix:
        for i in range(krylov_dim):
            for j in range(krylov_dim):
                if i >= j:
                    H_eff_circ[j, i] = H_first_row[i - j]
                else:
                    H_eff_circ[j, i] = np.conj(H_first_row[j - i])
        return H_eff_circ


import scipy as sp
# Based on similar function in IBM tutorial
def solve_regularized_gen_eig(h: np.ndarray, s: np.ndarray, threshold: float, print_eigs: bool = False) -> float:
    """
    Method for solving the generalized eigenvalue problem with regularization

    Args:
        h (numpy.ndarray):
            The effective representation of the matrix in our Krylov subspace
        s (numpy.ndarray):
            The matrix of overlaps between vectors of our Krylov subspace
        threshold (float):
            Cut-off value for the eigenvalue of s
        print_eigs (bool):
            Whether to print the eigenvalues of s

    Returns:
        lowest eigenvalue that is the solution to the regularized generalized eigenvalue problem
    """
    s_vals, s_vecs = sp.linalg.eigh(s)
    s_vecs = s_vecs.T
    good_vecs = np.array([vec for val, vec in zip(s_vals, s_vecs) if val > threshold])
    if print_eigs:
        print(
            s.shape[0],
            "minimum:", s_vals[0],
            "maximum:", s_vals[-1],
        )

    h_reg = good_vecs.conj() @ h @ good_vecs.T
    s_reg = good_vecs.conj() @ s @ good_vecs.T
    
    return sp.linalg.eigh(h_reg, s_reg, eigvals_only=True)[0]
    