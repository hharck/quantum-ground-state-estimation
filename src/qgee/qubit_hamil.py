import json
import pathlib
import numpy as np
import scipy.sparse.linalg
from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.circuit.quantumcircuit import ParameterValueType
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit.synthesis.evolution import SuzukiTrotter
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.mappers.fermionic_mapper import FermionicMapper
from qiskit_nature.second_q.problems.electronic_structure_problem import (
    ElectronicStructureProblem,
)

def json_to_geometry(data):
    symbols = data["symbols_list"][0]
    coordinates = data["geometry_list"][0]

    return "; ".join(
        f"{symbol} {x} {y} {z}"
        for symbol, (x, y, z) in zip(symbols, coordinates)
    )


def geometry_for_molecule(name: str) -> str:
    """Return geometry for the requested molecule."""
    if name == "H2":
        return "H 0 0 0; H 0 0 0.741"

    if name == "LiH":
        return "Li 0 0 0; H 0 0 1.595"

    if name == "BeH2":
        return "H 0 0 -1.326; Be 0 0 0; H 0 0 1.326"

    if name == "DMABN":
        geometry_path = (
            pathlib.Path(__file__).parent.parent.parent.resolve()
            / "geometries"
            / "dmabn"
            / "optimized_gfn2xtb.json"
        )

        with geometry_path.open("r", encoding="utf-8") as f:
            dmabn_json = json.load(f)

        return json_to_geometry(dmabn_json)

    if name == "ToyModel":
            geometry_path = (
                pathlib.Path(__file__).parent.parent.parent.resolve()
                / "geometries"
                / "toymodel"
                / "optimized_gfn2xtb.json"
            )
    
            with geometry_path.open("r", encoding="utf-8") as f:
                toymodel_json = json.load(f)
    
            return json_to_geometry(toymodel_json)

    raise ValueError(f"Unknown molecule: {name!r}")

class MolecularQubitHamiltonian:
    qubit_hamil: SparsePauliOp
    commuting_terms: list[SparsePauliOp]
    problem: ElectronicStructureProblem
    mapper: FermionicMapper

    def __init__(
        self,
        atom_geometry: str,
        basis: str = "sto3g",
        mapper: FermionicMapper | None = None,
        qubit_wise: bool = False,
        **kwargs,
    ):
        if mapper is None:
            mapper = JordanWignerMapper()

        driver = PySCFDriver(atom=atom_geometry, basis=basis, **kwargs)
        problem = driver.run()

        fermionic_hamiltonian = problem.hamiltonian.second_q_op()
        qubit_hamil = mapper.map(fermionic_hamiltonian)

        assert isinstance(qubit_hamil, SparsePauliOp)
        self.qubit_hamil = qubit_hamil
        self.problem = problem
        self.mapper = mapper
        self.commuting_terms = qubit_hamil.group_commuting(qubit_wise=qubit_wise)

    @property
    def num_qubits(self) -> int:
        assert isinstance(self.qubit_hamil.num_qubits, int)
        return self.qubit_hamil.num_qubits

    def hartree_fock_state(self) -> QuantumCircuit:
        hf_state = HartreeFock(
            num_spatial_orbitals=self.problem.num_spatial_orbitals,
            num_particles=self.problem.num_particles,
            qubit_mapper=self.mapper,
        )
        return hf_state

    def uccsd_state(self) -> QuantumCircuit:
        uccsd_state = UCCSD(
            num_spatial_orbitals=self.problem.num_spatial_orbitals,
            num_particles=self.problem.num_particles,
            qubit_mapper=self.mapper,
            initial_state=self.hartree_fock_state(),
        )
        return uccsd_state

    def time_evolution_gate(
        self,
        time: ParameterValueType = 1,
        reps: int = 1,
    ) -> PauliEvolutionGate:
        gate = PauliEvolutionGate(
            self.commuting_terms, time=time, synthesis=SuzukiTrotter(reps=reps)
        )
        return gate

    def expectation_value(
        self,
        circuit: QuantumCircuit,
    ) -> float:
        estimator = StatevectorEstimator()
        job = estimator.run([(circuit, self.commuting_terms)])
        result = job.result()[0]
        return float(np.sum(result.data["evs"]))

    def exact_ground_state_energy(self) -> float:
        sparse_matrix = self.qubit_hamil.to_matrix(sparse=True)
        eigenvalues, _ = scipy.sparse.linalg.eigsh(sparse_matrix, k=1, which="SA")
        return eigenvalues[0]

    @property
    def spectral_norm_upper_bound(self) -> float:
        return np.abs(self.qubit_hamil.coeffs).sum()


if __name__ == "__main__":
    h2_geometry = geometry_for_molecule("H2")
    hamil = MolecularQubitHamiltonian(
        h2_geometry, basis="sto3g", charge=0, spin=0
    )

    print(f"\nH2 Qubit Hamiltonian has {len(hamil.qubit_hamil)} Pauli terms.")
    print(
        "\nH2 Qubit Hamiltonian has exact ground state energy "
        f"{hamil.exact_ground_state_energy()}"
    )

    state = hamil.hartree_fock_state()
    print(state)

    gate = hamil.time_evolution_gate(time=5)
    state.append(gate, range(hamil.num_qubits))
    ex = hamil.expectation_value(state)
    print(ex)