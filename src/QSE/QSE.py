import numpy as np
from QSE.Krylov_sampling import Krylov_sampler, solve_regularized_gen_eig
from qgee.qubit_hamil import geometry_for_molecule, MolecularQubitHamiltonian
from QSE.Backends import QiskitBackend
from qiskit.circuit import Parameter
from qiskit.primitives import BasePrimitiveJob

class QSE:
    backend: QiskitBackend
    krylov_dim: int
    krylov_sampler: Krylov_sampler
    hamil: MolecularQubitHamiltonian
    dt: float
    job: BasePrimitiveJob | None = None
    _exact_ge: float | None = None

    def __init__(self, backend: QiskitBackend, molecule: str, num_trotter_steps: int = 2, krylov_dim: int = 2):
        assert isinstance(backend, QiskitBackend)
        assert isinstance(molecule, str)
        assert isinstance(num_trotter_steps, int) and num_trotter_steps >= 1

        self.backend = backend
        self.krylov_dim = krylov_dim

        geometry = geometry_for_molecule(molecule)
        self.hamil = MolecularQubitHamiltonian(geometry, basis="sto3g")
        self.dt = np.pi / self.hamil.spectral_norm_upper_bound 
        #dt_trot = self.dt / num_trotter_steps

        parameter = Parameter('t')
        self.krylov_sampler = Krylov_sampler(
            H=self.hamil.qubit_hamil.group_commuting(), 
            time_evolution_circuit=self.hamil.time_evolution_gate(time=parameter, reps=num_trotter_steps), 
            time_parameter=parameter, 
            dt=self.dt, 
            state_prep=self.hamil.hartree_fock_state(), 
            use_barriers=False
        )

        self.krylov_sampler.optimize_circuit(self.backend.preset_manager)

    def prepare_pub(self):
        pub = self.krylov_sampler.make_pub(dim=self.krylov_dim)
        return pub

    @property
    def exact_ge(self) -> float:
        if self._exact_ge is not None: return self._exact_ge
        self._exact_ge = self.hamil.exact_ground_state_energy()
        return self._exact_ge 

    def run(self, precision: float | None = None, print_curcuit_description: bool = False):
        pub = self.prepare_pub()

        if print_curcuit_description:
            circ = pub[0]
            print(
                f"""The optimized circuit
                #qubits,\tdepth,\t2Q gate depth,\tgate counts
                {circ.num_qubits},\t{circ.depth(lambda x: True)},\t{circ.depth(lambda x: x[0].num_qubits == 2)},\t{circ.count_ops()}
                """)
        self.job = self.backend.run_estimator(pubs=[pub], precision=precision)

    def energy_estimates(self, threshold: float = 0, print_eigs=False):
        assert self.job is not None, "Please use QSE.run() before requesting a result"
        print("\tAwating result")
        results = self.job.result()
        print("\tGot result")
        s = self.krylov_sampler.assemble_S(evs=results[0].data.evs, krylov_dim=self.krylov_dim)
        print("\tAssembled S")
        h = self.krylov_sampler.assemble_H(evs=results[0].data.evs, krylov_dim=self.krylov_dim)
        print("\tAssembled H")
        energies = [
            solve_regularized_gen_eig(h=h[:d, :d], s=s[:d, :d], threshold=threshold, print_eigs=print_eigs) 
            for d in range(1, self.krylov_dim + 1)
            ]
        print("\tGot eigenvalues")
        return energies
