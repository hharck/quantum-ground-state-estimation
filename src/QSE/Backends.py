from qiskit.transpiler.passmanager import StagedPassManager
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2 as IBMEstimator, Batch
from qiskit_aer import AerSimulator
from qiskit.providers import BackendV2
from qiskit.primitives import BasePrimitiveJob
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_aer.primitives import EstimatorV2 as AEREstimator
from qiskit_ibm_runtime.runtime_job_v2 import RuntimeJobV2


class QiskitBackend:
    backend: BackendV2
    preset_manager: StagedPassManager
    name: str

    def __init__(self, physical_device: str | None = None, simulate: bool = True, no_noise_override = False, num_threads: int = 1):
        if no_noise_override:
            self.backend = AerSimulator(max_parallel_experiments=0, max_parallel_threads=num_threads, max_parallel_shots=0, mps_omp_threads=num_threads)        
            self.name = "ideal"
        else:
            service = QiskitRuntimeService(channel="ibm_quantum_platform")
            if physical_device:
                device = service.backend(physical_device)
                self.name = physical_device
            else:
                device = service.least_busy(operational=True, simulator=False)
                self.name = device.name
            if simulate:
                self.backend = AerSimulator.from_backend(device, max_parallel_experiments=0, max_parallel_threads=num_threads, max_parallel_shots=0, mps_omp_threads=num_threads)
                self.name = "simulated_" + self.name
            else:
                self.backend = device
            
        self.preset_manager = generate_preset_pass_manager(backend=self.backend, optimization_level=3, seed_transpiler=1)

    @property
    def options(self):
        num_randomizations = 300
        num_randomizations_learning = 20
        max_batch_circuits = 20
        shots_per_randomization = 100
        learning_pair_depths = [0, 4, 24]
        noise_factors = [1, 1.3, 1.6]

        # Base option formatting
        return {
            # Builtin resilience settings for ZNE
            "resilience": {
                "measure_mitigation": True,
                "zne_mitigation": True,
                "zne": {"noise_factors": noise_factors},
                # TREX noise learning configuration
                "measure_noise_learning": {
                    "num_randomizations": num_randomizations_learning,
                    "shots_per_randomization": shots_per_randomization,
                },
                # PEA noise model configuration
                "layer_noise_learning": {
                    "max_layers_to_learn": 10,
                    "layer_pair_depths": learning_pair_depths,
                    "shots_per_randomization": shots_per_randomization,
                    "num_randomizations": num_randomizations_learning,
                },
            },
            # Randomization configuration
            "twirling": {
                "num_randomizations": num_randomizations,
                "shots_per_randomization": shots_per_randomization,
                "strategy": "all",
            },
            # Experimental settings for PEA method
            "experimental": {
                # # Just in case, disable any further qiskit transpilation not related to twirling / DD
                # "skip_transpilation": True,
                # Execution configuration
                "execution": {
                    "max_pubs_per_batch_job": max_batch_circuits,
                    "fast_parametric_update": True,
                },
                # Error Mitigation configuration
                "resilience": {
                    # ZNE Configuration
                    "zne": {
                        "amplifier": "pea",
                        "return_all_extrapolated": True,
                        "return_unextrapolated": True,
                        "extrapolated_noise_factors": [0] + noise_factors,
                    }
                },
            },
        }
    
    def run_estimator(self, pubs: list, precision: float | None) -> BasePrimitiveJob:
        if isinstance(self.backend, AerSimulator):
            estimator = AEREstimator.from_backend(self.backend)
            return estimator.run(pubs=pubs, precision=precision)
        else:
            with Batch(backend=self.backend) as batch:
                # Estimator
                estimator = IBMEstimator(mode=batch, options=self.options)
                job: RuntimeJobV2  = estimator.run(pubs, precision=precision)
            return job