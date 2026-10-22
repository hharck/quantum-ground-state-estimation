from QSE.Backends import QiskitBackend
from QSE.QSE import QSE
import argparse
import numpy as np
from pathlib import Path


def make_path(molecule, krylov_dim, precision, trotter_stepts, eigen_threshold, backend_name):
    all_params = [molecule, f"K={krylov_dim}", f"P={precision}", f"T={trotter_stepts}", f"e={eigen_threshold}", backend_name]
    return "/".join(all_params)

if __name__ == "__main__":
    print("Running...")

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--PrintEnergies", help="Print energies", action="store_true")
    parser.add_argument("-n", "--NoNoise", help="Make a noise free simulation", action="store_true")
    parser.add_argument("-d", "--device", type=str, default=None, help="Device")
    parser.add_argument("-P", "--precision", type=float, default=0.0,  help="Precision to use in Qiskit Estimator")
    parser.add_argument("-m", "--molecule", default="H2", help="Molecule to simulate")
    parser.add_argument("-T", "--trotterSteps", type=int, default=1, help="Number of trotter steps.")
    parser.add_argument("-K", "--maxKrylovDim", type=int, default=20, help="Highest Krylov dimension to compute")
    parser.add_argument("-N", "--numCores", type=int, default=1, help="Number of cpu cores to use")
    parser.add_argument("-e", "--eigenThreshold", type=float, default=0.0, help="Threshold on accepted eigenvalues")

    args = parser.parse_args()

    precision = float(args.precision)
    assert 1 >= precision >= 0

    molecule = args.molecule
    assert isinstance(molecule, str)

    trotterSteps = int(args.trotterSteps)
    assert trotterSteps >= 0

    krylov_dim = int(args.maxKrylovDim)
    assert krylov_dim > 0

    num_cores = int(args.numCores)
    assert num_cores > 0
    print(f"Running on {num_cores} cores")

    eigen_threshold = args.eigenThreshold
    assert eigen_threshold >= 0

    device = args.device
    no_noise = args.NoNoise

    assert (not no_noise) or (device is None), "No noise and physical device selection are mutually exclusive"

    backend = QiskitBackend(physical_device=device, simulate=True, no_noise_override=no_noise, num_threads=num_cores)

    # num_cores is excluded here as it should not effect the outcome. Noisy and device are merged when using backend.name
    param_string = make_path(molecule=molecule, krylov_dim=krylov_dim, precision=precision, trotter_stepts=trotterSteps, eigen_threshold=eigen_threshold, backend_name=backend.name)
    print(param_string)

    qse = QSE(backend=backend, molecule=molecule, num_trotter_steps=trotterSteps, krylov_dim=krylov_dim)
    print("Run...")
    qse.run(precision=precision, print_curcuit_description=True)

    print("Energies...")
    energies = qse.energy_estimates(threshold=eigen_threshold, print_eigs=True)
    print("Save...")
    
    save_path = Path(f"QSE_Data/{param_string}/data")
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(save_path, energies)

    if args.PrintEnergies:
        for j in range(krylov_dim):
            print(f"\tAt krylov dim {j} estimate is {energies[j]}")
  
    print("Done.")