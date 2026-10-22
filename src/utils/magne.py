from qdk import TargetProfile
from qdk._types import QirInputData
from qdk.qiskit import QSharpBackend, ResourceEstimatorBackend
from qdk.estimator import EstimatorResult
from qdk.simulation import NeutralAtomDevice, NoiseConfig
from qdk._device._atom._scheduler import Schedule
from qdk._device._atom._trace import Trace
from qsharp_widgets import Atoms
from pyqir import Context, Module
from IPython.display import display
import numpy as np
from qiskit import QuantumCircuit, transpile

NEUTRAL_ATOM_BASIS = ["rz", "sx", "cz", "measure"]


def magne_noise(
    one_qubit: float = 1e-3,
    two_qubit: float = 4e-3,
    spam: float = 2e-3,
    move: float = 1e-3,
    rz: float = 1e-3,
) -> NoiseConfig:
    """Noise at Magne's published physical-level specs, each fidelity bound taken as the error rate.

    Defaults: 1Q > 99.9%, 2Q > 99.6%, SPAM > 99.8%, movement > 99.9%, so an upper bound on
    the noise.

    Args:
        one_qubit: depolarizing error of each sx gate.
        two_qubit: depolarizing error of each cz gate.
        spam: bit-flip error of each measurement.
        move: depolarizing error of each atom move.
        rz: depolarizing error of each rz gate. rz may be virtual (error-free) on the
            hardware; pass 0 for that case.

    Returns:
        A `NoiseConfig` for `NeutralAtomDevice.simulate`.
    """
    noise = NoiseConfig()
    noise.sx.set_depolarizing(one_qubit)
    noise.rz.set_depolarizing(rz)
    noise.cz.set_depolarizing(two_qubit)
    noise.mresetz.set_bitflip(spam)
    noise.mov.set_depolarizing(move)
    return noise


def run_neutral_atom(
    qc: QuantumCircuit,
    shots: int = 2000,
    noise: NoiseConfig | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Run a circuit on the QDK's neutral-atom device model.

    Args:
        qc: circuit to run; compiled by `adaptive_qir` first.
        shots: how many runs.
        noise: a `NoiseConfig`, e.g. from `magne_noise`, or None for a noiseless run.
        seed: simulator seed, or None for a fresh one.

    Returns:
        An array of length 2**qc.num_clbits of outcome frequencies, indexed by the
        measured bit string read as an integer.
    """
    qir = _adaptive_qir(qc)
    p = np.zeros(2**qc.num_clbits)
    for shot in NeutralAtomDevice().simulate(
        qir, shots=shots, seed=seed, type="cpu", noise=noise
    ):
        parts = (
            shot if isinstance(shot, tuple) else (shot,)
        )  # one list per register, most significant bit first
        p[
            int(
                "".join(
                    "1" if str(r) == "One" else "0" for part in parts for r in part
                ),
                2,
            )
        ] += 1
    return p / shots


def show_atoms(program: QuantumCircuit | str | QirInputData) -> None:
    """Show how the device model lays out, moves and measures the atoms (needs qsharp-widgets).

    Takes a Qiskit circuit or compiled Q#. NeutralAtomDevice.show_trace refuses branching, so
    this runs its steps without that check. The scheduler places each branch block in turn, so
    the trace follows the real order, one measurement per digit, and draws every correction as
    applied. Uses QDK internals, checked against qdk 1.32.3.

    Args:
        program: a Qiskit circuit (compiled by `adaptive_qir`, with a reset appended to
            every qubit so the trace draws them), or already compiled Q#/QIR.

    Returns:
        None; displays the `Atoms` widget in the notebook.
    """
    if isinstance(program, QuantumCircuit):
        qc = program.copy()
        qc.reset(
            range(qc.num_qubits)
        )  # the trace only draws measure+reset, so end every qubit with a reset
        program = _adaptive_qir(qc)
    device = NeutralAtomDevice()
    module = Module.from_ir(Context(), str(device.compile(program)))
    Schedule(device).run(module)
    trace = Trace(device)
    trace.run(module)
    display(Atoms(machine_layout=device.get_layout(), trace_data=trace.trace))


def estimate(qc: QuantumCircuit, params: dict | None = None) -> EstimatorResult:
    """Ask the QDK's resource estimator what a circuit costs on an error-corrected machine.

    Args:
        qc: circuit to estimate; compiled to the device's native gates first, so the QDK
            never sees the controlled rotations it mishandles.
        params: estimator parameters, e.g. a different qubit model or error budget. None
            uses the estimator's defaults.

    Returns:
        The QDK's `EstimatorResult`, a dict-like object. `["logicalCounts"]` holds the
        logical qubits, rotations and T gates; `["physicalCounts"]` the physical qubits
        and runtime.
    """
    native = transpile(
        qc,
        basis_gates=NEUTRAL_ATOM_BASIS + ["reset"],
        optimization_level=1,
        seed_transpiler=1,
    )
    return (
        ResourceEstimatorBackend(skip_transpilation=True)
        .run(native, params=params)
        .result()
    )


def _reset_after_measure(qc: QuantumCircuit) -> QuantumCircuit:
    """Move each reset of a just-measured qubit to directly after its measurement.

    The qubit is untouched in between, so the circuit is unchanged. It works around a QDK
    device bug: a reset that isn't next to its measurement (e.g. with if_else blocks between)
    is compiled to its own mresetz with a result slot that doesn't exist, and the simulator
    crashes ("index out of bounds").

    Args:
        qc: circuit whose resets should be moved. Resets of a qubit touched after its
            measurement are left where they are.

    Returns:
        A new, equivalent circuit with those resets right after their measurements.
    """
    out = qc.copy_empty_like()
    data, moved = list(qc.data), set()
    for i, inst in enumerate(data):
        if i in moved:
            continue
        out.append(inst)
        if inst.operation.name == "measure":
            q = inst.qubits[0]
            nxt = next(
                (j for j in range(i + 1, len(data)) if q in data[j].qubits), None
            )
            if nxt is not None and data[nxt].operation.name == "reset":
                out.append(data[nxt])
                moved.add(nxt)
    return out


def _adaptive_qir(qc: QuantumCircuit) -> str:
    """Compile a circuit to Adaptive-profile QIR in the device's native gates.

    NeutralAtomBackend only emits Base-profile QIR (no branching on measurements), so compile
    to native gates in Qiskit and emit Adaptive QIR for the device directly.

    Args:
        qc: circuit to compile; may contain mid-circuit measurement, reset and if_else.

    Returns:
        Adaptive_RI-profile QIR, as text, in `NEUTRAL_ATOM_BASIS` gates.
    """
    native = _reset_after_measure(
        transpile(
            qc, basis_gates=NEUTRAL_ATOM_BASIS, optimization_level=1, seed_transpiler=1
        )
    )
    return QSharpBackend(skip_transpilation=True).qir(
        native, target_profile=TargetProfile.Adaptive_RI
    )
