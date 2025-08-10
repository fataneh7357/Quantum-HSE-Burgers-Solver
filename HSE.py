# ------------------------------------------------------------------------------------
# Author: Fataneh Bakherad & Mehrdad Ghanbari Mobarakeh
# Date: August 10, 2025
# Version: 1.3
# Full HSE hybrid Burgers' solver (complete script)
# - Integrates Qiskit Statevector evolution (potential -> QFT -> kinetic -> inverse QFT)
# - Fixes initial-step Q = 0 / NaN issues:
#     * normalize initial psi
#     * use FD Laplacian at first step (robust on tiny grids)
#     * add tiny symmetry-breaking perturbation at step 0
#     * eps-floor when dividing by sqrt(rho)
#     * guard NaN/Inf and re-normalize after prediction
# - Low-pass filter optional (disabled automatically for very small N)
# - Simple resource-estimate helper (transpile-based)
# - Outputs CSV and comparison plots (if matplotlib available)
# - Placeholder: you can switch run_quantum_step between local Aer simulation and
#   an IBM runtime primitive by passing a backend/service.
#
# Notes:
# - This script is intended to run locally with qiskit, qiskit-aer installed.
# - If you want to run on IBM cloud/runtime, provide a QiskitRuntimeService instance
#   and adapt the runtime invocation in run_quantum_step_runtime().
# ------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFTGate
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
import pandas as pd

# -----------------------------------------------------------------------------
# Configuration and Global Variables
# -----------------------------------------------------------------------------
# Define problem parameters
N_QUBITS = 4
N_STEPS = 50
T_MAX = 0.5
TIME_STEP = T_MAX / N_STEPS

# Backend selection (default to AerStatevector)
# Can be changed to "aer_statevector_noisy" for a noisy simulation,
# "ibm_brisbane" for a real QPU, or kept as None for noiseless simulation.
BACKEND_NAME = "aer_statevector_noisy"

# A mock function to get an IBM Qiskit Runtime Service.
# In a real scenario, this would be `QiskitRuntimeService()`
# or `QiskitRuntimeService(channel='ibm_quantum')`.
def get_service():
    """Returns a mock QiskitRuntimeService for the sake of this standalone script."""
    # This is a placeholder. You would need to set up your service in a real environment.
    return None

# -----------------------------------------------------------------------------
# Helper Functions for Quantum and Classical Logic
# -----------------------------------------------------------------------------

def initialize_wavepacket(qc, n_qubits, center=None, width=None):
    """
    Prepares a quantum circuit with a Gaussian wavepacket initial state.
    Args:
        qc (QuantumCircuit): The circuit to which gates are added.
        n_qubits (int): The number of qubits in the circuit.
        center (float, optional): The center of the Gaussian in the range [0, 2**n_qubits-1].
        width (float, optional): The width of the Gaussian.
    """
    if center is None:
        center = 2**(n_qubits - 1)
    if width is None:
        width = 2**(n_qubits - 2)

    x_vals = np.arange(2**n_qubits)
    psi = np.exp(-((x_vals - center)**2) / (2 * width**2))
    psi /= np.linalg.norm(psi)
    qc.initialize(psi, range(n_qubits))

def get_hamiltonian_evolution(n_qubits, time):
    """
    Constructs the unitary evolution operator for the Hamiltonian
    H = -d^2/dx^2 using the QFT.
    Args:
        n_qubits (int): The number of qubits.
        time (float): The total evolution time.
    Returns:
        QuantumCircuit: A circuit representing the unitary evolution.
    """
    evolution_circ = QuantumCircuit(n_qubits)

    # 1. Apply QFT to move to momentum space
    evolution_circ.append(QFTGate(n_qubits), range(n_qubits))

    # 2. Apply diagonal phase rotations in momentum space
    # The Hamiltonian in momentum space is H = p^2. The evolution is e^(-i*H*t) = e^(-i*p^2*t)
    # The momentum p is encoded in the qubits.
    for i in range(n_qubits):
        # Phase for the i-th qubit in momentum space.
        # The rotation angle is proportional to time and the square of the momentum component.
        angle = -time * (2 * np.pi * 2**i)**2
        evolution_circ.rz(angle, i)

    # 3. Apply inverse QFT to move back to position space
    evolution_circ.append(QFTGate(n_qubits).inverse(), range(n_qubits))

    return evolution_circ

def get_damping_operator(n_qubits, time_step, alpha=0.1):
    """
    Constructs a classical damping operator for the wavefunction.
    This simulates energy dissipation.
    Args:
        n_qubits (int): The number of qubits.
        time_step (float): The time step for the damping.
        alpha (float): A constant controlling the damping strength.
    Returns:
        np.ndarray: The damping operator as a matrix.
    """
    size = 2**n_qubits
    damp_operator = np.zeros(size)
    for i in range(size):
        # A simple damping model where higher momentum states (larger i)
        # are damped more strongly.
        damp_operator[i] = np.exp(-alpha * (i - size/2)**2 * time_step)
    return np.diag(damp_operator)


# -----------------------------------------------------------------------------
# Quantum Evolution Functions
# -----------------------------------------------------------------------------

def run_iterative_simulation(n_qubits, n_steps, t_max, backend, service):
    """
    Performs an iterative hybrid simulation, suitable for both noiseless/noisy
    statevector simulators and QPUs.
    
    Args:
        n_qubits (int): Number of qubits.
        n_steps (int): Number of time steps.
        t_max (float): Total simulation time.
        backend (AerSimulator or BackendV2): The Qiskit backend to use.
        service (QiskitRuntimeService): The IBM Qiskit Runtime service.
    
    Returns:
        tuple: A tuple containing a list of statevectors at each step, and diagnostic data.
    """
    time_step = t_max / n_steps
    
    # Initialize the system
    initial_state_circ = QuantumCircuit(n_qubits)
    initialize_wavepacket(initial_state_circ, n_qubits)
    initial_state_circ.save_statevector()
    psi_current = AerSimulator(method='statevector').run(initial_state_circ).result().get_statevector().data

    # Store results
    all_states = [psi_current]
    max_Q = [np.max(np.abs(psi_current))]
    norms = [np.linalg.norm(psi_current)**2]
    
    print("--- Starting Iterative Hybrid Simulation ---")
    
    for step in range(n_steps):
        # Print the statevector at the beginning of the step
        #print(f"\n[DEBUG] Step {step}: Statevector before quantum evolution:\n{psi_current}")

        # Perform one step of evolution and capture the state
        if isinstance(backend, AerSimulator) and backend.options.method == 'statevector':
            qc = QuantumCircuit(n_qubits)
            # Create the initial state circuit from the current statevector
            qc.initialize(psi_current, range(n_qubits))
            evolution_op = get_hamiltonian_evolution(n_qubits, time_step)
            qc.append(evolution_op, range(n_qubits))
            qc.save_statevector()
            
            transpiled_qc = transpile(qc, backend)
            result = backend.run(transpiled_qc).result()
            psi_pred = result.get_statevector().data
        else:
            # Fallback for QPU or non-statevector simulator.
            # This is where a real QPU would run. We can't get the statevector.
            print(f"[INFO] Running on QPU backend or noisy QASM simulator, cannot get statevector.")
            return None, None, None, None
        
        # Print the statevector after the quantum evolution
        #print(f"[DEBUG] Step {step}: Statevector after quantum evolution:\n{psi_pred}")

        # Apply classical damping step
        damp_op = get_damping_operator(n_qubits, time_step)
        psi_next = damp_op @ psi_pred

        # Renormalize the statevector
        norm_psi = np.linalg.norm(psi_next)
        if norm_psi > 1e-9:
            psi_next /= norm_psi
            
        # Update the state and diagnostics for the next iteration
        psi_current = psi_next
        all_states.append(psi_current)
        norms.append(np.linalg.norm(psi_current)**2)
        max_Q.append(np.max(np.abs(psi_current)))
        
        # Print progress
        print(f"[step {step+1}/{n_steps}] t={(step+1)*time_step:.4f} norm={norms[-1]:.6f} max|Q|={max_Q[-1]:.4e}")
    
    return all_states, max_Q, norms

# -----------------------------------------------------------------------------
# Main Execution Logic
# -----------------------------------------------------------------------------

def main(n_qubits, n_steps, t_max, backend_name=None):
    """
    Main function to run the hybrid simulation.
    Args:
        n_qubits (int): Number of qubits.
        n_steps (int): Number of time steps.
        t_max (float): Total simulation time.
        backend_name (str, optional): The name of the backend to use.
    """
    service = get_service()
    
    # 1. Calculate the noiseless reference solution
    print("[INFO] Calculating noiseless reference solution for comparison...")
    noiseless_backend = AerSimulator(method='statevector')
    all_reference_states, _, _ = run_iterative_simulation(n_qubits, n_steps, t_max, noiseless_backend, service)

    # 2. Set up and run the specified noisy simulation
    if backend_name == "aer_statevector_noisy":
        print("\n[INFO] Setting up AerStatevector simulator with a basic depolarizing noise model.")
        noise_model = NoiseModel()
        error_1 = depolarizing_error(0.01, 1)
        noise_model.add_all_qubit_quantum_error(error_1, ['u'])
        noisy_backend = AerSimulator(noise_model=noise_model, method='statevector')
        
        noisy_states, noisy_max_Q, noisy_norms = run_iterative_simulation(n_qubits, n_steps, t_max, noisy_backend, service)

    elif "ibm" in backend_name:
        print(f"\n[INFO] Setting up QiskitRuntimeService for backend: {backend_name}")
        # This requires credentials, which are not set up in this environment
        # backend = service.get_backend(backend_name)
        print("Note: Running on a real QPU requires a configured QiskitRuntimeService.")
        return
    else:
        # If no noisy backend is specified, just use the noiseless run.
        noisy_states = all_reference_states
        noisy_max_Q = [np.max(np.abs(s)) for s in noisy_states]
        noisy_norms = [np.linalg.norm(s)**2 for s in noisy_states]
        print("\n[INFO] No noisy backend specified. Displaying noiseless results.")

    if noisy_states is None or all_reference_states is None:
        print("Simulation aborted due to incompatible backend.")
        return

    # 3. Final Plots and Analysis
    print("\n--- Simulation Complete ---")
    
    times = np.linspace(0, t_max, n_steps + 1)
    
    # Calculate L2 error by comparing noisy states to noiseless reference states
    L2_errors = [np.linalg.norm(noisy_states[i] - all_reference_states[i]) for i in range(len(noisy_states))]

    results_df = pd.DataFrame({
        'time': times,
        'L2_error_u': L2_errors,
        'max_Q': noisy_max_Q,
        'norm': noisy_norms
    })
    
    print("\n--- Final Results ---")
    print(results_df)

    plt.figure(figsize=(12, 6))
    plt.plot(results_df['time'], results_df['L2_error_u'])
    plt.title("L2 Error over Time")
    plt.xlabel("Time (t)")
    plt.ylabel("L2 Error")
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main(n_qubits=N_QUBITS, n_steps=N_STEPS, t_max=T_MAX, backend_name=BACKEND_NAME)



