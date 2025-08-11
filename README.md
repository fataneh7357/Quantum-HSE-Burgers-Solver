# Project Name  
Quantum-Enhanced Simulation of the 1D Burgers’ Equation Using the Hydrodynamic Schrödinger Equation Framework

# Quantum-HSE-Burgers-Solver
The HSE framework is used in this resource-efficient quantum solver for the 1D viscous Burgers' equation. Its features include validation against traditional CFD baselines, real QPU execution with error mitigation, and noiseless/noisy simulation. The algorithm is intended for low-depth unitary evolution and minimal qubits.

# Team Name  
HSE Solvers team

# Team Members  
- Fataneh Bakherad — WISER Enrollment ID: gst-fTTTNAGPxE8rBql
- Mehrdad Ghanbari Mobarakeh — WISER Enrollment ID: gst-WGFCK7KwaNLETQf

# Project Description:
Click [here][1] to view the project description.

---

# Project Summary:

A basic nonlinear partial differential equation, the 1D Burgers' equation simulates traffic dynamics, shock wave formation, and viscous fluid flow. Because of the nonlinearity and the requirement for precise spatial and temporal discretization, it combines nonlinear convection and diffusion effects, making classical numerical simulation difficult.

Using the Hydrodynamic Schrödinger Equation (HSE) framework, our project proposes a hybrid quantum-classical algorithm to simulate the 1D Burgers' equation. By reformulating nonlinear PDEs into a Schrödinger-like form, this novel method makes it possible to employ quantum simulation methods that are normally only used for quantum mechanical systems.

Encoding the fluid velocity and density fields into a complex wavefunction that can be directly displayed on a quantum register is the main concept. To enable effective wavefunction encoding, we discretize the spatial domain into $N=2^n$ grid points, where $n$ is the number of qubits.

A series of unitary operations is used to implement the quantum evolution over a single time step:
Applying a possible phase shift to the position basis,
A kinetic phase shift is applied in momentum space, a quantum Fourier transform (QFT) is used to transition to momentum space, and an inverse QFT is used to go back to position space.

The linear kinetic and potential terms are naturally captured by this spectral split-step method. In momentum space, viscous dissipation is introduced as an exponential damping factor.

The Burgers' equation's nonlinear term is treated classically. We update the effective potential for the next quantum step and calculate the nonlinear quantum potential from the Laplacian of the wavefunction amplitude. The system is iteratively evolved by this hybrid loop, which combines classical processing for nonlinearities with quantum speedup in linear operators.

Our method accurately captures velocity evolution and shock smoothing, as shown by validation tests against the exact Cole-Hopf solution. Numerical experiments demonstrate stable long-term behavior and decreasing $L_2$ errors with increasing resolution.

Polynomial scaling of quantum gates dominated by QFT and diagonal phase operations is indicated by resource estimates derived from Qiskit transpilation, indicating feasibility on near-term quantum devices, particularly with approximate QFTs and error mitigation techniques.

The algorithm is appropriate for noisy intermediate-scale quantum (NISQ) processors because it strikes a balance between quantum advantage potential and classical numerical stability. The quantum component effectively simulates high-dimensional linear dynamics, while classical normalization and smoothing reduce noise effects.

Using the HSE framework to bridge the gap between quantum simulation and classical numerical methods, our project demonstrates a promising hybrid quantum-classical approach to solving nonlinear fluid dynamics PDEs. The foundation for scalable quantum simulations of intricate nonlinear systems is laid by this technique, which may find use in computational fluid dynamics and other fields.
---
# Bibliography:
- Meng, Z., Yang, J. Phys. Rev. Research 5 (2023). Hydrodynamic Schrödinger
Equation framework.[Link to paper][2]
- [Qiskit Documentation][3].

# Project Presentation Deck:
[Quantum Algorithm as a PDE Solver.pptx](https://github.com/user-attachments/files/21708316/Quantum.Algorithm.as.a.PDE.Solver.1.pptx)

[1]: https://www.thewiser.org/quantum-pde-solvers-for-cfd
[2]: https://arxiv.org/abs/2302.09741
[3]: https://qiskit.org/documentation/

---

