```

    Inverse problem script to learn ODE model from SXS waveform data

```
cd(@__DIR__)
using Pkg; Pkg.activate("."); Pkg.instantiate()
# using DifferentialEquations
# using DiffEqFlux
using LinearAlgebra
using Optim
using LineSearches
using DataFrames
using CSV
using Plots
using Lux
using Random
using SciMLBase
using SciMLSensitivity
using OptimizationOptimJL
using ForwardDiff
# using DiffEqSensitivity
using ComponentArrays
using Optimization
using OptimizationOptimisers
using OrdinaryDiffEq
gr()

include("/Users/rbari/Downloads/common_denominator-2020_scimlforbbhs-e19e6808aad2/paper/zenodo/GWaveInversion/myModelsMan.jl")
include("/Users/rbari/Downloads/common_denominator-2020_scimlforbbhs-e19e6808aad2/paper/zenodo/GWaveInversion/WaveformMan.jl")

t₀ = 5263.7607
tspan = (t₀, 5.569f3)
datasize = 1000
tsteps_data = range(tspan[1], tspan[2], length = datasize) 

x_ecc, y_ecc = file2trajectory(tsteps_data,"./input/trajectoryA_eccentric.txt")
x2_ecc, y2_ecc = file2trajectory(tsteps_data,"./input/trajectoryB_eccentric.txt")
waveform_real_ecc = file2waveform(tsteps_data,"./input/waveform_real_eccentric.txt")

function extract_effective_particle_initial_conditions(x1_traj, y1_traj, x2_traj, y2_traj, tsteps)
    println("Extracting effective particle initial conditions from binary trajectories...")
    
    # 1. Compute relative separation vector
    rel_x = x1_traj[1] - x2_traj[1]  
    rel_y = y1_traj[1] - y2_traj[1]
    
    # 2. Initial position in polar coordinates  
    r0 = sqrt(rel_x^2 + rel_y^2)
    φ0 = atan(rel_y, rel_x)
    
    # 3. Compute relative velocity using finite differences
    dt = tsteps[2] - tsteps[1]
    
    # Relative velocity components
    v_rel_x = ((x1_traj[2] - x2_traj[2]) - (x1_traj[1] - x2_traj[1])) / dt
    v_rel_y = ((y1_traj[2] - y2_traj[2]) - (y1_traj[1] - y2_traj[1])) / dt
    
    # 4. Convert to radial and angular velocities
    dr_dt = (rel_x * v_rel_x + rel_y * v_rel_y) / r0
    dφ_dt = (rel_x * v_rel_y - rel_y * v_rel_x) / r0^2
    
    # 5. Convert to canonical momenta for Schwarzschild coordinates
    f0 = 1 - 2/r0  # Schwarzschild factor (M = 1)
    
    pr0 = dr_dt / f0           # From dr/dt = (1-2M/r) * p_r  
    pφ0 = dφ_dt * r0^2         # From dφ/dt = p_φ/r²
    
    # 6. Determine energy (p_t) from the Hamiltonian constraint
    # For a bound orbit: H = (1/2)[p_t²*(-f₀⁻¹) + p_r²*f₀ + p_φ²/r₀²] = E_binding < 0
    
    # Use the effective potential minimum as initial guess
    L_eff = abs(pφ0)
    E_guess = -0.5  # Typical binding energy for close orbits
    
    function hamiltonian_constraint(E)
        E_val = E[1]
        pt = -E_val
        H_val = 0.5 * (pt^2 * (-1/f0) + pr0^2 * f0 + pφ0^2 / r0^2)
        return (H_val - (-0.5))^2
    end
    
    result = optimize(hamiltonian_constraint, [-1.0])
    E0_initial = result.minimizer[1]
    
    # Diagnostic checks (using correct variable names)
    H_check = 0.5 * (E0_initial^2 * (-1/(1-2/r0)) + pr0^2 * (1-2/r0) + pφ0^2 / r0^2)
    println("Initial Hamiltonian value: ", H_check, " (should be -0.5)")
    
    # More robust energy finding
    function find_correct_energy(r0, pr0, pφ0)  # Fixed variable names
        function energy_constraint(E_vec)
            E = E_vec[1]
            pt = -E
            f0 = 1 - 2/r0
            H_val = 0.5 * (pt^2 * (-1/f0) + pr0^2 * f0 + pφ0^2 / r0^2)
            return (H_val + 0.5)^2
        end
        
        results = []
        for E_init in [-0.1, -0.3, -0.5, -0.7, -0.9]
            result = optimize(energy_constraint, [E_init])
            push!(results, (result.minimizer[1], result.minimum))
        end
        
        best_E = results[argmin([r[2] for r in results])][1]
        return best_E
    end
    
    E0_corrected = find_correct_energy(r0, pr0, pφ0)  # Fixed variable names
    pt0_corrected = -E0_corrected
    
    # Use corrected energy in final state vector
    u0_effective = [0.0, r0, π/2, φ0, pt0_corrected, pr0, 0.0, pφ0, 0.0]
    println("Initial Energy: ", E0_initial)
    println("Corrected Energy: ", E0_corrected)
    return u0_effective
end

# Use it with your trajectory data:
# u0_extracted = extract_effective_particle_initial_conditions(x_ecc, y_ecc, x2, y2, tsteps)