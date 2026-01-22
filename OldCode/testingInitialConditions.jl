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

function pe_2_EL(semilatusRectum, eccentricity)
    p = semilatusRectum
    e = eccentricity
    
    M = 1
    E = sqrt( (p-2-2*e)*(p-2+2*e) / (p*(p-3-e^2)) )
    L = sqrt( (p^2 * M^2) / (p-3-e^2) )
  
    return [M, E, L]
  end


# # Gather waveform data

t₀ = 5263.7607
# χ₀ = 0 # optimized value is near 3.66
# p₀ = x₀ * (1+e₀*cos(χ₀))
# φ0 = atan(y[1], x[1])
# u0_pe_coordinates = Float32[χ₀, φ0, p₀, e₀] # χ₀, ϕ₀, p₀, e₀

tspan = (t₀, 6.769f3)
datasize = 1000
tsteps = range(tspan[1], tspan[2], length = datasize) 
dt_data = tsteps[2] - tsteps[1]
dt = 10.0
num_optimization_increments = 100

x_ecc, y_ecc = file2trajectory(tsteps,"./input/trajectoryA_eccentric.txt")
x2_ecc, y2_ecc = file2trajectory(tsteps,"./input/trajectoryB_eccentric.txt")

plot(x_ecc, y_ecc)
plot!(x2_ecc, y2_ecc)

function extract_initial_conditions_from_both_trajectories(x1_traj, y1_traj, x2_traj, y2_traj, tsteps)
    # Direct separation calculation
    r0 = sqrt((x1_traj[1] - x2_traj[1])^2 + (y1_traj[1] - y2_traj[1])^2)
    
    # Relative position angle
    rel_x = x1_traj[1] - x2_traj[1]
    rel_y = y1_traj[1] - y2_traj[1]
    φ0 = atan(rel_y, rel_x)
    
    # Time derivative
    dt = tsteps[2] - tsteps[1]
    
    # Relative velocity components
    d_rel_x_dt = ((x1_traj[2] - x2_traj[2]) - (x1_traj[1] - x2_traj[1])) / dt
    d_rel_y_dt = ((y1_traj[2] - y2_traj[2]) - (y1_traj[1] - y2_traj[1])) / dt
    
    # Convert to polar velocities
    dr_dt = (rel_x * d_rel_x_dt + rel_y * d_rel_y_dt) / r0
    dφ_dt = (rel_x * d_rel_y_dt - rel_y * d_rel_x_dt) / (r0^2)
    
    # Convert to canonical momenta
    f0 = 1 - 2/r0
    pr0 = dr_dt / f0
    pφ0 = dφ_dt * r0^2
    
    # Energy estimation (same as before)
    function energy_constraint(E_guess)
        pt_guess = -E_guess
        H_val = 0.5 * (pt_guess^2 * (-1/f0) + pr0^2 * f0 + pφ0^2 / r0^2)
        return abs(H_val)
    end
    
    result = optimize(energy_constraint, [-2.0])
    pt0 = -result.minimizer[1]
    
    u0_extracted = [0.0, r0, π/2, φ0, pt0, pr0, 0.0, pφ0, 0.0]
    
    println("Extracted from both trajectories:")
    println("Separation r0 = ", r0)
    println("Phase φ0 = ", φ0)
    # println("Energy pt0 = ", pt0)
    println("Radial momentum pr0 = ", pr0)
    println("Angular momentum pφ0 = ", pφ0)
    
    return u0_extracted
end

# Use both trajectories:
u0_extracted = extract_initial_conditions_from_both_trajectories(x_ecc, y_ecc, x2_ecc, y2_ecc, tsteps)