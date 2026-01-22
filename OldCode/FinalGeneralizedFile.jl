```

    Inverse problem script to learn ODE model from SXS waveform data

```
cd(@__DIR__)
using Pkg; Pkg.activate("."); Pkg.instantiate()

# Clean imports - no duplicates
using LinearAlgebra
using OrdinaryDiffEq
using Optim
using LineSearches
using DataFrames
using CSV
using Plots
using Random
using SciMLBase
using SciMLSensitivity
using OptimizationOptimJL
using ForwardDiff
using ComponentArrays
using Optimization
using OptimizationOptimisers
using Lux

gr()

include("/Users/rbari/Downloads/common_denominator-2020_scimlforbbhs-e19e6808aad2/paper/zenodo/GWaveInversion/ModelsGeneralized.jl")
include("/Users/rbari/Downloads/common_denominator-2020_scimlforbbhs-e19e6808aad2/paper/zenodo/GWaveInversion/WaveformMan.jl")


"""
DEFINE INITIAL CONDITIONS
"""
mass_ratio = 1
model_params = [mass_ratio] # Just the mass ratio, at least for now
mass1 = 1.0/(1.0+mass_ratio)
mass2 = mass_ratio/(1.0+mass_ratio)

tspan = (5264, 6770)
datasize = 1000
tsteps = range(tspan[1], tspan[2], length = datasize) 
dt_data = tsteps[2] - tsteps[1]
dt = 1.0
num_optimization_increments = 100

# Gather waveform data
x_ecc, y_ecc = file2trajectory(tsteps,"./input/trajectoryA_eccentric.txt")
x2_ecc, y2_ecc = file2trajectory(tsteps,"./input/trajectoryB_eccentric.txt")
waveform_real_ecc = file2waveform(tsteps,"./input/waveform_real_eccentric.txt")

e₀ = 0.085   # optimized value is near 0.099
x₀ = 6.484530581221468
χ₀ = 3.66 # optimized value is near 3.66
p₀ = x₀ * (1+e₀*cos(χ₀)) / mass1
u0_pe_coordinates = Float32[χ₀, 0.0, 6.484530872637143*2+2.1, 0.16384305990318113] # χ₀, ϕ₀, p₀, e₀
p = u0_pe_coordinates[3]
e = u0_pe_coordinates[4]
E0, L0 = pe_2_EL(p, e)[2:3]
# r0 = p / (1 + e)  # Correct periapsis radius = 12.368763
r0 = 12.9690618979
E0 = 0.8437969360351569
pᵣ0 = 0.01
L0 = 3.5061003277
u0 = [0, r0, pi/2, 0, E0, pᵣ0, 0, -L0, 0]

"""
CREATE NEURAL NETWORKS
"""
#  # Neural network setup 
NN_Conservative = Chain(
    Dense(1, 4, tanh), # Input: r only
    Dense(4, 4, tanh),
    Dense(4, 3), # Output: [g_tt, g_rr, g_ϕϕ]
)

NN_Dissipative = Chain(
    Dense(1, 4, tanh), # Input: r
    Dense(4, 4, tanh),
    Dense(4, 1),
)

"""
INITIALIZE NEURAL NETWORKS
"""
# Initialize parameters for both NNs
rng = MersenneTwister(222)
NN_Conservative_params, NN_Conservative_state = Lux.setup(rng, NN_Conservative)
NN_Dissipative_params, NN_Dissipative_state = Lux.setup(rng, NN_Dissipative)

# Convert to appropriate precision
precision = Float64
NN_Conservative_params = Lux.fmap(x -> precision.(x), NN_Conservative_params)
NN_Dissipative_params = Lux.fmap(x -> precision.(x), NN_Dissipative_params)

for (i, layer) in enumerate(NN_Conservative_params)
    if ~isempty(layer)
        if i == length(NN_Conservative_params)  # Final layer
            layer.weight .= 0.01 * randn(rng, eltype(layer.weight), size(layer.weight))
            layer.bias .= 0.0  # Force output near 0
        else  # Hidden layers
            layer.weight .= 0.1 * randn(rng, eltype(layer.weight), size(layer.weight))
            layer.bias .= 0.1 * randn(rng, eltype(layer.bias), size(layer.bias))
        end
    end
end 

for (i, layer) in enumerate(NN_Dissipative_params)
    if ~isempty(layer)
        if i == length(NN_Dissipative_params)  # Final layer
            layer.weight .= 0.01 * randn(rng, eltype(layer.weight), size(layer.weight))
            layer.bias .= 0.0  # Force output near 0
        else  # Hidden layers
            layer.weight .= 0.1 * randn(rng, eltype(layer.weight), size(layer.weight))
            layer.bias .= 0.1 * randn(rng, eltype(layer.bias), size(layer.bias))
        end
    end
end 

"""
ASSIGN NEURAL NETWORK INPUTS
EXTRACT NEURAL NETWORK OUTPUTS
"""
# Now, create adapter functions that match the calling pattern expected by AbstractNROrbitModel
function NN_adapter_dual(u, params)
    # Conservative network
    conservative_features = [u[2]]
    conservative_output, _ = NN_Conservative(conservative_features, params.conservative, NN_Conservative_state)
    
    # Dissipative network
    dissipative_features = [u[2]]  # t, r, θ, φ, p_t, p_r, p_θ, p_ϕ
    dissipative_output, _ = NN_Dissipative(dissipative_features, params.dissipative, NN_Dissipative_state)

    return (conservative = conservative_output, dissipative = dissipative_output)
end

NN_params = ComponentArray(
    conservative = NN_Conservative_params, 
    dissipative = NN_Dissipative_params
)

"""
DEFINE ODE MODEL FOR BINARY BLACK HOLE SYSTEM!
"""
function ODE_model_dual(du, u, p, t)
    du = GENERIC(du, u, model_params, t,
                                    NN=NN_adapter_dual, 
                                    NN_params=p)
    return du
end

"""
SET UP ODE MODEL + SOLVE IT + CONVERT TO WAVEFORM
"""
prob_nn_dual = ODEProblem(ODE_model_dual, u0, tspan, NN_params)
soln_nn_dual = Array(solve(prob_nn_dual, Tsit5(), saveat = tsteps, dt = dt, adaptive=false, verbose=false))
waveform_nn_real_dual, waveform_nn_imag_dual = compute_waveform(dt_data, soln_nn_dual, mass_ratio; coorbital=false)

"""
DEFINE A LOSS FUNCTION
"""
# # ## Define the objective function
function loss(NN_params; saveat=tsteps)
    tspan = (saveat[1],saveat[end])
    pred_soln = solve(remake(prob_nn_dual, p = NN_params, tspan=tspan), Tsit5(),
                            saveat = saveat, dt = dt, adaptive=false, verbose = false, sensealg=BacksolveAdjoint(checkpointing=true))
    pred_waveform_real, pred_waveform_imag = compute_waveform(dt_data, pred_soln, mass_ratio)

    N = length(pred_waveform_real)
    loss = ( sum(abs2, waveform_real_ecc[1:N] .- pred_waveform_real))
    return loss, pred_waveform_real, pred_waveform_imag, pred_soln
end

# # # # Test loss function
loss(NN_params)

losses = []
# predicted_wave = []


"""
RUN OPTIMIZATION ALGORITHM
"""
# # ## Train with BFGS (gives best results because the Newtonian model seems to give a very good initial guess)
optimization_increments = [collect(40:10:num_optimization_increments-10)..., num_optimization_increments-3, num_optimization_increments-2, num_optimization_increments-1,  num_optimization_increments]
for i in optimization_increments
    println("optimization increment :: ", i, " of ", num_optimization_increments)
    tsteps_increment = tsteps[tsteps .<= tspan[1]+i*(tspan[2]-tspan[1])/num_optimization_increments]
    tmp_loss(p) = loss(p,saveat=tsteps_increment)
    
    function scalar_loss(p)
        loss_val, pred_waveform_real, pred_waveform_imag, pred_soln = tmp_loss(p)
        return loss_val  # Return only the loss value for gradient computation
    end

    function opt_callback(state, args...; kwargs...)
        p = state.u
    
        loss_val, pred_waveform_real, pred_waveform_imag, pred_soln = tmp_loss(p)
    
        push!(losses, loss_val)
        display(loss_val)
    
        N = length(pred_waveform_real)
        startPoint = 1
        z_axis  = tsteps[startPoint:N]
        # psuh!(predicted_wave, pred_waveform_real[N])
        # print("real waveform: ", waveform_real_ecc, 
        #     "\npred waveform: ", pred_waveform_real)
        plt1 = plot(tsteps[startPoint:N], waveform_real_ecc[startPoint:N], 
                    markershape=:circle, markeralpha = 0.25,
                    linewidth = 2, alpha = 0.5, label = "True Gravitational Wave")
        plot!(plt1, tsteps[startPoint:N], pred_waveform_real[startPoint:N],
              markershape=:circle, markeralpha = 0.25,
              linewidth = 2, alpha = 0.5, label = "Predicted Gravitational Wave")
            
        display(plt1)
        # N = 10
        pred_orbit_relative = soln2orbit(pred_soln)
        pred_orbit1, pred_orbit2 = one2two(pred_orbit_relative, 1, mass_ratio)
        plt2 = plot3d(x_ecc[startPoint:N], y_ecc[startPoint:N], z_axis;
              lw=2, alpha=1, label="orbit data",
              camera=(35,20))  # azimuth,elevation
        plot!(plt2, x2_ecc[startPoint:N], y2_ecc[startPoint:N], z_axis;
              lw=2, alpha=0.9, color=:orange, label="orbit 2 data")
        plot!(plt2, pred_orbit1[1,startPoint:N], pred_orbit1[2,startPoint:N], z_axis;
      lw=2, alpha=0.9, ls=:dash, color=:red, label="orbit NN")
        plot!(plt2, pred_orbit2[1,startPoint:N], pred_orbit2[2,startPoint:N], z_axis;
              linewidth = 2, alpha = 0.8, color = "green",
              label = "orbit 2 NN", ls=:dash, aspect_ratio=:equal)
        display(plt2)
        χ = pred_soln[1,:]
        # layout = @layout [
        #     a; b
        #     ]
    
        # full_plot = plot(plt1, plt2, layout=layout, size = (600, 400))
        # display(plot(full_plot))
        
        return false
    end

    global NN_params = Lux.fmap(x -> x .+ Float64(1e-5)*randn(eltype(x), size(x)), NN_params) 

    p_init = ComponentArray(NN_params)
    adtype = Optimization.AutoZygote()
    optf = Optimization.OptimizationFunction((x, p) -> scalar_loss(x), adtype)
    optprob = Optimization.OptimizationProblem(optf, p_init)

    if i < 80
        local res = Optimization.solve(optprob, Optim.BFGS(; initial_stepnorm=0.5, linesearch = LineSearches.BackTracking()), callback=opt_callback, maxiters = 50, allow_f_increases=true)
    elseif i == 80
        local res = Optimization.solve(optprob, Optim.BFGS(; initial_stepnorm=0.5, linesearch = LineSearches.BackTracking()), callback=opt_callback, maxiters = 100, allow_f_increases=true)
    elseif i == optimization_increments[end-3]  # 97
        local res = Optimization.solve(optprob, Optim.BFGS(; initial_stepnorm=1e-1, linesearch = LineSearches.BackTracking()), callback=opt_callback, maxiters = 200, allow_f_increases=true)
    elseif i == optimization_increments[end-2]  # 98
        local res = Optimization.solve(optprob, Optim.BFGS(; initial_stepnorm=1e-2, linesearch = LineSearches.BackTracking()), callback=opt_callback, maxiters = 300, allow_f_increases=true)
    elseif i == optimization_increments[end-1]  # 99
        local res = Optimization.solve(optprob, Optim.BFGS(; initial_stepnorm=1e-3, linesearch = LineSearches.BackTracking()), callback=opt_callback, maxiters = 400, allow_f_increases=true)
    else  # 100
        local res = Optimization.solve(optprob, Optim.BFGS(; initial_stepnorm=1e-4, linesearch = LineSearches.BackTracking()), callback=opt_callback, maxiters = 500, allow_f_increases=true)
    end

    global NN_params = res.minimizer
    # local plt = plot(losses, yaxis=:log, linewidth = 2, xlabel = "Iteration", ylabel = "Objective value", legend = false)
    # display(plot(plt))
end

optimized_solution = solve(remake(prob_nn_dual, p = NN_params, tspan=tspan), Tsit5(), saveat = tsteps, dt = dt, adaptive=false)

pred_waveform_real, pred_waveform_imag = compute_waveform(dt_data, optimized_solution, mass_ratio)

pred_orbit = soln2orbit(optimized_solution)
pred_orbit1_nn, pred_orbit2_nn = one2two(pred_orbit, 1, 1)

plt1 = plot(x_ecc, y_ecc, linewidth = 2, label = "truth")
plot!(plt1, pred_orbit1_nn[1,:], pred_orbit1_nn[2,:], linestyle = :dash, linewidth = 2, label = "prediction")
plt2 = plot(losses, yaxis=:log, linewidth = 2, xlabel = "Iteration", ylabel = "Objective value", legend = false)
plt = plot(plt1, plt2, layout = (1,2))
display(plot(plt))