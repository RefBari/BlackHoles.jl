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
using ProgressMeter
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

## Define the experiment parameters
mass_ratio = 1.0
model_params = [mass_ratio] # Just the mass ratio, at least for now
mass1 = 1.0/(1.0+mass_ratio)
mass2 = mass_ratio/(1.0+mass_ratio)

# e₀ = 0.164   # optimized value is near 0.099
# x₀ = 14.256812850825423
t₀ = 5263.7607
# χ₀ = 0 # optimized value is near 3.66
# p₀ = x₀ * (1+e₀*cos(χ₀))
# φ0 = atan(y[1], x[1])
# u0_pe_coordinates = Float32[χ₀, φ0, p₀, e₀] # χ₀, ϕ₀, p₀, e₀

e₀ = 0.16384305990318113  # SXS reference eccentricity
r₀ = 6.484530872637143*2     # Actual SXS initial separation  
χ₀ = 0                     # Confirmed at periapsis
p₀ = r₀ * (1 + e₀)         # = 6.485 * 1.164 = 7.55


tspan = (t₀, 5.569f3)
datasize = 1000
tsteps = range(tspan[1], tspan[2], length = datasize) 
dt_data = tsteps[2] - tsteps[1]
dt = 10.0
num_optimization_increments = 100

x, y = file2trajectory(tsteps,"./input/trajectoryA_eccentric.txt")
x2, y2 = file2trajectory(tsteps,"./input/trajectoryB_eccentric.txt")
waveform_real = file2waveform(tsteps,"./input/waveform_real_eccentric.txt")
waveform_imag = file2waveform(tsteps,"./input/waveform_imag_eccentric.txt")

# φ₀ = atan(y[1], x[1])
# u0_pe_coordinates = Float32[χ₀, φ₀, p₀, e₀]

# p = u0_pe_coordinates[3]
# e = u0_pe_coordinates[4]  
# E0, L0 = pe_2_EL(p, e)[2:3]

# u0 = Float32[0, r₀, pi/2, φ₀, E0, 0, 0, -L0, 0]
u0_pe_coordinates = Float32[0.0, 0.0, 6.484530872637143*2+2.1, 0.16384305990318113]  # χ₀ = 0, φ₀ = 0 (periapsis)
p = u0_pe_coordinates[3]
e = u0_pe_coordinates[4]
E0, L0 = pe_2_EL(p, e)[2:3]
r0 = p / (1 + e)  # Correct periapsis radius = 12.368763

u0 = [0, r0, pi/2, 0, E0, 0, 0, -L0, 0]

plt = plot(tsteps, waveform_real,
           markershape=:circle, markeralpha = 0.25,
           linewidth = 2, alpha = 0.5,
           label="waveform data (Re)")

## Define neural network models (seems to help to build up the NNs with a bunch of ansatz functions)
# Define the Lux neural networks

#  # Neural network setup 
 NN = Chain(
    Dense(1, 4, tanh), # Learns correction term in terms of 1 parameter: r
    Dense(4, 4, tanh),
    Dense(4, 1),
)

# Initialize parameters
rng = MersenneTwister(222)
NN_params, NN_state = Lux.setup(rng, NN)
precision = Float64
NN_params = Lux.fmap(x -> precision.(x), NN_params)

for (i, layer) in enumerate(NN_params)
    if ~isempty(layer)
        if i == length(NN_params)  # Final layer
            layer.weight .= 0.01 * randn(rng, eltype(layer.weight), size(layer.weight))
            layer.bias .= 0.0  # Force output near 0
        else  # Hidden layers
            layer.weight .= 0.1 * randn(rng, eltype(layer.weight), size(layer.weight))
            layer.bias .= 0.1 * randn(rng, eltype(layer.bias), size(layer.bias))
        end
    end
end

# Now, create adapter functions that match the calling pattern expected by AbstractNROrbitModel
function NN_adapter(u, params)
    features = [u[2]]  # t, r, θ, φ, p_t, p_r, p_θ, p_ϕ
    output, _ = NN(features, params, NN_state)
    return output
end

NN_params = ComponentArray(params_Schwarzschild = NN_params)

# Update ODE_model to use ComponentArray and adapters
function ODE_model(du, u, p, t)
    du = AbstractNROrbitModel(du, u, model_params, t,
                          NN=NN_adapter, 
                          NN_params=p.params_Schwarzschild)
    return du
end

prob_nn = ODEProblem(ODE_model, u0, tspan, NN_params)
soln_nn = Array(solve(prob_nn, RK4(), saveat = tsteps, dt = dt, adaptive=false, verbose=false))
waveform_nn_real, waveform_nn_imag = compute_waveform(dt_data, soln_nn, 1.0; coorbital=false)

plot!(plt, tsteps, waveform_nn_real,
           markershape=:circle, markeralpha = 0.25,
           linewidth = 2, alpha = 0.5,
           label="waveform NN (Re)")
display(plt)

# # ## Define the objective function
function loss(NN_params; saveat=tsteps)
    tspan = (saveat[1],saveat[end])
    pred_soln = solve(remake(prob_nn, p = NN_params, tspan=tspan), RK4(),
                            saveat = saveat, dt = dt, adaptive=false, verbose = false, sensealg=BacksolveAdjoint(checkpointing=true))
    pred_waveform_real, pred_waveform_imag = compute_waveform(dt_data, pred_soln, mass_ratio)

    N = length(pred_waveform_real)
    loss = ( sum(abs2, waveform_real[1:N] .- pred_waveform_real) )
    return loss, pred_waveform_real, pred_waveform_imag, pred_soln
end

# # # Test loss function
loss(NN_params)

losses = []

# # ## Train with BFGS (gives best results because the Newtonian model seems to give a very good initial guess)
optimization_increments = [collect(40:10:num_optimization_increments-10)..., num_optimization_increments-3, num_optimization_increments-2, num_optimization_increments-1,  num_optimization_increments]
@showprogress for i in optimization_increments
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
    
        plt1 = plot(tsteps[1:N], waveform_real[1:N], 
                    markershape=:circle, markeralpha = 0.25,
                    linewidth = 2, alpha = 0.5, label = "True Gravitational Wave")
        plot!(plt1, tsteps[1:N], pred_waveform_real,
              markershape=:circle, markeralpha = 0.25,
              linewidth = 2, alpha = 0.5, label = "Predicted Gravitational Wave")
            
        # display(plt1)
        # N = 10
        pred_orbit_relative = soln2orbit(pred_soln)
        pred_orbit1, pred_orbit2 = one2two(pred_orbit_relative, 1, 1)
    
        plt2 = plot(x[1:N], y[1:N],
                    linewidth = 2, alpha = 1, 
                    label = "orbit data", aspect_ratio=:equal)
        plot!(plt2, pred_orbit1[1,1:N], pred_orbit1[2,1:N],
              linewidth = 2, alpha = 0.8, color = "red",
              label = "orbit NN", ls=:dash, aspect_ratio=:equal)

              # display(plt2)
        # χ = pred_soln[1,:]
        # ϕ = pred_soln[2,:]
        # p_vals = pred_soln[3,:]
        # e_vals = pred_soln[4,:]
    
        # plt3 = plot(tsteps[1:N], p_vals, linewidth = 2, alpha = 0.5, label="p", legend=:best)
        # plt4 = plot(tsteps[1:N], e_vals, linewidth = 2, alpha = 0.5, label="e", legend=:topleft)
        # plt5 = plot(tsteps[1:N], χ, linewidth = 2, alpha = 0.5, label="χ", legend=:topright)
        # plt6 = plot(tsteps[1:N], ϕ, linewidth = 2, alpha = 0.5, label="ϕ", legend=:best)

        layout = @layout [
            a; b
            ]
    
        full_plot = plot(plt1, plt2, layout=layout, size = (600, 400))
        display(plot(full_plot))
        
        return false
    end

    global NN_params = Lux.fmap(x -> x .+ Float64(1e-5)*randn(eltype(x), size(x)), NN_params) 

    p_init = ComponentArray(NN_params)
    adtype = Optimization.AutoZygote()
    optf = Optimization.OptimizationFunction((x, p) -> scalar_loss(x), adtype)
    optprob = Optimization.OptimizationProblem(optf, p_init)

    if i < 80
        local res = Optimization.solve(optprob, Optim.BFGS(; initial_stepnorm=5e-2, linesearch = LineSearches.BackTracking()), callback=opt_callback, maxiters = 50, allow_f_increases=true)
    elseif i == 80
        local res = Optimization.solve(optprob, Optim.BFGS(; initial_stepnorm=1e-2, linesearch = LineSearches.BackTracking()), callback=opt_callback, maxiters = 100, allow_f_increases=true)
    elseif i == optimization_increments[end-3]  # 97
        local res = Optimization.solve(optprob, Optim.BFGS(; initial_stepnorm=1e-3, linesearch = LineSearches.BackTracking()), callback=opt_callback, maxiters = 200, allow_f_increases=true)
    elseif i == optimization_increments[end-2]  # 98
        local res = Optimization.solve(optprob, Optim.BFGS(; initial_stepnorm=1e-4, linesearch = LineSearches.BackTracking()), callback=opt_callback, maxiters = 300, allow_f_increases=true)
    elseif i == optimization_increments[end-1]  # 99
        local res = Optimization.solve(optprob, Optim.BFGS(; initial_stepnorm=1e-5, linesearch = LineSearches.BackTracking()), callback=opt_callback, maxiters = 400, allow_f_increases=true)
    else  # 100
        local res = Optimization.solve(optprob, Optim.BFGS(; initial_stepnorm=1e-6, linesearch = LineSearches.BackTracking()), callback=opt_callback, maxiters = 500, allow_f_increases=true)
    end

    global NN_params = res.minimizer
    # local plt = plot(losses, yaxis=:log, linewidth = 2, xlabel = "Iteration", ylabel = "Objective value", legend = false)
    # display(plot(plt))
end

# ## plot final solutions
optimized_solution = solve(remake(prob_nn, p = NN_params, tspan=tspan), RK4(), saveat = tsteps, dt = dt, adaptive=false)

pred_waveform_real, pred_waveform_imag = compute_waveform(dt_data, optimized_solution, mass_ratio)

pred_orbit = soln2orbit(optimized_solution)
pred_orbit1_nn, pred_orbit2_nn = one2two(pred_orbit, 1, 1)

plt1 = plot(x, y, linewidth = 2, label = "truth")
# plot!(plt1, pred_orbit1_nn[1,:], pred_orbit2_nn[2,:], linestyle = :dash, linewidth = 2, label = "prediction")
# plot!(plt1, pred_orbit1_nn[1,:], pred_orbit1_nn[2,:], linestyle = :dash, linewidth = 1, label = "pred orbit 1")
plot!(plt1, pred_orbit1_nn[1,:], pred_orbit1_nn[2,:], linestyle = :dash, linewidth = 2, label = "pred orbit 1", aspect_ratio=:equal)
plt2 = plot(losses, yaxis=:log, linewidth = 2, xlabel = "Iteration", ylabel = "Objective value", legend = false)
plt = plot(plt1, plt2, layout = (1,2))
display(plot(plt))