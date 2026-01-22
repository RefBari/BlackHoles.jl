```

    Inverse problem script to recover geodesic equations from waveform data

```
cd(@__DIR__)
using Pkg; Pkg.activate("."); Pkg.instantiate()
using OrdinaryDiffEq
using DiffEqFlux
using Optim
using LineSearches
using DataFrames
using CSV
using Plots
using Lux
using Random
using SciMLBase
using OptimizationOptimJL
using ComponentArrays
using Optimization
using OptimizationOptimisers

gr()

include("utils.jl")
include("models.jl")

## Define the experimental parameters
mass_ratio = 0.0
u0 = Float64[pi, 0.0]
datasize = 250
tspan = (0.0f0, 6.0f4)
tsteps = range(tspan[1], tspan[2], length = datasize)
dt_data = tsteps[2] - tsteps[1]
dt = 100.0
model_params = [100.0, 1.0, 0.5] # p, M, e

# Generate waveform data
prob = ODEProblem(RelativisticOrbitModel, u0, tspan, model_params)
soln = Array(solve(prob, RK4(), saveat = tsteps, dt = dt, adaptive=false))
waveform = compute_waveform(dt_data, soln, mass_ratio, model_params)[1]

plt = plot(tsteps, waveform,
           markershape=:circle, markeralpha = 0.25,
           linewidth = 2, alpha = 0.5,
           label="waveform data")

## Define neural network model
NN = Chain(x -> [cos(x[1])],
                Dense(1, 32, cos),
                Dense(32, 32, cos),
                Dense(32, 2))
rng = Random.default_rng()
NN_params, NN_st = Lux.setup(rng, NN)

for layer in NN_params
    if ~isempty(layer)
        layer.weight .*= 0
        layer.bias .*= 0
    end
end

NN_params = Lux.fmap(x -> Float64.(x), NN_params)

function ODE_model(u, NN_params, t)
    du = AbstractNNOrbitModel(u, model_params, t, NN=NN, NN_params=NN_params, NN_st=NN_st)
    return du
end

prob_nn = ODEProblem(ODE_model, u0, tspan, NN_params)
soln_nn = Array(solve(prob_nn, RK4(), saveat = tsteps, dt = dt, adaptive=false))
waveform_nn = compute_waveform(dt_data, soln_nn, mass_ratio, model_params)[1]

plot!(plt, tsteps, waveform_nn,
           markershape=:circle, markeralpha = 0.25,
           linewidth = 2, alpha = 0.5,
           label="waveform NN")
display(plt)

## Define objective function

function predict_waveform(NN_params)
    _prob = remake(prob_nn, p = NN_params)
    soln_nn = Array(solve(_prob, RK4(), saveat=tsteps, dt = dt, adaptive=false))
    return compute_waveform(dt_data, soln_nn, mass_ratio, model_params)[1]
end

function loss_with_prediction(NN_params, p=nothing)
    pred_waveform = predict_waveform(NN_params)
    loss = sum(abs2, waveform .- pred_waveform)
    return loss, pred_waveform
end

function loss(NN_params)
    pred_waveform = predict_waveform(NN_params)
    return sum(abs2, waveform .- pred_waveform)
end

# Test loss function
loss_with_prediction(NN_params)
losses = []

function callback(state, l; doplot = true)
    push!(losses, l) # after calculating the loss, add it to the losses array
    display(l) #print the loss
    if doplot    
    # Plot current prediction against data.
        pred_waveform = predict_waveform(state.u)
        plt = plot(tsteps, waveform,   
            markershape = :circle, markeralpha = 0.25,
            linewidth = 2, alpha = 0.5,
            label = "wform data (h22)", legend = :topleft)
        plot!(plt, tsteps, pred_waveform,
            markershape = :circle, markeralpha = 0.25,
            linewidth = 2, alpha = 0.5,
            label = "wform NN")
        display(plt)
    end
    # Tell the optimizer to continue (return false).
    return false
end

## Train with BFGS (gives best results because the Newtonian model seems to give a very good initial guess)
NN_params = Lux.fmap(x -> x .+ Float64(1e-3)*randn(eltype(x), size(x)), NN_params) 
p_init = ComponentArray(NN_params)
 
# Literally straight from the docs, except I fiddled aroudn with the parameters
adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x, p) -> loss(x), adtype)
optprob = Optimization.OptimizationProblem(optf, p_init)

res = Optimization.solve(optprob, Optim.BFGS(; initial_stepnorm = 0.05), callback=callback, maxiters=50)

final_params = res.u
optimization_info = res.original;
display(optimization_info)

plt = plot(losses, label = "loss", xlabel = "iteration", ylabel = "loss", yaxis=:log)

## Plot learned solutions

reference_solution = solve(remake(prob, p = model_params, saveat = tsteps, tspan=tspan),
                            RK4(), dt = dt, adaptive=false)
optimized_solution = solve(remake(prob_nn, p = res.minimizer, saveat = tsteps, tspan=tspan),
                            RK4(), dt = dt, adaptive=false)
Newtonian_prob = ODEProblem(NewtonianOrbitModel, u0, tspan, model_params)
Newtonian_solution = solve(remake(Newtonian_prob, p = model_params, saveat = tsteps, tspan=tspan),
                            RK4(), dt = dt, adaptive=false)
true_orbit = soln2orbit(reference_solution, model_params)
pred_orbit = soln2orbit(optimized_solution, model_params)
Newt_orbit = soln2orbit(Newtonian_solution, model_params)

df_learned_trajectories = DataFrame(time = tsteps,
                       true_orbit_x = true_orbit[1,:],
                       true_orbit_y = true_orbit[2,:],
                       pred_orbit_x = pred_orbit[1,:],
                       pred_orbit_y = pred_orbit[2,:],
                       Newt_orbit_x = Newt_orbit[1,:],
                       Newt_orbit_y = Newt_orbit[2,:])

true_waveform = compute_waveform(dt_data, reference_solution, mass_ratio, model_params)[1]
pred_waveform = compute_waveform(dt_data, optimized_solution, mass_ratio, model_params)[1]
Newt_waveform = compute_waveform(dt_data, Newtonian_solution, mass_ratio, model_params)[1]

df_learned_waveforms = DataFrame(time = tsteps,
                                true_waveform = true_waveform,
                                pred_waveform = pred_waveform,
                                Newt_waveform = Newt_waveform)

## Plot extrapolated solutions
factor=5

extended_tspan = (tspan[1], factor*tspan[2])
extended_tsteps = range(tspan[1], factor*tspan[2], length = factor*datasize)
reference_solution = solve(remake(prob, p = model_params, saveat = extended_tsteps, tspan=extended_tspan),
                            RK4(), dt = dt, adaptive=false)
optimized_solution = solve(remake(prob_nn, p = res.minimizer, saveat = extended_tsteps, tspan=extended_tspan),
                            RK4(), dt = dt, adaptive=false)
Newtonian_prob = ODEProblem(NewtonianOrbitModel, u0, tspan, model_params)
Newtonian_solution = solve(remake(Newtonian_prob, p = model_params, saveat = extended_tsteps, tspan=extended_tspan),
                            RK4(), dt = dt, adaptive=false)
true_orbit = soln2orbit(reference_solution, model_params)
pred_orbit = soln2orbit(optimized_solution, model_params)
Newt_orbit = soln2orbit(Newtonian_solution, model_params)
plt = plot(true_orbit[1,:], true_orbit[2,:], linewidth = 2, label = "truth")
plot!(plt, pred_orbit[1,:], pred_orbit[2,:], linestyle = :dash, linewidth = 2, label = "prediction")
plot!(plt, Newt_orbit[1,:], Newt_orbit[2,:], linewidth = 2, label = "Newtonian")
display(plot(plt))

true_waveform = compute_waveform(dt_data, reference_solution, mass_ratio, model_params)[1]
pred_waveform = compute_waveform(dt_data, optimized_solution, mass_ratio, model_params)[1]
Newt_waveform = compute_waveform(dt_data, Newtonian_solution, mass_ratio, model_params)[1]

df_predicted_trajectories = DataFrame(time = extended_tsteps,
                         true_orbit_x = true_orbit[1,:],
                         true_orbit_y = true_orbit[2,:],
                         pred_orbit_x = pred_orbit[1,:],
                         pred_orbit_y = pred_orbit[2,:],
                         Newt_orbit_x = Newt_orbit[1,:],
                         Newt_orbit_y = Newt_orbit[2,:])

df_predicted_waveforms = DataFrame(time = extended_tsteps,
                                true_waveform = true_waveform,
                                pred_waveform = pred_waveform,
                                Newt_waveform = Newt_waveform)
