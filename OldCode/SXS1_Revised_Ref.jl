```

    Inverse problem script to learn ODE model from SXS waveform data

```
cd(@__DIR__)
using Pkg; Pkg.activate("."); Pkg.instantiate()
# using DifferentialEquations
# using DiffEqFlux
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
# using DiffEqSensitivity
using ComponentArrays
using Optimization
using OptimizationOptimisers
using OrdinaryDiffEq
gr()

include("utils.jl")
include("models.jl")

## Define the experiment parameters
u0 = Float32[pi, 0.0, 12.37, 1.0f-4] # χ₀ (mean anomaly), ϕ₀ (true anomaly), p₀ (semi-latus rectum), e₀ (eccentricity)
tspan = (5.20913f3, 6.78f3) # why this time span, specifically?
datasize = 1000
tsteps = range(tspan[1], tspan[2], length = datasize) 
dt_data = tsteps[2] - tsteps[1]
dt = 10.0
num_optimization_increments = 100
mass_ratio = 1.0
model_params = [mass_ratio] # Just the mass ratio, at least for now
mass1 = mass_ratio/(1.0+mass_ratio)
mass2 = 1.0/(1.0+mass_ratio)

# Gather waveform data
x, y = file2trajectory(tsteps,"./input/trajectoryA.txt")
x2, y2 = file2trajectory(tsteps,"./input/trajectoryB.txt")
waveform_real = file2waveform(tsteps,"./input/waveform_real.txt") 

plt = plot(tsteps, waveform_real,
           markershape=:circle, markeralpha = 0.25,
           linewidth = 2, alpha = 0.5,
           label="waveform data (Re)")

## Define neural network models (seems to help to build up the NNs with a bunch of ansatz functions)
# Define the Lux neural networks

# two separate neural networks: one for the angular parameters (chi, phi)
# and one for the parameters (p, e)
NN_chiphi = Chain(
    x -> [cos(x[1]), sin(x[1]), 1/abs(x[3]),1/sqrt(abs(x[3])),sqrt(abs(x[3])),x[3],sqrt(abs(x[3]))^3,x[3]^2,x[4],x[4]^2, x[3]*x[4]], # these are all different ansatz functions that the neural network can try out for chi and phi?
    Dense(11, 32, tanh),
    Dense(32, 2)
)

NN_pe = Chain(
    x -> [1/sqrt(abs(x[3]))^3, 1/abs(x[3]), 1/sqrt(abs(x[3])), sqrt(abs(x[3])), 
          x[3], sqrt(abs(x[3]))^3, x[3]^2, x[4], x[4]^2, x[3]*x[4]],
    Dense(10, 32, tanh),
    Dense(32, 2)
)

# Initialize parameters
rng = Random.default_rng()
rng2 = Random.default_rng()
NN_chiphi_params, NN_st = Lux.setup(rng, NN_chiphi)
NN_pe_params, NN_st2 = Lux.setup(rng2, NN_pe)

# Zero out parameters
for layer in NN_chiphi_params
    if ~isempty(layer)
        layer.weight .*= 0
        layer.bias .*= 0
    end
end

for layer in NN_pe_params
    if ~isempty(layer)
        layer.weight .*= 0
        layer.bias .*= 0
    end
end

NN_chiphi_params = Lux.fmap(x -> Float64.(x), NN_chiphi_params)
NN_pe_params = Lux.fmap(x -> Float64.(x), NN_pe_params)

# Now, create adapter functions that match the calling pattern expected by AbstractNROrbitModel
function chiphi_adapter(u, params)
    features = [cos(u[1]), sin(u[1]), 1/abs(u[3]),1/sqrt(abs(u[3])),sqrt(abs(u[3])),u[3],sqrt(abs(u[3]))^3,u[3]^2,u[4],u[4]^2, u[3]*u[4]]
    output, _ = NN_chiphi(features, params, NN_st)
    return output
end

function pe_adapter(u, params)
    features = [1/sqrt(abs(u[3]))^3, 1/abs(u[3]), 1/sqrt(abs(u[3])), sqrt(abs(u[3])), 
                u[3], sqrt(abs(u[3]))^3, u[3]^2, u[4], u[4]^2, u[3]*u[4]]
    output, _ = NN_pe(features, params, NN_st2)
    return output
end

# Create a structured parameter container
NN_params = ComponentArray(chiphi = NN_chiphi_params, pe = NN_pe_params)
l1 = length(NN_chiphi_params)

# Update ODE_model to use ComponentArray and adapters
function ODE_model(u, p, t)
    du = AbstractNROrbitModel(u, model_params, t,
                          NN_chiphi=chiphi_adapter, 
                          NN_chiphi_params=p.chiphi,
                          NN_pe=pe_adapter, 
                          NN_pe_params=p.pe)
    return du
end

prob_nn = ODEProblem(ODE_model, u0, tspan, NN_params)
soln_nn = Array(solve(prob_nn, RK4(), saveat = tsteps, dt = dt, adaptive=false))
waveform_nn_real, waveform_nn_imag = compute_waveform(dt_data, soln_nn, mass_ratio)

plot!(plt, tsteps, waveform_nn_real,
           markershape=:circle, markeralpha = 0.25,
           linewidth = 2, alpha = 0.5,
           label="waveform NN (Re)")
display(plt)

## Define the objective function
function loss(NN_params; saveat=tsteps)
    tspan = (saveat[1],saveat[end])
    pred_soln = solve(remake(prob_nn, p = NN_params, tspan=tspan), RK4(),
                            saveat = saveat, dt = dt, adaptive=false, sensealg=BacksolveAdjoint(checkpointing=true))
    pred_waveform_real, pred_waveform_imag = compute_waveform(dt_data, pred_soln, mass_ratio)
    p = pred_soln[3,:]
    e = pred_soln[4,:]

    N = length(pred_waveform_real)
    loss = 1/N * (
            sum(abs2, waveform_real[1:N] .- pred_waveform_real)
           + 1.0f3*sum(abs2, max.(d_dt(p,dt_data),0.0))
           + 1.0f2*sum(abs2, max.(d2_dt2(p,dt_data),0.0))
           + 1.0f1*sum(abs2, max.(-e,0.0))
           + 1.0f0*sum(abs2, max.(e[p .>= 6 + 2*e[1]] .- e[1],0.0))
           )
           + 1f-3*sum(abs2, NN_params)

    return loss, pred_waveform_real, pred_waveform_imag, pred_soln
end

# Test loss function
loss(NN_params)

losses = []

## Train with BFGS (gives best results because the Newtonian model seems to give a very good initial guess)
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
    
        plt1 = plot(tsteps[1:N], waveform_real[1:N], 
                    markershape=:circle, markeralpha = 0.25,
                    linewidth = 2, alpha = 0.5)
        plot!(plt1, tsteps[1:N], pred_waveform_real,
              markershape=:circle, markeralpha = 0.25,
              linewidth = 2, alpha = 0.5)
            
        pred_orbit = soln2orbit(pred_soln)
        orbit_nn1, orbit_nn2 = one2two(pred_orbit, mass1, mass2)
    
        plt2 = plot(x[1:N], y[1:N],
                    linewidth = 2, alpha = 0.5, 
                    label = "orbit data")
    
        plot!(plt2, orbit_nn1[1,:], orbit_nn1[2,:],
              linewidth = 2, alpha = 0.5, 
              label = "orbit NN")
        
        χ = pred_soln[1,:]
        ϕ = pred_soln[2,:]
        p_vals = pred_soln[3,:]
        e_vals = pred_soln[4,:]
    
        plt3 = plot(tsteps[1:N], p_vals, linewidth = 2, alpha = 0.5, label="p", legend=:best)
        plt4 = plot(tsteps[1:N], e_vals, linewidth = 2, alpha = 0.5, label="e", legend=:topleft)
        plt5 = plot(tsteps[1:N], χ, linewidth = 2, alpha = 0.5, label="χ", legend=:topright)
        plt6 = plot(tsteps[1:N], ϕ, linewidth = 2, alpha = 0.5, label="ϕ", legend=:best)

        layout = @layout [
            a; b; [c{0.5w} d{0.5w}]; [e{0.5w} f{0.5w}]
            ]
    
        full_plot = plot(plt1, plt2, plt3, plt4, plt5, plt6, layout=layout, size = (600, 400))
        display(plot(full_plot))
        
        return false
    end

    global NN_params = Lux.fmap(x -> x .+ Float64(1e-5)*randn(eltype(x), size(x)), NN_params) 

    p_init = ComponentArray(NN_params)
    adtype = Optimization.AutoZygote()
    optf = Optimization.OptimizationFunction((x, p) -> scalar_loss(x), adtype)
    optprob = Optimization.OptimizationProblem(optf, p_init)

    if i < optimization_increments[end-1]
        local res = Optimization.solve(optprob, Optim.BFGS(; initial_stepnorm=0.001, linesearch = LineSearches.BackTracking()), callback=opt_callback, maxiters = 100, allow_f_increases=true)
    elseif i == optimization_increments[end-3]
        local res = Optimization.solve(optprob, Optim.BFGS(; initial_stepnorm=0.0005, linesearch = LineSearches.BackTracking()), callback=opt_callback, maxiters = 200, allow_f_increases=true)
    elseif i == optimization_increments[end-2]
        local res = Optimization.solve(optprob, Optim.BFGS(; initial_stepnorm=0.0001, linesearch = LineSearches.BackTracking()), callback=opt_callback, maxiters = 300, allow_f_increases=true)
    elseif i == optimization_increments[end-1]
        local res = Optimization.solve(optprob, Optim.BFGS(; initial_stepnorm=0.00001, linesearch = LineSearches.BackTracking()), callback=opt_callback, maxiters = 400, allow_f_increases=true)
    else
        local res = Optimization.solve(optprob, Optim.BFGS(; initial_stepnorm=0.000001, linesearch = LineSearches.BackTracking()), callback=opt_callback, maxiters = 500, allow_f_increases=true)
    end
    global NN_params = res.minimizer
    local plt = plot(losses, yaxis=:log, linewidth = 2, xlabel = "Iteration", ylabel = "Objective value", legend = false)
    display(plot(plt))
end

## plot final solutions
optimized_solution = solve(remake(prob_nn, p = NN_params, tspan=tspan), RK4(), saveat = tsteps, dt = dt, adaptive=false)
x, y = file2trajectory(tsteps,"./input/trajectoryA.txt")
x2, y2 = file2trajectory(tsteps,"./input/trajectoryB.txt")
pred_waveform_real, pred_waveform_imag = compute_waveform(dt_data, optimized_solution, mass_ratio)

pred_orbit = soln2orbit(optimized_solution)
orbit_nn1, orbit_nn2 = one2two(pred_orbit, mass1, mass2)
plt1 = plot(x, y, linewidth = 2, label = "truth")
plot!(plt1, orbit_nn1[1,:], orbit_nn1[2,:], linestyle = :dash, linewidth = 2, label = "prediction")
plt2 = plot(losses, yaxis=:log, linewidth = 2, xlabel = "Iteration", ylabel = "Objective value", legend = false)
plt = plot(plt1, plt2, layout = (1,2))
display(plot(plt))

## save data
save = false

df_solution = DataFrame(time = tsteps[1:length(optimized_solution)],
                        χ = optimized_solution[1,:],
                        ϕ = optimized_solution[2,:],
                        p = optimized_solution[3,:],
                        e = optimized_solution[4,:])

df_trajectories = DataFrame(time = tsteps,
                         true_orbit_x1 = x,
                         true_orbit_y1 = y,
                         true_orbit_x2 = x2,
                         true_orbit_y2 = y2,
                         pred_orbit_x1 = orbit_nn1[1,:],
                         pred_orbit_y1 = orbit_nn1[2,:],
                         pred_orbit_x2 = orbit_nn2[1,:],
                         pred_orbit_y2 = orbit_nn2[2,:])

df_waveforms = DataFrame(time = tsteps,
                        true_waveform_real = waveform_real,
                        true_waveform_imag = 0*waveform_real, # not recorded
                        pred_waveform_real = pred_waveform_real,
                        pred_waveform_imag = pred_waveform_imag,
                        error_real = waveform_real .- pred_waveform_real,
                        error_imag = 0*waveform_real) # not recorded

df_losses = DataFrame(losses = losses)
df_parameters = DataFrame(parameters = NN_params)

if save
    CSV.write("./output/SXS1_solution.csv", df_solution)
    CSV.write("./output/SXS1_trajectories.csv", df_trajectories)
    CSV.write("./output/SXS1_waveforms.csv", df_waveforms)
    CSV.write("./output/SXS1_losses.csv", df_losses)
    CSV.write("./output/SXS1_parameters.csv", df_parameters)
end
