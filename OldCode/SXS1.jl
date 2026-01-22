```

    Inverse problem script to learn ODE model from SXS waveform data

```
cd(@__DIR__)
using Pkg; Pkg.activate("."); Pkg.instantiate()
using OrdinaryDiffEq
using Optim
using LineSearches
using DiffEqFlux
using DiffEqSensitivity
using Plots
using DataFrames
using CSV
gr()

include("utils.jl")
include("models.jl")

## Define the experiment parameters
u0 = Float32[pi, 0.0, 12.37, 1.0f-4] # χ₀, ϕ₀, p₀, e₀
tspan = (5.20913f3, 6.78f3)
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
waveform_real = file2waveform(tsteps,"./input/waveform_real.txt")
x2, y2 = file2trajectory(tsteps,"./input/trajectoryB.txt")

plt = plot(tsteps, waveform_real,
           markershape=:circle, markeralpha = 0.25,
           linewidth = 2, alpha = 0.5,
           label="waveform data (Re)")

## Define neural network models (seems to help to build up the NNs with a bunch of ansatz functions)
NN_chiphi = FastChain((x, NN_chiphi_params) -> [cos(x[1]),1/abs(x[3]),1/sqrt(abs(x[3])),sqrt(abs(x[3])),x[3],sqrt(abs(x[3]))^3,x[3]^2,x[4],x[4]^2],
                FastDense(9, 32, tanh),
                FastDense(32, 2))
NN_chiphi_params = initial_params(NN_chiphi) .* 0
NN_pe = FastChain((x, NN_pe_params) -> [1/sqrt(abs(x[3]))^3,1/abs(x[3]),1/sqrt(abs(x[3])),sqrt(abs(x[3])),x[3],sqrt(abs(x[3]))^3,x[3]^2,x[4],x[4]^2,x[3]*x[4]],
                FastDense(10, 32, tanh),
                FastDense(32, 2))
NN_pe_params = initial_params(NN_pe) .* 0
NN_params = vcat(NN_chiphi_params,NN_pe_params)
l1 = length(NN_chiphi_params)

function ODE_model(u, NN_params, t)
    NN_params1 = NN_params[1:l1]
    NN_params2 = NN_params[l1+1:end]
    du = AbstractNROrbitModel(u, model_params, t,
                              NN_chiphi=NN_chiphi, NN_chiphi_params=NN_params1,
                              NN_pe=NN_pe, NN_pe_params=NN_params2)
    return du
end

prob_nn = ODEProblem(ODE_model, u0, tspan, NN_params)
soln_nn = Array(solve(prob_nn, RK4(), u0 = u0, p = NN_params, saveat = tsteps, dt = dt, adaptive=false))
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
           + 1.0f-1*sum(abs2, NN_params)

    return loss, pred_waveform_real, pred_waveform_imag, pred_soln
end

# Test loss function
loss(NN_params)

const losses = []

callback(θ,l,pred_waveform_real, pred_waveform_imag, pred_soln; doplot = true) = begin
    push!(losses, l)
    display(l)
    plt1 = plot(tsteps, waveform_real,
               markershape=:circle, markeralpha = 0.25,
               linewidth = 2, alpha = 0.5,
               label="wform data (Re)", legend=:topleft)
    N = length(pred_waveform_real)
    plot!(plt1, tsteps[1:N], pred_waveform_real,
               markershape=:circle, markeralpha = 0.25,
               linewidth = 2, alpha = 0.5,
               label="wform NN (Re)")

    pred_orbit = soln2orbit(pred_soln)
    orbit_nn1, orbit_nn2 = one2two(pred_orbit, mass1, mass2)
    N = size(orbit_nn1,2)
    plt2 = plot(x[1:N],y[1:N],
               linewidth = 2, alpha = 0.5,
               label="orbit data")
    plot!(plt2, orbit_nn1[1,1:end], orbit_nn1[2,1:end],
               linewidth = 2, alpha = 0.5,
               label="orbit NN")
    χ = pred_soln[1,:]
    ϕ = pred_soln[2,:]
    p = pred_soln[3,:]
    e = pred_soln[4,:]
    plt3 = plot(tsteps[1:N], p, linewidth = 2, alpha = 0.5, label="p", legend=:best)
    plt4 = plot(tsteps[1:N], e, linewidth = 2, alpha = 0.5, label="e", legend=:topleft)
    l = @layout [
        a; [b{0.6w} [c;d]]
        ]
    plt = plot(plt1, plt2, plt3, plt4, layout=l)

    if doplot
        display(plot(plt))
    end
    # Tell sciml_train to not halt the optimization. If return true, then
    # optimization stops.
    return false
end

## Train with BFGS (gives best results because the Newtonian model seems to give a very good initial guess)
optimization_increments = [collect(40:10:num_optimization_increments-5)..., num_optimization_increments-1,  num_optimization_increments]
for i in optimization_increments
    println("optimization increment :: ", i, " of ", num_optimization_increments)
    tsteps_increment = tsteps[tsteps .<= tspan[1]+i*(tspan[2]-tspan[1])/num_optimization_increments]
    tmp_loss(p) = loss(p,saveat=tsteps_increment)
    global NN_params = NN_params + Float64(1e-6)*randn(eltype(NN_params), size(NN_params))
    if i < optimization_increments[end-1]
        local res = DiffEqFlux.sciml_train(tmp_loss, NN_params, BFGS(initial_stepnorm=0.001, linesearch = LineSearches.BackTracking()), cb=callback, maxiters = 50)
    elseif i == optimization_increments[end-1]
        local res = DiffEqFlux.sciml_train(tmp_loss, NN_params, BFGS(initial_stepnorm=0.001, linesearch = LineSearches.BackTracking()), cb=callback, maxiters = 100, allow_f_increases=true)
    else
        local res = DiffEqFlux.sciml_train(tmp_loss, NN_params, BFGS(initial_stepnorm=0.001, linesearch = LineSearches.BackTracking()), cb=callback, maxiters = 250, allow_f_increases=true)
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
