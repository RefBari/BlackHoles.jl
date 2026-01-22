```

    Plot SNR vs. MSE in the gravitational waveform

```
cd(@__DIR__)
using Pkg; Pkg.activate("."); Pkg.instantiate()
using Random
using OrdinaryDiffEq
using Optim
using LineSearches
using DiffEqFlux
using DiffEqSensitivity
using Plots
using DataFrames
using CSV
import Printf
gr()

include("utils.jl")
include("models.jl")

## Define the experiment parameters (Same as SXS2.jl)
df = CSV.read("./output/SXS2_parameters.csv", DataFrame)
θ = vec(convert(Matrix{Float64},df))

mass_ratio = 1.0
model_params = [mass_ratio] # Just the mass ratio, at least for now
mass1 = 1.0/(1.0+mass_ratio)
mass2 = mass_ratio/(1.0+mass_ratio)

e₀ = θ[1]
x₀ = 6.484530581221468
t₀ = 5263.7607
χ₀ = θ[2]
p₀ = x₀ * (1+e₀*cos(χ₀)) / mass1
u0 = Float64[χ₀, 0.0, p₀, e₀] # χ₀, ϕ₀, p₀, e₀
tspan = (t₀, 6.769f3)
datasize = 1000
tsteps = range(tspan[1], tspan[2], length = datasize)
dt_data = tsteps[2] - tsteps[1]
dt = 10.0
num_optimization_increments = 100

NN_params = θ[3:end]

# Gather waveform data
x, y = file2trajectory(tsteps,"./input/trajectoryA_eccentric.txt")
x2, y2 = file2trajectory(tsteps,"./input/trajectoryB_eccentric.txt")
waveform_real = file2waveform(tsteps,"./input/waveform_real_eccentric.txt")
waveform_imag = file2waveform(tsteps,"./input/waveform_imag_eccentric.txt")

plt = plot(tsteps, waveform_real,
           markershape=:circle, markeralpha = 0.25,
           linewidth = 2, alpha = 0.5,
           label="waveform data")

# Add noise to the waveform (optional)
SNR = 0.0
if SNR > 0
    println("Generate noisy waveform")
    Random.seed!(8000);
    waveform_real = set_SNR(waveform_real, dt, SNR)
end

plot!(plt, tsteps, waveform_real,
           markershape=:circle, markeralpha = 0.25,
           linewidth = 2, alpha = 0.5,
           label="noisy waveform data")

## Define neural network models
NN_chiphi = FastChain((x, NN_chiphi_params) -> [cos(x[1]),1/abs(x[3]),1/sqrt(abs(x[3])),sqrt(abs(x[3])),x[3],sqrt(abs(x[3]))^3,x[3]^2,x[4],x[4]^2],
                FastDense(9, 32, tanh),
                FastDense(32, 2))
NN_pe = FastChain((x, NN_pe_params) -> [1/sqrt(abs(x[3]))^3,1/abs(x[3]),1/sqrt(abs(x[3])),sqrt(abs(x[3])),x[3],sqrt(abs(x[3]))^3,x[3]^2,x[4],x[4]^2,x[3]*x[4]],
                FastDense(10, 32, tanh),
                FastDense(32, 2))
l1 = length(initial_params(NN_chiphi))

function ODE_model(u, NN_params, t)
    NN_params1 = NN_params[1:l1]
    NN_params2 = NN_params[l1+1:end]
    du = AbstractNROrbitModel(u, model_params, t,
                              NN_chiphi=NN_chiphi, NN_chiphi_params=NN_params1,
                              NN_pe=NN_pe, NN_pe_params=NN_params2)
end

prob_nn = ODEProblem(ODE_model, u0, tspan, NN_params)

## Define the objective function
function loss(θ; waveform_real=waveform_real, SNR=0.0)
    e0=θ[1]
    χ0=θ[2]
    p0 = x₀ * (1+e0*cos(χ0)) / mass1
    NN_params = θ[3:end]
    pred_soln = solve(remake(prob_nn, u0 = [χ0, 0.0, p0, e0], p = NN_params), RK4(),
                            saveat = tsteps, dt = dt, adaptive=false, sensealg=BacksolveAdjoint(checkpointing=true))
    pred_waveform_real, pred_waveform_imag = compute_waveform(dt_data, pred_soln, mass_ratio)
    p = pred_soln[3,:]
    e = pred_soln[4,:]

    N = length(pred_waveform_real)
    loss = 1/N * (
             sum(abs2, waveform_real .- pred_waveform_real)
           + 1.0f3*sum(abs2, max.(d_dt(p,dt_data),0.0))
           + 1.0f2*sum(abs2, max.(d2_dt2(p,dt_data),0.0))
           + 1.0f1*sum(abs2, max.(-e,0.0))
           + 1.0f0*sum(abs2, max.(e[p .>= 6 + 2*e[1]] .- e[1],0.0))
           )
           + 1.0f-1*sum(abs2, NN_params)
    return loss, waveform_real, pred_waveform_real, pred_waveform_imag, pred_soln
end

# Test loss function
loss(θ)

losses = []

callback(θ,l, waveform_real, pred_waveform_real, pred_waveform_imag, pred_soln; doplot = true) = begin
    push!(losses, l)
    display(l)
    N = length(pred_waveform_real)
    plt1 = plot(tsteps, waveform_real,
               markershape=:circle, markeralpha = 0.25,
               linewidth = 2, alpha = 0.5,
               label="wform data (Re)", legend=:topleft)
    plot!(plt1, tsteps[1:N], pred_waveform_real[1:end],
               markershape=:circle, markeralpha = 0.25,
               linewidth = 2, alpha = 0.5,
               label="wform NN (Re)")
    plt2 = plot(tsteps, waveform_imag,
              markershape=:circle, markeralpha = 0.25,
              linewidth = 2, alpha = 0.5,
              label="wform data (Im)", legend=:topleft)
    plot!(plt2, tsteps[1:N], pred_waveform_imag[1:end],
              markershape=:circle, markeralpha = 0.25,
              linewidth = 2, alpha = 0.5,
              label="wform NN (Im)")

    pred_orbit = soln2orbit(pred_soln)
    orbit_nn1, orbit_nn2 = one2two(pred_orbit, mass1, mass2)
    N = size(orbit_nn1,2)
    plt3 = plot(x[1:N],y[1:N],
               linewidth = 2, alpha = 0.5,
               label="orbit data")
    plot!(plt3, orbit_nn1[1,1:end-1], orbit_nn1[2,1:end-1],
               linewidth = 2, alpha = 0.5,
               label="orbit NN")
    χ = pred_soln[1,:]
    ϕ = pred_soln[2,:]
    p = pred_soln[3,:]
    e = pred_soln[4,:]
    plt4 = plot(tsteps[1:N], p, linewidth = 2, alpha = 0.5, label="p", legend=:best)
    plt5 = plot(tsteps[1:N], e, linewidth = 2, alpha = 0.5, label="e", legend=:topleft)
    l = @layout [
        a; b; [c{0.6w} [d;e]]
        ]
    plt = plot(plt1, plt2, plt3, plt4, plt5, layout=l, size=(600,600), legendfontsize=6)

    if doplot
        display(plot(plt))
    end
    # Tell sciml_train to not halt the optimization. If return true, then
    # optimization stops.
    return false
end

## Train with BFGS:
#   * same VERY GOOD initial guess
#   * different SNRs
train = false

if train
    global df_params = DataFrame()
    insertcols!(df_params, string(0)=>θ)
    for i in 12:-1:1
        Random.seed!(8000);
        local SNR = 2.0^i
        waveform_real_noisy = set_SNR(waveform_real, dt, SNR)
        tmp_loss(p) = loss(p, waveform_real=waveform_real_noisy, SNR=SNR)
        local res = DiffEqFlux.sciml_train(tmp_loss, θ, BFGS(alphaguess = InitialQuadratic(α0=1.0), linesearch = LineSearches.BackTracking()), cb=callback, maxiters = 100, allow_f_increases = true)
        local plt = plot(losses, yaxis=:log, linewidth = 2, xlabel = "Iteration", ylabel = "Objective value", legend = false)
        display(plot(plt))
        insertcols!(df_params, string(SNR)=>res.minimizer)
    end
    NN_params_filename = "./output/SNR_vs_MSE_params.csv"
    CSV.write(NN_params_filename, df_params)
else
    df_params = CSV.read("./output/SNR_vs_MSE_params_2.csv", DataFrame)
end

function RMSE(θ)
    e0=θ[1]
    χ0=θ[2]
    p0 = x₀ * (1+e0*cos(χ0)) / mass1
    NN_params = θ[3:end]
    pred_soln = solve(remake(prob_nn, u0 = [χ0, 0.0, p0, e0], p = NN_params), RK4(),
    # pred_soln = solve(remake(prob_nn, u0 = u0, p = θ), RK4(),
                            saveat = tsteps, dt = dt, adaptive=false, sensealg=BacksolveAdjoint(checkpointing=true))
    pred_waveform_real, pred_waveform_imag = compute_waveform(dt_data, pred_soln, mass_ratio)

    N = length(pred_waveform_real)
    mse = (sum(abs2, waveform_real .- pred_waveform_real) + sum(abs2, waveform_imag .- pred_waveform_imag)) * dt
    norm2 = (sum(abs2, waveform_real) + sum(abs2, waveform_imag)) * dt
    return sqrt(mse), sqrt(norm2)
end

vals = []
SNRs = []
for i in 1:12
    θ_tmp = df_params[!, i+1]
    rmse, norm = RMSE(θ_tmp)
    println(rmse)
    snr = 2.0^(13-i)
    push!(SNRs, snr)
    # push!(vals, rmse)
    push!(vals, rmse/norm)
end

df_rmse = DataFrame(SNR = SNRs, rmse = vals)
CSV.write("./output/SNR_vs_MSE_rmse.csv", df_rmse)

plt = plot(SNRs, vals, xaxis=:log, linewidth = 2, xlabel = "SNR", ylabel = "RMSE", legend = false)
xticks = 2 .^ collect(1:12)
xticklabels = [ Printf.@sprintf("2^%1i",i) for i in 1:12 ]
plot!(plt, xticks=(xticks,xticklabels))
display(plot(plt))
vals = []
SNRs = []
for i in 1:12
    θ_tmp = df_params[!, i+1]
    rmse, norm = RMSE(θ_tmp)
    println(rmse)
    snr = 2.0^(13-i)
    push!(SNRs, snr)
    # push!(vals, rmse)
    push!(vals, rmse/norm)
end

plt = plot(SNRs, vals, xaxis=:log, linewidth = 2, xlabel = "SNR", ylabel = "RMSE", legend = false)
xticks = 2 .^ collect(1:12)
xticklabels = [ Printf.@sprintf("2^%1i",i) for i in 1:12 ]
plot!(plt, xticks=(xticks,xticklabels))
display(plot(plt))
