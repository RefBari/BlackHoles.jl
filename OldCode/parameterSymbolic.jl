
# # --- Data (make sure Float64 and finite) ---
# u = collect(1.0 ./ rvals)         # u = 1/r
# g = collect(g_rr_pred)            # your NN-predicted g^{rr}(u)

# mask = isfinite.(u) .& isfinite.(g) .& (abs.(g) .< 1e12)
# u, g = u[mask], g[mask]

# # --- Simple model: ĝ(u; α) = 1 - α u ---
# # Robust squared-error loss
# lossα(θ) = begin
#     α = float(θ[1])
#     pred = 1 .- α .* u
#     r = pred .- g
#     if any(!isfinite, r)
#         return Inf
#     end
#     sum(r.^2)
# end

# # Optional: closed-form LS init to help Adam
# A   = [ones(length(u))  u]
# β   = A \ g
# α0  = -β[2]                         # because g ≈ 1 - α u
# θ0  = [float(α0)]

# # --- Build the optimization problem (use AD for Adam) ---
# optf = Optimization.OptimizationFunction((θ,p)->lossα(θ), Optimization.AutoZygote())
# prob = Optimization.OptimizationProblem(optf, θ0)

# # --- Adam pass ---
# lr = 1e-2
# maxiters = 3000
# res_adam = Optimization.solve(prob, Optimisers.Adam(lr); maxiters=maxiters)

# α_adam = res_adam.u[1]
# @show α_adam res_adam.minimum

# # (Optional) polish with BFGS starting from Adam’s result
# prob2 = Optimization.OptimizationProblem(optf, res_adam.u)
# res_bfgs = Optimization.solve(prob2, Optim.BFGS(); allow_f_increases=true)
# α = res_bfgs.u[1]
# @show α res_bfgs.minimum

# # --- Plot for sanity ---
# using Plots, LaTeXStrings
# g_fit  = 1 .- α .* u
# g_schw = 1 .- 2 .* u
# plt = plot(u, g, lw=2, label="Predicted: g_{NN}^{rr}(u)", color = "red")
# plot!(plt, u, g_fit,  lw=2, ls=:dashdot, label="Parameter Estimation Fit: (1 - αu), α = 1.38", color = "green")
# plot!(plt, u, g_schw, lw=2,  label="Schwarzschild: (1 - 2u)", xlabel = L"u=1/r", ylabel = L"g^{rr}(u)", color = "blue", ls=:dash)
# xlabel(plt, L"u=1/r"); ylabel!(plt, L"g^{rr}(u)"); legend!(plt, :topleft)
# display(plt)
