```
    DissipationRicci.jl
    Contains the NN predictions for the metric
    and the functions to compute the Riemann tensor, Ricci tensor, and Ricci scalar from the metric.

```

using TensorOperations
using ForwardDiff
using LinearAlgebra
using StaticArrays

ϵ = 1e-6

# Calculate derivatives by pushing one element of 
# x vector at a time using ForwardDiff
function ∂(f, x)
    derivs = map(eachindex(x)) do k
        e_k = [i == k ? 1.0 : 0.0 for i in eachindex(x)]
        ForwardDiff.derivative(t -> f(x + t * e_k), 0.0)
    end
    stack(derivs; dims=1)
end

# Metric --> Christoffel Symbols
function Γ(g, x)
    ginv = inv(g(x))
    ∂g = ∂(g, x)
    
    @tensor Γ[λ, μ, ν] := 0.5 * ginv[λ, σ] * (∂g[μ, σ, ν] + ∂g[ν, σ, μ] - ∂g[σ, μ, ν])
end

# Christoffel Symbols --> Riemann Tensor
function Riemann(g, x)
    ∂Γ = ∂(y -> Γ(g, y), x)
    @tensor Riemann[ρ, σ, μ, ν] := ∂Γ[μ, ρ, σ, ν] - ∂Γ[ν, ρ, σ, μ] + 
                                   Γ(g, x)[ρ, λ, μ] * Γ(g, x)[λ, σ, ν] - 
                                   Γ(g, x)[ρ, λ, ν] * Γ(g, x)[λ, σ, μ]
end

# Riemann Tensor --> Ricci Tensor
function RicciTensor(Riemann_tensor)
    @tensor Ricci[μ, ν] := Riemann_tensor[λ, μ, λ, ν]
end

# Ricci Tensor --> Ricci Scalar
function RicciScalar(Ricci_tensor, g, x)
    @tensor R = inv(g(x))[μ, ν] * Ricci_tensor[μ, ν]
    return R
end

# --- 3. Metric Definition ---

g_Random(x) = [
    -x[2]^3 0 0 0;
    0 x[2]^2 0 0; 
    0 0 x[2]^1 0;
    0 0 0 x[2]
]

g_Minkowski(x) = [
    -1 0 0 0;
    0 1 0 0; 
    0 0 1 0;
    0 0 0 1
]

g_Schwarzschild(x) = [
    -(1 - 2/x[2]) 0 0 0; 
    0 (1 - 2/x[2])^(-1) 0 0; 
    0 0 x[2]^2 0; 
    0 0 0 x[2]^2 * sin(x[3])^2
]

g_Kerr_Equatorial(x; M = 1.0, a = 0.9) = [
    -(1 - 2/x[2]) 0 0 -(2*M*a)/x[2]; 
    0 (x[2])^2/(x[2]^2 - 2*M*x[2] + a^2) 0 0; 
    0 0 x[2]^2 0; 
    -(2*M*a)/x[2] 0 0 x[2]^2 + a^2 + 2*M*a^2/x[2]
]

function g_Kerr(x; M = 1.0, a = 0.95)
    a = a

    Σ = x[2]^2 + a^2 * cos(x[3])^2
    Δ = x[2]^2 - 2*M*x[2] + a^2
    A = (x[2]^2 + a^2)^2 - Δ * a^2 * sin(x[3])^2

    g_tt = -(1 - 2*M*x[2]/Σ)
    g_rr = Σ / Δ
    g_θθ = Σ
    g_ϕϕ = A / Σ * sin(x[3])^2
    g_tϕ = -(2 * M * a * x[2] * (sin(x[3]))^2) / Σ

    return [
        g_tt 0 0 g_tϕ; 
        0 g_rr 0 0; 
        0 0 g_θθ 0; 
        g_tϕ 0 0 g_ϕϕ
    ]
end

function g_NeuralNetwork(x, nn_outputs)
    M = 1.0
    
    t = x[1]
    r = x[2]
    θ = x[3]
    ϕ = x[4]
    
    # Σ = r^2 + a^2 * cos(θ)^2
    # Δ = r^2 - 2*M*r + a^2
    # A = (r^2 + a^2)^2 - Δ * a^2 * sin(θ)^2

    # δ = softplus(nn_outputs[1]) - softplus(zero(nn_outputs[1]))
    #             clamp to:            log(32) ≈ 3.47        log(800) ≈ 6.68
    # Σ_nn = r^2 * (1 + δ)
    
    # Δ_nn = max(r^2 - softplus(nn_outputs[2]), 1e-4) # Ensure Δ_nn is positive to avoid singularities

    # --- KERR METRIC CORRECT EQUATIONS --- #
    # g_tt = -(1 - 2*M*x[2]/Σ)
    # g_rr = Σ / Δ
    # g_θθ = Σ
    # g_ϕϕ = A / Σ * sin(x[3])^2
    # g_tϕ = -(2 * M * a * x[2] * (sin(x[3]))^2) / Σ

    # --- KERR METRIC SANITY CHECK --- #
    # g_tt = -(1 - 2*M*x[2]/Σ) + nn_outputs[1]
    # g_rr = Σ / Δ + nn_outputs[2]
    # g_θθ = Σ + nn_outputs[3]
    # g_ϕϕ = A / Σ * sin(x[3])^2 + nn_outputs[4]
    # g_tϕ = -(2 * M * a * x[2] * (sin(x[3]))^2) / Σ + nn_outputs[5]

    # g_tt_base = -1
    # g_rr_base = +1
    # g_θθ_base = r^2
    # g_ϕϕ_base = r^2 * sin(θ)^2

    # zero_shift = softplus(zero(nn_outputs[1]))

    # δ_tt = softplus(nn_outputs[1]) - zero_shift
    # δ_rr = softplus(nn_outputs[2]) - zero_shift
    # δ_θθ = softplus(nn_outputs[3]) - zero_shift
    # δ_ϕϕ = softplus(nn_outputs[4]) - zero_shift
    # δ_tϕ = -1 * sin(θ)^2 * softplus(nn_outputs[5])

    # g_tt = g_tt_base + δ_tt
    # g_rr = g_rr_base + δ_rr
    # g_θθ = g_θθ_base + δ_θθ
    # g_ϕϕ = g_ϕϕ_base + δ_ϕϕ
    # g_tϕ = δ_tϕ

    # g_tt = -(1 + nn_outputs[1])
    # g_rr = Σ / Δ
    # g_θθ = Σ
    # g_ϕϕ = A / Σ * sin(θ)^2
    # g_tϕ = -(2 * M * a * r * (sin(θ))^2) / Σ

    g_tt = -(1 + nn_outputs[1] / r)                        # Minkowski: -1,  correction pushes toward -0.75
    g_rr = 1 + (nn_outputs[2] / r)                        # Minkowski: +1,  correction pushes toward ~1.15
    g_θθ = r^2 + (nn_outputs[3] / r)                                          # Minkowski: r²,  correction tiny (~1%)
    g_ϕϕ = sin(θ)^2 * (r^2 + (nn_outputs[4] / r))                         # Minkowski: r²sin²θ, correction tiny
    g_tϕ = (nn_outputs[5]/r) * sin(θ)^2              # Minkowski: 0,   NN learns sign itself

    # g_rr = Σ / Δ * (1 + softplus(nn_outputs[2]) - s0)
    # g_θθ = Σ * (1 + softplus(nn_outputs[3]) - s0)
    # g_ϕϕ = (A / Σ * sin(x[3])^2) * (1 + softplus(nn_outputs[4]) - s0)
    # g_tϕ = (-(2 * M * a * x[2] * (sin(x[3]))^2) / Σ) * (1 + softplus(nn_outputs[5]) - s0)
    
    # g_rr = (1 + softplus(nn_outputs[2]) - s0)
    # g_θθ = r^2 * (1 + softplus(nn_outputs[3]) - s0)
    # g_ϕϕ = r^2 * sin(θ)^2 * (1 + softplus(nn_outputs[4]) - s0)
    # g_tϕ = -sin(θ)^2 * softplus(nn_outputs[5])

    return [
        g_tt 0 0 g_tϕ;
        0 g_rr 0 0;
        0 0 g_θθ 0;
        g_tϕ 0 0 g_ϕϕ
    ]
end

# --- 4. Execution ---

g_Kerr_true(x) = g_Kerr(x; M = 1.0, a = 0.95)

# Define spacetime |point (t, r, θ, φ)
x_val = @SVector [100.0, 20.0, π/6, 0.0]

println("Computing Riemann Tensor...")
# Pass the function 'g_Random' itself, not the result
R_tensor = Riemann(g_Kerr_true, x_val)

println("Computing Ricci Tensor...")
Ric_tensor = RicciTensor(R_tensor)
# print(Ric_tensor)

println("Computing Scalar Curvature...")
R_scalar = RicciScalar(Ric_tensor, g_Kerr_true, x_val)

println("------------------------------------------------")
println("Ricci Scalar result: ", R_scalar)
println("Expected result:     0.0 (Vacuum solution)")
println("------------------------------------------------")