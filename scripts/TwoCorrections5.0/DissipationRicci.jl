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

function g_NeuralNetwork(x, nn_outputs)
    t = x[1]
    r = x[2]
    θ = x[3]
    ϕ = x[4]
    
    f_tt_output = nn_outputs[1]
    f_rr_output = nn_outputs[2]

    f_tt_correction = 1 - softplus(f_tt_output)
    f_rr_correction = 1 + softplus(f_rr_output)
    
    f = 1 - (2/r)

    return [
        -1 * f_tt_correction 0 0 0;
        0 1 * f_rr_correction 0 0;
        0 0 r^2 0; 
        0 0 0 r^2 * sin(θ)^2
    ]
end

# --- 4. Execution ---

# Define spacetime point (t, r, θ, φ)
x_val = @SVector [0.0, 12.0, π/2, 0.0]

println("Random: Computing Riemann Tensor...")
# Pass the function 'g_Random' itself, not the result
R_tensor = Riemann(g_Schwarzschild, x_val)

println("Random: Computing Ricci Tensor...")
Ric_tensor = RicciTensor(R_tensor)

println("Random: Computing Scalar Curvature...")
R_scalar = RicciScalar(Ric_tensor, g_Schwarzschild, x_val)

println("------------------------------------------------")
println("Ricci Scalar result: ", R_scalar)
println("Expected result:     0.0 (Vacuum solution)")
println("------------------------------------------------")