using TensorOperations
using ForwardDiff
using LinearAlgebra
using StaticArrays

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

# Wrapper to compute Ricci components analytically
function compute_analytical_ricci_costs(r_point, NN, NN_params, NN_state, scale_factor)
    
    # Define A(r) and B(r) closure
    function get_metric_components(r)
        # Fix 1: Pass NN_state and unpack the tuple (output, state)
        nn_out, _ = NN([r / scale_factor], NN_params, NN_state)
        
        raw_tt = nn_out[1]
        
        a1, a2, a3 = nn_out[2], nn_out[3], nn_out[4]

        a1 = softplus(a1) + 1e-3
        a2 = softplus(a2) + 1e-3
        a3 = softplus(a3) + 1e-3
        
        rr_correction = 1.0 + a1 / r + a2 / r^2 + a3 / r^3
        
        # g_tt component (A)
        A = 1 - softplus(raw_tt)
        
        # g_rr component (B)
        B = max(rr_correction, 0.05)
        
        return A, B
    end

    # Define helper functions for ForwardDiff
    A_func(r) = get_metric_components(r)[1]
    B_func(r) = get_metric_components(r)[2]

    # Compute values and derivatives
    A_val = A_func(r_point)
    B_val = B_func(r_point)
    
    Ap = ForwardDiff.derivative(A_func, r_point)
    Bp = ForwardDiff.derivative(B_func, r_point)
    
    # Second derivative needed for R_tt and R_rr
    App = ForwardDiff.derivative(r -> ForwardDiff.derivative(A_func, r), r_point)

    # --- Analytical Ricci Component ---
    # R_θθ (The Golden Constraint)
    # Formula: 1 - (1/B) + rB'/2B^2 - rA'/2AB
    R_th = 1.0 - (1.0 / B_val) + (r_point * Bp) / (2 * B_val^2) - (r_point * Ap) / (2 * A_val * B_val)
    
    # open("training_log.txt", "a") do io
    #     A_print = ForwardDiff.value(A_val)
    #     B_print = ForwardDiff.value(B_val)
    #     Ap_print = ForwardDiff.value(Ap)
    #     Bp_print = ForwardDiff.value(Bp)
    #     R_print = ForwardDiff.value(R_th)
        
    #     println(io, "    [Ricci] r=$(r_point): A=$(round(A_print, digits=4)), B=$(round(B_print, digits=4)), A'=$(round(Ap_print, sigdigits=3)), B'=$(round(Bp_print, sigdigits=3)), R_θθ=$(round(R_print, sigdigits=4))")
    # end
     
    return R_th^2
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

    f_tt_correction = 1 - softplus(f_tt_output)

    a1, a2, a3 = nn_outputs[2], nn_outputs[3], nn_outputs[4]

    a1 = softplus(a1) + 1e-3
    a2 = softplus(a2) + 1e-3
    a3 = softplus(a3) + 1e-3
    
    rr_correction = 1.0 + a1 / r + a2 / r^2 + a3 / r^3   

    f_rr_correction = max(rr_correction, 0.05)
    # f_ϕϕ_correction = softplus(f_ϕϕ_output) + ϵ
    
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