# Numerical Kludge Scheme

To calculate the gravitational waveform for a Kerr geodesic, we need a function which can: 

```julia
function Compute_Gravitational_Wave(Geodesic, Observer)
    return h+, hx
end
```

As a first check, we must ensure that we are in the correct temporal coordinates. For Kerr geodesics, there are not one, not two, but three possible temporal variables!: Proper Time, Coordinate Time, and Mino Time. 

To ensure that we are in Coordinate Time, we return to our geodesic equations of motion, which are calculated in the `Kerr()` function: 

```julia
function Kerr(du, u, model_params, t; M = 1.0, a = 0.9)
    x = @view u[1:8]
    q = model_params[1]

    function H(state_vec)
        t, r, θ, φ, p_t, p_r, p_θ, p_ϕ = state_vec

        p = [p_t, p_r, p_θ, p_ϕ]

        Σ = r^2 + a^2 * cos(θ)^2
        Δ = r^2 - 2*M*r + a^2

        g_tt = -(1 - (2*M*r)/ Σ )
        g_rr = Σ / Δ
        g_θθ = Σ
        g_ϕϕ = (r^2 + a^2 + (2*M*r*a^2*sin(θ)^2) / Σ) * sin(θ)^2
        g_tϕ = - (2*M*r*a*sin(θ)^2) / Σ
        g_ϕt = g_tϕ

        # Base Metric: Kerr Metric
        g = [
                g_tt 0 0 g_tϕ;
                0 g_rr 0 0;
                0 0 g_θθ 0;
                g_ϕt 0 0 g_ϕϕ
            ]

        inv_g = inv(g)
        H_kerr = (1/2) * p' * inv_g * p

        return H_kerr # Returns equations of motion in PROPER time
    end

    grad_H = ForwardDiff.gradient(H, x)

    J = [
            zeros(4, 4) I(4);
            -I(4) zeros(4, 4)
        ]

    Conservative = J * grad_H
    
    du_dτ = Conservative

    dH_dpₜ = grad_H[5]
    dτ_dt = (dH_dpₜ)^(-1)

    @inbounds for i = 1:8
        du[i] = du_dτ[i] * dτ_dt
    end

    du[1] = 1

    return [du[1], du[2], du[3], du[4], du[5], du[6], du[7], du[8]]
end
```

This can be a lot to look at at first sight, so here's a diagram summarizing the main components of this function, which returns 8 equations of motion for Kerr geodesics: 

![KerrHamiltonianFunction](KerrGeodesics.png)

The essential idea is as follows: First, we form the hamiltonian via 

$$H = \frac{1}{2}p_{\mu}g^{\mu\nu}p_{\nu}$$

Then, we can use Hamilton's equations of motion:

$$\dot q = \frac{\partial H}{\partial p}, \dot p = - \frac{\partial H}{\partial q}$$

If we write this in matrix form, it looks like: 

$$\begin{pmatrix}
\dot p \\ 
\dot q
\end{pmatrix} = \begin{pmatrix}
0 & +1 \\ 
-1 & 0
\end{pmatrix}
\begin{pmatrix}
\partial H/\partial p \\ 
\partial H/\partial q
\end{pmatrix}$$

Canonically, these time derivates are in terms of proper time: 

$$\dot p = \frac{\partial p}{\partial \tau},\dot q = \frac{\partial q}{\partial \tau}$$

To convert these to coordinate time, we use the chain rule: 

$$\dot t = \frac{\partial H}{\partial p_t}$$

That means 

$$\dot t = \frac{dt}{d\tau} = \frac{\partial H}{\partial p_t}$$

Thus, in terms of coordinate time, the equations of motion are: 

$$\frac{dx^{\mu}}{dt}=\frac{dx^{\mu}}{d\tau}*\frac{d\tau}{dt}$$

Great! With this in order, we turn to the orbit-to-waveform mapping. We will employ the so-called "Numerical Kludge" formalism, which uses the Quadrupole approximation (in the original paper, the Quadrupole-Octope approximation or Press formula are also tested).
