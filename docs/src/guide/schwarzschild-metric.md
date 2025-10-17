# Schwarzschild Metric

We seek to describe the two body problem in general relativity. In particular, we seek the equations of motion for two body dynamics. In the limit that one black hole is much larger than the other, we can describe the smaller black hole as a particle following geodesics of the Schwarzschild metric around the larger black hole. We shall now proceed to describe how the two body problem can thus be reduced to an effective one-body problem. 

Consider two black holes of masses $m_1$ and $m_2$ orbiting around their common center-of-mass. We have two cases: 
> - $m_1>>m_2$: The equivalent one-body picture is a particle orbiting a central Schwarzschild black hole of mass $M=m_1+m_2$
> - $m_1\sim m_2:$ The equivalent one-body picture is a given by the Effective One-Body formalism, wherein the two-body dynamics is mapped to a single-body of reduced mass $\mu=m_1m_2 / (m_1+m_2)$ orbiting a central mass $M = m_1+m_2$.

For reasons of simplicity and application to EMRIs, we will focus on the first option above. The Hamiltonian describing a particle orbiting a Schwarzschild black hole is 

$$H = \frac{p^2}{2}= \frac{1}{2}p^\mu g_{\mu \nu} p^\nu$$

The Schwarzschild Metric $g_{\mu\nu}$ is 

$$g^{\text{Schwarzschild}}_{\mu\nu} = 
\begin{pmatrix}
    -\left(1-\frac{2M}{r} \right) & 0 & 0 & 0 \\
    0 & \left(1-\frac{2M}{r} \right)^{-1} & 0 & 0 \\ 
    0  & 0 & r^2 & 0 \\ 
    0 & 0 & 0 & r^2 \sin^2 \theta
\end{pmatrix}$$

Thus, the Hamiltonian for the Schwarzschild Metric is 

$$H_{Schwarzschild}=- \left(1-\frac{2M}{r}\right)^{-1}\frac{p_t^2}{2} + \left(1-\frac{2M}{r}\right) \frac{p_r^2}{2} + \frac{p_\phi^2}{2r^2}$$

$$H=\frac{1}{2}\left(p_t, p_r, p_\theta, p_\phi \right)^T \begin{pmatrix}
    -\left(1-\frac{2M}{r}\right)^{-1} & 0 & 0 & 0\\
    0 & \left(1-\frac{2M}{r}\right) & 0 & 0 \\ 
    0 & 0 & 0 & 0 \\
    0 & 0 & 0 & r^{-2}
    \end{pmatrix}	 
    \left(p_t, p_r, p_\theta, p_\phi \right)$$

We can now use Hamilton's Equations of Motion to obtain the orbit: 

$$\dot{q}=\frac{\partial H}{\partial p}, -\dot{p}=\frac{\partial H}{\partial q}$$

Thus, for a particle on an equatorial orbit around a Schwarzschild black hole, we obtain the following geodesic equations of motion: 
$$\frac{dt}{d\tau} = \left(1-\frac{2M}{r}\right)^{-1}E, \frac{dr}{d\tau} = \left(1-\frac{2M}{r}\right){p_r}, \frac{d\theta}{d\tau} =0, \frac{d\phi}{d\tau} = \frac{L}{r^2}$$

Similarly, the momentum vector $p^{\mu}$ evolves as

$$&\frac{dp_t}{d\tau} = \dot E = 0, \frac{dp_r}{d\tau} =  -\frac{1}{2}\left[\left(1-\frac{2M}{r}\right)^{-2}\left( \frac{2M}{r^2}\right) (p_t)^2 + \frac{2M}{r^2}(p_r)^2-2r^{-3} (p_{\phi})^2\right] \\ 
    &\dot p_\theta = 0, \dot p_\phi = \dot L = 0$$

We have thus obtained all the equations of motion for a test particle orbiting a Schwarzschild Black Hole. 
