# Boundary Conditions

Consider a given metric in coordinates $(t, r, \theta, \phi)$. The horizon predicted by such a metric will be given by 

$$g^{rr}(r_H, \theta) = 0$$

Consider the Schwarzschild Metric: 

$$g_{rr}=\left(1-\frac{2M}{r}\right)^{-1} \to g^{rr} = \left(1-\frac{2M}{r}\right) = 0 \to r_H = 2M$$

Similarly, consider the Kerr Metric: 

$$g_{rr}=\frac{\Sigma}{\Delta}$$

The nice thing about the Kerr metric is this: although it's not a diagonal matrix, finding it's inverse isn't hard! The off-diagonal component is slotted away on the upper right corner (and the metric is symmetric). Thus, for the metric

$$
g_{\mu\nu} = \begin{pmatrix}
g_{tt} & 0 & 0 & g_{t\phi} \\ 
0 & \frac{\Sigma}{\Delta} & 0 & 0 \\ 
0 & 0 & \Sigma & 0 \\ 
g_{t\phi} & 0 & 0 & \frac{A}{\Sigma}\sin^2\theta
\end{pmatrix}
$$

To compute the inverse metric $g^{\mu\nu}$, we only need to invert the $t\phi$ block, because the $r\theta$ block is diagonal: 

$$g^{rr} = \frac{\Delta}{\Sigma}, g^{\theta\theta} = \frac{1}{\Sigma}$$

To invert the $t\phi$ block: 

$$
\tilde g_{ab} = 
\begin{pmatrix}
g_{tt} & g_{t\phi} \\ 
g_{t\phi} & g_{\phi\phi}
\end{pmatrix}
$$

The inverse of a $2x2$ matrix is simple: 

$$
\tilde g^{ab} = 
\frac{1}{Det|\tilde g_{ab}|}\begin{pmatrix}
g_{\phi\phi} & -g_{t\phi} \\ 
-g_{t\phi} & g_{tt}
\end{pmatrix}
$$

It turns out 

$$Det|\tilde g_{ab}|=g_{tt}g_{\phi\phi}-g^2_{t\phi}=-\Delta \cdot \sin^2\theta$$

Once you've gone through the crank (no pun intended\footnote{If you're thinking: There was no pun to begin with, you'd be right.}), you'll find the following:

$$g^{tt}=-\frac{1}{\Delta \sin^2\theta}g_{\phi\phi}$$
$$g^{\phi\phi}=-\frac{1}{\Delta \sin^2\theta}g_{tt}$$
$$g^{t\phi}=\frac{1}{\Delta \sin^2\theta}g_{t\phi}$$

Of course, $g^{rr}$ and $g^{\theta\theta}$ are trivial and written upstairs. 
