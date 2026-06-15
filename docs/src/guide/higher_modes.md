# Boundary Conditions

Consider a given metric in coordinates $(t, r, \theta, \phi)$. The horizon predicted by such a metric will be given by 

$$g^{rr}(r_H, \theta) = 0$$

Consider the Schwarzschild Metric: 

$$g_{rr}=\left(1-\frac{2M}{r}\right)^{-1} \to g^{rr} = \left(1-\frac{2M}{r}\right) = 0 \to r_H = 2M$$

Similarly, consider the Kerr Metric: 

$$g_{rr}=\frac{\Sigma}{\Delta}$$

The nice thing about the Kerr metric is this: although it's not a diagonal matrix, finding it's inverse isn't hard! The off-diagonal component is slotted away on the upper right corner (and the metric is symmetric). Thus, for the metric

$$g_{\mu\nu} = \begin{pmatrix}
g_{tt} & 0 & 0 & g_{t\phi} \\ 
0 & \frac{\Sigma}{\Delta} & 0 & 0 \\ 
0 & 0 & \Sigma & 0 \\ 
g_{t\phi} & 0 & 0 & \frac{A}{\Sigma}\sin^2\theta
\end{pmatrix}$$

To compute the inverse metric $g^{\mu\nu}$, we only need to invert the $t\phi$ block, because the $r\theta$ block is diagonal: 

$$g^{rr} = \frac{\Delta}{\Sigma}, g^{\theta\theta} = \frac{1}{\Sigma}$$

To invert the $t\phi$ block: 

$$\tilde g_{ab} = 
\begin{pmatrix}
g_{tt} & g_{t\phi} \\ 
g_{t\phi} & g_{\phi\phi}
\end{pmatrix}$$

The inverse of a $2x2$ matrix is simple: 

$$\tilde g^{ab} = 
\frac{1}{Det|\tilde g_{ab}|}\begin{pmatrix}
g_{\phi\phi} & -g_{t\phi} \\ 
-g_{t\phi} & g_{tt}
\end{pmatrix}$$

It turns out 

$$Det|\tilde g_{ab}|=g_{tt}g_{\phi\phi}-g^2_{t\phi}=-\Delta \cdot \sin^2\theta$$

Once you've gone through the crank (no pun intended\footnote{If you're thinking: There was no pun to begin with, you'd be right.}), you'll find the following:

$$g^{tt}=-\frac{1}{\Delta \sin^2\theta}g_{\phi\phi}$$
$$g^{\phi\phi}=-\frac{1}{\Delta \sin^2\theta}g_{tt}$$
$$g^{t\phi}=\frac{1}{\Delta \sin^2\theta}g_{t\phi}$$

Of course, $g^{rr}$ and $g^{\theta\theta}$ are trivial and written upstairs. 

Thus, here is the key idea: 
!!! warning "The Big Idea"
    For a given metric, $g^{rr}=0$ typically signals the horizon predicted by the metric. For Schwarzschild, $g^{rr}=0$ predicts $r=2M$. For Kerr, $g^{rr}=0$ predicts $r=M \pm \sqrt{M^2 - a^2}$. Thus, for the metric predicted by the neural network, one can solve $g^{rr}=0$ to find its predicted horizon. And then one can penalize the location of the predicted horizon against the true horizon using $(r_{H, predicted}-r_{H, true})^2$.

!!! danger "The Problem"
    You gotta be careful: If you directly penalize against the true location of the horizon, you are injecting _a priori information_ about the true metric into the loss function. Thus, is there a way to leverage the outer Kerr event horizon as an inner boundary condition without explicitly specifying the location of the kerr horizon? Yes! But it is a tad bit subtle. Instead of penalizing the _location_ of the horizon, since that would give away information about the metric in the loss function, we simply penalize against the smoothness and continuity of the horizon. This will be discussed in the next blurb. 

As I discussed in "The Problem", one must engineer a clever workaround to penalizing the exact location of the Kerr horizon. Instead, we shall penalize the smoothness and regularity of the horizon. In particular, say we solve for the roots of the predicted metric component: 

$$g^{rr}_{pred}(r, \theta)=0$$

Ideally, we will find a series of points $(r_i, \theta_i)$ which are the roots of this equation. Now, for a real black hole, the event horizon is a constant-r surface. Thus, ideally, all the $r_i$ values should be identical and furthermore, the radial value should be independent of theta: $r \neq r(\theta)$. To implement this in practice, we do the following: 
