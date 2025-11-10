# Attempts to Optimize Neural Network Fits

In this section, I recount five experiments I ran on the weekend of 11/08 and week of 11/10 to optimize the neural network fits: (1) Tikhonov Regularization, (2) Testing Different Parameters in the `Optim.jl` library, (3) Variable Transformation to $u=1/r$, (4) Neural Network Size, and (5) Learnable Initial Conditions. 

!!! note "Tikhonov Regularization"
    Testing test test

!!! note "`Optim.jl` Parameters"
    The heart and soul of the optimizer is the BFGS Algorithm. We now provide a brief intro to the BFGS algorithm. The idea is simple: we have some function $f(x)$ which we seek to minimize. To understand BFGS, we turn to its predecessor, Newton's method. Newton's method forms a second order approximation of $f(x)$ at a given point and finds the minimum of this approximation. Specifically, say we'd like to minimize $f(x)$ starting at $x_0$. If we take a linear Taylor expansion of $f'(x)$, we have $f'(x) \approx f'(x_0)+f''(x_0)(x-x_0)$. If we seek to minimize $f(x)$, then we seek $f'(x) = 0$. In that case, $f'(x_0) + f''(x_0)x-f''(x_0)x_0 = 0$. If we now solve for $x$, we find that $x^* = x_0 - \frac{f'(x_0)}{f''(x_0)}$. This method of finding the minimum of a function is known as Newton's method. Newton's method requires both the gradient $\nabla f = f'(x)$ and the Hessian $\nabla^2 f = f''(x)$. We can write Newton's method as $x^* = x_0 - [\nabla^2 f(x_0)]^{-1} \nabla f(x_0)$. But this can be difficult for several reasons: finding the inverse Hessian could be difficult. What if there was a way we could approximate the Hessian (or its inverse)?
    We now consider the BFGS algorithm. Consider again our goal of minimizing the function $f(x)$. We are iteratively updating the step $x_k$ where $k=1, 2, ..., T$, where $T$ is determined by some convergence criterion. Suppose we use a quadratic approximation to $f$ at each iteration. Denote this approximation at step $k$ as $\hat{f}_k(x)$. Specifically, $\hat{f}_k=f(x_k) + [\nabla f(x_k)]^T (x-x_k) + \frac{1}{2}(x-x_k)^T [\nabla f(x_k)]^2 (x-x_k)$.

!!! note "Neural Network Size"
    Breadth and Depth of Neural Network

!!! note "Learnable Initial Conditions"
    Breadth and Depth of Neural Network

!!! note "Variable Transformation to $u=1/r$"
    One idea I am considering is a variable transformation from $r$ to $u \equiv 1/r$. In this new set of variables, consider the angular component of the metric. The newtonian angular component looks as follows under this variable transformation. In terms of $r$, the angular component was originally $g_{N}^{\phi\phi}(r)=r^{-2} (1+\frac{2}{r})^{-1}$. In terms of $u$, this becomes $g_{N}^{\phi\phi}(u)=u^2 * (1+2*u)^{-1}$. Thus, if written in the form $g_{S}^{\phi\phi}=r^{-2}$, the goal of the NN under this variable transformation would be to learn $(1+2*u)$. Alternatively, if formulated in the form $g^{\phi\phi}_{S}=g^{\phi\phi}_{N} * (1+f^{\phi\phi}_{NN})$, then the NN must learn only $f^{\phi\phi}_{NN} = 2*u$. 
