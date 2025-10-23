# Newtonian Metric

The goal of this module is to explain the Newtonian Metric which is used as the base model for the Neural Network for Conservative Dynamics. This module should answer the following questions: 

- What is the Newtonian Metric?
- How does it give rise to Newton's Equations of Motion?
- In what limit does the Schwarzschild Metric reduce to the Newtonian Metric?

According to Einstein's Theory of General Relativity, Flat Space is given by the Minkowski Metric: 

$$g=\begin{pmatrix} 
    -1 & 0 & 0 & 0 \\ 
    0 & 1 & 0 & 0 \\ 
    0 & 0 & 1 & 0 \\ 
    0 & 0 & 0 & 1
    \end{pmatrix} \rightarrow ds^2 = -dt^2 + dx^2 + dy^2 + dz^2$$

  The Newtonian Metric is given by
  
  $$g=\begin{pmatrix} 
    -\left(1-\frac{2GM}{r} \right) & 0 & 0 & 0 \\ 
    0 & \left(1+\frac{2GM}{r} \right) & 0 & 0 \\ 
    0 & 0 & \left(1+\frac{2GM}{r} \right) & 0 \\ 
    0 & 0 & 0 & \left(1+\frac{2GM}{r} \right)
    \end{pmatrix} \rightarrow ds^2 = -\left(1-\frac{2GM}{r} \right)dt^2 + \left(1+\frac{2GM}{r} \right)(dx^2 + dy^2 + dz^2)$$

  This metric is valid in the limit that $\frac{GM}{r}<<1$. The Newtonian Metric represents a small deviation from flat spacetime due to gravity. Indeed, observe that for $r>>M$, we see that the Newtonian Metric reduces back to the Minkowski Metric.
