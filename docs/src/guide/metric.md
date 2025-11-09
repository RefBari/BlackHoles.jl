# Attempts

!!! note "Variable Transformation to $u=1/r$"
    One idea I am considering is a variable transformation from $r$ to $u \equiv 1/r$. In this new set of variables, consider the angular component of the metric. The newtonian angular component looks as follows under this variable transformation. In terms of $r$, the angular component was originally $g_{N}^{\phi\phi}(r)=r^{-2} (1+\frac{2}{r})^{-1}$. In terms of $u$, this becomes $g_{N}^{\phi\phi}(u)=u^2 * (1+2*u)^{-1}$. Thus, if written in the form $g_{S}^{\phi\phi}=r^{-2}$, the goal of the NN under this variable transformation would be to learn $(1+2*u)$. Alternatively, if formulated in the form $g^{\phi\phi}_{S}=g^{\phi\phi}_{N} * (1+f^{\phi\phi}_{NN})$, then the NN must learn only $f^{\phi\phi}_{NN} = 2*u$. 
