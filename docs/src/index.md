# Binary Black Holes: From Gravity Waves to Orbits
## Ref Bari

Binary Black Holes are a fascinating physical system, unifying the small (amplitudes on the order of $\sim 10^{-21}$) and the grand (lumonisities $>10^{47}$ ergs, brighter than all the stars in the universe!). For a binary black hole system, the dominant $h^{22}$ mode is given by 

$$h^{22}(t) \sim \frac{1}{r}(\ddot{I}_{xx} - \ddot{I}_{yy} - 2i\ddot{I}_{xy})$$

The above terms are given by 

$$I_{xx}= 2(x^2 - \frac{1}{3}(x^2+y^2))$$
$$I_{yy}= 2(y^2 - \frac{1}{3}(x^2+y^2))$$
$$I_{xy}= xy$$

Formulating this in Julia is simple: 
```julia
function orbit2tensor(orbit, component, mass=1.0)
    x, y = orbit[1,:], orbit[2,:]

    Ixx, Iyy, Ixy = x^2, y^2, x*y
    trace = Ixx + Iyy

    if I[1,1]:
        I = Ixx .- (1.0 ./ 3.0).*trace
    elseif I[2,2]:
        I = Iyy .- (1.0 ./ 3.0).*trace
    else
        I = Ixy
    end

    return mass .* I
end
```
