# Binary Black Holes: From Gravity Waves to Orbits

Binary Black Holes are a fascinating physical system, unifying the small (amplitudes on the order of $\sim 10^{-21}$) and the grand (lumonisities $>10^{47}$ ergs, brighter than all the stars in the universe!). For a binary black hole system, the dominant $h^{22}$ mode is given by 

$$h^{22}(t) \sim \frac{1}{r}(\ddot{I}_{xx} - \ddot{I}_{yy} - 2i\ddot{I}_{xy})$$

The above terms are given by 

$$I_{xx}= 2(x^2 - \frac{1}{3}(x^2+y^2))$$
$$I_{yy}= 2(y^2 - \frac{1}{3}(x^2+y^2))$$
$$I_{xy}= xy$$

Expressing this in Julia is simple: 
```julia
function orbit2tensor(orbit, component, mass=1.0)
    #=
        Construct trace-free moment tensor Ι(t) for orbit from BH orbit (x(t),y(t))

        component defines the Cartesion indices in x,y. For example,
        I_{22} is the yy component of the moment tensor.
    =#
    x = orbit[1,:]
    y = orbit[2,:]

    Ixx = x.^2
    Iyy = y.^2
    Ixy = x.*y
    trace = Ixx .+ Iyy

    if component[1] == 1 && component[2] == 1
        tmp = Ixx .- (1.0 ./ 3.0).*trace
    elseif component[1] == 2 && component[2] == 2
        tmp = Iyy .- (1.0 ./ 3.0).*trace
    else
        tmp = Ixy
    end

    return mass .* tmp
end
```
