# Binary Black Holes: From Gravity Waves to Orbits
#### Ref Bari (Brown University, Physics MS Thesis)

A binary black hole system consists of two black holes orbiting each other. As these black holes orbit each other, they release ripples through the fabric of space-time. These ripples are known as "gravitational waves". They were first predicted in 1916 by Einstein himself, shortly after he discovered the General Theory of Relativity. Einstein speculated that gravitational waves would be too faint to detect. Nearly exactly a century to his discovery, the Laser Interferometer Gravitational-Wave Obsevatory (LIGO) made the first direct detection of a gravitational wave on September 14, 2015 (hence the gravitational wave signal was aptly-named GW150914). This landmark discovery resulted in the 2017 Nobel Prize in Physics being awarded to Rainer Weiss, Barry Barish, and Kip Thorne -- the three founding scientists behind LIGO. This project, Neural DynAMO, is dedicated to the discovery and prediction of the spacetime metric of a binary black hole system from its gravitational wave signal alone. 

![Gravitational Wave Signal](GW250114.mp4)

Binary Black Holes are a fascinating physical system, unifying the small [^Abott2016] and the grand [^Thorne1995]. For a binary black hole system, the dominant $h^{22}$ mode is given by 
 
[^Abott2016]:
    > Amplitudes on the order of $\sim 10^{-21}$
    Abbott, Benjamin P., et al. "Properties of the binary black hole merger GW150914." Physical review letters 116.24 (2016): 241102.

[^Thorne1995]:
    > Lumonisities $>10^{47}$ ergs, brighter than all the stars in the universe
    Thorne, Kip S. "Gravitational waves." arXiv preprint gr-qc/9506086 (1995).
    
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
