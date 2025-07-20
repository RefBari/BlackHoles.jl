module BlackHoles

export greet, greetalien


greet() = print("Hello World!")

"""
Add two numbers together
"""
function addition(a::Int, b::Int)
    return a + b
end



"""
Convert semi-latus rectum and eccentricity to mass, energy, and angular momentum.
"""
function pe_2_EL(semilatusRectum::Float64, eccentricity::Float64)
    p = semilatusRectum
    e = eccentricity

    M = 1
    E = (e^2 - 1)/(2*p)
    L = sqrt(p)

    return [M, E, L]
end

end # module BlackHoles
