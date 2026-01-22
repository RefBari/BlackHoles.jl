open("output.txt", "w") do io
    println(io, "Hello from Julia!")
    # your existing stuff:
    x = 42
    println(io, "x = $x")
end
