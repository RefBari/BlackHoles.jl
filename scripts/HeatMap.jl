using ProgressMeter

include("/Users/rbari/Work/BlackHoles/scripts/NewtonianMetricTest.jl")

optimizeBlackHole(
    learningRate = 5e-3,
    epochsPerIteration = 16,
    numberOfCycles = 8,
    totalTrainingPercent = 0.50,
    true_parameters = [125, 0.8], # Create training data for these (p_0, e_0) values
    initial_guess = [125, 0.8],
)

x = range(6, 15, length = 100)
y = (x .- 6) ./ 2

p_vals = []
e_vals = []

function create_sampling_points(grid_size)
    global p_vals = []
    global e_vals = []

    p_max = 12
    e_max = 1
    e_min = 0.01

    # Maximum eccentricity = 1, so if we want a 3x3 grid, we need ecc = 1/3, so that the ecc values are [0.333, 0.666, 0.999]
    e_spacing = (e_max - e_min)/grid_size
    ecc_vals = e_min:e_spacing:e_max
    push!(e_vals, ecc_vals)

    for e in ecc_vals
        p_min = 6.001+2*e
        p_rows = collect(range(start = p_min, stop = p_max, length = grid_size))
        push!(p_vals, p_rows)

        return p_vals, ecc_vals
    end
end

create_sampling_points(5)

p_values = p_vals[1]
e_values = e_vals[1]
pe_values = []

p = plot(
    x,
    y,
    linewidth = 2,
    xlabel = L"p",
    ylabel = L"e",
    ylims = (0, 1),
    label = "Separatrix",
    top_margin = 10mm,
    bottom_margin = 10mm,
    left_margin = 10mm,
    right_margin = 10mm,
    xlims = (6, 12),
)

for i in p_values[2:end]
    for j in e_values[1:end]
        if i > 6.001 + 2*j
            push!(pe_values, (i, j))
            scatter!([i], [j], legend = false, markersize = 5, color = "blue")
        end
    end
end

M = length(pe_values)
losses = Vector{Float64}(undef, M)
display(p)
@showprogress for i = 1:M
    p, e = pe_values[i]
    losses[i] = optimizeBlackHole(
        learningRate = 5e-3,
        epochsPerIteration = 16,
        numberOfCycles = 6,
        totalTrainingPercent = 0.50,
        true_parameters = [p, e],
        initial_guess = [p, e],
    )
end

ps = first.(pe_values)
es = last.(pe_values)

scatter(ps, es; marker_z = losses, colorbar = true, xlabel = L"p", ylabel = L"e")

display(p)
