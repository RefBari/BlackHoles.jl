using ScatteringOptics
using CairoMakie
using VLBISkyModels

im = load_fits("data/jason_mad_eofn.fits", IntensityMap);

imageviz(im, size=(600, 500), colormap=:afmhot)

sm = ScatteringModel()

νref = VLBISkyModels.metadata(im).frequency
print("Frequency: ", νref/1e9," GHz")

skm = kernelmodel(sm, νref=νref)

im_ea = convolve(im, skm);

imageviz(im_ea, size=(600, 500), colormap=:afmhot)

u = LinRange(0, 10e9, 1000)
vis = [visibility_point(skm, (U=u, V=0, Fr=νref)) for u=u]

f = Figure()
ax = CairoMakie.Axis(f[1, 1],
            xlabel = "Baseline Length(Gλ)",
            ylabel = "Kernel Amplitudes")
lines!(ax, u/1e9, abs.(vis))
f

rps = refractivephasescreen(sm, im)

using StableRNGs

rng = StableRNG(123)
noise_screen = generate_gaussian_noise(rps; rng = rng)
im_a = scatter_image(rps, im; noise_screen = noise_screen);
imageviz(im_a, size = (600, 500), colormap=:afmhot)