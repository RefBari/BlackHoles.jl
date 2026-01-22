
"""
CREATE ANIMATION OF GRAVITATIONAL WAVE (OPTIONAL)
"""
# Compare apples-to-apples with your predicted waveform

# plot(tsteps, waveform_real_ecc, label = "True (original)")
# plot!(tsteps, h_plus_true, label="True (used compute_waveform on true x(t), y(t))", legend=:topright)# plot!(tsteps, waveform_nn_real_dual, label="Predicted")
# plot!(tsteps, waveform_nn_real_dual, label = "ODE Model")

# === Concurrent orbit + waveform animation (Plots.jl / GR backend) ===
# using Plots
# gr()

# # Pull out arrays for readability
# x1, y1 = blackHole_r1[1, :], blackHole_r1[2, :]
# x2, y2 = blackHole_r2[1, :], blackHole_r2[2, :]
# t      = collect(tsteps)                    # ensure it's a Vector
# hplus  = h_plus_true

# # Axes limits that remain fixed (keeps the movie from "breathing")
# xymax = 1.05 * maximum(abs.([x1; y1; x2; y2]))
# ylims_wave = 1.10 .* (minimum(hplus), maximum(hplus))

# # Animation tuning knobs
# stride    = 10      # plot every Nth sample for speed (lower = smoother)
# trail_len = 400     # how many past points to keep as a tail
# fps       = 30
# # choose fixed colors so the faint + bright traces match every frame
# c1, c2 = :royalblue, :crimson

# anim = @animate for k in 1:stride:length(t)
#     i0 = max(1, k - trail_len):k

#     # --- Left: full faint paths + bright recent trails + current positions ---
#     plt_orbit = plot(
#         xlims=(-xymax, xymax), ylims=(-xymax, xymax),
#         aspect_ratio=:equal, legend=false,
#         xlabel=L"x", ylabel=L"y", title="Black Hole Orbits (Ref)",
#         left_margin=6mm, right_margin=4mm, top_margin=4mm, bottom_margin=6mm
#     )

#     # full paths so far (low opacity, stay on screen)
#     plot!(plt_orbit, x1[1:k], y1[1:k], lw=2, color=c1, linealpha=0.25)
#     plot!(plt_orbit, x2[1:k], y2[1:k], lw=2, color=c2, linealpha=0.25)

#     # recent trails (bright)
#     plot!(plt_orbit, x1[i0], y1[i0], lw=3, color=c1)
#     plot!(plt_orbit, x2[i0], y2[i0], lw=3, color=c2)

#     # current positions
#     scatter!(plt_orbit, [x1[k]], [y1[k]], ms=6, color=c1)
#     scatter!(plt_orbit, [x2[k]], [y2[k]], ms=6, color=c2)

#     # center of mass
#     scatter!(plt_orbit, [0.0], [0.0], ms=3, color=:black)

#     # --- Right: waveform panel stays as you had it ---
#     plt_wave = plot(
#         t[1:k], hplus[1:k], lw=2, label="h₊",
#         xlims=(t[1], t[end]), ylims=ylims_wave,
#         xlabel=L"t", ylabel=L"h(t)", title="Gravitational Wave",
#         left_margin=6mm, right_margin=6mm, top_margin=4mm, bottom_margin=6mm
#     )
#     scatter!(plt_wave, [t[k]], [hplus[k]], ms=5)

#     plot(plt_orbit, plt_wave, layout=(1,2), size=(1200, 480), margin=10mm)  
# end


# # Save to GIF (requires ImageMagick or ffmpeg via Plots.jl)
# gif(anim, "bbh_orbit_plus_waveform.gif", fps=fps)

# # Optional MP4 (requires ffmpeg installed and detected by Plots.jl)
# # mp4(anim, "bbh_orbit_plus_waveform.mp4", fps=fps)
