import matplotlib.pyplot as plt

# HLT paths with (start run, duration in runs)
hlt_paths = [
    ("HLT_TripleMu_12_10_5_v", (297046, 306460 - 297046)),  # full year
    ("HLT_TripleMu_10_5_5_DZ_v", (297046, 306460 - 297046)),  # full year
    ("HLT_TripleMu_5_3_3_Mass3p8to60_DZ_v", (302509, 306460 - 302509)),  # second half
]

# Colors for each path
colors = ["skyblue", "skyblue", "lightgreen", "salmon"]

# Plot
fig, ax = plt.subplots(figsize=(12, 3))

for i, (name, (start, duration)) in enumerate(hlt_paths):
    ax.broken_barh(
        [(start, duration)], (i - 0.4, 0.8), facecolors=colors[i], label=name
    )

# Customize plot
ax.set_ylim(-1, len(hlt_paths))
ax.set_xlim(297000, 307000)  # covers full range
ax.set_xlabel("Run Number")
ax.set_yticks(range(len(hlt_paths)))
ax.set_yticklabels([name for name, _ in hlt_paths])
ax.grid(True)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2)

plt.tight_layout()
plt.savefig("hlt_ranges.pdf")
