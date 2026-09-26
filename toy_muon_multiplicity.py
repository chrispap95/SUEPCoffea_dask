"""Simple toy: two event populations (2-jet and 8-jet) where each jet's b quark
decays to a muon with 10% probability. Count muons per event and fill a
HEP-style histogram (scikit-hep `hist` + `mplhep`)."""

import hist
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

rng = np.random.default_rng(42)

N_EVENTS = 10000
P_TWO_JET = 0.99  # fraction of events with 2 jets
P_MUON = 0.10  # probability a jet gives a muon

# Choose number of jets for each event: 2 jets (99%) or 8 jets (1%)
n_jets = np.where(rng.random(N_EVENTS) < P_TWO_JET, 2, 8)

# For each event, dice each jet and count muons
n_muons = np.array([np.sum(rng.random(nj) < P_MUON) for nj in n_jets])

# Fill a HEP-style integer-binned histogram, with a category axis splitting the
# two jet populations so each can be drawn as its own histogram.
h = hist.Hist(
    hist.axis.Integer(0, 9, name="n_muons", label="Number of muons"),
    hist.axis.StrCategory(["2-jet", "8-jet"], name="pop", label="Population"),
)
h.fill(n_muons=n_muons, pop=np.where(n_jets == 2, "2-jet", "8-jet"))

print(
    f"Generated {N_EVENTS} events "
    f"({np.sum(n_jets == 2)} with 2 jets, {np.sum(n_jets == 8)} with 8 jets)\n"
)
print(h)

# Plot with CMS style, log-y to expose the high-multiplicity tail
hep.style.use("CMS")
fig, ax = plt.subplots()
h[:, "2-jet"].plot(ax=ax, label="2-jet")
h[:, "8-jet"].plot(ax=ax, label="8-jet")
h[:, sum].plot(ax=ax, color="black", label="total")
ax.set_yscale("log")
ax.set_ylim(0.5, None)
ax.set_ylabel("Events")
ax.legend()
hep.cms.label("Toy", ax=ax, data=False, rlabel="")

outfile = "toy_muon_multiplicity.png"
fig.savefig(outfile, bbox_inches="tight")
print(f"\nSaved plot to {outfile}")
