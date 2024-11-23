higgs_modes_xs = {
    "ggH": 48.58,
    "VBF": 3.782,
    "WminusH": 0.5328,
    "WplusH": 0.8400,
    "ZH": 0.8839,
    "ttH": 0.5071,
    # "WminusHToLNuH": 0.05983,
    # "WplusHToLNuH": 0.09426,
}

higgs_brs = {
    "HToZZ": 0.02619,
    "HTobb": 0.5824,
    "HToNonbb": 0.4176,
}

Z_brs = {
    "ZToEE": 0.03366,
    "ZToMuMu": 0.03366,
    "ZToTauTau": 0.03366,
}

for mode in higgs_modes_xs:
    xs = (
        higgs_modes_xs[mode]
        * higgs_brs["HToZZ"]
        * (Z_brs["ZToEE"] + Z_brs["ZToMuMu"]) ** 2
    )
    # print(f"{mode}, HToZZTo4L (L=e, μ): {xs:.5f} pb")
    xs = (
        higgs_modes_xs[mode]
        * higgs_brs["HToZZ"]
        * (Z_brs["ZToEE"] + Z_brs["ZToMuMu"] + Z_brs["ZToTauTau"]) ** 2
    )
    print(f"{mode}, HToZZTo4L (L=e, μ, τ): {xs:.5f} pb")
    print()

zz_xs = 16.5
# print(f"ZZ, ZToLL (L=e, μ): {zz_xs * (Z_brs['ZToEE'] + Z_brs['ZToMuMu'])**2:.5f} pb")
print(
    f"ZZTo4L (L=e, μ, τ): {zz_xs * (Z_brs['ZToEE'] + Z_brs['ZToMuMu'] + Z_brs['ZToTauTau'])**2:.5f} pb"
)
print(
    f"if we include neutrinos: {4 * zz_xs * (Z_brs['ZToEE'] + Z_brs['ZToMuMu'] + Z_brs['ZToTauTau'])**2:.5f} pb"
)
