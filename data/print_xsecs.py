import json

datasets = [
    ("DYto2L-2Jets_MLL-50_0J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8", "DY", "NLO(NLO)"),
    ("DYto2L-2Jets_MLL-50_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8", "DY", "NLO(NLO)"),
    ("DYto2L-2Jets_MLL-50_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8", "DY", "NLO(NLO)"),
    ("DYto2L-4Jets_MLL-10to50_TuneCP5_13p6TeV_madgraphMLM-pythia8", "DY", "LO(NLO)"),
    ("QCD_PT-15to20_MuEnrichedPt5_TuneCP5_13p6TeV_pythia8", "QCD", "LO(LO)"),
    ("QCD_PT-20to30_MuEnrichedPt5_TuneCP5_13p6TeV_pythia8", "QCD", "LO(LO)"),
    ("QCD_PT-30to50_MuEnrichedPt5_TuneCP5_13p6TeV_pythia8", "QCD", "LO(LO)"),
    ("QCD_PT-50to80_MuEnrichedPt5_TuneCP5_13p6TeV_pythia8", "QCD", "LO(LO)"),
    ("QCD_PT-80to120_MuEnrichedPt5_TuneCP5_13p6TeV_pythia8", "QCD", "LO(LO)"),
    ("QCD_PT-120to170_MuEnrichedPt5_TuneCP5_13p6TeV_pythia8", "QCD", "LO(LO)"),
    ("QCD_PT-170to300_MuEnrichedPt5_TuneCP5_13p6TeV_pythia8", "QCD", "LO(LO)"),
    ("QCD_PT-300to470_MuEnrichedPt5_TuneCP5_13p6TeV_pythia8", "QCD", "LO(LO)"),
    ("QCD_PT-470to600_MuEnrichedPt5_TuneCP5_13p6TeV_pythia8", "QCD", "LO(LO)"),
    ("QCD_PT-600to800_MuEnrichedPt5_TuneCP5_13p6TeV_pythia8", "QCD", "LO(LO)"),
    ("QCD_PT-800to1000_MuEnrichedPt5_TuneCP5_13p6TeV_pythia8", "QCD", "LO(LO)"),
    ("QCD_PT-1000_MuEnrichedPt5_TuneCP5_13p6TeV_pythia8", "QCD", "LO(LO)"),
    ("TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8", "TT", "NLO(NLO)"),
    ("TTto4Q_TuneCP5_13p6TeV_powheg-pythia8", "TT", "NLO(NLO)"),
    ("TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8", "TT", "NLO(NLO)"),
    (
        "Wto2Q-3Jets_HT-200to400_TuneCP5_13p6TeV_madgraphMLM-pythia8",
        "WJets",
        "LO(NLO)",
    ),
    (
        "Wto2Q-3Jets_HT-400to600_TuneCP5_13p6TeV_madgraphMLM-pythia8",
        "WJets",
        "LO(NLO)",
    ),
    (
        "Wto2Q-3Jets_HT-600to800_TuneCP5_13p6TeV_madgraphMLM-pythia8",
        "WJets",
        "LO(NLO)",
    ),
    ("Wto2Q-3Jets_HT-800_TuneCP5_13p6TeV_madgraphMLM-pythia8", "WJets", "LO(NLO)"),
    ("WtoLNu-2Jets_0J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8", "WJets", "NLO(NLO)"),
    ("WtoLNu-2Jets_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8", "WJets", "NLO(NLO)"),
    ("WtoLNu-2Jets_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8", "WJets", "NLO(NLO)"),
    ("WWto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8", "VV+VVV", "NLO(NLO)"),
    ("WWto4Q_TuneCP5_13p6TeV_powheg-pythia8", "VV+VVV", "NLO(NLO)"),
    ("WWtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8", "VV+VVV", "NLO(NLO)"),
    ("WZto2L2Q_TuneCP5_13p6TeV_powheg-pythia8", "VV+VVV", "NLO(NLO)"),
    ("WZto3LNu_TuneCP5_13p6TeV_powheg-pythia8", "VV+VVV", "NLO(NLO)"),
    ("WZto4Q-1Jets-4FS_TuneCP5_13p6TeV_amcatnloFXFX-pythia8", "VV+VVV", "NLO(NLO)"),
    ("ZZto2L2Q_TuneCP5_13p6TeV_powheg-pythia8", "VV+VVV", "NLO(NLO)"),
    ("ZZto4L_TuneCP5_13p6TeV_powheg-pythia8", "VV+VVV", "NLO(NLO)"),
    ("WWW_4F_TuneCP5_13p6TeV_amcatnlo-madspin-pythia8", "VV+VVV", "NLO(NLO)"),
    ("WWZ_4F_TuneCP5_13p6TeV_amcatnlo-pythia8", "VV+VVV", "NLO(NLO)"),
    ("WZZ_TuneCP5_13p6TeV_amcatnlo-pythia8", "VV+VVV", "NLO(NLO)"),
    ("ZZZ_TuneCP5_13p6TeV_amcatnlo-pythia8", "VV+VVV", "NLO(NLO)"),
    (
        "GluGluHtoZZto4L_M-125_TuneCP5_13p6TeV_powheg2-JHUGenV752-pythia8",
        "Higgs",
        "NLO(NNLO)",
    ),
    (
        "VBFHto2Zto4L_M125_TuneCP5_13p6TeV_powheg-jhugenv752-pythia8",
        "Higgs",
        "NLO(NNLO)",
    ),
    (
        "WminusH_Hto2Zto4L_M-125_TuneCP5_13p6TeV_powheg2-minlo-HWJ-JHUGenV752-pythia8",
        "Higgs",
        "NLO(NNLO)",
    ),
    (
        "WplusH_Hto2Zto4L_M-125_TuneCP5_13p6TeV_powheg2-minlo-HWJ-JHUGenV752-pythia8",
        "Higgs",
        "NLO(NNLO)",
    ),
    (
        "ZHto2Zto4L_M125_TuneCP5_13p6TeV_powheg2-minlo-HZJ-JHUGenV752-pythia8",
        "Higgs",
        "NLO(NNLO)",
    ),
    ("bbH_Hto2Zto4L_M-125_TuneCP5_13p6TeV_JHUGenV752-pythia8", "Higgs", "NLO(NNLO)"),
    (
        "TTH_Hto2Z_M-125_4LFilter_TuneCP5_13p6TeV_powheg2-JHUGenV752-pythia8",
        "Higgs",
        "NLO(NNLO)",
    ),
]

with open("xsections_bkg_13p6TeV.json", "r") as f:
    xsections = json.load(f)
for dataset, group, accuracy in datasets:
    if dataset in xsections:
        xsec = (
            xsections[dataset]["xsec"]
            * xsections[dataset]["br"]
            / xsections[dataset]["eff"]
        )
        print(dataset.replace("_", r"\_") + f" & {xsec} & {group} & {accuracy} \\\\")
    else:
        raise KeyError(f"{dataset}: Not found in xsections data")
