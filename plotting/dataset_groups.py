"""
Define groups of datasets here. The keys are the group names and the values are
lists of regular expressions that match the dataset names. A group can contain
multiple datasets, other groups, or even single datasets (for simpler naming).

The dataset_group_new dictionary is used for the new UL18 datasets:
    Central MINIAOD -> SUEPNano, November 2024.

The dataset_group_old dictionary is used for the old UL18 datasets:
    Central NANOAOD -> SUEPSkimmer.
"""

dataset_groups_old = {
    "QCD_Pt_MuEnrichedPt5": [
        r"QCD_Pt-.*_MuEnrichedPt5_TuneCP5_13TeV-pythia8.*UL18.*NANOAODSIM$",
    ],
    "TT_powheg": [
        r"TTTo.*_TuneCP5_13TeV-powheg-pythia8.*UL18.*NANOAODSIM$",
    ],
    "DY_inclusive_NLO": [
        r"DYJetsToLL_M-.*_TuneCP5_13TeV-amcatnloFXFX-pythia8.*UL18.*NANOAODSIM$",
    ],
    "ST_NLO": [
        r"ST_t-channel_.*_5f_InclusiveDecays_TuneCP5_13TeV-powheg-pythia8.*UL18.*NANOAODSIM$",
        r"ST_tW_Dilept_5f_DR_TuneCP5_13TeV-amcatnlo-pythia8.*UL18.*NANOAODSIM$",
    ],
    "WJetsToLNu_HT_LO": [
        r"WJetsToLNu_HT-.*_TuneCP5_13TeV-madgraphMLM-pythia8.*UL18.*NANOAODSIM$",
    ],
    "WJetsToLNu_inclusive_NLO": [
        r"WJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-pythia8.*UL18.*NANOAODSIM$",
    ],
    "WJetsToLNu": [
        "WJetsToLNu_HT_LO",
        "WJetsToLNu_inclusive_NLO",
    ],
    "VV_NLO": [
        r"WWTo1L1Nu2Q_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8.*UL18.*NANOAODSIM$",
        r"WWTo2L2Nu_TuneCP5_13TeV-powheg-pythia8.*UL18.*NANOAODSIM$",
        r"WZTo1L1Nu2Q_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8.*UL18.*NANOAODSIM$",
        r"WZTo1L3Nu_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8.*UL18.*NANOAODSIM$",
        r"WZTo2Q2L_mllmin4p0_TuneCP5_13TeV-amcatnloFXFX-pythia8.*UL18.*NANOAODSIM$",
        r"WZTo3LNu_mllmin4p0_TuneCP5_13TeV-powheg-pythia8.*UL18.*NANOAODSIM$",
        r"ZZTo4L_TuneCP5_13TeV_powheg_pythia8.*UL18.*NANOAODSIM$",
    ],
    "VVV_NLO": [
        r"WWZ_4F_TuneCP5_13TeV-amcatnlo-pythia8.*UL18.*NANOAODSIM$",
        r"ZZZ_TuneCP5_13TeV-amcatnlo-pythia8.*UL18.*NANOAODSIM$",
    ],
    "TTZ_inclusive_LO": [
        r"ttZJets_TuneCP5_13TeV_madgraphMLM_pythia8.*UL18.*NANOAODSIM$",
    ],
}


dataset_groups_new = {
    "DY_NJets_LO": [
        r"DY(1|2|3|4)JetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8",
    ],
    "DY_LHEFilterPtZ_NLO": [
        r"DYJetsToLL_LHEFilterPtZ-.*_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8",
    ],
    "DY_M-10to50_inclusive_NLO": [
        r"DYJetsToLL_M-10to50_TuneCP5_13TeV-amcatnloFXFX-pythia8",
    ],
    "DY_M-10to50_inclusive_LO": [
        r"DYJetsToLL_M-10to50_TuneCP5_13TeV-madgraphMLM-pythia8",
    ],
    "DY_M-50_inclusive_NLO": [
        r"DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8",
    ],
    "QCD_Pt_MuEnrichedPt5": [
        r"QCD_Pt-.*_MuEnrichedPt5_TuneCP5_13TeV-pythia8",
    ],
    "TT_powheg": [
        r"TTTo(Hadronic|SemiLeptonic|2L2Nu)_TuneCP5_13TeV-powheg-pythia8",
    ],
    "TTW_NLO": [
        r"TTWJetsTo(LNu|QQ)_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8",
    ],
    "TTZ_NLO": [
        r"TTZTo(QQ|LL)_TuneCP5_13TeV.amcatnlo-pythia8",
    ],
    "TTZToLLNuNu_M-10_NLO": [
        r"TTZToLLNuNu_M-10_TuneCP5_13TeV-amcatnlo-pythia8",
    ],
    "TTZToLL_LO": [
        r"TTZToLL_5f_TuneCP5_13TeV-madgraphMLM-pythia8",
    ],
    "TTZ_inclusive_LO": [
        r"ttZJets_TuneCP5_13TeV_madgraphMLM_pythia8",
    ],
    "TTTT_NLO": [
        r"TTTT_TuneCP5_13TeV-amcatnlo-pythia8",
    ],
    "ST_s-channel_NLO": [
        r"ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-amcatnlo-pythia8",
    ],
    "ST_t-channel_powheg": [
        r"ST_t-channel_(anti)?top_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8",
    ],
    "ST_tW_Dilept_NLO": [
        r"ST_tW_Dilept_5f_DR_TuneCP5_13TeV-amcatnlo-pythia8",
    ],
    "ST_NLO": [
        r"ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-amcatnlo-pythia8",
        r"ST_t-channel_(anti)?top_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8",
        r"ST_tW_Dilept_5f_DR_TuneCP5_13TeV-amcatnlo-pythia8",
    ],
    "ST_tW_powheg": [
        r"ST_tW_(anti)?top_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8",
    ],
    "WJetsToLNu_HT_LO": [
        r"WJetsToLNu_HT-.*_TuneCP5_13TeV-madgraphMLM-pythia8",
    ],
    "WJetsToLNu_Pt_NLO": [
        r"WJetsToLNu_Pt-.*_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8",
    ],
    "WJetsToLNu_inclusive_NLO": [
        r"WJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-pythia8",
    ],
    "WJetsToLNu_total_inclusive": [
        r"WJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-pythia8",
        r"WJetsToLNu_HT-.*_TuneCP5_13TeV-madgraphMLM-pythia8",
    ],
    "WW_NLO": [
        r"WWTo(1L1Nu2Q|2L2Nu)(_4f)?_TuneCP5_13TeV.(amcatnloFXFX|powheg).pythia8",
    ],
    "WZ_NLO": [
        r"WZTo(1L1Nu2Q|2Q2L|3LNu)(_4f)?(_mllmin4p0)?_TuneCP5_13TeV.(amcatnloFXFX|powheg).pythia8",
    ],
    "ZZ_NLO": [
        r"ZZTo(2L2Nu|2Q2L|4L)(_mllmin4p0)?_TuneCP5_13TeV.(amcatnloFXFX|powheg).pythia8",
    ],
    "VV_NLO": [
        r"(WW|WZ|ZZ)To(1L1Nu2Q|2L2Nu|2Q2L|3LNu|4L)(_4f)?(_mllmin4p0)?_TuneCP5_13TeV.(amcatnloFXFX|powheg).pythia8",
    ],
    "VVV_NLO": [
        r"(WWW|WWZ|ZZZ)(_4F)?_TuneCP5_13TeV-amcatnlo-pythia8",
    ],
    "WH_HToBB_powheg": [
        r"W(minus|plus)H_HToBB_WToLNu_M-125_TuneCP5_13TeV-powheg-pythia8",
    ],
    "ttH_powheg": [
        r"ttHTo(Non)?bb_M125_TuneCP5_13TeV-powheg-pythia8",
    ],
    "Higgs": [
        r"GluGluHToZZTo4L_M125_TuneCP5_13TeV_powheg2_minloHJJ_JHUGenV7011_pythia8",
        r"VBF_HToZZTo4L_M125_TuneCP5_13TeV_powheg2_JHUGenV7011_pythia8",
        r"ZH_HToZZ_4LFilter_M125_TuneCP5_13TeV_powheg2-minlo-HZJ_JHUGenV7011_pythia8",
        r"W(minus|plus)H_HToZZTo4L_M125_TuneCP5_13TeV_powheg2-minlo-HWJ_JHUGenV7011_pythia8",
        r"W(minus|plus)H_HToBB_WToLNu_M-125_TuneCP5_13TeV-powheg-pythia8",
        r"GluGluToZH_HToZZTo4L_M125_TuneCP5_13TeV-jhugenv723-pythia8",
        r"ttH_HToZZ_4LFilter_M125_TuneCP5_13TeV_powheg2_JHUGenV7011_pythia8",
        # r"ttHTo(Non)?bb_M125_TuneCP5_13TeV-powheg-pythia8",
        # r"VHToNonbb_M125_TuneCP5_13TeV-amcatnloFXFX_madspin_pythia8",
    ],
}
