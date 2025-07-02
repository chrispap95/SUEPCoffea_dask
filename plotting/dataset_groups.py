"""
Define groups of datasets here. The keys are the group names and the values are
lists of regular expressions that match the dataset names. A group can contain
multiple datasets, other groups, or even single datasets (for simpler naming).

The dataset_group_new dictionary is used for the new UL18 datasets:
    Central MINIAOD -> SUEPNano, November 2024.
"""

dataset_groups_Run2 = {
    "SingleMuon": [
        r"^SingleMuon_Run2016.-UL2016_MiniAODv2_NanoAODv9-v._NANOAOD$",
        r"^SingleMuon_Run2016.-HIPM_UL2016_MiniAODv2_NanoAODv9-v._NANOAOD$",
        r"^SingleMuon_Run2017.-UL2017_MiniAODv2_NanoAODv9_GT36-v._NANOAOD$",
    ],
    "JetHT": [
        r"^JetHT_Run2016.*HIPM_UL2016_MiniAODv2_NanoAODv9-v._NANOAOD$",
        r"^JetHT_Run2016.-UL2016_MiniAODv2_NanoAODv9-v._NANOAOD$",
        r"^JetHT_Run2017.-UL2017_MiniAODv2_NanoAODv9-v._NANOAOD$",
    ],
    "Data": [
        r"^DoubleMuon_Run2016(B-ver1_|B-ver2_|C-|D-|E-|F-)HIPM_UL2016_MiniAODv2-v._MINIAOD$",
        r"^DoubleMuon_Run2016(F|G|H)-UL2016_MiniAODv2-v._MINIAOD$",
        r"^DoubleMuon_Run2017(B|C|D|E|F)-UL2017_MiniAODv2-v._MINIAOD$",
        r"^DoubleMuon_Run2018(A|B|C|D)-UL2018_MiniAODv2_GT36-v._MINIAOD$",
    ],
    "DY_NJets_LO": [
        r"^DY(1|2|3|4)JetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8$",
    ],
    "DY_LHEFilterPtZ_NLO": [
        r"^DYJetsToLL_LHEFilterPtZ-.*_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8$",
    ],
    "DY_M-10to50_inclusive_NLO": [
        r"^DYJetsToLL_M-10to50_TuneCP5_13TeV-amcatnloFXFX-pythia8$",
    ],
    "DY_M-10to50_inclusive_LO": [
        r"^DYJetsToLL_M-10to50_TuneCP5_13TeV-madgraphMLM-pythia8$",
    ],
    "DY_M-50_inclusive_NLO": [
        r"^DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8$",
    ],
    "DYToMuMu_M-10to50_NLO": [
        r"^DYJetsToMuMu_M-10to50_H2ErratumFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos$",
    ],
    "DYToMuMu_M-50_NLO": [
        r"^DYJetsToMuMu_M-50_massWgtFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos$",
    ],
    "DY": ["^DY_LHEFilterPtZ_NLO$", "^DY_M-10to50_inclusive_LO$"],
    "QCD_HT": [
        r"^QCD_HT.*_TuneCP5_13TeV-madgraphMLM-pythia8$",
    ],
    "QCD_Pt_MuEnrichedPt5": [
        r"^QCD_Pt-.*_MuEnrichedPt5_TuneCP5_13TeV-pythia8$",
    ],
    "TT_powheg": [
        r"^TTTo(Hadronic|SemiLeptonic|2L2Nu)_TuneCP5_13TeV-powheg-pythia8$",
    ],
    "TTW_NLO": [
        r"^TTWJetsTo(LNu|QQ)_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8$",
    ],
    "TTZ_NLO": [
        r"^TTZTo(QQ|LL)_TuneCP5_13TeV.amcatnlo-pythia8$",
    ],
    "TTV": ["^TTW_NLO$", "^TTZ_NLO$"],
    "TTZToLLNuNu_M-10_NLO": [
        r"^TTZToLLNuNu_M-10_TuneCP5_13TeV-amcatnlo-pythia8$",
    ],
    "TTZToLL_LO": [
        r"^TTZToLL_5f_TuneCP5_13TeV-madgraphMLM-pythia8$",
    ],
    "TTZ_inclusive_LO": [
        r"^ttZJets_TuneCP5_13TeV_madgraphMLM_pythia8$",
    ],
    "TTTT_NLO": [
        r"^TTTT_TuneCP5_13TeV-amcatnlo-pythia8$",
    ],
    "ST_s-channel_NLO": [
        r"^ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-amcatnlo-pythia8$",
    ],
    "ST_t-channel_powheg": [
        r"^ST_t-channel_(anti)?top_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8$",
    ],
    "ST_tW_Dilept_NLO": [
        r"^ST_tW_Dilept_5f_DR_TuneCP5_13TeV-amcatnlo-pythia8$",
    ],
    "ST_NLO": [
        r"^ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-amcatnlo-pythia8$",
        r"^ST_t-channel_(anti)?top_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8$",
        r"^ST_tW_Dilept_5f_DR_TuneCP5_13TeV-amcatnlo-pythia8$",
    ],
    "ST_tW_powheg": [
        r"^ST_tW_(anti)?top_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8$",
    ],
    "WJetsToLNu_HT_LO": [
        r"^WJetsToLNu_HT-.*_TuneCP5_13TeV-madgraphMLM-pythia8$",
    ],
    "WJetsToLNu_Pt_NLO": [
        r"^WJetsToLNu_Pt-.*_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8$",
    ],
    "WJetsToLNu_inclusive_NLO": [
        r"^WJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-pythia8$",
    ],
    "WJets": [
        r"^WJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-pythia8$",
        r"^WJetsToLNu_HT-.*_TuneCP5_13TeV-madgraphMLM-pythia8$",
    ],
    "WW_NLO": [
        r"^WWTo(1L1Nu2Q|2L2Nu)(_4f)?_TuneCP5_13TeV.(amcatnloFXFX|powheg).pythia8$",
    ],
    "WZ_NLO": [
        r"^WZTo(1L1Nu2Q|2Q2L|3LNu)(_4f)?(_mllmin4p0)?_TuneCP5_13TeV.(amcatnloFXFX|powheg).pythia8$",
    ],
    "ZZ_NLO": [
        r"^ZZTo(2L2Nu|2Q2L|4L)(_mllmin4p0)?_TuneCP5_13TeV.(amcatnloFXFX|powheg).pythia8$",
    ],
    "VV_NLO": [
        r"^(WW|WZ|ZZ)To(1L1Nu2Q|2L2Nu|2Q2L|3LNu|4L)(_4f)?(_mllmin4p0)?_TuneCP5_13TeV.(amcatnloFXFX|powheg).pythia8$",
    ],
    "VVV_NLO": [
        r"^(WWW|WWZ|ZZZ)(_4F)?_TuneCP5_13TeV-amcatnlo-pythia8$",
    ],
    "VV+VVV": ["^VV_NLO$", "^VVV_NLO$"],
    "Higgs": [
        r"^GluGluHToZZTo4L_M125_TuneCP5_13TeV_powheg2_minloHJJ_JHUGenV7011_pythia8$",
        r"^VBF_HToZZTo4L_M125_TuneCP5_13TeV_powheg2_JHUGenV7011_pythia8$",
        r"^ZH_HToZZ_4LFilter_M125_TuneCP5_13TeV_powheg2-minlo-HZJ_JHUGenV7011_pythia8$",
        r"^W(minus|plus)H_HToZZTo4L_M125_TuneCP5_13TeV_powheg2-minlo-HWJ_JHUGenV7011_pythia8$",
        r"^W(minus|plus)H_HToBB_WToLNu_M-125_TuneCP5_13TeV-powheg-pythia8$",
        r"^GluGluToZH_HToZZTo4L_M125_TuneCP5_13TeV-jhugenv723-pythia8$",
        r"^ttH_HToZZ_4LFilter_M125_TuneCP5_13TeV_powheg2_JHUGenV7011_pythia8$",
        # r"^ttHTo(Non)?bb_M125_TuneCP5_13TeV-powheg-pythia8$",
        # r"^VHToNonbb_M125_TuneCP5_13TeV-amcatnloFXFX_madspin_pythia8$",
    ],
    "WH_HToBB_powheg": [
        r"^W(minus|plus)H_HToBB_WToLNu_M-125_TuneCP5_13TeV-powheg-pythia8$",
    ],
    "ttH_powheg": [
        r"^ttHTo(Non)?bb_M125_TuneCP5_13TeV-powheg-pythia8$",
    ],
    "GluGluHToZZTo4L": [
        r"^GluGluHToZZTo4L_M125_TuneCP5_13TeV_powheg2_minloHJJ_JHUGenV7011_pythia8$",
    ],
    "VBF_HToZZTo4L": [
        r"^VBF_HToZZTo4L_M125_TuneCP5_13TeV_powheg2_JHUGenV7011_pythia8$",
    ],
    "ZH_HToZZ_4LFilter": [
        r"^ZH_HToZZ_4LFilter_M125_TuneCP5_13TeV_powheg2-minlo-HZJ_JHUGenV7011_pythia8$",
    ],
    "WH_HToZZTo4L": [
        r"^W(minus|plus)H_HToZZTo4L_M125_TuneCP5_13TeV_powheg2-minlo-HWJ_JHUGenV7011_pythia8$",
    ],
    "WH_HToZZTo4L": [
        r"^W(minus|plus)H_HToZZTo4L_M125_TuneCP5_13TeV_powheg2-minlo-HWJ_JHUGenV7011_pythia8$",
    ],
    "total_bkg": [
        "^Higgs$",
        "^TTV$",
        "^ST_NLO$",
        "^WJets$",
        r"^VV\+VVV$",
        "^TT_powheg$",
        "^DY$",
        "^QCD_Pt_MuEnrichedPt5$",
    ],
}

dataset_groups_Run3 = {
    "Data": [
        "^DoubleMuon_Run2022C-22Sep2023-v1_MINIAOD$",
        "^Muon_Run2022(C|D|E|F|G)-(22Sep2023|19Dec2023)-v._MINIAOD$",
        "^Muon(0|1)_Run2023(C|D)-22Sep2023_v.-v._MINIAOD$",
    ],
    "QCD_Pt_MuEnrichedPt5": [r"^QCD_PT-.*_MuEnrichedPt5_TuneCP5_13p6TeV_pythia8$"],
    "DY_NJets_NLO": [r"^DYto2L-2Jets_MLL-50_.J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8$"],
    "DY_NJets_LO": [r"^DYto2L-4Jets_MLL-50_.J_TuneCP5_13p6TeV_madgraphMLM-pythia8$"],
    "DY_low_mass": ["^DYto2L-4Jets_MLL-10to50_TuneCP5_13p6TeV_madgraphMLM-pythia8$"],
    "DY": [
        "^DY_low_mass$",
        r"^DYto2L-2Jets_MLL-50_(0|1|2)J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8$",
        # r"^DYto2L-4Jets_MLL-50_(3|4)J_TuneCP5_13p6TeV_madgraphMLM-pythia8$",
    ],
    "TT_powheg": [r"^TTto(LNu2Q|2L2Nu|4Q)_TuneCP5_13p6TeV_powheg-pythia8$"],
    "ST_NLO": [
        r"^T(Bbar|barB)Q_t-channel_4FS_TuneCP5_13p6TeV_powheg-madspin-pythia8$",
        r"^T(BbartoLplusNuBbar|barBtoLminusNuB)-s-channel-4FS_TuneCP5_13p6TeV_amcatnlo-pythia8$",
        r"^T(Wminus|barWplus)to2L2Nu_TuneCP5_13p6TeV_powheg-pythia8$",
    ],
    "TTZ_NLO": [
        r"^TTZ_Zto2L_SMEFT_TuneCP5_13p6TeV_amcatnlo-pythia8$",
        r"^TTZ-ZtoQQ-1Jets_TuneCP5_13p6TeV_amcatnloFXFX-pythia8$",
    ],
    "TTV": ["^TTZ_NLO$"],
    "WW": [r"^WWto(2L2Nu|4Q|LNu2Q)_TuneCP5_13p6TeV_powheg-pythia8$"],
    "WZ": [
        r"^WZto(2L2Q|3LNu)_TuneCP5_13p6TeV_powheg-pythia8$",
        "^WZto4Q-1Jets-4FS_TuneCP5_13p6TeV_amcatnloFXFX-pythia8$",
    ],
    "ZZ": [r"^ZZto(2L2Q|4L)_TuneCP5_13p6TeV_powheg-pythia8$"],
    "VV": ["^WW$", "^WZ$", "^ZZ$"],
    "WWW": ["^WWW_4F_TuneCP5_13p6TeV_amcatnlo-madspin-pythia8$"],
    "WWZ": ["^WWZ_4F_TuneCP5_13p6TeV_amcatnlo-pythia8$"],
    "WZZ": ["^WZZ_TuneCP5_13p6TeV_amcatnlo-pythia8$"],
    "ZZZ": ["^ZZZ_TuneCP5_13p6TeV_amcatnlo-pythia8$"],
    "VVV": ["^WWW$", "^WWZ$", "^WZZ$", "^ZZZ$"],
    "VV+VVV": ["^VV$", "^VVV$"],
    "WtoLNu_NLO": [r"^WtoLNu-2Jets_.J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8$"],
    "WtoLNu_LO": [r"^WtoLNu-4Jets_.J_TuneCP5_13p6TeV_madgraphMLM-pythia8$"],
    "WJets": [
        r"^WtoLNu-2Jets_(0|1|2)J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8$",
        # r"^WtoLNu-4Jets_(3|4)J_TuneCP5_13p6TeV_madgraphMLM-pythia8$",
        r"^Wto2Q-3Jets_HT-.*_TuneCP5_13p6TeV_madgraphMLM-pythia8$",
    ],
    "Higgs": [
        # NOTE: Some of these appear to be duplicates, but they correspond only to 2022 or 2023
        "^GluGluHtoZZto4L_M-125_TuneCP5_13p6TeV_powheg2-JHUGenV752-pythia8$",
        "^GluGluHtoZZto4L_M-125_TuneCP5_13p6TeV_powheg-jhugen-pythia8$",
        "^VBFHto2Zto4L_M125_TuneCP5_13p6TeV_powheg-jhugenv752-pythia8$",
        "^VBFHto2Zto4L_M-125_TuneCP5_13p6TeV_powheg-jhugen-pythia8$",
        r"^W(minus|plus)H_Hto2Zto4L_M-125_TuneCP5_13p6TeV_powheg2-minlo-HWJ-JHUGenV752-pythia8$",
        "^ZHto2Zto4L_M125_TuneCP5_13p6TeV_powheg2-minlo-HZJ-JHUGenV752-pythia8$",
        "^ZH_Hto2Z_4LFilter_M-125_TuneCP5_13p6TeV_powheg-jhugenv752-pythia8$",
        "^TTH_Hto2Z_M-125_4LFilter_TuneCP5_13p6TeV_powheg2-JHUGenV752-pythia8$",
        "^TTH_Hto2Z_4LFilter_M-125_TuneCP5_13p6TeV_powheg-jhugenv752-pythia8$",
        "^bbH_Hto2Zto4L_M-125_TuneCP5_13p6TeV_JHUGenV752-pythia8$",
    ],
    "total_bkg": [
        "^Higgs$",
        "^WJets$",
        r"^VV\+VVV$",
        "^TT_powheg$",
        "^DY$",
        "^QCD_Pt_MuEnrichedPt5$",
    ],
}
