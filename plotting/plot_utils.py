import os
import glob
import re
import pickle
from typing import Optional

# https://twiki.cern.ch/twiki/bin/viewauth/CMS/RA2b13TeVProduction#Dataset_luminosities_2016_pb_1
lumis = {
    "2016_apv": 19497.914,
    "2016": 16810.813,
    "2017": 41471.589,
    # NOTE: 2018 lumi is only for the main trigger path
    "2018": 54540.000,
}


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


# dataset_groups_old = {
#     "QCD_Pt_MuEnrichedPt5": [
#         "QCD_Pt-1000_MuEnrichedPt5_TuneCP5_13TeV-pythia8+RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2+NANOAODSIM",
#         "QCD_Pt-120To170_MuEnrichedPt5_TuneCP5_13TeV-pythia8+RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2+NANOAODSIM",
#         "QCD_Pt-15To20_MuEnrichedPt5_TuneCP5_13TeV-pythia8+RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2+NANOAODSIM",
#         "QCD_Pt-170To300_MuEnrichedPt5_TuneCP5_13TeV-pythia8+RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2+NANOAODSIM",
#         "QCD_Pt-20To30_MuEnrichedPt5_TuneCP5_13TeV-pythia8+RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2+NANOAODSIM",
#         "QCD_Pt-300To470_MuEnrichedPt5_TuneCP5_13TeV-pythia8+RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2+NANOAODSIM",
#         "QCD_Pt-30To50_MuEnrichedPt5_TuneCP5_13TeV-pythia8+RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2+NANOAODSIM",
#         "QCD_Pt-470To600_MuEnrichedPt5_TuneCP5_13TeV-pythia8+RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2+NANOAODSIM",
#         "QCD_Pt-50To80_MuEnrichedPt5_TuneCP5_13TeV-pythia8+RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2+NANOAODSIM",
#         "QCD_Pt-600To800_MuEnrichedPt5_TuneCP5_13TeV-pythia8+RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2+NANOAODSIM",
#         "QCD_Pt-800To1000_MuEnrichedPt5_TuneCP5_13TeV-pythia8+RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2+NANOAODSIM",
#         "QCD_Pt-80To120_MuEnrichedPt5_TuneCP5_13TeV-pythia8+RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2+NANOAODSIM",
#     ],
#     "TT_powheg": [
#         "TTToHadronic_TuneCP5_13TeV-powheg-pythia8+RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1+NANOAODSIM",
#         "TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8+RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1+NANOAODSIM",
#         "TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8+RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1+NANOAODSIM",
#     ],
#     "DY_inclusive_NLO": [
#         "DYJetsToLL_M-10to50_TuneCP5_13TeV-amcatnloFXFX-pythia8+RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1+NANOAODSIM",
#         "DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8+RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2+NANOAODSIM",
#     ],
#     "ST_NLO": [
#         "ST_t-channel_antitop_5f_InclusiveDecays_TuneCP5_13TeV-powheg-pythia8+RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1+NANOAODSIM",
#         "ST_t-channel_top_5f_InclusiveDecays_TuneCP5_13TeV-powheg-pythia8+RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1+NANOAODSIM",
#         "ST_tW_Dilept_5f_DR_TuneCP5_13TeV-amcatnlo-pythia8+RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2+NANOAODSIM",
#     ],
#     "WJetsToLNu_HT_LO": [
#         "WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8+RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1+NANOAODSIM",
#         "WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8+RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1+NANOAODSIM",
#         "WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8+RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1+NANOAODSIM",
#         "WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8+RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2+NANOAODSIM",
#         "WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8+RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1+NANOAODSIM",
#         "WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8+RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1+NANOAODSIM",
#         "WJetsToLNu_HT-70To100_TuneCP5_13TeV-madgraphMLM-pythia8+RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1+NANOAODSIM",
#         "WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8+RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1+NANOAODSIM",
#     ],
#     "WJetsToLNu_inclusive_NLO": [
#         "WJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-pythia8+RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2+NANOAODSIM",
#     ],
#     "WJetsToLNu": [
#         "WJetsToLNu_HT_LO",
#         "WJetsToLNu_inclusive_NLO",
#     ],
#     "VV_NLO": [
#         "WWTo1L1Nu2Q_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8+RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1+NANOAODSIM",
#         "WWTo2L2Nu_TuneCP5_13TeV-powheg-pythia8+RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2+NANOAODSIM",
#         "WZTo1L1Nu2Q_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8+RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1+NANOAODSIM",
#         "WZTo1L3Nu_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8+RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1+NANOAODSIM",
#         "WZTo2Q2L_mllmin4p0_TuneCP5_13TeV-amcatnloFXFX-pythia8+RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1+NANOAODSIM",
#         "WZTo3LNu_mllmin4p0_TuneCP5_13TeV-powheg-pythia8+RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2+NANOAODSIM",
#         "ZZTo4L_TuneCP5_13TeV_powheg_pythia8+RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2+NANOAODSIM",
#     ],
#     "VVV_NLO": [
#         "WWZ_4F_TuneCP5_13TeV-amcatnlo-pythia8+RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1_ext1-v2+NANOAODSIM",
#         "ZZZ_TuneCP5_13TeV-amcatnlo-pythia8+RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1_ext1-v2+NANOAODSIM",
#     ],
#     "TTZ_inclusive_LO": [
#         "ttZJets_TuneCP5_13TeV_madgraphMLM_pythia8+RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2+NANOAODSIM",
#     ],
# }


# dataset_groups = {
#     "DY_NJets_LO": [
#         "/DY1JetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer19UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1/MINIAODSIM",
#         "/DY2JetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer19UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1/MINIAODSIM",
#         "/DY3JetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer19UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1/MINIAODSIM",
#         "/DY4JetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer19UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1/MINIAODSIM",
#     ],
#     "DY_LHEFilterPtZ_NLO": [
#         "/DYJetsToLL_LHEFilterPtZ-0To50_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/MINIAODSIM",
#         "/DYJetsToLL_LHEFilterPtZ-100To250_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/MINIAODSIM",
#         "/DYJetsToLL_LHEFilterPtZ-250To400_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/MINIAODSIM",
#         "/DYJetsToLL_LHEFilterPtZ-400To650_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/MINIAODSIM",
#         "/DYJetsToLL_LHEFilterPtZ-50To100_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/MINIAODSIM",
#         "/DYJetsToLL_LHEFilterPtZ-650ToInf_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/MINIAODSIM",
#     ],
#     "DY_M-10to50_inclusive_NLO": [
#         "/DYJetsToLL_M-10to50_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/MINIAODSIM",
#     ],
#     "DY_M-10to50_inclusive_LO": [
#         "/DYJetsToLL_M-10to50_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1/MINIAODSIM",
#     ],
#     "DY_M-50_inclusive_NLO": [
#         "/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/MINIAODSIM",
#     ],
#     "QCD_Pt_MuEnrichedPt5": [
#         "/QCD_Pt-1000_MuEnrichedPt5_TuneCP5_13TeV-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/MINIAODSIM",
#         "/QCD_Pt-120To170_MuEnrichedPt5_TuneCP5_13TeV-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/MINIAODSIM",
#         "/QCD_Pt-15To20_MuEnrichedPt5_TuneCP5_13TeV-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/MINIAODSIM",
#         "/QCD_Pt-170To300_MuEnrichedPt5_TuneCP5_13TeV-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/MINIAODSIM",
#         "/QCD_Pt-20To30_MuEnrichedPt5_TuneCP5_13TeV-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/MINIAODSIM",
#         "/QCD_Pt-300To470_MuEnrichedPt5_TuneCP5_13TeV-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/MINIAODSIM",
#         "/QCD_Pt-30To50_MuEnrichedPt5_TuneCP5_13TeV-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/MINIAODSIM",
#         "/QCD_Pt-470To600_MuEnrichedPt5_TuneCP5_13TeV-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/MINIAODSIM",
#         "/QCD_Pt-50To80_MuEnrichedPt5_TuneCP5_13TeV-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/MINIAODSIM",
#         "/QCD_Pt-600To800_MuEnrichedPt5_TuneCP5_13TeV-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/MINIAODSIM",
#         "/QCD_Pt-800To1000_MuEnrichedPt5_TuneCP5_13TeV-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/MINIAODSIM",
#         "/QCD_Pt-80To120_MuEnrichedPt5_TuneCP5_13TeV-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/MINIAODSIM",
#     ],
#     "TT_powheg": [
#         "/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1/MINIAODSIM",
#         "/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1/MINIAODSIM",
#         "/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/MINIAODSIM",
#     ],
#     "TTW_NLO": [
#         "/TTWJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1/MINIAODSIM",
#         "/TTWJetsToQQ_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1/MINIAODSIM",
#     ],
#     "TTZ_NLO": [
#         "/TTZToLL_TuneCP5_13TeV_amcatnlo-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v3/MINIAODSIM",
#         "/TTZToQQ_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1/MINIAODSIM",
#         "/TTZToQQ_TuneCP5_13TeV_amcatnlo-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v3/MINIAODSIM",
#     ],
#     "TTZToLLNuNu_M-10_NLO": [
#         "/TTZToLLNuNu_M-10_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/MINIAODSIM",
#     ],
#     "TTZToLL_LO": [
#         "/TTZToLL_5f_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/MINIAODSIM",
#     ],
#     "TTZ_inclusive_LO": [
#         "/ttZJets_TuneCP5_13TeV_madgraphMLM_pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/MINIAODSIM",
#     ],
#     "TTTT_NLO": [
#         "/TTTT_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/MINIAODSIM",
#     ],
#     "ST_NLO": [
#         "/ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1/MINIAODSIM",
#         "/ST_t-channel_antitop_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1/MINIAODSIM",
#         "/ST_t-channel_top_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1/MINIAODSIM",
#         "/ST_tW_Dilept_5f_DR_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/MINIAODSIM",
#         # "/ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/MINIAODSIM",
#         # "/ST_tW_top_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/MINIAODSIM",
#     ],
#     "WJetsToLNu_HT_LO": [
#         "/WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/MINIAODSIM",
#         "/WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/MINIAODSIM",
#         "/WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1_ext1-v2/MINIAODSIM",
#         "/WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1/MINIAODSIM",
#         "/WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/MINIAODSIM",
#         "/WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1/MINIAODSIM",
#         "/WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1_ext1-v2/MINIAODSIM",
#         "/WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1/MINIAODSIM",
#         "/WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1_ext1-v2/MINIAODSIM",
#         "/WJetsToLNu_HT-70To100_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1/MINIAODSIM",
#         "/WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1/MINIAODSIM",
#         "/WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1_ext1-v2/MINIAODSIM",
#     ],
#     "WJetsToLNu_Pt_NLO": [
#         "/WJetsToLNu_Pt-100To250_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1/MINIAODSIM",
#         "/WJetsToLNu_Pt-250To400_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1/MINIAODSIM",
#         "/WJetsToLNu_Pt-400To600_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1/MINIAODSIM",
#         "/WJetsToLNu_Pt-600ToInf_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1/MINIAODSIM",
#     ],
#     "WJetsToLNu_inclusive_NLO": [
#         "/WJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/MINIAODSIM",
#     ],
#     "VV_NLO": [
#         "/WWTo1L1Nu2Q_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/MINIAODSIM",
#         "/WWTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/MINIAODSIM",
#         "/WZTo1L1Nu2Q_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/MINIAODSIM",
#         "/WZTo2Q2L_mllmin4p0_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/MINIAODSIM",
#         "/WZTo3LNu_mllmin4p0_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/MINIAODSIM",
#         "/ZZTo2L2Nu_TuneCP5_13TeV_powheg_pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1/MINIAODSIM",
#         "/ZZTo2Q2L_mllmin4p0_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/MINIAODSIM",
#         "/ZZTo4L_TuneCP5_13TeV_powheg_pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/MINIAODSIM",
#     ],
#     "VVV_NLO": [
#         "/WWW_4F_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1_ext1-v2/MINIAODSIM",
#         "/WWZ_4F_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1_ext1-v2/MINIAODSIM",
#         "/ZZZ_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1_ext1-v2/MINIAODSIM",
#     ],
#     "Higgs": [
#         "/GluGluHToZZTo4L_M125_TuneCP5_13TeV_powheg2_minloHJJ_JHUGenV7011_pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/MINIAODSIM",
#         # "/GluGluToZH_HToZZTo4L_M125_TuneCP5_13TeV-jhugenv723-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/MINIAODSIM",
#         "/ZH_HToZZ_4LFilter_M125_TuneCP5_13TeV_powheg2-minlo-HZJ_JHUGenV7011_pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/MINIAODSIM",
#         "/WminusH_HToBB_WToLNu_M-125_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1/MINIAODSIM",
#         "/WplusH_HToBB_WToLNu_M-125_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1/MINIAODSIM",
#         # "/VHToNonbb_M125_TuneCP5_13TeV-amcatnloFXFX_madspin_pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/MINIAODSIM",
#         "/ttHToNonbb_M125_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/MINIAODSIM",
#         "/ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/MINIAODSIM",
#         # "/ttH_HToZZ_4LFilter_M125_TuneCP5_13TeV_powheg2_JHUGenV7011_pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/MINIAODSIM",
#     ],
# }


def lumi_Label(year: str) -> float:
    if year == "2016":
        return round((lumis[year] + lumis[year + "_apv"]) / 1000, 1)
    return round(lumis[year] / 1000, 1)


def find_lumi(
    infile_name: str,
    auto_lumi: Optional[bool] = True,
    year: Optional[str | None] = None,
) -> float:
    if auto_lumi:
        if "20UL16MiniAODv2" in infile_name:
            lumi = lumis["2016"]
        if "20UL17MiniAODv2" in infile_name:
            lumi = lumis["2017"]
        if "20UL16MiniAODAPVv2" in infile_name:
            lumi = lumis["2016_apv"]
        if "20UL18" in infile_name:
            lumi = lumis["2018"]
        if "SUEP-m" in infile_name or "SUEP_m" in infile_name:
            lumi = lumis["2018"]
        if "DoubleMuon" in infile_name:
            lumi = 1
    if year and not auto_lumi:
        lumi = lumis[year]
    if year and auto_lumi:
        raise Exception("Apply lumis automatically or based on year")
    return lumi


def load_samples(
    infile_names: list[str],
    year: Optional[int | str | None] = None,
    auto_lumi: Optional[bool | None] = None,
    custom_lumi: Optional[float | None] = None,
    is_data: Optional[bool | None] = False,
) -> dict:
    if isinstance(year, int):
        year = str(year)

    plots_ = {}
    # Load histograms and scale to lumi
    for infile_name in infile_names:
        if not os.path.isfile(infile_name):
            print("WARNING:", infile_name, "doesn't exist")
            continue
        elif ".pkl" not in infile_name:
            print("WARNING:", infile_name, "is not a .pkl file")
            continue

        # set the lumi based on year or override using a custom value (in /pb). Data shouldn't be scaled.
        lumi = find_lumi(
            infile_name,
            auto_lumi,
            year,
        )
        if custom_lumi is not None:
            lumi = custom_lumi
        if is_data:
            lumi = 1

        sample = infile_name.split("/")[-1].replace("_histograms.pkl", "")

        with open(infile_name, "rb") as f:
            plots_[sample] = pickle.load(f)
        for plot in plots_[sample]:
            plots_[sample][plot] = plots_[sample][plot] * lumi

    # Create combined histograms
    for combined_dataset in dataset_groups_old:
        plots_[combined_dataset] = {}
        for pattern in dataset_groups_old[combined_dataset]:
            for sample in plots_:
                if re.search(pattern, sample):
                    for plot in plots_[sample]:
                        if plot not in plots_[combined_dataset]:
                            plots_[combined_dataset][plot] = plots_[sample][plot].copy()
                        else:
                            plots_[combined_dataset][plot] += plots_[sample][plot]

    return plots_


def loader(
    tag="test",
    custom_lumi=None,
    load_data=False,
    verbosity=0,
):
    # input .pkl files
    plotDir = f"../../processor_output_files/{tag}_output_histograms/"
    filenames = glob.glob(plotDir + "*histograms.pkl")

    # separate the files into signal, background, and data
    files_SUEP = [f for f in filenames if ("SUEP" in f) or ("ggHBSMpythia" in f)]
    files_bkg = [
        f
        for f in filenames
        if ("pythia8" in f) and ("SUEP" not in f) and ("ggHBSMpythia" not in f)
    ]
    files_data = [f for f in filenames if ("DoubleMuon" in f)]
    if verbosity > 0:
        print(files_bkg)

    # load histograms and scale to lumi
    plots_SUEP_2018 = load_samples(files_SUEP, year=2018, custom_lumi=custom_lumi)
    plots_bkg_2018 = load_samples(files_bkg, year=2018, custom_lumi=custom_lumi)
    if load_data:
        plots_data_2018 = load_samples(files_data, year=2018, is_data=True)

    if verbosity > 1:
        print(plots_SUEP_2018)

    # put everything in one dictionary
    plots = {}
    for plot in plots_SUEP_2018:
        plots[plot + "_2018"] = plots_SUEP_2018[plot]
    for plot in plots_bkg_2018:
        plots[plot + "_2018"] = plots_bkg_2018[plot]
    if load_data:
        for plot in plots_data_2018:
            plots[plot + "_2018"] = plots_data_2018[plot]

    return plots
