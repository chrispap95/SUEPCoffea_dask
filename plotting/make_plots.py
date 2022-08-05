# Make plots for SUEP analysis. Reads in hdf5 files and outputs to pickle and root files
import os, sys, subprocess
import pandas as pd 
import numpy as np
import argparse
import getpass
import uproot
import getpass
import pickle
import json
from tqdm import tqdm
from hist import Hist
from collections import defaultdict

#Import our own functions
import pileup_weight
from plot_utils import *

parser = argparse.ArgumentParser(description='Famous Submitter')
parser.add_argument("-dataset", "--dataset"  , type=str, default="QCD", help="dataset name", required=True)
parser.add_argument("-t"   , "--tag"   , type=str, default="IronMan"  , help="production tag", required=False)
parser.add_argument("-o"   , "--output"   , type=str, default="IronMan"  , help="output tag", required=False)
parser.add_argument("-e"   , "--era"   , type=int, default=2018  , help="era", required=False)
parser.add_argument('--doSyst', type=int, default=0, help="make systematic plots")
parser.add_argument('--isMC', type=int, default=1, help="Is this MC or data")
parser.add_argument('--scouting', type=int, default=0, help="Is this scouting or no")
parser.add_argument('--blind', type=int, default=1, help="Blind the data (default=True)")
parser.add_argument('--weights', type=str, default="None", help="Pass the filename of the weights, e.g. --weights weights.npy")
parser.add_argument('--xrootd', type=int, default=0, help="Local data or xrdcp from hadoop (default=False)")
options = parser.parse_args()

# parameters for script
output_label = options.output
outDir = "/work/submit/{}/SUEP/outputs/".format(getpass.getuser())
redirector = "root://t3serv017.mit.edu/"

"""
Define output plotting methods, each draws from an input_method (outputs of SUEPCoffea),
and can have its own selections, ABCD regions, and signal region.
Multiple plotting methods can be defined for the same input method, as different
selections and ABCD methods can be applied.
N.B.: Include lower and upper bounds for all ABCD regions.
"""
config = {
    'ISRRemoval' : {
        'input_method' : 'IRM',
        'xvar' : 'SUEP_S1_IRM',
        'xvar_regions' : [0.35, 0.4, 0.5, 1.0],
        'yvar' : 'SUEP_nconst_IRM',
        'yvar_regions' : [10, 20, 40, 1000],
        'SR' : [['SUEP_S1_IRM', '>=', 0.5], ['SUEP_nconst_IRM', '>=', 40]],
        'selections' : [['ht', '>', 1200], ['ntracks','>', 0], ["SUEP_S1_IRM", ">=", 0.0]]
    },
    
    'Cluster' : {
        'input_method' : 'CL',
        'label_out' : 'CL',
        'xvar' :'SUEP_S1_CL',
        'xvar_regions' : [0.35, 0.4, 0.5, 1.0],
        'yvar' : 'SUEP_nconst_CL',
        'yvar_regions' : [20, 40, 80, 1000],
        'SR' : [['SUEP_S1_CL', '>=', 0.5], ['SUEP_nconst_CL', '>=', 80]],
        'selections' : [['ht', '>', 1200], ['ntracks','>', 0], ["SUEP_S1_CL", ">=", 0.0]]
    },
    
    'ClusterInverted' : {
        'input_method' : 'CL',
        'xvar' : 'ISR_S1_CL',
        'xvar_regions' : [0.35, 0.4, 0.5, 1.0],
        'yvar' : 'ISR_nconst_CL',
        'yvar_regions' : [20, 40, 80, 1000],
        'SR' : [['ISR_S1_CL', '>=', 0.5], ['ISR_nconst_CL', '>=', 80]],
        'selections' : [['ht', '>', 1200], ['ntracks','>', 10], ["ISR_S1_CL", ">=", 0.0]]
    },
    
#     'ResNet' : {
#         'input_method' : 'ML',
#         'label_out' : 'ML',
#         'xvar' : 'resnet_SUEP_pred_ML',
#         'xvar_regions' : [0.0, 0.5, 1.0],
#         'yvar' : 'ntracks',
#         'yvar_regions' : [0, 100, 1000],
#         'SR' : [['resnet_SUEP_pred_ML', '>=', 0.5], ['ntracks', '>=', 100]],
#         'selections' : [['ht', '>', 600], ['ntracks','>',0]]
#     },
    
#     'Cone' : {
#         'input_method' : 'CO',
#         'xvar' : 'SUEP_S1_CO',
#         'xvar_regions' : [0.35, 0.4, 0.5, 1.0],
#         'yvar' : 'SUEP_nconst_CO',
#         'yvar_regions' : [20, 40, 80, 1000],
#         'SR' : [['SUEP_S1_CO', '>=', 0.5], ['SUEP_nconst_CO', '>=', 80]],
#         'selections' : [['ht', '>', 600], ['ntracks','>', 10], ["SUEP_S1_CO", ">=", 0.35]]
#     } 
}

#############################################################################################################
    
def auto_fill(df, output, abcd, label_out, do_abcd=False):
    
    input_method = abcd['input_method']

    #####################################################################################
    # ---- Fill Histograms
    # Automatically fills all histograms that are declared in the output dict.
    #####################################################################################
    
    # 1. fill the distributions as they are saved in the dataframes
    # 1a. Plot event wide variables
    plot_labels = [key for key in df.keys() if key+"_"+label_out in list(output.keys())]  
    for plot in plot_labels: output[plot+"_"+label_out].fill(df[plot], weight=df['event_weight']) 
    # 1b. Plot method variables
    plot_labels = [key for key in df.keys() if key.replace(input_method, label_out) in list(output.keys())]
    for plot in plot_labels: output[plot.replace(input_method, label_out)].fill(df[plot], weight=df['event_weight'])  
    # FIXME: plot ABCD 2d
    
    # 2. fill some 2D distributions  
    keys = list(output.keys())
    keys_2Dhists = [k for k in keys if '2D' in k]
    for key in keys_2Dhists:
        if not key.endswith(label_out): continue
        string = key[len("2D")+1:-(len(label_out)+1)] # cut out "2D_" and output label
        var1 = string.split("_vs_")[0]
        var2 = string.split("_vs_")[1]
        if var1 not in list(df.keys()): var1 += "_" + input_method
        if var2 not in list(df.keys()): var2 += "_" + input_method
        output[key].fill(df[var1], df[var2], weight=df['event_weight'])

    # 3. divide the dfs by region
    if do_abcd:
        regions = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        xvar = abcd['xvar']
        yvar = abcd['yvar']
        xvar_regions = abcd['xvar_regions']
        yvar_regions = abcd['yvar_regions']
        iRegion = 0
        for i in range(len(xvar_regions)-1):
            x_val_lo = xvar_regions[i]
            x_val_hi = xvar_regions[i+1]

            for j in range(len(yvar_regions)-1):
                y_val_lo = yvar_regions[j]
                y_val_hi = yvar_regions[j+1]

                x_cut = (make_selection(df, xvar, '>=', x_val_lo, False) & make_selection(df, xvar, '<', x_val_hi, False))
                y_cut = (make_selection(df, yvar, '>=', y_val_lo, False) & make_selection(df, yvar, '<', y_val_hi, False))
                df_r = df.loc[(x_cut & y_cut)]

                r = regions[iRegion] + "_"
                iRegion += 1

                # double check blinding
                if iRegion == (len(xvar_regions)-1)*(len(yvar_regions)-1) and not options.isMC:
                    if df_r.shape[0] > 0: 
                        sys.exit(label_out+": You are not blinding correctly! Exiting.")

                # 3a. Plot event wide variables
                plot_labels = [key for key in df_r.keys() if r+key+"_"+label_out in list(output.keys())]   # event wide variables
                for plot in plot_labels: output[r+plot+"_"+label_out].fill(df_r[plot], weight=df_r['event_weight']) 
                # 3b. Plot method variables
                plot_labels = [key for key in df_r.keys() if r+key.replace(input_method, label_out) in list(output.keys())]  # method vars
                for plot in plot_labels: output[r+plot.replace(input_method, label_out)].fill(df_r[plot], weight=df_r['event_weight'])  
            
def plot(df, output, abcd, label_out):
    """
    INPUTS:
        df: input file DataFrame.
        output: dictionary of histograms to be filled.
        abcd: definitions of ABCD regions, signal region, event selections.
        label_out: label associated with the output (e.g. "ISRRemoval"), as keys in 
                   the config dictionary.
        
    OUTPUTS: 
        output: now with updated histograms.
        
    EXPLANATION:
    The DataFrame generated by ../workflows/SUEP_coffea.py has the form:
    event variables (ht, ...)   IRM vars (SUEP_S1_IRM, ...)  ML vars  Other Methods
          0                                 0                   0          ...
          1                                 NaN                 1          ...
          2                                 NaN                 NaN        ...
          3                                 1                   2          ...
    (The event vars are always filled, while the vars for each method are filled only
    if the event passes the method's selections, hence the NaNs).
    
    This function will plot, for each 'label_out':
        1. All event variables, e.g. output histogram = ht_label_out
        2. All columns from 'input_method', e.g. SUEP_S1_IRM column will be
           plotted to histogram SUEP_S1_ISRRemoval.
        3. 2D variables are automatically plotted, as long as hstogram is
           initialized in the output dict as "2D_var1_vs_var2"
    
    N.B.: Histograms are filled only if they are initialized in the output dictionary.

    e.g. We want to plot CL. 
    Event Selection:
        1. Grab only events that don't have NaN for CL variables.
        2. Blind for data! Use SR to define signal regions and cut it out of df.
        3. Apply selections as defined in the 'selections' in the dict.

    Fill Histograms:
        1. Plot variables from the DataFrame. 
           1a. Event wide variables
           1b. Cluster method variables
        2. Plot 2D variables.
        3. Plot variables from the different ABCD regions as defined in the abcd dict.
           3a. Event wide variables
           3b. Cluster method variables
    """

    input_method = abcd['input_method']

    #####################################################################################
    # ---- Event Selection
    #####################################################################################
    
    # 1. keep only events that passed this method
    df = df[~df[abcd['xvar']].isnull()]
        
    # 2. blind
    if options.blind and not options.isMC:       
        SR = abcd['SR']
        if len(SR) != 2: sys.exit(label_out + ": Make sure you have correctly defined your signal region. Exiting.")
        df = df.loc[~(make_selection(df, SR[0][0], SR[0][1], SR[0][2], apply=False) & make_selection(df, SR[1][0], SR[1][1], SR[1][2], apply=False))]
        
    # 3. apply selections
    for sel in abcd['selections']: 
        df = make_selection(df, sel[0], sel[1], sel[2], apply=True)
        
    # auto fill all histograms in the output dictionary
    auto_fill(df, output, abcd, label_out, do_abcd=True)
           
    return output
        
#############################################################################################################

# get list of files
username = getpass.getuser()
if options.xrootd:
    dataDir = "/scratch/{}/SUEP/{}/{}/merged/".format(username,options.tag,options.dataset)
    result = subprocess.check_output(["xrdfs",redirector,"ls",dataDir])
    result = result.decode("utf-8")
    files = result.split("\n")
    files = [f for f in files if len(f) > 0]
else:
    dataDir = "/data/submit/{}/{}/{}/merged/".format(username, options.tag, options.dataset)
    files = [dataDir + f for f in os.listdir(dataDir)]

# get cross section
xsection = 1.0
if options.isMC: xsection = getXSection(options.dataset, options.era)

# pileup weights
puweights, puweights_up, puweights_down = pileup_weight.pileup_weight(options.era)   

# custom per region weights
weights = None
if options.weights != "None":
    w = np.load(options.weights, allow_pickle=True)
    weights = defaultdict(lambda: np.zeros(2))
    weights.update(w.item())

# output histos
def create_output_file(label, abcd):

    # don't recreate histograms if called multiple times with the same output label
    if label in output["labels"]: return output
    else: output["labels"].append(label)
    
    # ABCD histogram
    xvar = abcd['xvar']
    yvar = abcd['yvar']
    xvar_regions = abcd['xvar_regions']
    yvar_regions = abcd['yvar_regions']
    output.update({"ABCDvars_"+label : Hist.new.Reg(100, 0, yvar_regions[-1], name=xvar).Reg(100, 0, xvar_regions[-1], name=yvar).Weight()})
 
    # defnie all the regions, will be used to make historgams for each region
    regions = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    n_regions = ((len(xvar_regions) - 1) * (len(yvar_regions) - 1))
    regions_list =  [""] + [regions[i]+"_" for i in range(n_regions)]
    
    # variables from the dataframe for all the events, and those in A, B, C regions
    for r in regions_list:
        output.update({
            r+"ht_" + label : Hist.new.Reg(100, 0, 10000, name=r+"ht_"+label, label='HT').Weight(),
            r+"ntracks_" + label : Hist.new.Reg(101, 0, 500, name=r+"ntracks_"+label, label='# Tracks in Event').Weight(),
            r+"ngood_fastjets_" + label : Hist.new.Reg(9,0, 10, name=r+"ngood_fastjets_"+label, label='# FastJets in Event').Weight(),
            r+"PV_npvs_"+label : Hist.new.Reg(199,0, 200, name=r+"PV_npvs_"+label, label="# PVs in Event ").Weight(),
            r+"Pileup_nTrueInt_"+label : Hist.new.Reg(199,0, 200, name=r+"Pileup_nTrueInt_"+label, label="# True Interactions in Event ").Weight(),
            r+"ngood_ak4jets_" + label : Hist.new.Reg(19,0, 20, name=r+"ngood_ak4jets_"+label, label= '# ak4jets in Event').Weight(),
            r+"ngood_tracker_ak4jets_" + label : Hist.new.Reg(19,0, 20, name=r+"ngood_tracker_ak4jets_"+label, label= r'# ak4jets in Event ($|\eta| < 2.4$)').Weight(),
            r+"FNR_" + label : Hist.new.Reg(50,0, 1, name=r+"FNR_"+label, label= r'# SUEP Tracks in ISR / # SUEP Tracks').Weight(),
            r+"ISR_contamination_" + label : Hist.new.Reg(50,0, 1, name=r+"ISR_contamination_"+label, label= r'# SUEP Tracks in ISR / # ISR Tracks').Weight(),
        })
        # for i in range(10):
        #     output.update({
        #         r+"eta_ak4jets"+str(i)+"_"+label : Hist.new.Reg(100,-5,5, name=r+"eta_ak4jets"+str(i)+"_"+label, label=r"ak4jets"+str(i)+" $\eta$").Weight(),
        #         r+"phi_ak4jets"+str(i)+"_"+label : Hist.new.Reg(100,-6.5,6.5, name=r+"phi_ak4jets"+str(i)+"_"+label, label=r"ak4jets"+str(i)+" $\phi$").Weight(),
        #         r+"pt_ak4jets"+str(i)+"_"+label : Hist.new.Reg(100, 0, 2000, name=r+"pt_ak4jets"+str(i)+"_"+label, label=r"ak4jets"+str(i)+" $p_T$").Weight(),
        #     })
        # for i in range(2):
        #     output.update({
        #         r+"eta_ak4jets"+str(i)+"_4jets_"+label : Hist.new.Reg(100,-5,5, name=r+"eta_ak4jets"+str(i)+"_4jets_"+label, label=r"ak4jets"+str(i)+" (4 jets) $\eta$").Weight(),
        #         r+"phi_ak4jets"+str(i)+"_4jets_"+label : Hist.new.Reg(100,-6.5,6.5, name=r+"phi_ak4jets"+str(i)+"_4jets_"+label, label=r"ak4jets"+str(i)+" (4 jets) $\phi$").Weight(),
        #         r+"pt_ak4jets"+str(i)+"_4jets_"+label : Hist.new.Reg(100, 0, 2000, name=r+"pt_ak4jets"+str(i)+"_4jets_"+label, label=r"ak4jets"+str(i)+" (4 jets) $p_T$").Weight(),
        #     })
    
    if label == 'ISRRemoval' or label == 'Cluster' or label=='Cone':
        # 2D histograms
        output.update({
            "2D_SUEP_S1_vs_ntracks_"+label : Hist.new.Reg(100, 0, 1.0, name="SUEP_S1_"+label, label='$Sph_1$').Reg(100, 0, 500, name="ntracks_"+label, label='# Tracks').Weight(),
            "2D_SUEP_S1_vs_SUEP_nconst_"+label : Hist.new.Reg(100, 0, 1.0, name="SUEP_S1_"+label, label='$Sph_1$').Reg(200, 0, 500, name="nconst_"+label, label='# Constituents').Weight(),     
            "2D_SUEP_nconst_vs_SUEP_pt_avg_"+label : Hist.new.Reg(200, 0, 500, name="SUEP_nconst_"+label, label='# Const').Reg(200, 0, 500, name="SUEP_pt_avg_"+label, label='$p_T Avg$').Weight(), 
            "2D_SUEP_nconst_vs_SUEP_pt_avg_b_"+label : Hist.new.Reg(200, 0, 500, name="SUEP_nconst_"+label, label='# Const').Reg(50, 0, 50, name="SUEP_pt_avg_b_"+label, label='$p_T Avg (Boosted frame)$').Weight(), 
            "2D_SUEP_S1_vs_SUEP_pt_mean_scaled_"+label : Hist.new.Reg(100, 0, 1, name="SUEP_S1_"+label, label='$Sph_1$').Reg(100, 0, 1, name="SUEP_pt_mean_scaled_"+label, label='$p_T Avg / p_T Max (Boosted frame)$').Weight(), 
            "2D_SUEP_nconst_vs_SUEP_pt_mean_scaled_"+label : Hist.new.Reg(200, 0, 500, name="SUEP_nconst_"+label, label='# Const').Reg(100, 0, 1, name="SUEP_pt_mean_scaled_"+label, label='$p_T Avg / p_T Max (Boosted frame)$').Weight(),  
        })
        
        # variables from the dataframe for all the events, and those in A, B, C regions
        for r in regions_list:
            output.update({
                r+"SUEP_nconst_"+label : Hist.new.Reg(199, 0, 500, name=r+"SUEP_nconst_"+label, label="# Tracks in SUEP").Weight(),
                r+"SUEP_pt_"+label : Hist.new.Reg(100, 0, 2000, name=r+"SUEP_pt_"+label, label=r"SUEP $p_T$ [GeV]").Weight(),
                r+"SUEP_pt_avg_"+label : Hist.new.Reg(200, 0, 500, name=r+"SUEP_pt_avg_"+label, label=r"SUEP Components $p_T$ Avg.").Weight(),
                r+"SUEP_pt_avg_b_"+label : Hist.new.Reg(50, 0, 50, name=r+"SUEP_pt_avg_b_"+label, label=r"SUEP Components $p_T$ avg (Boosted Frame)").Weight(),
                r+"SUEP_pt_mean_scaled_"+label : Hist.new.Reg(100, 0, 1, name=r+"SUEP_pt_mean_scaled_"+label, label=r"SUEP Components $p_T$ Mean / Max (Boosted Frame)").Weight(),
                r+"SUEP_eta_"+label : Hist.new.Reg(100,-5,5, name=r+"SUEP_eta_"+label, label=r"SUEP $\eta$").Weight(),
                r+"SUEP_phi_"+label : Hist.new.Reg(100,-6.5,6.5, name=r+"SUEP_phi_"+label, label=r"SUEP $\phi$").Weight(),
                r+"SUEP_mass_"+label : Hist.new.Reg(150, 0, 2000, name=r+"SUEP_mass_"+label, label="SUEP Mass [GeV]").Weight(),
                r+"SUEP_delta_mass_genMass_"+label : Hist.new.Reg(400, -2000, 2000, name=r+"SUEP_delta_mass_genMass_"+label, label="SUEP Mass - genSUEP Mass [GeV]").Weight(),
                r+"SUEP_S1_"+label : Hist.new.Reg(100, 0, 1, name=r+"SUEP_S1_"+label, label='$Sph_1$').Weight(),
                r+"SUEP_girth": Hist.new.Reg(50, 0, 1.0, name=r+"SUEP_girth_"+label, label=r"SUEP Girth").Weight(),
                r+"SUEP_rho0_"+label : Hist.new.Reg(50, 0, 20, name=r+"SUEP_rho0_"+label, label=r"SUEP $\rho_0$").Weight(),
                r+"SUEP_rho1_"+label : Hist.new.Reg(50, 0, 20, name=r+"SUEP_rho1_"+label, label=r"SUEP $\rho_1$").Weight(),
            })
    
    if label == 'ClusterInverted':
        output.update({
            # 2D histograms
            "2D_ISR_S1_vs_ntracks_"+label : Hist.new.Reg(100, 0, 1.0, name="ISR_S1_"+label, label='$Sph_1$').Reg(200, 0, 500, name="ntracks_"+label, label='# Tracks').Weight(),
            "2D_ISR_S1_vs_ISR_nconst_"+label : Hist.new.Reg(100, 0, 1.0, name="ISR_S1_"+label, label='$Sph_1$').Reg(200, 0, 500, name="nconst_"+label, label='# Constituents').Weight(),     
            "2D_ISR_nconst_vs_ISR_pt_avg_"+label : Hist.new.Reg(200, 0, 500, name="ISR_nconst_"+label).Reg(500, 0, 500, name="ISR_pt_avg_"+label).Weight(), 
            "2D_ISR_nconst_vs_ISR_pt_avg_b_"+label : Hist.new.Reg(200, 0, 500, name="ISR_nconst_"+label).Reg(100, 0, 100, name="ISR_pt_avg_"+label).Weight(), 
            "2D_ISR_S1_vs_ISR_pt_mean_scaled_"+label : Hist.new.Reg(100, 0, 1, name="ISR_S1_"+label).Reg(100, 0, 1, name="ISR_pt_mean_scaled_"+label).Weight(),
            "2D_ISR_nconst_vs_ISR_pt_mean_scaled_"+label : Hist.new.Reg(200, 0, 500, name="ISR_nconst_"+label).Reg(100, 0, 1, name="ISR_pt_mean_scaled_"+label).Weight(),  
            
        })
        # variables from the dataframe for all the events, and those in A, B, C regions
        for r in regions_list:
            output.update({
                r+"ISR_nconst_"+label : Hist.new.Reg(199, 0, 500, name=r+"ISR_nconst_"+label, label="# Tracks in ISR").Weight(),
                r+"ISR_pt_"+label : Hist.new.Reg(100, 0, 2000, name=r+"ISR_pt_"+label, label=r"ISR $p_T$ [GeV]").Weight(),
                r+"ISR_pt_avg_"+label : Hist.new.Reg(500, 0, 500, name=r+"ISR_pt_avg_"+label, label=r"ISR Components $p_T$ Avg.").Weight(),
                r+"ISR_pt_avg_b_"+label : Hist.new.Reg(100, 0, 100, name=r+"ISR_pt_avg_b_"+label, label=r"ISR Components $p_T$ avg (Boosted Frame)").Weight(),
                r+"ISR_pt_mean_scaled_"+label : Hist.new.Reg(100, 0, 1, name=r+"ISR_pt_mean_scaled_"+label, label=r"ISR Components $p_T$ Mean / Max (Boosted Frame)").Weight(),
                r+"ISR_eta_"+label : Hist.new.Reg(100,-5,5, name=r+"ISR_eta_"+label, label=r"ISR $\eta$").Weight(),
                r+"ISR_phi_"+label : Hist.new.Reg(100,-6.5,6.5, name=r+"ISR_phi_"+label, label=r"ISR $\phi$").Weight(),
                r+"ISR_mass_"+label : Hist.new.Reg(150, 0, 4000, name=r+"ISR_mass_"+label, label="ISR Mass [GeV]").Weight(),
                r+"ISR_S1_"+label : Hist.new.Reg(100, 0, 1, name=r+"ISR_S1_"+label, label='$Sph_1$').Weight(),
                r+"ISR_girth": Hist.new.Reg(50, 0, 1.0, name=r+"ISR_girth_"+label, label=r"ISR Girth").Weight(),
                r+"ISR_rho0_"+label : Hist.new.Reg(100, 0, 20, name=r+"ISR_rho0_"+label, label=r"ISR $\rho_0$").Weight(),
                r+"ISR_rho1_"+label : Hist.new.Reg(100, 0, 20, name=r+"ISR_rho1_"+label, label=r"ISR $\rho_1$").Weight(),
            })
    
    if label == 'ML':
        for r in regions_list:
            output.update({
                r+"resnet_pred_"+label : Hist.new.Reg(100, 0, 1, name=r+"resnet_SUEP_pred_"+label, label="Resnet Output").Weight(),
                r+"ntracks_"+label : Hist.new.Reg(100, 0, 500, name=r+"ntracks"+label, label="# Tracks in Event").Weight(),
            })
                        
    return output

# fill ABCD hists with dfs from hdf5 files
nfailed = 0
weight = 0
fpickle =  open(outDir + options.dataset+ "_" + output_label + '.pkl', "wb")
output = {"labels":[]}

### Plotting loop #######################################################################
for ifile in tqdm(files):
    
    #####################################################################################
    # ---- Load file
    #####################################################################################

    if options.xrootd:
        if os.path.exists(options.dataset+'.hdf5'): os.system('rm ' + options.dataset+'.hdf5')
        xrd_file = redirector + ifile
        os.system("xrdcp -s {} {}.hdf5".format(xrd_file, options.dataset))
        df, metadata = h5load(options.dataset+'.hdf5', 'vars')   
    else:
        df, metadata = h5load(ifile, 'vars')   
 
    # check if file is corrupted
    if type(df) == int: 
        nfailed += 1
        continue
            
    # update the gensumweight
    if options.isMC and metadata != 0: weight += metadata['gensumweight']

    # check if file is empty
    if 'empty' in list(df.keys()): continue
    if df.shape[0] == 0: continue    

    #####################################################################################
    # ---- Additional weights
    # Currently applies pileup weights through nTrueInt
    # and optionally (options.weights) scaling weights that are derived to force
    # MC to agree with data in one variable. Usage:
    # df['event_weight'] *= another event weight, etc
    #####################################################################################
    event_weight = np.ones(df.shape[0])
    df['event_weight'] = event_weight
    
    # pileup weights
    if options.isMC == 1 and options.scouting != 1 and False:
        Pileup_nTrueInt = np.array(df['Pileup_nTrueInt']).astype(int)
        pu = puweights[Pileup_nTrueInt]
        df['event_weight'] *= pu
    
    # scaling weights
    if options.isMC == 1 and weights is not None:
        
        regions = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        x_var = 'SUEP_S1_CL'
        y_var = 'SUEP_nconst_CL'
        z_var = 'ht'
        x_var_regions = []
        y_var_regions = []
        iRegion = 0
        
        # S1 regions
        for i in range(len(x_var_regions)-1):
            x_val_lo = x_var_regions[i]
            x_val_hi = x_var_regions[i+1]

            # nconst regions
            for j in range(len(y_var_regions)-1):
                y_val_lo = y_var_regions[j]
                y_val_hi = y_var_regions[j+1]
                
                r = regions[iRegion]
                
                # from the weights
                bins = weights[r]['ht_bins']
                ratios = weights[r]['ratios']
                
                # ht bins
                for k in range(len(bins)-1):
                    z_val_lo = bins[k]
                    z_val_hi = bins[k+1]
                    ratio = ratios[k]
                
                    zslice = (df[z_var] >= z_val_lo) & (df[z_var] < z_val_hi)
                    yslice = (df[y_var] >= y_val_lo) & (df[y_var] < y_val_hi)
                    xslice = (df[x_var] >= x_val_lo) & (df[x_var] < x_val_hi)
                                        
                    df.loc[xslice & yslice & zslice, 'event_weight'] *= ratio
                
                iRegion += 1
    
    #####################################################################################
    # ---- Make plots
    #####################################################################################
    
    for label_out, config_out in config.items():
        output.update(create_output_file(label_out, config_out))
        output = plot(df.copy(), output, config_out, label_out)
        
    #####################################################################################
    # ---- End
    #####################################################################################
    
    # remove file at the end of loop   
    if options.xrootd: os.system('rm ' + options.dataset+'.hdf5')    

### End plotting loop ###################################################################
    
# apply normalization
output.pop("labels")
if options.isMC:
    if weight > 0.0:
        for plot in list(output.keys()): output[plot] = output[plot]*xsection/weight
    else:
        print("Weight is 0")
        
#Save to pickle
pickle.dump(output, fpickle)
print("Number of files that failed to be read:", nfailed)

# save to root
with uproot.recreate(outDir + options.dataset+ "_" + output_label + '.root') as froot:
    for h, hist in output.items():
        froot[h] = hist