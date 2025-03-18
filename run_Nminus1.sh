#!/bin/bash -e

# Input options
# -s : signal
# -b : background

# By default, run signal and background

all=1
signal=0
background=0
tag=Nminus1_Mar2025

while getopts 'sbt:' flag; do
  case "${flag}" in
    s) all=0; signal=1 ;;
    b) all=0; background=1 ;;
    t) tag="${OPTARG}" ;;
    *) echo "Unexpected option ${flag}" ;;
  esac
done

if [ $all -eq 1 ]; then
    signal=1
    background=1
fi

if [ $signal -eq 1 ]; then
    echo -n ""
    # echo "Processing signal SR_low_temp..."
    # python runner.py \
    #     --workflow SUEP_coffea_SR_low_temp_Nminus1 -o "processor_output_files/${tag}_SR_low_temp" \
    #     --json filelists/signal/GluGluToSUEP_central_UL18_Nov2024.json --era 2018 --skimmed \
    #     --isMC --executor futures -j 38 --chunk 4000
    # echo "Processing signal SR_high_temp..."
    # python runner.py \
    #     --workflow SUEP_coffea_SR_high_temp_Nminus1 -o "processor_output_files/${tag}_SR_high_temp" \
    #     --json filelists/signal/GluGluToSUEP_central_UL18_Nov2024.json --era 2018 --skimmed \
    #     --isMC --executor futures -j 48 --chunk 80000
    echo "Processing signal CRs..."
    python runner.py \
        --workflow SUEP_coffea_CRs_Nminus1 -o "processor_output_files/${tag}_CRs" \
        --json filelists/signal/GluGluToSUEP_central_UL18_Nov2024.json --era 2018 --skimmed \
        --isMC --executor futures -j 48 --chunk 80000
fi

if [ $background -eq 1 ]; then
    echo -n ""
    # echo "Processing BKG SR_low_temp..."
    # python runner.py \
    #     --workflow SUEP_coffea_SR_low_temp_Nminus1 -o "processor_output_files/${tag}_SR_low_temp" \
    #     --json filelists/mc_collections/SUEPNano_UL18_Nov2024.json --era 2018 --skimmed \
    #     --isMC --executor futures -j 44 --chunk 7000
    # echo "Processing BKG SR_high_temp..."
    # python runner.py \
    #     --workflow SUEP_coffea_SR_high_temp_Nminus1 -o "processor_output_files/${tag}_SR_high_temp" \
    #     --json filelists/mc_collections/SUEPNano_UL18_Nov2024.json --era 2018 --skimmed \
    #     --isMC --executor futures -j 48 --chunk 80000
    echo "Processing BKG CRs..."
    python runner.py \
        --workflow SUEP_coffea_CRs_Nminus1 -o "processor_output_files/${tag}_SR_high_temp" \
        --json filelists/mc_collections/SUEPNano_UL18_Nov2024.json --era 2018 --skimmed \
        --isMC --executor futures -j 48 --chunk 80000
fi
