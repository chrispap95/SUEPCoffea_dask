#!/bin/bash -e

# Input options
# -s : signal
# -b : background

# By default, run signal and background

all=1
signal=0
background=0
data=0
tag=full_analysis_Dec2024
blind=0 # 0 for unblinded, 1 for blinded

while getopts 'sbdt:cw:' flag; do
  case "${flag}" in
    s) all=0; signal=1 ;;
    b) all=0; background=1 ;;
    d) all=0; data=1 ;;
    t) tag="${OPTARG}" ;;
    *) echo "Unexpected option ${flag}" ;;
  esac
done

if [ $all -eq 1 ]; then
    signal=1
    background=1
    data=1
fi

if [ $signal -eq 1 ]; then
    echo "Processing signal SR high temp..."
    python runner.py \
        --workflow SUEP_coffea_SR_high_temp -o "processor_output_files/${tag}_SR_high_temp" \
        --json filelists/signal/GluGluToSUEP_central_UL18_Nov2024.json --era 2018 \
        --executor futures -j 8 --chunk 50000 --skimmed --isMC --do_syst
    echo "Processing signal SR low temp..."
    python runner.py \
        --workflow SUEP_coffea_SR_low_temp -o "processor_output_files/${tag}_SR_low_temp" \
        --json filelists/signal/GluGluToSUEP_central_UL18_Nov2024.json --era 2018 \
        --executor dask/lpc --chunk 3000 --skimmed --isMC --do_syst
    echo "Processing signal CR..."
    python runner.py \
        --workflow SUEP_coffea_CRs -o "processor_output_files/${tag}_CR" --do_syst \
        --json filelists/signal/GluGluToSUEP_central_UL18_Nov2024.json --era 2018 \
        --executor futures -j 8 --chunk 50000 --skimmed --isMC
    echo "Processing signal VR..."
    python runner.py \
        --workflow SUEP_coffea_VR -o "processor_output_files/${tag}_VR" --do_syst \
        --json filelists/signal/GluGluToSUEP_central_UL18_Nov2024.json --era 2018 \
        --executor futures -j 8 --chunk 50000 --skimmed --isMC
fi

if [ $background -eq 1 ]; then
    echo "Processing BKG SR high temp..."
    python runner.py \
        --workflow SUEP_coffea_SR_high_temp -o "processor_output_files/${tag}_SR_high_temp" \
        --json filelists/mc_collections/SUEPNano_UL18_Nov2024.json --era 2018 --do_syst \
        --executor futures -j 8 --chunk 50000 --skimmed --isMC
    echo "Processing BKG SR low temp..."
    python runner.py \
        --workflow SUEP_coffea_SR_low_temp -o "processor_output_files/${tag}_SR_low_temp" \
        --json filelists/mc_collections/SUEPNano_UL18_Nov2024.json --era 2018 --do_syst \
        --executor dask/lpc --chunk 3000 --skimmed --isMC
    echo "Processing BKG CR..."
    python runner.py \
        --workflow SUEP_coffea_CRs -o "processor_output_files/${tag}_CR" --do_syst \
        --json filelists/mc_collections/SUEPNano_UL18_Nov2024.json --era 2018 \
        --executor futures -j 8 --chunk 50000 --skimmed --isMC
    echo "Processing BKG VR..."
    python runner.py \
        --workflow SUEP_coffea_VR -o "processor_output_files/${tag}_VR" --do_syst \
        --json filelists/mc_collections/SUEPNano_UL18_Nov2024.json --era 2018 \
        --executor futures -j 8 --chunk 50000 --skimmed --isMC
fi

if [ $data -eq 1 ]; then
    if [ $blind -eq 0 ]; then
        echo "Processing data SR high temp..."
        python runner.py \
            --workflow SUEP_coffea_SR_high_temp -o "processor_output_files/${tag}_SR_high_temp" \
            --json filelists/data/data_Run2018A_0p6fb_1file_unskimmed.json --do_syst \
            --executor futures -j 8 --chunk 50000 --era 2018
        echo "Processing data SR low temp..."
        python runner.py \
            --workflow SUEP_coffea_SR_low_temp -o "processor_output_files/${tag}_SR_low_temp" \
            --json filelists/data/data_Run2018A_0p6fb_1file_unskimmed.json --do_syst \
            --executor futures -j 8 --chunk 50000 --era 2018
    fi
    echo "Processing data CR..."
    python runner.py \
        --workflow SUEP_coffea_CRs -o "processor_output_files/${tag}_CR" \
        --json filelists/data/data_Run2018A_0p6fb_1file_unskimmed.json --do_syst \
        --executor futures -j 8 --chunk 50000 --era 2018
    echo "Processing data VR..."
    python runner.py \
        --workflow SUEP_coffea_VR -o "processor_output_files/${tag}_VR" --do_syst \
        --json filelists/data/data_Run2018A_0p6fb_1file_unskimmed.json --era 2018 \
        --executor futures -j 8 --chunk 50000 --isMC
fi
