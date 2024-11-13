#!/bin/bash -e

# Input options
# -s : signal
# -b : background

# By default, run signal and background

all=1
signal=0
background=0
data=0
tag=SR_high_temp
blind=1 # 0 for unblinded, 1 for blinded

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
    echo "Processing signal SR..."
    python runner.py \
        --workflow SUEP_coffea_SR_high_temp -o "processor_output_files/${tag}_SR" \
        --json filelists/signal/SUEP_signal_central_2018_working.json \
        --executor futures --chunk 50000 --trigger TripleMu --era 2018 --isMC
    echo "Processing signal CR..."
    python runner.py \
        --workflow SUEP_coffea_CRs -o "processor_output_files/${tag}_CR" \
        --json filelists/signal/SUEP_signal_central_2018_working.json \
        --executor futures --chunk 50000 --trigger TripleMu --era 2018 --isMC
fi

if [ $background -eq 1 ]; then
    echo "Processing BKG SR..."
    python runner.py \
        --workflow SUEP_coffea_SR_high_temp -o "processor_output_files/${tag}_SR" \
        --json filelists/mc_collections/full_mc_skimmed_merged_new_trigger.json \
        --executor futures --chunk 50000 --skimmed --trigger TripleMu --era 2018 --isMC
    echo "Processing BKG CR..."
    python runner.py \
        --workflow SUEP_coffea_CRs -o "processor_output_files/${tag}_CR" \
        --json filelists/mc_collections/full_mc_skimmed_merged_new_trigger.json \
        --executor futures --chunk 50000 --skimmed --trigger TripleMu --era 2018 --isMC
fi

if [ $data -eq 1 ]; then
    if [ $blind -eq 0 ]; then
        echo "Processing data SR..."
        python runner.py \
            --workflow SUEP_coffea_SR_high_temp -o "processor_output_files/${tag}_SR" \
            --json filelists/data/data_Run2018A_5p3fb_unskimmed.json \
            --executor futures --chunk 50000 --trigger TripleMu --era 2018
    fi
    echo "Processing data CR..."
    python runner.py \
        --workflow SUEP_coffea_CRs -o "processor_output_files/${tag}_CR" \
        --json filelists/data/data_Run2018A_5p3fb_unskimmed.json \
        --executor futures --chunk 50000 --trigger TripleMu --era 2018
fi
