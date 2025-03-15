#!/bin/bash -e

# Input options
# -s : signal
# -b : background

# By default, run signal and background

all=1
signal=0
background=0
tag=VR_quantities_Jan2025

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
    echo "Processing signal VR..."
    python runner.py \
        --workflow SUEP_coffea_VR_quantities -o "processor_output_files/${tag}_VR"\
        --json filelists/signal/GluGluToSUEP_central_UL18_Nov2024.json --era 2018 \
        --executor futures -j 44 --chunk 80000 --skimmed --isMC
fi

if [ $background -eq 1 ]; then
    echo "Processing BKG VR..."
    python runner.py \
        --workflow SUEP_coffea_VR_quantities -o "processor_output_files/${tag}_VR"\
        --json filelists/mc_collections/SUEPNano_UL18_Nov2024.json --era 2018 \
        --executor futures -j 44 --chunk 80000 --skimmed --isMC
fi
