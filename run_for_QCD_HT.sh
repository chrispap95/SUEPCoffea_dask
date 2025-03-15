#!/bin/bash -e

# Input options
# -s : signal
# -b : background

# By default, run signal and background

all=1
background=0
signal=0
tag=QCD_HT_Feb2025

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
    sleep 0
    # echo "Processing signal SRs..."
    # python runner.py \
    #     --workflow SUEP_coffea_SRs -o "processor_output_files/${tag}_SRs" \
    #     --json filelists/signal/GluGluToSUEP_central_UL18_Nov2024.json --era 2018 \
    #     --executor futures -j 40 --chunk 4000 --skimmed --isMC --do_syst
    # echo "Processing signal CR..."
    # python runner.py \
    #     --workflow SUEP_coffea_CRs -o "processor_output_files/${tag}_CR" --do_syst \
    #     --json filelists/signal/GluGluToSUEP_central_UL18_Nov2024.json --era 2018 \
    #     --executor futures -j 48 --chunk 70000 --skimmed --isMC
    # echo "Processing signal VR..."
    # python runner.py \
    #     --workflow SUEP_coffea_VR -o "processor_output_files/${tag}_VR" --do_syst \
    #     --json filelists/signal/GluGluToSUEP_central_UL18_Nov2024.json --era 2018 \
    #     --executor futures -j 40 --chunk 70000 --skimmed --isMC
fi

if [ $background -eq 1 ]; then
    # echo "Processing BKG SRs..."
    # python runner.py \
    #     --workflow SUEP_coffea_SRs -o "processor_output_files/${tag}_SRs" \
    #     --json filelists/mc_collections/SUEPNano_UL18_Nov2024.json --era 2018 --do_syst \
    #     --executor futures -j 40 --chunk 7000 --skimmed --isMC
    echo "Processing BKG CR..."
    python runner.py \
        --workflow SUEP_coffea_CRs -o "processor_output_files/${tag}_CR" --do_syst \
        --json filelists/mc_processes//QCD_HT_nanoaodsim_copied.json --era 2018 \
        --executor futures -j 48 --chunk 300000 --isMC
    # echo "Processing BKG VR..."
    # python runner.py \
    #     --workflow SUEP_coffea_VR -o "processor_output_files/${tag}_VR" --do_syst \
    #     --json filelists/mc_collections/SUEPNano_UL18_Nov2024.json --era 2018 \
    #     --executor futures -j 40 --chunk 70000 --skimmed --isMC
fi
