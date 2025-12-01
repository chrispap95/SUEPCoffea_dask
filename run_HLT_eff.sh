#!/bin/bash -e

# Need pythia
source source_pythia.sh

# By default, run signal and background

all=1
background=0
data=0
era=2018
tag=HLT_eff_Dec2025

while getopts 'bde:t:' flag; do
  case "${flag}" in
    b) all=0; background=1 ;;
    d) all=0; data=1 ;;
    e) era="${OPTARG}" ;;
    t) tag="${OPTARG}" ;;
    *) echo "Unexpected option ${flag}" ;;
  esac
done

if [ $all -eq 1 ]; then
    background=1
    data=1
fi

# Set files according to the year
background_filelist="filelists/unskimmed_samples/qcd_${era}_local_nano.json"
data_filelist="filelists/unskimmed_samples/DoubleMuon_${era}_local_nano.json"

if [ $background -eq 1 ]; then
    echo "Processing BKG for ${era}..."
    python runner.py \
        --workflow SUEP_coffea_HLT_eff -o "processor_output_files/${tag}_${era}" \
        --json "$background_filelist" --era "$era" --isMC --executor futures \
        -j "$(awk "BEGIN { print int($(nproc) * 1) }")" --chunk 150000
fi

if [ $data -eq 1 ]; then
    echo "Processing data for ${era}..."
    python runner.py \
        --workflow SUEP_coffea_HLT_eff -o "processor_output_files/${tag}_${era}" \
        --json "$data_filelist" --era "$era" --executor futures \
        -j "$(awk "BEGIN { print int($(nproc) * 1) }")" --chunk 150000
fi
