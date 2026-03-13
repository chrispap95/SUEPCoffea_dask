#!/bin/bash -e

# Input options
# -s : signal
# -b : background

# Need pythia
source source_pythia.sh

# By default, run signal and background

all=1
background=0
data=0
era=2018
tag=test_Feb2026
blind=1 # 0 for unblinded, 1 for blinded

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


if [ $background -eq 1 ]; then
    echo "Processing BKG SRs for ${era}..."
    python runner.py \
        --workflow SUEP_coffea_SRs -o "processor_output_files/${tag}_${era}_SRs" \
        --json "filelists/mc_collections/${era}/test.json" --era "$era" --skimmed \
        --isMC --do_syst --do_lhepdfsyst --executor futures -j 4 --chunk 10000
    echo "Processing BKG CR for ${era}..."
    python runner.py \
        --workflow SUEP_coffea_CRs -o "processor_output_files/${tag}_${era}_CR" \
        --json "filelists/mc_collections/${era}/test.json" --era "$era" --skimmed \
        --isMC --do_syst --do_lhepdfsyst --executor iterative --chunk 10000
    python runner.py \
        --workflow SUEP_coffea_VR -o "processor_output_files/${tag}_${era}_VR" \
        --json "filelists/mc_collections/${era}/test.json" --era "$era" --skimmed  \
        --isMC --executor futures -j 4 --chunk 10000
fi

if [ $data -eq 1 ]; then
    if [ $blind -eq 0 ]; then
        echo "Processing data SR for ${era}..."
        python runner.py \
            --workflow SUEP_coffea_SRs -o "processor_output_files/${tag}_${era}_SRs" \
            --json "filelists/data/${era}/test.json" --era "$era" --executor futures -j 4 --chunk 4000
    fi
    echo "Processing data CR for ${era}..."
    python runner.py \
        --workflow SUEP_coffea_CRs -o "processor_output_files/${tag}_${era}_CR" \
        --json "filelists/data/${era}/test.json" --era "$era" --executor futures -j 4 --chunk 4000
    echo "Processing data VR for ${era}..."
    python runner.py \
        --workflow SUEP_coffea_VR -o "processor_output_files/${tag}_${era}_VR" \
        --json "filelists/data/${era}/test.json" --era "$era" --executor futures -j 4 --chunk 4000
fi
