#!/bin/bash -e

# Input options
# -s : signal
# -b : background

# By default, run signal and background

all=1
signal=0
background=0
data=0
tag=post_gensum_bug
workflow=SUEP_pgb_scans
extra_commands=0

while getopts 'sbdt:cw:' flag; do
  case "${flag}" in
    s) all=0; signal=1 ;;
    b) all=0; background=1 ;;
    d) all=0; data=1 ;;
    t) tag="${OPTARG}" ;;
    w) workflow="${OPTARG}" ;;
    c) extra_commands=1 ;;
    *) echo "Unexpected option ${flag}" ;;
  esac
done

if [ $all -eq 1 ]; then
    signal=1
    background=1
    data=1
fi

#        --json filelists/signal/SUEP_signal_central_2018_from_mini.json \
        # --json filelists/signal/SUEP_signal_central_2018.json \
if [ $signal -eq 1 ]; then
    echo "Processing signal..."
    python runner.py \
        --workflow "$workflow" -o "$tag" \
        --json filelists/signal/SUEP_signal_central_2018_working.json \
        --executor dask/lpc --chunk 50000 \
        --trigger TripleMu --era 2018 --isMC
fi

if [ $background -eq 1 ]; then
        # --json filelists/mc_processes/qcd_mu_enriched_skimmed_merged_new_trigger.json \
        # --json filelists/mc_processes/qcd_muenriched_jul2024.json \
    echo "Processing BKG..."
    python runner.py \
        --workflow "$workflow" -o "$tag" \
        --json filelists/mc_collections/full_mc_skimmed_merged_new_trigger.json \
        --executor dask/lpc --chunk 100000 \
        --skimmed --trigger TripleMu \
        --era 2018 --isMC
fi

if [ $data -eq 1 ]; then
        # --json filelist/data_Run2018A_1fb_unskimmed.json \
    echo "Processing data..."
    python runner.py \
        --workflow "$workflow" -o "$tag" \
        --json filelists/data/data_Run2018A_5p3fb_unskimmed.json \
        --executor futures --chunk 30000 \
        --trigger TripleMu --era 2018
fi

if [ $extra_commands -eq 1 ]; then
    # Needs +3 to account for the ./ in the beginning and the _ in the end
    tag_length=$((${#tag} + 3))
    for mode in cutflow histograms; do
        if [ ! -d "plotting/${tag}_output_${mode}" ]; then
            mkdir "plotting/${tag}_output_${mode}"
        fi
        for i in ./"${tag}"*_"${mode}".pkl; do
            mv "$i" "plotting/${tag}_output_${mode}/${i:$tag_length}"
        done
    done
    # for f in ./condor_"${tag}"*hdf5; do
    #     rm "$f"
    # done
fi
