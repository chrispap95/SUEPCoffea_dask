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
extra_commands=0

while getopts 'sbdt:c' flag; do
  case "${flag}" in
    s) all=0; signal=1 ;;
    b) all=0; background=1 ;;
    d) all=0; data=1 ;;
    t) tag="${OPTARG}" ;;
    c) extra_commands=1 ;;
    *) echo "Unexpected option ${flag}" ;;
  esac
done

if [ $all -eq 1 ]; then
    signal=1
    background=1
    data=1
fi

#        --json filelist/SUEP_signal_central_2018_from_mini.json \
        # --json filelist/SUEP_signal_central_2018.json \
if [ $signal -eq 1 ]; then
    echo "Processing signal..."
    python dask/runner.py \
        --workflow SUEP_post_gensum_bug -o "$tag" \
        --json filelist/SUEP_signal_central_2018_working.json \
        --executor futures -j 8 --chunk 10000 \
        --trigger TripleMu --era 2018 --isMC
fi

if [ $background -eq 1 ]; then
        # --json filelist/qcd_mu_enriched_skimmed_merged_new_trigger.json \
        # --json filelist/qcd_muenriched_jul2024.json \
    echo "Processing BKG..."
    python dask/runner.py \
        --workflow SUEP_post_gensum_bug -o "$tag" \
        --json filelist/full_mc_skimmed_merged_new_trigger.json \
        --executor futures -j 8 --chunk 20000 \
        --skimmed --trigger TripleMu \
        --era 2018 --isMC
fi

if [ $data -eq 1 ]; then
        # --json filelist/data_Run2018A_1fb_unskimmed.json \
    echo "Processing data..."
    python dask/runner.py \
        --workflow SUEP_post_gensum_bug -o "$tag" \
        --json filelist/data_Run2018A_5p3fb_unskimmed.json \
        --executor futures --chunk 30000 \
        --trigger TripleMu --era 2018
fi

if [ $extra_commands -eq 1 ]; then
    # Needs +3 to account for the ./ in the beginning and the _ in the end
    tag_length=$((${#tag} + 3))
    for mode in cutflow histograms; do 
        if [ ! -d "plotting/${tag}_output_${mode}" ]; then
            mkdir plotting/${tag}_output_${mode}
        fi
        for i in ./${tag}*_${mode}.pkl; do 
            mv $i plotting/${tag}_output_${mode}/${i:$tag_length}
        done
    done
    for f in ./condor_${tag}*hdf5; do
        rm $f
    done
fi
