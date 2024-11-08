#!/bin/bash -e

tag=gen_bug_comparisons
workflow=SUEP_gen_bug_comparisons
extra_commands=0

while getopts 'sbdt:cw:' flag; do
  case "${flag}" in
    t) tag="${OPTARG}" ;;
    w) workflow="${OPTARG}" ;;
    c) extra_commands=1 ;;
    *) echo "Unexpected option ${flag}" ;;
  esac
done

echo "Processing signal..."
python runner.py \
    --workflow "$workflow" -o "$tag" \
    --json filelists/signal/signal_bug_comparisons/SUEP_after_bugfix.json \
    --executor futures -j 8 --chunk 50000 \
    --trigger TripleMu --era 2018 --isMC

echo "Processing signal..."
python runner.py \
    --workflow "$workflow" -o "$tag" \
    --json filelists/signal/signal_bug_comparisons/SUEP_before_bugfix.json \
    --executor futures -j 8 --chunk 50000 \
    --trigger TripleMu --era 2018 --isMC

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
fi
