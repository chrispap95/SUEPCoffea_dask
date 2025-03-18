#!/bin/bash -e

tag=scans_Mar2025

while getopts 't:' flag; do
  case "${flag}" in
    t) tag="${OPTARG}" ;;
    *) echo "Unexpected option ${flag}" ;;
  esac
done

echo "Processing BKG SR_low_temp..."
python runner.py \
    --workflow SUEP_coffea_SR_low_temp_scans -o "processor_output_files/${tag}_SR_low_temp" \
    --json filelists/mc_collections/SUEPNano_UL18_Nov2024_QCD_DY.json --era 2018 --skimmed \
    --isMC --executor futures -j 48 --chunk 7000

echo "Processing BKG SR_high_temp..."
python runner.py \
    --workflow SUEP_coffea_SR_high_temp_scans -o "processor_output_files/${tag}_SR_high_temp" \
    --json filelists/mc_collections/SUEPNano_UL18_Nov2024_QCD_DY.json --era 2018 --skimmed \
    --isMC --executor futures -j 48 --chunk 60000
