#!/bin/bash -e

# Need pythia
source source_pythia.sh

era=2018
background_filelist="filelists/mc_collections/2018/QCD_HT_joint_UL18.json"
tag=full_analysis_Dec2025

while getopts 't:' flag; do
  case "${flag}" in
    t) tag="${OPTARG}" ;;
    *) echo "Unexpected option ${flag}" ;;
  esac
done


echo "Processing QCD_HT SRs for ${era}..."
time python runner.py \
    --workflow SUEP_coffea_SRs -o "processor_output_files/${tag}_${era}_SRs" \
    --json $background_filelist --era "$era" --skimmed --isMC --do_syst --do_lhepdfsyst \
    --executor futures -j "$(awk "BEGIN { print int($(nproc) * 0.15) }")" --chunk 6000
echo "Processing QCD_HT CR for ${era}..."
time python runner.py \
    --workflow SUEP_coffea_CRs -o "processor_output_files/${tag}_${era}_CR" \
    --json $background_filelist --era "$era" --skimmed --isMC --do_syst --do_lhepdfsyst \
    --executor futures -j "$(awk "BEGIN { print int($(nproc) * 0.45) }")" --chunk 30000
echo "Processing QCD_HT VR for ${era}..."
time python runner.py \
    --workflow SUEP_coffea_VR -o "processor_output_files/${tag}_${era}_VR" \
    --json $background_filelist --era "$era" --skimmed --isMC \
    --executor futures -j "$(awk "BEGIN { print int($(nproc) * 0.45) }")" --chunk 30000
