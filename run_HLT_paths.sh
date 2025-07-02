#!/bin/bash -e
tag=HLT_paths_Apr2025

for era in UL16 UL16APV UL17 UL18; do #2022 2022EE 2023 2023BPix; do
    python runner.py \
        --workflow SUEP_coffea_HLT_paths -o "processor_output_files/${tag}_${era}" \
        --json "filelists/mc_processes/QCD_Pt_MuEnrichedPt5_NanoAOD_copied_${era}.json" \
        --era "$era" --isMC --executor futures -j 48 --chunk 400000
done
