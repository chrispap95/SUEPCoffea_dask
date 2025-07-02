#!/bin/bash -e

source source_pythia.sh

tag=test_norm_May2025

for era in 2016APV 2016 2017; do
    input_file="filelists/data_${era}_local_nano.json"
    # input_file="filelists/JetHT_${era}_local.json"
    # input_file="filelists/SingleMuon_${era}_local.json"

    python runner.py \
        --workflow SUEP_coffea_test_norm -o "processor_output_files/${tag}_${era}" \
        --verbose --json $input_file --era $era \
        --executor futures -j "$(awk "BEGIN { print int($(nproc) * 0.8) }")" --chunk 500000

    # python runner.py \
    #     --workflow SUEP_coffea_test_norm -o "processor_output_files/${tag}_${era}" \
    #     --verbose --json "filelists/qcd_${era}_local_nano.json" --era $era --isMC \
    #     --executor futures -j "$(nproc)" --chunk 1000000
done

# tag=test_norm_skimmed_May2025
# for era in 16APV; do
#     python runner.py \
#         --workflow SUEP_coffea_test_norm -o "processor_output_files/${tag}_20${era}" \
#         --verbose --json "filelists/data/20${era}/DoubleMuon_UL${era}_Nov2024.json" --era "20${era}" \
#         --executor futures -j "$(awk "BEGIN { print int($(nproc) * 0.6) }")" --chunk 200000

#     python runner.py \
#         --workflow SUEP_coffea_test_norm -o "processor_output_files/${tag}_20${era}" \
#         --verbose --json "filelists/mc_collections/20${era}/QCD.json" --era "20${era}" \
#         --skimmed --isMC --executor futures -j "$(awk "BEGIN { print int($(nproc) * 0.6) }")" --chunk 200000
# done
