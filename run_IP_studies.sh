#!/bin/bash -e

# Need pythia
source source_pythia.sh

# By default, run signal and background

all=1
background=0
data=0
era=2018
tag=IP_studies_Nov2025

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
if [ "$era" = 2018 ]; then
    background_filelist="filelists/mc_collections/2018/SUEPNano_UL18_Nov2024.json"
    data_filelist="filelists/data/2018/DoubleMuon_UL18_Nov2024.json"
else
    echo "Invalid era specified. Available options are:"
    echo -e "\n\t2016, 2016APV, 2017, 2018, 2022, 2022EE, 2023, or 2023BPix.\n"
    exit 1
fi

if [ $background -eq 1 ]; then
    echo "Processing BKG for ${era}..."
    python runner.py \
        --workflow SUEP_coffea_IP_studies -o "processor_output_files/${tag}_${era}" \
        --json $background_filelist --era "$era" --isMC --executor futures \
        -j "$(awk "BEGIN { print int($(nproc) * 1) }")" --chunk 50000
fi

if [ $data -eq 1 ]; then
    echo "Processing data for ${era}..."
    python runner.py \
        --workflow SUEP_coffea_IP_studies -o "processor_output_files/${tag}_${era}" \
        --json $data_filelist --era "$era" --executor futures \
        -j "$(awk "BEGIN { print int($(nproc) * 1) }")" --chunk 100000
fi
