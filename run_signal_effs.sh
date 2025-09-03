#!/bin/bash -e

# Input options
# -s : signal
# -b : background

# Need pythia
source source_pythia.sh

# By default, run signal and background

era=2018
tag=signal_effs_Jul2025

while getopts 'e:t:' flag; do
  case "${flag}" in
    e) era="${OPTARG}" ;;
    t) tag="${OPTARG}" ;;
    *) echo "Unexpected option ${flag}" ;;
  esac
done

# Set files according to the year
if [ "$era" = 2016 ]; then
    signal_filelist="filelists/signal/2016/GluGluToSUEP_central_UL16_May2025.json"
elif [ "$era" = 2016APV ]; then
    signal_filelist="filelists/signal/2016APV/GluGluToSUEP_central_UL16APV_May2025.json"
elif [ "$era" = 2017 ]; then
    signal_filelist="filelists/signal/2017/GluGluToSUEP_central_UL17_May2025.json"
elif [ "$era" = 2018 ]; then
    signal_filelist="filelists/signal/2018/GluGluToSUEP_central_UL18_May2025.json"
elif [ "$era" = 2022 ]; then
    signal_filelist="filelists/signal/2022/GluGluToSUEP_central_2022_May2025.json"
elif [ "$era" = 2022EE ]; then
    signal_filelist="filelists/signal/2022EE/GluGluToSUEP_central_2022EE_May2025.json"
elif [ "$era" = 2023 ]; then
    signal_filelist="filelists/signal/2023/GluGluToSUEP_central_2023_May2025.json"
elif [ "$era" = 2023BPix ]; then
    signal_filelist="filelists/signal/2023BPix/GluGluToSUEP_central_2023BPix_May2025.json"
else
    echo "Invalid era specified. Available options are:"
    echo -e "\n\t2016, 2016APV, 2017, 2018, 2022, 2022EE, 2023, or 2023BPix.\n"
    exit 1
fi

echo "Processing signal CR for ${era}..."
python runner.py \
    --workflow SUEP_coffea_signal_effs -o "processor_output_files/${tag}_${era}" \
    --json $signal_filelist --era "$era" --skimmed --isMC \
    --executor futures -j "$(awk "BEGIN { print int($(nproc) * 0.8) }")" --chunk 40000
