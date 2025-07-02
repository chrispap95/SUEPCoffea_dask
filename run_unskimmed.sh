#!/bin/bash -e

# Need pythia
source source_pythia.sh

era=2018
tag=full_analysis_12_10_5_unskimmed_June2025

while getopts 'e:t:' flag; do
  case "${flag}" in
    e) era="${OPTARG}" ;;
    t) tag="${OPTARG}" ;;
    *) echo "Unexpected option ${flag}" ;;
  esac
done

# Set files according to the year
if [ "$era" = 2016 ]; then
    data_filelist="filelists/data_2016_local_nano.json"
elif [ "$era" = 2016APV ]; then
    data_filelist="filelists/data_2016APV_local_nano.json"
elif [ "$era" = 2017 ]; then
    data_filelist="filelists/data_2017_local_nano.json"
else
    echo "Invalid era specified. Available options are:"
    echo -e "\n\t2016, 2016APV, or 2017.\n"
    exit 1
fi

# Copy original output directory
output_dir="processor_output_files/${tag}_${era}_CR_output_histograms"
cp -r "${output_dir/_unskimmed/}" "$output_dir"

echo "Processing unskimmed data for ${era}..."
python runner.py \
    --workflow SUEP_coffea_CRs -o "processor_output_files/${tag}_${era}_CR" --json $data_filelist \
    --era "$era" --executor futures -j "$(awk "BEGIN { print int($(nproc) * 0.5) }")" --chunk 400000
