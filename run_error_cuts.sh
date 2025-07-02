#!/bin/bash -e

# Need pythia
source source_pythia.sh

era=2018
tag=error_cuts_Jun2025

while getopts 'e:t:' flag; do
  case "${flag}" in
    e) era="${OPTARG}" ;;
    t) tag="${OPTARG}" ;;
    *) echo "Unexpected option ${flag}" ;;
  esac
done

# Set files according to the year
if [ "$era" = 2016 ]; then
    background_filelist="filelists/mc_collections/2016/SUEPNano_UL16_Nov2024.json"
    data_filelist="filelists/data/2016/DoubleMuon_UL16_Nov2024.json"
elif [ "$era" = 2016APV ]; then
    background_filelist="filelists/mc_collections/2016APV/SUEPNano_UL16APV_Nov2024.json"
    data_filelist="filelists/data/2016APV/DoubleMuon_UL16APV_Nov2024.json"
elif [ "$era" = 2017 ]; then
    background_filelist="filelists/mc_collections/2017/SUEPNano_UL17_Nov2024.json"
    data_filelist="filelists/data/2017/DoubleMuon_UL17_Nov2024.json"
elif [ "$era" = 2018 ]; then
    background_filelist="filelists/mc_collections/2018/SUEPNano_UL18_Nov2024.json"
    data_filelist="filelists/data/2018/DoubleMuon_UL18_Nov2024.json"
elif [ "$era" = 2022 ]; then
    background_filelist="filelists/mc_collections/2022/SUEPNano_2022_full.json"
    data_filelist="filelists/data/2022/DoubleMuon_Muon_2022_Apr2025.json"
elif [ "$era" = 2022EE ]; then
    background_filelist="filelists/mc_collections/2022EE/SUEPNano_2022EE_full.json"
    data_filelist="filelists/data/2022EE/Muon_2022EE_Apr2025.json"
elif [ "$era" = 2023 ]; then
    background_filelist="filelists/mc_collections/2023/SUEPNano_2023_full.json"
    data_filelist="filelists/data/2023/Muon0_Muon1_2023_Apr2025.json"
elif [ "$era" = 2023BPix ]; then
    background_filelist="filelists/mc_collections/2023BPix/SUEPNano_2023BPix_full.json"
    data_filelist="filelists/data/2023BPix/Muon0_Muon1_2023BPix_Apr2025.json"
else
    echo "Invalid era specified. Available options are:"
    echo -e "\n\t2016, 2016APV, 2017, 2018, 2022, 2022EE, 2023, or 2023BPix.\n"
    exit 1
fi

echo "Processing BKG for ${era}..."
python runner.py \
    --workflow SUEP_coffea_CRs_error_cuts -o "processor_output_files/${tag}_${era}" \
    --json $background_filelist --era "$era" --skimmed --isMC \
    --executor futures -j "$(nproc)" --chunk 35000

echo "Processing data for ${era}..."
python runner.py \
    --workflow SUEP_coffea_CRs_error_cuts -o "processor_output_files/${tag}_${era}" \
    --json $data_filelist --era "$era" --executor futures \
    -j "$(nproc)" --chunk 70000
