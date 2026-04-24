#!/bin/bash -e

# Input options
# -s : signal
# -b : background

# Need pythia
source source_pythia.sh

# By default, run signal and background

all=1
signal=0
background=0
era=2018
tag=full_analysis_noIPcorr_Apr2026

while getopts 'sbe:t:' flag; do
  case "${flag}" in
    s) all=0; signal=1 ;;
    b) all=0; background=1 ;;
    e) era="${OPTARG}" ;;
    t) tag="${OPTARG}" ;;
    *) echo "Unexpected option ${flag}" ;;
  esac
done

if [ $all -eq 1 ]; then
    signal=1
    background=1
fi

# Set files according to the year
if [ "$era" = 2016 ]; then
    signal_filelist="filelists/signal/2016/GluGluToSUEP_central_UL16_May2025.json"
    background_filelist="filelists/mc_collections/2016/SUEPNano_UL16_Nov2024.json"
elif [ "$era" = 2016APV ]; then
    signal_filelist="filelists/signal/2016APV/GluGluToSUEP_central_UL16APV_May2025.json"
    background_filelist="filelists/mc_collections/2016APV/SUEPNano_UL16APV_Nov2024.json"
elif [ "$era" = 2017 ]; then
    signal_filelist="filelists/signal/2017/GluGluToSUEP_central_UL17_May2025.json"
    background_filelist="filelists/mc_collections/2017/SUEPNano_UL17_Nov2024.json"
elif [ "$era" = 2018 ]; then
    signal_filelist="filelists/signal/2018/GluGluToSUEP_central_UL18_May2025.json"
    background_filelist="filelists/mc_collections/2018/SUEPNano_UL18_Nov2024.json"
elif [ "$era" = 2022 ]; then
    signal_filelist="filelists/signal/2022/GluGluToSUEP_central_2022_May2025.json"
    background_filelist="filelists/mc_collections/2022/SUEPNano_2022_full.json"
elif [ "$era" = 2022EE ]; then
    signal_filelist="filelists/signal/2022EE/GluGluToSUEP_central_2022EE_May2025.json"
    background_filelist="filelists/mc_collections/2022EE/SUEPNano_2022EE_full.json"
elif [ "$era" = 2023 ]; then
    signal_filelist="filelists/signal/2023/GluGluToSUEP_central_2023_May2025.json"
    background_filelist="filelists/mc_collections/2023/SUEPNano_2023_full.json"
elif [ "$era" = 2023BPix ]; then
    signal_filelist="filelists/signal/2023BPix/GluGluToSUEP_central_2023BPix_May2025.json"
    background_filelist="filelists/mc_collections/2023BPix/SUEPNano_2023BPix_full.json"
else
    echo "Invalid era specified. Available options are:"
    echo -e "\n\t2016, 2016APV, 2017, 2018, 2022, 2022EE, 2023, or 2023BPix.\n"
    exit 1
fi


if [ $signal -eq 1 ]; then
    echo "Processing signal SRs for ${era}..."
    python runner.py \
        --workflow SUEP_coffea_SRs_noIPcorr -o "processor_output_files/${tag}_${era}_SRs" \
        --json $signal_filelist --era "$era" --skimmed --isMC \
        --executor dask/lpc --max-scaleout 100 --memory 8GB --chunk 5000
    echo "Processing signal CR for ${era}..."
    python runner.py \
        --workflow SUEP_coffea_CRs_noIPcorr -o "processor_output_files/${tag}_${era}_CR" \
        --json $signal_filelist --era "$era" --skimmed --isMC \
        --executor dask/lpc --mild-scaleout --max-scaleout 100 --memory 8GB --chunk 8000
    echo "Processing signal VR for ${era}..."
    python runner.py \
        --workflow SUEP_coffea_VR_noIPcorr -o "processor_output_files/${tag}_${era}_VR" \
        --json $signal_filelist --era "$era" --skimmed --isMC \
        --executor dask/lpc --max-scaleout 100 --memory 8GB --chunk 10000
fi

if [ $background -eq 1 ]; then
    echo "Processing BKG SRs for ${era}..."
    python runner.py \
        --workflow SUEP_coffea_SRs_noIPcorr -o "processor_output_files/${tag}_${era}_SRs" \
        --json $background_filelist --era "$era" --skimmed --isMC \
        --executor dask/lpc --max-scaleout 100 --memory 4GB --chunk 10000
    echo "Processing BKG CR for ${era}..."
    python runner.py \
        --workflow SUEP_coffea_CRs_noIPcorr -o "processor_output_files/${tag}_${era}_CR" \
        --json $background_filelist --era "$era" --skimmed --isMC \
        --executor dask/lpc --mild-scaleout --max-scaleout 100 --memory 4GB --chunk 10000
    echo "Processing BKG VR for ${era}..."
    python runner.py \
        --workflow SUEP_coffea_VR_noIPcorr -o "processor_output_files/${tag}_${era}_VR" \
        --json $background_filelist --era "$era" --skimmed --isMC \
        --executor dask/lpc --max-scaleout 100 --memory 4GB --chunk 20000
fi
