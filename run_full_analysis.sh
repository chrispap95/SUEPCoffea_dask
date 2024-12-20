#!/bin/bash -e

# Input options
# -s : signal
# -b : background

# By default, run signal and background

all=1
signal=0
background=0
data=0
tag=full_analysis_Dec2024
blind=0 # 0 for unblinded, 1 for blinded

while getopts 'sbdt:cw:' flag; do
  case "${flag}" in
    s) all=0; signal=1 ;;
    b) all=0; background=1 ;;
    d) all=0; data=1 ;;
    t) tag="${OPTARG}" ;;
    *) echo "Unexpected option ${flag}" ;;
  esac
done

if [ $all -eq 1 ]; then
    signal=1
    background=1
    data=1
fi

if [ $signal -eq 1 ]; then
    echo "Processing signal SRs..."
    python runner.py \
        --workflow SUEP_coffea_SRs -o "processor_output_files/${tag}_SRs" \
        --json filelists/signal/GluGluToSUEP_central_UL18_Nov2024.json --era 2018 \
        --executor dask/lpc --chunk 3000 --skimmed --isMC --do_syst
    echo "Processing signal CR..."
    python runner.py \
        --workflow SUEP_coffea_CRs -o "processor_output_files/${tag}_CR" --do_syst \
        --json filelists/signal/GluGluToSUEP_central_UL18_Nov2024.json --era 2018 \
        --executor dask/lpc --chunk 3000 --skimmed --isMC --do_syst
    echo "Processing signal VR..."
    python runner.py \
        --workflow SUEP_coffea_VR -o "processor_output_files/${tag}_VR" --do_syst \
        --json filelists/signal/GluGluToSUEP_central_UL18_Nov2024.json --era 2018 \
        --executor dask/lpc --chunk 3000 --skimmed --isMC --do_syst
fi

if [ $background -eq 1 ]; then
    echo "Processing BKG SRs..."
    python runner.py \
        --workflow SUEP_coffea_SRs -o "processor_output_files/${tag}_SRs" \
        --json filelists/mc_collections/SUEPNano_UL18_Nov2024.json --era 2018 --do_syst \
        --executor dask/lpc --chunk 3000 --skimmed --isMC
    echo "Processing BKG CR..."
    python runner.py \
        --workflow SUEP_coffea_CRs -o "processor_output_files/${tag}_CR" --do_syst \
        --json filelists/mc_collections/SUEPNano_UL18_Nov2024.json --era 2018 \
        --executor dask/lpc --chunk 3000 --skimmed --isMC --do_syst
    echo "Processing BKG VR..."
    python runner.py \
        --workflow SUEP_coffea_VR -o "processor_output_files/${tag}_VR" --do_syst \
        --json filelists/mc_collections/SUEPNano_UL18_Nov2024.json --era 2018 \
        --executor dask/lpc --chunk 3000 --skimmed --isMC --do_syst
fi

if [ $data -eq 1 ]; then
    if [ $blind -eq 0 ]; then
        echo "Processing data SR high temp..."
        python runner.py \
            --workflow SUEP_coffea_SRs -o "processor_output_files/${tag}_SRs" \
            --json filelists/data/data_Run2018A_0p6fb_1file_unskimmed.json \
            --executor dask/lpc --chunk 3000 --era 2018 --do_syst
    fi
    echo "Processing data CR..."
    python runner.py \
        --workflow SUEP_coffea_CRs -o "processor_output_files/${tag}_CR" \
        --json filelists/data/data_Run2018A_0p6fb_1file_unskimmed.json \
        --executor dask/lpc --chunk 3000 --skimmed --isMC --do_syst
    echo "Processing data VR..."
    python runner.py \
        --workflow SUEP_coffea_VR -o "processor_output_files/${tag}_VR" \
        --json filelists/data/data_Run2018A_0p6fb_1file_unskimmed.json \
        --executor dask/lpc --chunk 3000 --skimmed --isMC --do_syst
fi
